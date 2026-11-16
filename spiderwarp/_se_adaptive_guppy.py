"""Standalone Guppy emitter for an already certified adaptive decision graph."""

from __future__ import annotations

import hashlib
import importlib.util
from itertools import groupby
from collections import deque
from pathlib import Path
from pprint import pformat
import sys
import tempfile

from spiderwarp._se_decisions import XorDecisions
from spiderwarp._se_circuit import row_mask


COMPACT_POLICY_THRESHOLD = 1000


def factor_masks(table):
    """Express masks in a basis chosen by first independent table value.

    Keeping the original basis vectors makes the coefficient table invariant
    under changes of output coordinates. Physical frame masks remain exact.
    """
    vectors, pivots, coefficients = [], {}, {}
    for key, original in sorted(table.items()):
        value, coefficient = original, 0
        for pivot, (row, representation) in sorted(pivots.items(), reverse=True):
            if (value >> pivot) & 1:
                value ^= row
                coefficient ^= representation
        if value:
            bit = 1 << len(vectors)
            vectors.append(original)
            pivots[value.bit_length() - 1] = value, coefficient ^ bit
            coefficient = bit
        coefficients[key] = coefficient
    return tuple(vectors), coefficients


def generate_adaptive_source(protocol, *, table_writer=None):
    """Choose between shared direct decoding and standard decoding plus patches.

    Compare the complete constant payloads: a local improvement can destroy
    sharing elsewhere. Keep this decision at build time, never at runtime.
    """
    import json
    candidates = []
    for baseline in (False, True):
        captured = []

        def capture(tables):
            captured.append(tables)
            return '# SELECTED_DECODER_TABLES'

        source = _generate_adaptive_source(protocol, baseline=baseline, table_writer=capture)
        tables = captured[0]
        size = len(json.dumps(tables, separators=(',', ':')))
        candidates.append((size, source, tables))
    _, source, tables = min(candidates, key=lambda item: item[0])
    data = (table_writer(tables) if table_writer else '\n\n'.join(
        f'{name} = {pformat(value, width=100, compact=True)}' for name, value in tables.items()))
    sizes = f'DIRECT_TABLE_BYTES = {candidates[0][0]}\nPATCHED_TABLE_BYTES = {candidates[1][0]}\n'
    return source.replace('# SELECTED_DECODER_TABLES', sizes + data)


def _generate_adaptive_source(protocol, *, baseline, table_writer):
    policy, code = protocol.policy, protocol.code
    if not policy.report.verified:
        raise ValueError("Verify the adaptive policy before exporting")
    if code.n > 63:
        raise NotImplementedError("Guppy frame masks support at most 63 data qubits")
    used_actions = {node.action for node in policy.nodes if node.kind == 'extract'}
    actions = sorted((name, action) for name, action in policy.actions.items()
                     if name in used_actions or not used_actions)
    ids = {name: i for i, (name, _) in enumerate(actions)}
    if any(a.gadget.num_detectors > 63 for _, a in actions):
        raise NotImplementedError("Guppy outcome masks support at most 63 combined flags/checks")
    metadata = dict(
        N_DATA=code.n,
        DECODER_BASELINE_MODE='standard-plus-deltas' if baseline else 'shared-direct',
        CODE_DISTANCE=code.d,
        MANDATORY_ACCEPTANCE_ORDER=policy.contract.correctable,
        CLEAN_INPUT_ACCEPTANCE_ORDER=policy.contract.accept_clean_through,
        LOGICAL_FAULT_ORDER=policy.contract.logical_order,
        ANALYZED_FAULT_ORDER=policy.contract.fault_order,
        ACCEPTED_FILTER_ORDER=policy.contract.accepted_filter_order,
        INPUT_SCOPE=policy.contract.input_scope,
        VERIFIED_AT_BUILD=policy.report.verified,
        WORST_CASE_CNOTS=policy.report.worst_case_cnots,
        LOWER_BOUND_CNOTS=policy.report.lower_bound_cnots,
        DISTANCE_LADDER="label >= max(1, 2 * maximum_remaining_faults)",
        PEAK_QUBITS=policy.report.peak_qubits,
        MAX_EXTRACTIONS=policy.report.max_extractions,
        MINIMAX_OPTIMAL=policy.optimal,
        ROOT_NODE=0,
        IDEAL_DECODER_HASH=policy.decoder.fingerprint,
        VERIFICATION_MODEL=policy.report.noise_model,
        CIRCUIT_HASHES={name: hashlib.sha256(str(a.gadget.circuit).encode()).hexdigest()
                        for name, a in actions},
        ACTION_NAMES=tuple(name for name, _ in actions),
        ACTION_FLAG_WIDTHS=[a.gadget.num_flags for _, a in actions],
        ACTION_IS_Z=[int(a.gadget.basis == "Z") for _, a in actions],
        FRAME_X_CHECK_ROWS=[row_mask(row) for row in code.H_z],
        FRAME_Z_CHECK_ROWS=[row_mask(row) for row in code.H_x],
    )
    circuits = []

    def call(prefix, values):
        operands = pformat(values, width=max(40, 96 - len(prefix)), compact=True)
        first, *rest = operands.splitlines()
        statement = prefix + first + ''.join('\n' + ' ' * len(prefix) + s for s in rest) + ')'
        circuits.extend('    ' + s for s in statement.splitlines())

    arg = 'data: array[qubit, comptime(N_DATA)]'
    for i, (_, action) in enumerate(actions):
        g = action.gadget
        circuits.extend(['@guppy.comptime', f'def run_action_{i}({arg}) -> ExtractionResult:',
                         '    register = dict(enumerate(data))', '    measurements = []'])
        for name, operations in groupby(g.operations, key=lambda op: op[0]):
            targets = tuple(qs if len(qs) == 2 else qs[0] for _, qs in operations)
            if name == 'R':
                call('prepare(register, ', targets)
            elif name == 'M':
                call('read(register, measurements, ', targets)
            elif name in {'H', 'X', 'Y', 'Z', 'CX', 'CZ'}:
                call(f'apply({name.lower()}, register, ', targets)
            else:
                raise ValueError(f'Basis-lower the action before export: {name}')
        call('flags = pack_parities(measurements, ', g.readout.flags)
        call('syndrome = pack_parities(measurements, ', g.readout.syndromes)
        circuits.extend(['    return ExtractionResult(flags, syndrome)', '', ''])
    circuits.extend(['@guppy', f'def execute_action({arg}, action: int) -> ExtractionResult:'])
    for i in range(len(actions)):
        circuits.extend([f'    if action == {i}:', f'        return run_action_{i}(data)'])
    circuits.extend(['    return ExtractionResult(-1, 0)', ''])

    decisions = XorDecisions()
    roots = []
    # Every acceptance can share one state: its correction belongs on the edge.
    node_rows = [(0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0)]
    terminal_tables = []
    terminal_ids = {}
    bases, basis_ids = [()], {(): 0}
    control_specs = []

    def root_id(table, width, *, partial=False):
        roots.append(decisions.build(table, width, partial=partial))
        return len(roots) - 1

    def basis_id(vectors):
        if vectors not in basis_ids:
            basis_ids[vectors] = len(bases)
            bases.append(vectors)
        return basis_ids[vectors]

    def correction_table(table, action):
        """Choose exact corrections or standard-syndrome corrections plus deltas."""
        def factor(values):
            offset = values[min(values)] if values else 0
            xb, xs = factor_masks({key: (frame ^ offset) & ((1 << code.n) - 1)
                                   for key, frame in values.items()})
            zb, zs = factor_masks({key: (frame ^ offset) >> code.n
                                   for key, frame in values.items()})
            cost = len(xb) + len(zb)
            if baseline:
                trial = XorDecisions()
                for coefficients in (xs, zs):
                    trial.build(coefficients, max(1, action.gadget.num_detectors), partial=True)
                cost += len(trial.nodes)
            return cost, offset, xb, xs, zb, zs

        direct = factor(table)
        if not baseline:
            return False, direct[1:]
        component = 'X' if action.gadget.basis == 'Z' else 'Z'
        standard = policy.decoder.classes[component]
        patched = factor({key: frame ^ standard.get(key >> action.gadget.num_flags, 0)
                          for key, frame in table.items()})
        # Account for the extra baseline lookup. Ties keep the direct decoder.
        use_baseline = patched[0] + 1 < direct[0]
        return use_baseline, (patched if use_baseline else direct)[1:]

    baseline_x = root_id({s: frame & ((1 << code.n) - 1)
                          for s, frame in policy.decoder.classes['X'].items()}, len(code.H_z))
    baseline_z = root_id({s: frame >> code.n for s, frame in policy.decoder.classes['Z'].items()},
                         len(code.H_x))
    large_export = len(policy.nodes) >= COMPACT_POLICY_THRESHOLD
    # Terminal extraction tables already contain their accept frames. Those
    # accept nodes must not also occupy a large, unreachable Guppy constant.
    reachable, pending = {}, deque([policy.root])
    while pending:
        index = pending.popleft()
        while policy.nodes[index].kind == 'frame':
            index = policy.nodes[index].next
        if index in reachable:
            continue
        reachable[index] = len(reachable)
        node = policy.nodes[index]
        if node.kind == 'extract':
            targets = []
            for outcome, child in sorted(node.children.items(), key=lambda item: (item[0][1], item[0][0])):
                while policy.nodes[child].kind == 'frame':
                    child = policy.nodes[child].next
                targets.append(child)
            if not all(policy.nodes[child].kind in {'accept', 'reject'} for child in targets):
                pending.extend(targets)
    exported_ids = {old: new for new, old in enumerate(
        (i for i in reachable if policy.nodes[i].kind == 'extract'), 2)}

    def successor(index):
        frame = 0
        while policy.nodes[index].kind == "frame":
            frame ^= policy.nodes[index].frame
            index = policy.nodes[index].next
        target = policy.nodes[index]
        exported = 1 if target.kind == 'accept' else exported_ids.get(index, 0)
        return exported, frame, target

    root, root_frame, target = successor(policy.root)
    if target.kind == 'accept':
        root_frame ^= target.frame
    metadata.update(ROOT_NODE=root, ROOT_FRAME_X=root_frame & ((1 << code.n) - 1),
                    ROOT_FRAME_Z=root_frame >> code.n)
    for index in exported_ids:
        node = policy.nodes[index]
        if node.kind == 'extract':
            action = policy.actions[node.action]
            width = action.gadget.num_flags
            successors = {outcome: successor(child) for outcome, child in node.children.items()}
            terminal = all(target.kind in {'accept', 'reject'}
                           for _, _, target in successors.values())
            if terminal:
                table = {f | (s << width): frame ^ target.frame
                         for (f, s), (_, frame, target) in successors.items()
                         if target.kind == 'accept'}
                use_baseline, (offset, xb, xs, zb, zs) = correction_table(table, action)
                normalized = tuple((key, xs[key], zs[key]) for key in sorted(table))
                spec = max(1, action.gadget.num_detectors), normalized
                if spec not in terminal_ids:
                    terminal_ids[spec] = len(terminal_tables)
                    terminal_tables.append(spec)
                node_rows.append((3 + 4 * use_baseline, ids[node.action], terminal_ids[spec],
                                  offset & ((1 << code.n) - 1), offset >> code.n,
                                  basis_id(xb), basis_id(zb)))
            else:
                table = {f | (s << width): (child, frame ^ (target.frame if target.kind == 'accept' else 0), target)
                         for (f, s), (child, frame, target) in successors.items() if target.kind != 'reject'}
                input_width = max(1, action.gadget.num_detectors)
                first = table[min(table)] if table else (0, 0, None)
                offset, frame_offset = first[:2]
                accepted = root_id({key: 1 for key in table}, input_width)
                normalized = {key: child ^ offset for key, (child, _, _) in table.items()}
                active_bits = 0
                for child in normalized.values():
                    active_bits |= child
                bits = [(bit, root_id({key: (child >> bit) & 1 for key, child in normalized.items()},
                                      input_width, partial=True))
                        for bit in range(active_bits.bit_length()) if (active_bits >> bit) & 1]
                use_baseline, (frame_offset, xb, xs, zb, zs) = correction_table(
                    {key: frame for key, (_, frame, _) in table.items()}, action)
                control_specs.append((accepted, offset, tuple(bits),
                                      root_id(xs, input_width, partial=True), root_id(zs, input_width, partial=True)))
                node_rows.append((2 + 4 * use_baseline, ids[node.action], len(control_specs) - 1,
                                  frame_offset & ((1 << code.n) - 1), frame_offset >> code.n,
                                  basis_id(xb), basis_id(zb)))
    control_decisions = len(decisions.nodes)
    decoder_start = len(roots)
    for width, entries in terminal_tables:
        roots.extend(decisions.build(values, width, partial=i > 0) for i, values in enumerate((
            {key: 1 for key, _, _ in entries},
            {key: x for key, x, _ in entries},
            {key: z for key, _, z in entries},
        )))
    rows, references, deltas, decision_metadata = decisions.compact(roots)
    terminal_roots = [tuple(references[i:i + 3])
                      for i in range(decoder_start, len(references), 3)]
    control_rows, control_bits, bit_lists = [], [], {}
    control_frames = []
    for accepted, offset, bits, xs, zs in control_specs:
        shared = tuple((bit, references[root]) for bit, root in bits)
        if shared not in bit_lists:
            bit_lists[shared] = len(control_bits), len(shared)
            control_bits.extend(shared)
        start, count = bit_lists[shared]
        control_rows.append((references[accepted], offset, start, count))
        control_frames.append((references[xs], references[zs]))
    basis_specs, basis_vectors = [], []
    for vectors in bases:
        basis_specs.append((len(basis_vectors), len(vectors)))
        basis_vectors.extend(vectors)
    metadata.update(N_POLICY_NODES=len(node_rows), N_DECISION_NODES=len(rows),
                    N_TERMINAL_DECODERS=len(terminal_roots), N_FRAME_BASES=len(bases),
                    N_CONTROL_DECISIONS_BEFORE_COMPACTION=control_decisions,
                    N_CONTROL_STATES=sum(row[0] & 3 == 2 for row in node_rows),
                    N_BASELINE_STATES=sum(row[0] >= 4 for row in node_rows),
                    BASELINE_X_ROOT=references[baseline_x], BASELINE_Z_ROOT=references[baseline_z])
    metadata.update(decision_metadata)
    if decision_metadata['DECISION_ROWS_PACKED']:
        unpack_decision = """        word = decisions[node - 1]
        bit = word & comptime(DECISION_BIT_MASK)
        low = (word >> comptime(DECISION_BIT_BITS)) & comptime(DECISION_REFERENCE_MASK)
        high = (word >> comptime(DECISION_HIGH_SHIFT)) & comptime(DECISION_REFERENCE_MASK)
        delta = word >> comptime(DECISION_DELTA_SHIFT)"""
    else:
        unpack_decision = '        bit, low, high, delta = decisions[node - 1]'
    if large_export:
        # Pool the repeated frame offsets; pack the remaining small fields.
        columns = (0, 1, 2, 5, 6)
        widths = [max(1, max(row[c] for row in node_rows).bit_length()) for c in columns]
        if sum(widths) > 63:
            raise ValueError('Packed policy controls exceed a signed machine word')
        shifts, shift = [], 0
        for width in widths:
            shifts.append(shift)
            shift += width
        frame_offsets = list(dict.fromkeys((row[3], row[4]) for row in node_rows))
        frame_ids = {frame: i for i, frame in enumerate(frame_offsets)}
        controls = list(dict.fromkeys(sum(row[c] << s for c, s in zip(columns, shifts))
                                      for row in node_rows))
        control_ids = {word: i for i, word in enumerate(controls)}
        policy_control_bits = max(1, (len(controls) - 1).bit_length())
        if policy_control_bits + max(1, (len(frame_offsets) - 1).bit_length()) > 63:
            raise ValueError('Packed policy references exceed a signed machine word')
        node_rows = [(control_ids[sum(row[c] << s for c, s in zip(columns, shifts))]
                      | (frame_ids[row[3], row[4]] << policy_control_bits)) for row in node_rows]
        metadata['POLICY_ROWS_PACKED'] = True
        metadata.update(POLICY_CONTROL_BITS=policy_control_bits, POLICY_CONTROL_MASK=(1 << policy_control_bits) - 1)
        unpack_policy = ['    nodes = comptime(POLICY_NODES)',
                         '    controls = comptime(POLICY_CONTROLS)',
                         '    offsets = comptime(POLICY_FRAME_OFFSETS)',
                         '    ref = nodes[node]',
                         '    word = controls[ref & comptime(POLICY_CONTROL_MASK)]',
                         '    frame_x, frame_z = offsets[ref >> comptime(POLICY_CONTROL_BITS)]']
        for field, shift, width in zip(('kind', 'action', 'transition', 'basis_x', 'basis_z'), shifts, widths):
            metadata['POLICY_' + field.upper() + '_SHIFT'] = shift
            metadata['POLICY_' + field.upper() + '_MASK'] = (1 << width) - 1
            unpack_policy.append(f'    {field} = (word >> comptime(POLICY_{field.upper()}_SHIFT)) & comptime(POLICY_{field.upper()}_MASK)')
        unpack_policy.append('    return kind, action, transition, frame_x, frame_z, basis_x, basis_z')
    else:
        controls, frame_offsets = [0], [(0, 0)]
        unpack_policy = ['    nodes = comptime(POLICY_NODES)', '    return nodes[node]']
    empty_decisions = [0] if decision_metadata['DECISION_ROWS_PACKED'] else [(0, 0, 0, 0)]
    table_data = dict(
        POLICY_NODES=node_rows,
        POLICY_CONTROLS=controls,
        POLICY_FRAME_OFFSETS=frame_offsets,
        TERMINAL_DECODERS=terminal_roots or [(-1, -1, -1)],
        CONTROL_SPECS=control_rows or [(-1, 0, 0, 0)],
        CONTROL_FRAMES=control_frames or [(-1, -1)],
        CONTROL_BITS=control_bits or [(0, -1)],
        FRAME_BASIS_SPECS=basis_specs,
        FRAME_BASIS_VECTORS=basis_vectors or [0],
        DECISION_DELTAS=deltas,
        DECISION_NODES=rows or empty_decisions,
    )
    tables = table_writer(table_data)
    template = Path(__file__).with_name('_se_adaptive_guppy_template.py.in').read_text()
    template = template.replace('# GENERATED_POLICY_ROW', '\n'.join(unpack_policy))
    return (template.replace('# GENERATED_METADATA', '\n'.join(
                f'{name} = {value!r}' for name, value in metadata.items()))
            .replace('# GENERATED_DECISION_ROW', unpack_decision)
            .replace('# GENERATED_GADGETS', '\n'.join(circuits))
            .replace('# GENERATED_TABLES', tables))


def compile_adaptive_protocol(protocol):
    return compile_adaptive_source(generate_adaptive_source(protocol))


def write_adaptive_source(protocol, path):
    """Write readable standalone code plus hash-checked compile-time data.

    Only Python's standard library reads the sidecar. HUGR embeds the decoded
    constants and never opens a file, decompresses data, or calls Python.
    """
    import json
    import zlib
    from spiderwarp.se_protocol_io import atomic_write
    path = Path(path)

    def write_tables(tables):
        payload = zlib.compress(json.dumps(tables, separators=(',', ':')).encode(), level=9)
        sha = hashlib.sha256(payload).hexdigest()
        name = f'steane_se_tables_{sha[:16]}.json.zlib'
        atomic_write(path.with_name(name), payload)
        return f'''# BEGIN_COMPILE_TIME_TABLES
# Immutable decoder data, loaded once by Python when compiling this module.
# The compiled HUGR contains these constants and needs no companion file.
TABLES_FILE = {name!r}
TABLES_SHA256 = {sha!r}

from pathlib import Path as _Path
import hashlib as _hashlib
import json as _json
import zlib as _zlib

_payload = _Path(__file__).with_name(TABLES_FILE).read_bytes()
if _hashlib.sha256(_payload).hexdigest() != TABLES_SHA256:
    raise ValueError("Decoder table checksum mismatch")
_tables = _json.loads(_zlib.decompress(_payload))

def _table(name):
    return [tuple(row) if isinstance(row, list) else row for row in _tables[name]]

''' + '\n'.join(f'{name} = _table({name!r})' for name in tables) + '\n# END_COMPILE_TIME_TABLES'

    source = generate_adaptive_source(protocol, table_writer=write_tables)
    atomic_write(path, source)
    return source


def inline_adaptive_source(source, source_path):
    """Resolve a saved module's sidecar for compilation or independent audits."""
    import ast
    import json
    import zlib
    start = '# BEGIN_COMPILE_TIME_TABLES'
    end = '# END_COMPILE_TIME_TABLES'
    if start not in source:
        return source
    header, rest = source.split(start, 1)
    loader, tail = rest.split(end, 1)
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse(loader).body if isinstance(node, ast.Assign)
                 and node.targets[0].id in {'TABLES_FILE', 'TABLES_SHA256'}}
    payload = Path(source_path).with_name(constants['TABLES_FILE']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != constants['TABLES_SHA256']:
        raise ValueError('Decoder table checksum mismatch')
    tables = {name: [tuple(row) if isinstance(row, list) else row for row in rows]
              for name, rows in json.loads(zlib.decompress(payload)).items()}
    return header + '\n\n'.join(f'{name} = {pformat(value, width=100, compact=True)}'
                                for name, value in tables.items()) + tail


def compile_adaptive_source(source, *, entrypoint="se_cycle", source_path=None):
    """Compile source emitted from an already verified policy."""
    if source_path is not None:
        source = inline_adaptive_source(source, source_path)
    path = Path(tempfile.mkdtemp(prefix='spiderwarp_adaptive_se_')) / 'steane_se_protocol.py'
    path.write_text(source)
    name = '_spiderwarp_adaptive_' + hashlib.sha256(source.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        function = getattr(module, entrypoint)
        if entrypoint == "se_cycle":
            package = module.compile_se_cycle()
        else:
            target = function
            minimal_opt = getattr(target, "with_minimal_opt", None)
            if minimal_opt is not None:
                target = minimal_opt()
            package = target.compile_function()
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return source, path, module, function, package

"""Standalone Guppy emitter for an already certified adaptive decision graph."""

from __future__ import annotations

import hashlib
import importlib.util
from itertools import groupby
from pathlib import Path
from pprint import pformat
import sys
import tempfile

from spiderwarp._se_guppy import IntegerDecisions
from spiderwarp._se_circuit import row_mask


COMPACT_POLICY_THRESHOLD = 50000


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


def generate_adaptive_source(protocol):
    policy, code = protocol.policy, protocol.code
    if not policy.report.verified:
        raise ValueError("Verify the adaptive policy before exporting")
    if code.n > 63:
        raise NotImplementedError("Guppy frame masks support at most 63 data qubits")
    actions = list(sorted(policy.actions.items()))
    ids = {name: i for i, (name, _) in enumerate(actions)}
    if any(a.gadget.num_detectors > 63 for _, a in actions):
        raise NotImplementedError("Guppy outcome masks support at most 63 combined flags/checks")
    metadata = dict(
        N_DATA=code.n,
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

    decisions = IntegerDecisions()
    roots = []
    node_rows = []
    terminal_tables = []
    terminal_ids = {}
    bases, basis_ids = [()], {(): 0}
    control_specs = []

    def root_id(table, width):
        roots.append(decisions.build(table, width))
        return len(roots) - 1

    def basis_id(vectors):
        if vectors not in basis_ids:
            basis_ids[vectors] = len(bases)
            bases.append(vectors)
        return basis_ids[vectors]
    large_export = len(policy.nodes) >= COMPACT_POLICY_THRESHOLD
    if large_export:
        # Terminal extraction tables already contain their accept frames. Those
        # accept nodes must not also occupy a large, unreachable Guppy constant.
        reachable, pending = {0}, [policy.root]
        while pending:
            index = pending.pop()
            while policy.nodes[index].kind == 'frame':
                index = policy.nodes[index].next
            if index in reachable:
                continue
            reachable.add(index)
            node = policy.nodes[index]
            if node.kind == 'extract':
                targets = []
                for child in node.children.values():
                    while policy.nodes[child].kind == 'frame':
                        child = policy.nodes[child].next
                    targets.append(child)
                if not all(policy.nodes[child].kind in {'accept', 'reject'} for child in targets):
                    pending.extend(targets)
        exported_ids = {old: new for new, old in enumerate(sorted(reachable))}
    else:
        exported_ids = {old: new for new, old in enumerate(
            i for i, node in enumerate(policy.nodes) if node.kind != "frame")}

    def successor(index):
        frame = 0
        while policy.nodes[index].kind == "frame":
            frame ^= policy.nodes[index].frame
            index = policy.nodes[index].next
        return exported_ids.get(index, 0), frame, policy.nodes[index]

    root, root_frame, _ = successor(policy.root)
    metadata.update(ROOT_NODE=root, ROOT_FRAME_X=root_frame & ((1 << code.n) - 1),
                    ROOT_FRAME_Z=root_frame >> code.n)
    for index, node in enumerate(policy.nodes):
        if index not in exported_ids:
            continue
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
                offset = table[min(table)] if table else 0
                normalized = {key: frame ^ offset for key, frame in table.items()}
                xb, xs = factor_masks({key: frame & ((1 << code.n) - 1)
                                       for key, frame in normalized.items()})
                zb, zs = factor_masks({key: frame >> code.n for key, frame in normalized.items()})
                normalized = tuple((key, xs[key], zs[key]) for key in sorted(table))
                spec = max(1, action.gadget.num_detectors), normalized
                if spec not in terminal_ids:
                    terminal_ids[spec] = len(terminal_tables)
                    terminal_tables.append(spec)
                node_rows.append((3, ids[node.action], terminal_ids[spec],
                                  offset & ((1 << code.n) - 1), offset >> code.n,
                                  basis_id(xb), basis_id(zb)))
            else:
                table = {f | (s << width): child
                         for (f, s), child in successors.items() if child[2].kind != "reject"}
                input_width = max(1, action.gadget.num_detectors)
                first = table[min(table)] if table else (0, 0, None)
                offset, frame_offset = first[:2]
                accepted = root_id({key: 1 for key in table}, input_width)
                normalized = {key: child ^ offset for key, (child, _, _) in table.items()}
                active_bits = 0
                for child in normalized.values():
                    active_bits |= child
                bits = []
                while active_bits:
                    low = active_bits & -active_bits
                    bit = low.bit_length() - 1
                    bits.append((bit, root_id({key: 1 for key, child in normalized.items()
                                              if child & low}, input_width)))
                    active_bits ^= low
                xb, xs = factor_masks({key: (frame ^ frame_offset) & ((1 << code.n) - 1)
                                       for key, (_, frame, _) in table.items()})
                zb, zs = factor_masks({key: (frame ^ frame_offset) >> code.n
                                       for key, (_, frame, _) in table.items()})
                control_specs.append((accepted, offset, tuple(bits),
                                      root_id(xs, input_width), root_id(zs, input_width)))
                node_rows.append((2, ids[node.action], len(control_specs) - 1,
                                  frame_offset & ((1 << code.n) - 1), frame_offset >> code.n,
                                  basis_id(xb), basis_id(zb)))
        elif node.kind == 'accept':
            node_rows.append((1, 0, 0, node.frame & ((1 << code.n) - 1), node.frame >> code.n, 0, 0))
        else:
            node_rows.append((0, 0, 0, 0, 0, 0, 0))
    control_decisions = len(decisions.nodes)
    decoder_start = len(roots)
    for width, entries in terminal_tables:
        roots.extend(decisions.build(values, width) for values in (
            {key: 1 for key, _, _ in entries},
            {key: x for key, x, _ in entries},
            {key: z for key, _, z in entries},
        ))
    rows, references = decisions.compact(roots)
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
                    N_CONTROL_STATES=sum(row[0] == 2 for row in node_rows))
    # Intern predicates and leaf values so most decision rows fit one signed
    # machine word. Besides reducing file size, this avoids constructing four
    # Guppy constants per row in large real-code exports.
    predicates = list(dict.fromkeys((m, p) for m, p, _, _ in rows))
    leaves = sorted({-(r + 1) for _, _, a, b in rows for r in (a, b) if r < 0})
    predicate_ids = {value: i for i, value in enumerate(predicates)}
    leaf_ids = {value: i + len(rows) for i, value in enumerate(leaves)}
    predicate_bits = max(1, (len(predicates) - 1).bit_length())
    reference_bits = max(1, (len(rows) + len(leaves) - 1).bit_length())
    packed = predicate_bits + 2 * reference_bits <= 63
    if packed:
        def encode(reference):
            return reference if reference >= 0 else leaf_ids[-(reference + 1)]
        decision_data = [(predicate_ids[m, p]
                          | (encode(a) << predicate_bits)
                          | (encode(b) << (predicate_bits + reference_bits)))
                         for m, p, a, b in rows]
        lookup = """    decisions = comptime(DECISION_NODES)
    predicates = comptime(DECISION_PREDICATES)
    leaves = comptime(DECISION_LEAVES)
    while node >= 0:
        word = decisions[node]
        mask, pattern = predicates[word & comptime(DECISION_PREDICATE_MASK)]
        if (key & mask) == pattern:
            following = (word >> comptime(DECISION_PREDICATE_BITS)) & comptime(DECISION_REFERENCE_MASK)
        else:
            following = word >> comptime(DECISION_OTHER_SHIFT)
        if following >= comptime(N_DECISION_NODES):
            return leaves[following - comptime(N_DECISION_NODES)]
        node = following
    return -(node + 1)"""
    else:
        decision_data = rows
        lookup = """    decisions = comptime(DECISION_NODES)
    while node >= 0:
        mask, pattern, matched, other = decisions[node]
        if (key & mask) == pattern:
            node = matched
        else:
            node = other
    return -(node + 1)"""
    metadata.update(DECISION_ROWS_PACKED=packed, DECISION_PREDICATE_BITS=predicate_bits,
                    DECISION_PREDICATE_MASK=(1 << predicate_bits) - 1,
                    DECISION_REFERENCE_MASK=(1 << reference_bits) - 1,
                    DECISION_OTHER_SHIFT=predicate_bits + reference_bits)
    if large_export:
        # A tuple of seven constants per row exceeds the serializer's per-value
        # text limit on full order-two controllers. Five small control fields
        # fit in one positive signed machine word; retain the two frame masks.
        columns = (0, 1, 2, 5, 6)
        widths = [max(1, max(row[c] for row in node_rows).bit_length()) for c in columns]
        if sum(widths) > 63:
            raise ValueError('Packed policy controls exceed a signed machine word')
        shifts, shift = [], 0
        for width in widths:
            shifts.append(shift)
            shift += width
        node_rows = [(sum(row[c] << s for c, s in zip(columns, shifts)), row[3], row[4])
                     for row in node_rows]
        metadata['POLICY_ROWS_PACKED'] = True
        unpack_policy = ['        word, frame_x, frame_z = nodes[node]']
        for field, shift, width in zip(('kind', 'action', 'transition', 'basis_x', 'basis_z'), shifts, widths):
            metadata['POLICY_' + field.upper() + '_SHIFT'] = shift
            metadata['POLICY_' + field.upper() + '_MASK'] = (1 << width) - 1
            unpack_policy.append(f'        {field} = (word >> comptime(POLICY_{field.upper()}_SHIFT)) & comptime(POLICY_{field.upper()}_MASK)')
    tables = '\n\n'.join(f'{name} = {pformat(value, width=100, compact=True)}'
                         for name, value in [('POLICY_NODES', node_rows),
                                             ('TERMINAL_DECODERS', terminal_roots or [(-1, -1, -1)]),
                                             ('CONTROL_SPECS', control_rows or [(-1, 0, 0, 0)]),
                                             ('CONTROL_FRAMES', control_frames or [(-1, -1)]),
                                             ('CONTROL_BITS', control_bits or [(0, -1)]),
                                             ('FRAME_BASIS_SPECS', basis_specs),
                                             ('FRAME_BASIS_VECTORS', basis_vectors or [0]),
                                             ('DECISION_PREDICATES', predicates or [(0, 0)]),
                                             ('DECISION_LEAVES', leaves or [0]),
                                             ('DECISION_NODES', decision_data or ([0] if packed else [(0, 0, -1, -1)]))])
    template = Path(__file__).with_name('_se_adaptive_guppy_template.py.in').read_text()
    if large_export:
        template = template.replace('        kind, action, transition, frame_x, frame_z, basis_x, basis_z = nodes[node]',
                                    '\n'.join(unpack_policy))
    return (template.replace('# GENERATED_METADATA', '\n'.join(
                f'{name} = {value!r}' for name, value in metadata.items()))
            .replace('# GENERATED_LOOKUP', lookup)
            .replace('# GENERATED_GADGETS', '\n'.join(circuits))
            .replace('# GENERATED_TABLES', tables))


def compile_adaptive_protocol(protocol):
    return compile_adaptive_source(generate_adaptive_source(protocol))


def compile_adaptive_source(source, *, entrypoint="se_cycle"):
    """Compile source emitted from an already verified policy."""
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

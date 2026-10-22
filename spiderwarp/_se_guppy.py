"""Emit standalone Guppy source; no Stim, NumPy or Python decoder at runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from itertools import groupby
from pathlib import Path
from pprint import pformat


class IntegerDecisions:
    """Reduced, shared bit decisions for sparse integer-valued functions."""

    def __init__(self):
        self.nodes = []
        self._ids = {}

    def build(self, table, width):
        entries = tuple(sorted((k, v) for k, v in table.items() if v))

        def visit(values, bit):
            if not values:
                return False, 0
            first = values[0][1]
            if len(values) == 1 << (bit + 1) and all(v == first for _, v in values):
                return False, first
            mask = 1 << bit
            low = visit(tuple((k, v) for k, v in values if k < mask), bit - 1)
            high = visit(tuple((k - mask, v) for k, v in values if k >= mask), bit - 1)
            if low == high:
                return low
            key = bit, low, high
            if key not in self._ids:
                self._ids[key] = len(self.nodes)
                self.nodes.append(key)
            return True, self._ids[key]

        return visit(entries, width - 1)

    @staticmethod
    def reference(ref):
        """Nonnegative node indices; negative values encode terminal integers."""
        node, value = ref
        return value if node else -value - 1

    def evaluate(self, ref, syndrome):
        while ref[0]:
            bit, low, high = self.nodes[ref[1]]
            ref = high if (syndrome >> bit) & 1 else low
        return ref[1]

    def compact(self, roots):
        """Fold consecutive tests with a common fallback into one masked test.

        Sparse tables have long runs of bits that simply reject. A single
        ``key & mask == pattern`` test represents that entire run. Prune nodes
        made private/unreachable by folding, while retaining sharing at roots.
        """
        nodes, ids, cache = [], {}, {}

        def visit(ref):
            if not ref[0]:
                return self.reference(ref)
            if ref[1] in cache:
                return cache[ref[1]]
            bit, low, high = self.nodes[ref[1]]
            low, high = visit(low), visit(high)
            mask = 1 << bit
            pattern, matched, other = mask, high, low
            if low >= 0 and high < 0:
                pattern, matched, other = 0, low, high
            while matched >= 0 and nodes[matched][3] == other:
                child_mask, child_pattern, matched, _ = nodes[matched]
                mask |= child_mask
                pattern |= child_pattern
            row = mask, pattern, matched, other
            if row not in ids:
                ids[row] = len(nodes)
                nodes.append(row)
            cache[ref[1]] = ids[row]
            return ids[row]

        references = [visit(root) for root in roots]
        reachable = set()
        pending = references.copy()
        while pending:
            index = pending.pop()
            if index >= 0 and index not in reachable:
                reachable.add(index)
                pending.extend(nodes[index][2:])
        mapping = {old: new for new, old in enumerate(sorted(reachable))}

        def remap(ref):
            return mapping[ref] if ref >= 0 else ref

        return ([(mask, pattern, remap(matched), remap(other))
                 for i in sorted(reachable) for mask, pattern, matched, other in [nodes[i]]],
                [remap(ref) for ref in references])


def share_recovery_decoders(recoveries):
    """Merge compatible sparse decoders, preserving every reachable rejection.

    A syndrome absent from a plan is unreachable for its observed prefix under
    the verified fault budget. It can share another plan's rule. Explicitly
    discarded syndromes are constraints and must never become accepted.
    """
    specs = {}
    for key, plan in recoveries.items():
        entries = dict(plan.corrections)
        entries.update((s, None) for s in plan.discarded_syndromes)
        specs[key] = tuple(sorted(entries.items()))
    tables = []
    ids = {}
    for spec in sorted(set(specs.values()), key=lambda s: (-len(s), repr(s))):
        table_id = next((i for i, table in enumerate(tables)
                         if all(s not in table or table[s] == p for s, p in spec)), None)
        if table_id is None:
            table_id = len(tables)
            tables.append({})
        tables[table_id].update(spec)
        ids[spec] = table_id
    return tables, {key: ids[spec] for key, spec in specs.items()}


def generate_source(code, gadgets, policy) -> str:
    if code.n > 63:
        raise NotImplementedError("The current Guppy correction-mask emitter supports at most 63 data qubits")
    if any(max(len(g.readout.flags), len(g.readout.syndromes)) > 63 for g in gadgets.values()):
        raise NotImplementedError("Guppy measurement masks currently support at most 63 flags/checks")
    if any(p.cleanup_distance is not None for p in policy.recoveries.values()) and code.H_x.shape[0] * 2 > 63:
        raise NotImplementedError("Two-stage recovery requires at most 31 checks per extraction")
    metadata = [
        f'N_DATA = {code.n}',
        f'N_CHECKS = {code.H_x.shape[0]}',
        f'N_X_FLAGS = {gadgets["X", code.d].num_flags}',
        f'N_Z_FLAGS = {gadgets["Z", code.d].num_flags}',
        f'CODE_DISTANCE = {code.d}',
        f'CORRECTABLE_FAULTS = {policy.correctable}',
        f'DETECTABLE_FAULTS = {policy.detectable}',
        f'RECOVERY_OBJECTIVE = {getattr(policy, "recovery_objective", "qubits")!r}',
        f'VERIFIED_AT_BUILD = {policy.report.verified!r}',
        f'EC_A_INPUT_SYNDROMES = {policy.report.ec_a_input_syndromes}',
        f'VERIFICATION_MODEL = {policy.report.noise_model!r}',
        f'RESIDUAL_METRIC = {policy.report.residual_metric!r}',
        f'ACCEPTANCE_CONTRACT = {policy.report.acceptance_contract!r}',
        'CIRCUIT_HASHES = {',
    ]
    for key, gadget in sorted(gadgets.items()):
        metadata.append(f'    {key!r}: {hashlib.sha256(str(gadget.circuit).encode()).hexdigest()!r},')
    metadata.append('}')

    circuits = []

    def call(prefix, values):
        """Wrap circuit operands without hiding the schedule in long lines."""
        operands = pformat(values, width=max(40, 96 - len(prefix)), compact=True)
        first, *rest = operands.splitlines()
        statement = prefix + first
        statement += ''.join('\n' + ' ' * len(prefix) + line for line in rest)
        circuits.extend('    ' + line for line in (statement + ')').splitlines())

    data_arg = 'data: array[qubit, comptime(N_DATA)]'
    for (basis, distance), gadget in sorted(gadgets.items()):
        circuits.extend([
            '@guppy.comptime',
            f'def run_{basis.lower()}_d{distance}({data_arg}) -> ExtractionResult:',
            '    register = dict(enumerate(data))',
            '    measurements = []',
        ])
        # Batch adjacent operations only: neither their order nor the lifetime
        # of any ancilla may change, because those define the fault locations.
        for name, operations in groupby(gadget.operations, key=lambda op: op[0]):
            targets = [qs if len(qs) == 2 else qs[0] for _, qs in operations]
            if name == 'R':
                call('prepare(register, ', tuple(targets))
            elif name == 'M':
                call('read(register, measurements, ', tuple(targets))
            elif name in {'H', 'X', 'Y', 'Z', 'CX', 'CZ'}:
                call(f'apply({name.lower()}, register, ', tuple(targets))
            else:
                raise ValueError(f'Guppy export requires basis-lowered operations, got {name}')
        call('flags = pack_parities(measurements, ', gadget.readout.flags)
        call('syndrome = pack_parities(measurements, ', gadget.readout.syndromes)
        circuits.extend([
            '    return ExtractionResult(flags, syndrome)', '', '',
        ])
    for basis in ('x', 'z'):
        circuits.extend(['@guppy', f'def extract_{basis}({data_arg}, distance: int) -> ExtractionResult:'])
        for distance in sorted(d for b, d in gadgets if b.lower() == basis):
            circuits.extend([f'    if distance == {distance}:', f'        return run_{basis}_d{distance}(data)'])
        circuits.extend(['    return ExtractionResult(-1, 0)', '', ''])

    decisions = IntegerDecisions()
    decoders = []
    tables, decoder_ids = share_recovery_decoders(policy.recoveries)
    for table in [policy.standard['Z'], policy.standard['X']] + [
            {s: p for s, p in table.items() if p is not None} for table in tables]:
        width = max(1, max(table, default=0).bit_length())
        refs = [decisions.build(values, width) for values in (
            {s: 1 for s in table},
            {s: p & ((1 << code.n) - 1) for s, p in table.items()},
            {s: p >> code.n for s, p in table.items()},
        )]
        decoders.append((width, *refs))

    routes = []
    route_ids = {}
    z_routes, x_routes = {}, {}
    for key, plan in sorted(policy.recoveries.items()):
        basis, flags, previous = key
        route = plan.distance, plan.cleanup_distance or 0, decoder_ids[key]
        if route not in route_ids:
            route_ids[route] = len(routes) + 1  # zero means unknown/reject
            routes.append(route)
        if basis == 'Z':
            z_routes[flags] = route_ids[route]
        else:
            x_routes.setdefault(previous, {})[flags] = route_ids[route]
    z_root = decisions.build(z_routes, gadgets['Z', code.d].num_flags)
    x_roots = [decisions.build(table, gadgets['X', code.d].num_flags)
               for _, table in sorted(x_routes.items())]
    previous_width = max(1, max(x_routes, default=0).bit_length())
    previous_root = decisions.build({s: i + 1 for i, s in enumerate(sorted(x_routes))}, previous_width)
    nodes, roots = decisions.compact([ref for _, *refs in decoders for ref in refs]
                                    + [z_root, previous_root] + x_roots)
    decoder_roots = iter(roots[:3 * len(decoders)])
    decoders = [(width, next(decoder_roots), next(decoder_roots), next(decoder_roots))
                for width, *_ in decoders]
    z_root, previous_root, *x_roots = roots[3 * len(decoders):]
    metadata.extend([
        f'PREVIOUS_WIDTH = {previous_width}',
        f'Z_RECOVERY_ROOT = {z_root}',
        f'X_PREVIOUS_ROOT = {previous_root}',
    ])

    data = [
        '# Decoder data: (syndrome width, acceptance root, X root, Z root).',
        '# Decoder 0 is standard Z; decoder 1 is standard X; the rest are recovery.',
    ]
    for name, rows in [('DECODERS', decoders),
                       ('RECOVERY_ROUTES', routes or [(-1, 0, -1)]),
                       ('X_RECOVERY_ROOTS', x_roots or [-1]),
                       ('DECISION_NODES', nodes or [(0, 0, -1, -1)])]:
        if name == 'RECOVERY_ROUTES':
            data.append('# Routes: (recovery distance, cleanup distance or zero, decoder id).')
        elif name == 'DECISION_NODES':
            data.extend(['# Shared decision nodes: (mask, pattern, matching branch, other branch).',
                         '# A nonnegative reference is a node index; -(value + 1) is a leaf.'])
        data.append(f'{name} = [')
        if name == 'DECISION_NODES':
            # Compact numeric data, with a bounded line width. The human-facing
            # circuit and control-flow definitions remain separate above it.
            formatted = pformat(rows, width=92, compact=True)[1:-1]
            data.extend('    ' + line.lstrip() for line in formatted.splitlines())
        else:
            data.extend(f'    {row!r},' for row in rows)
        data.extend([']', ''])
    template = Path(__file__).with_name('_se_guppy_template.py.in').read_text()
    return (template.replace('# GENERATED_METADATA', '\n'.join(metadata))
            .replace('# GENERATED_GADGETS', '\n'.join(circuits).rstrip())
            .replace('# GENERATED_TABLES', '\n'.join(data)))


def compile_protocol(code, gadgets, policy):
    source = generate_source(code, gadgets, policy)
    directory = Path(tempfile.mkdtemp(prefix='spiderwarp_se_'))
    path = directory / 'steane_se_protocol.py'
    path.write_text(source)
    name = '_spiderwarp_se_' + hashlib.sha256(source.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        function = module.se_cycle
        package = module.compile_se_cycle()
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return source, path, module, function, package

"""Emit standalone Guppy source; no Stim, NumPy or Python decoder at runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from collections import Counter
from pathlib import Path


class IntegerDecisions:
    """Reduced, shared bit decisions for sparse integer-valued functions."""

    def __init__(self):
        self.nodes = []
        self._ids = {}
        self.roots = set()

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

        root = visit(entries, width - 1)
        if root[0]:
            self.roots.add(root[1])
        return root

    @staticmethod
    def expression(ref):
        node, value = ref
        return f'decision_{value}(syndrome)' if node else str(value)

    def evaluate(self, ref, syndrome):
        while ref[0]:
            bit, low, high = self.nodes[ref[1]]
            ref = high if (syndrome >> bit) & 1 else low
        return ref[1]

    def functions(self):
        """Inline private nodes; emit functions only for roots/shared nodes."""
        uses = Counter(ref[1] for _, low, high in self.nodes
                       for ref in (low, high) if ref[0])
        retained = self.roots | {i for i, count in uses.items() if count > 1}

        def body(ref, root, indent=''):
            node, value = ref
            if not node or (value in retained and value != root):
                return [indent + 'return ' + self.expression(ref)]
            bit, low, high = self.nodes[value]
            return ([indent + f'if (syndrome & {1 << bit}) == 0:']
                    + body(low, root, indent + '    ') + body(high, root, indent))

        return [(i, body((True, i), i)) for i in sorted(retained)]


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
    lines = [
        '"""Generated adaptive SE: Z primary, then X primary; recovery ends the cycle.',
        '',
        'Import se_cycle(data) -> bool, or postselected_se_cycle(data) -> None.',
        'All integer measurement masks use bit i for check/flag i.',
        'Use compile_se_cycle() for compatibility with supported Guppy versions.',
        '"""',
        'from guppylang import guppy',
        'from guppylang.std.builtins import array, exit, owned',
        'from guppylang.std import quantum as _quantum',
        'from guppylang.std.quantum import qubit, h, cx, cz, x, y, z, measure',
        '',
        f'N_DATA = {code.n}',
        f'CODE_DISTANCE = {code.d}',
        f'CORRECTABLE_FAULTS = {policy.correctable}',
        f'DETECTABLE_FAULTS = {policy.detectable}',
        f'VERIFIED_AT_BUILD = {policy.report.verified!r}',
        f'EC_A_INPUT_SYNDROMES = {policy.report.ec_a_input_syndromes}',
        f'VERIFICATION_MODEL = {policy.report.noise_model!r}',
        f'RESIDUAL_METRIC = {policy.report.residual_metric!r}',
        f'ACCEPTANCE_CONTRACT = {policy.report.acceptance_contract!r}',
        'CIRCUIT_HASHES = {',
    ]
    for key, gadget in sorted(gadgets.items()):
        lines.append(f'    {key!r}: {hashlib.sha256(str(gadget.circuit).encode()).hexdigest()!r},')
    lines.extend([
        '}', '',
        'if hasattr(_quantum, "Measurement"):',
        '    @guppy',
        '    def measure_bit(q: qubit @ owned) -> bool:',
        '        return measure(q).read()',
        'else:',
        '    @guppy',
        '    def measure_bit(q: qubit @ owned) -> bool:',
        '        return measure(q)',
        '',
    ])

    def function(name, args, returns, body):
        lines.extend(['@guppy', f'def {name}({args}) -> {returns}:'])
        lines.extend('    ' + line if line else '' for line in body)
        lines.append('')

    data_arg = f'data: array[qubit, {code.n}]'
    for (basis, distance), gadget in sorted(gadgets.items()):
        body = []
        record = 0

        def qref(q):
            return f'data[{q}]' if q < code.n else f'a{q}'

        for name, qs in gadget.operations:
            a = qref(qs[0])
            if name in {'R', 'RX'}:
                body.append(f'{a} = qubit()')
                if name == 'RX':
                    body.append(f'h({a})')
            elif name in {'M', 'MX'}:
                if name == 'MX':
                    body.append(f'h({a})')
                body.append(f'm{record} = measure_bit({a})')
                record += 1
            elif name == 'XCX':
                b = qref(qs[1])
                body.extend([f'h({a})', f'h({b})', f'cz({a}, {b})', f'h({a})', f'h({b})'])
            elif name in {'CX', 'CZ'}:
                body.append(f'{name.lower()}({a}, {qref(qs[1])})')
            else:
                body.append(f'{name.lower()}({a})')
        body.extend(['flags = 0', 'syndrome = 0'])
        for label, group in [('flags', gadget.readout.flags), ('syndrome', gadget.readout.syndromes)]:
            for bit, parity in enumerate(group):
                if parity:
                    expression = ' ^ '.join(f'm{i}' for i in parity)
                    body.extend([f'if {expression}:', f'    {label} ^= {1 << bit}'])
        body.append('return flags, syndrome')
        function(f'run_{basis.lower()}_d{distance}', data_arg, 'tuple[int, int]', body)

    body = []
    for q in range(code.n):
        body.extend([f'if (x_mask & {1 << q}) != 0:', f'    x(data[{q}])'])
    for q in range(code.n):
        body.extend([f'if (z_mask & {1 << q}) != 0:', f'    z(data[{q}])'])
    function('apply_correction', data_arg + ', x_mask: int, z_mask: int', 'None', body)

    decisions = IntegerDecisions()

    def correction_decoder(name, table):
        if not table:
            function(name, 'syndrome: int', 'tuple[bool, int, int]', ['return False, 0, 0'])
            return
        width = max(1, max(table).bit_length())
        accept = decisions.build({s: 1 for s in table}, width)
        xs = decisions.build({s: p & ((1 << code.n) - 1) for s, p in table.items()}, width)
        zs = decisions.build({s: p >> code.n for s, p in table.items()}, width)
        body = ['if syndrome < 0:', '    return False, 0, 0']
        if width < 63:
            body.extend([f'if syndrome >= {1 << width}:', '    return False, 0, 0'])
        if accept != (False, 1):
            body.extend([f'if {decisions.expression(accept)} == 0:', '    return False, 0, 0'])
        body.append(f'return True, {decisions.expression(xs)}, {decisions.expression(zs)}')
        function(name, 'syndrome: int', 'tuple[bool, int, int]', body)

    for basis in ('Z', 'X'):
        correction_decoder(f'decode_{basis.lower()}_standard', policy.standard[basis])

    tables, decoder_ids = share_recovery_decoders(policy.recoveries)
    routing = {'Z': {}, 'X': {}}
    for key, plan in sorted(policy.recoveries.items()):
        basis, flags, previous = key
        routing[basis][flags, previous] = plan.distance, plan.cleanup_distance or 0, decoder_ids[key]
    for table_id, table in enumerate(tables):
        correction_decoder(f'decode_recovery_{table_id}', {s: p for s, p in table.items() if p is not None})
    for node_id, body in decisions.functions():
        function(f'decision_{node_id}', 'syndrome: int', 'int', body)
    body = []
    for table_id in range(len(tables)):
        body.extend([f'if decoder_id == {table_id}:', f'    return decode_recovery_{table_id}(syndrome)'])
    body.append('return False, 0, 0')
    function('decode_recovery', 'decoder_id: int, syndrome: int', 'tuple[bool, int, int]', body)

    for basis in ('Z', 'X'):
        grouped = {}
        for (flags, previous), value in sorted(routing[basis].items()):
            condition = f'flags == {flags}'
            if basis == 'X':
                condition = f'({condition} and previous == {previous})'
            grouped.setdefault(value, []).append(condition)
        body = []
        for (distance, cleanup_distance, table_id), conditions in grouped.items():
            # Keep generated Boolean expressions and parser recursion bounded.
            for start in range(0, len(conditions), 16):
                body.extend([f'if {" or ".join(conditions[start:start + 16])}:',
                             f'    return {distance}, {cleanup_distance}, {table_id}'])
        body.append('return -1, 0, -1')
        args = 'flags: int' + (', previous: int' if basis == 'X' else '')
        function(f'select_{basis.lower()}_recovery', args, 'tuple[int, int, int]', body)

        dual = 'x' if basis == 'Z' else 'z'
        distances = sorted(d for b, d in gadgets if b == dual.upper())
        body = [f'distance, cleanup_distance, decoder_id = select_{basis.lower()}_recovery(flags'
                + (', previous)' if basis == 'X' else ')')]
        for i, distance in enumerate(distances):
            body.extend([f'{"if" if i == 0 else "elif"} distance == {distance}:',
                         f'    recovery_flags, syndrome = run_{dual}_d{distance}(data)'])
        body.extend(['else:', '    return False', 'if recovery_flags != 0:',
                     '    return False', 'if cleanup_distance != 0:'])
        for i, distance in enumerate(sorted(d for b, d in gadgets if b == basis)):
            body.extend([f'    {"if" if i == 0 else "elif"} cleanup_distance == {distance}:',
                         f'        cleanup_flags, cleanup_syndrome = run_{basis.lower()}_d{distance}(data)'])
        body.extend(['    else:', '        return False', '    if cleanup_flags != 0:',
                     '        return False', f'    syndrome |= cleanup_syndrome << {code.H_x.shape[0]}',
                     'ok, x_mask, z_mask = decode_recovery(decoder_id, syndrome)',
                     'if not ok:', '    return False', 'apply_correction(data, x_mask, z_mask)',
                     'return True'])
        function(f'recover_after_{basis.lower()}', data_arg + ', ' + args, 'bool', body)

    function('se_cycle', data_arg, 'bool', [
        f'flags, z_syndrome = run_z_d{code.d}(data)',
        'if flags != 0:',
        '    return recover_after_z(data, flags)',
        'ok, x_mask, z_mask = decode_z_standard(z_syndrome)',
        'if not ok:',
        '    return False',
        'apply_correction(data, x_mask, z_mask)',
        f'flags, x_syndrome = run_x_d{code.d}(data)',
        'if flags != 0:',
        '    return recover_after_x(data, flags, z_syndrome)',
        'ok, x_mask, z_mask = decode_x_standard(x_syndrome)',
        'if not ok:',
        '    return False',
        'apply_correction(data, x_mask, z_mask)',
        'return True',
    ])
    function('postselected_se_cycle', data_arg, 'None', [
        'if not se_cycle(data):',
        '    exit("SE protocol rejected", 1)',
    ])
    lines.extend([
        'def compile_se_cycle():',
        '    """Compile with minimal optimization on Guppy versions offering it."""',
        '    target = se_cycle',
        '    minimal_opt = getattr(target, "with_minimal_opt", None)',
        '    if minimal_opt is not None:',
        '        target = minimal_opt()',
        '    return target.compile_function()',
        '',
    ])
    return '\n'.join(lines)


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

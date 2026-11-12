"""Standalone Guppy emitter for the conventional Steane RUS baselines."""
from itertools import groupby
from pathlib import Path


def generate_source(protocol):
    template = Path(__file__).with_name("_se_guppy_template.py.in").read_text()
    # Share the existing basis-compatible measurement and circuit helpers.
    helpers = template[template.index('from guppylang import guppy'):template.index('# GENERATED_GADGETS')]
    helpers = helpers.replace('# GENERATED_METADATA', f'N_DATA = {protocol.code.n}')
    lines = [helpers, '''
@guppy.struct
class CycleResult:
    accepted: bool
    frame_x: int
    frame_z: int

@guppy.struct
class Prepared:
    ancilla: array[qubit, comptime(N_DATA)]
    level: int
''']
    n = protocol.code.n
    for basis in ("Z", "X"):
        for d in (1, 2, 4):
            g = protocol.gadgets[f"prep_{basis}_d{d}"]
            lines += ['@guppy.comptime',
                      f'def prep_{basis.lower()}_d{d}(ancilla: array[qubit, comptime(N_DATA)]) -> int:',
                      '    register = dict(enumerate(ancilla))', '    measurements = []']
            operations = []
            for op in g.circuit:
                qs = [t.value - n for t in op.targets_copy()]
                for targets in (list(zip(qs[::2], qs[1::2])) if op.name in {"CX", "CZ"} else [(q,) for q in qs]):
                    # Output block is freshly allocated by the caller.
                    if op.name == 'R' and targets[0] < n:
                        continue
                    operations.append((op.name, targets))
            for name, group in groupby(operations, key=lambda op: op[0]):
                targets = tuple(qs if len(qs) == 2 else qs[0] for _, qs in group)
                if name == 'R':
                    statement = f'prepare(register, {targets!r})'
                elif name == 'M':
                    statement = f'read(register, measurements, {targets!r})'
                else:
                    statement = f'apply({name.lower()}, register, {targets!r})'
                lines.append('    ' + statement)
            lines += [f'    return pack_parities(measurements, {g.readout.flags!r})', '']
        low = basis.lower()
        lines += ['@guppy', f'def prepare_{low}(level: int) -> Prepared:',
                  '    while True:',
                  '        ancilla = array(qubit() for _ in range(comptime(N_DATA)))',
                  f'        flags = prep_{low}_d4(ancilla) if level == 4 else prep_{low}_d2(ancilla) if level == 2 else prep_{low}_d1(ancilla)',
                  '        if flags == 0:', '            return Prepared(ancilla, level)',
                  '        for q in ancilla:', '            measure_bit(q)']
        if protocol.mode != 'd4':
            lines += ['        level = 2 if level == 4 else 1']
        lines += ['', '@guppy.comptime',
                  f'def syndrome_{low}(measurements: array[bool, comptime(N_DATA)]) -> int:',
                  f'    return pack_parities(measurements, {protocol.gadgets[f"read_{basis}"].readout.syndromes!r})',
                  '', '@guppy',
                  f'def read_{low}(data: array[qubit, comptime(N_DATA)], ancilla: array[qubit, comptime(N_DATA)] @ owned) -> int:',
                  '    for i in range(comptime(N_DATA)):']
        lines += ['        cx(data[i], ancilla[i])' if basis == 'Z' else '        cx(ancilla[i], data[i])']
        if basis == 'X':
            lines += ['    for i in range(comptime(N_DATA)):', '        h(ancilla[i])']
        lines += ['    measurements = array(measure_bit(q) for q in ancilla)',
                  f'    return syndrome_{low}(measurements)', '']
        checks = protocol.code.H_z if basis == 'Z' else protocol.code.H_x
        from spiderwarp._se_circuit import row_mask
        masks = [row_mask(row) for row in checks]
        lines += [f'CHECKS_{basis} = {masks!r}', f'DECODER_{basis} = {list(protocol.decoders[basis])!r}', '',
                  '@guppy', f'def frame_syndrome_{low}(frame: int) -> int:',
                  f'    checks = comptime(CHECKS_{basis})', '    syndrome = 0',
                  f'    for i in range(comptime(len(CHECKS_{basis}))):',
                  '        parity = False', '        for q in range(comptime(N_DATA)):',
                  '            if (checks[i] & frame & (1 << q)) != 0:', '                parity = not parity',
                  '        syndrome = append_bit(syndrome, parity, i)', '    return syndrome', '']
    lines += ['@guppy',
              'def se_cycle_with_frame(data: array[qubit, comptime(N_DATA)], frame_x: int, frame_z: int) -> CycleResult:',
              '    level = 4']
    for basis, frame in (("z", "frame_x"), ("x", "frame_z")):
        lines += ['    previous = -1', '    done = False', '    while not done:']
        if protocol.mode != 'ladder-cycle':
            lines += ['        level = 4']
        lines += [f'        prepared = prepare_{basis}(level)',
                  '        level = prepared.level',
                  f'        syndrome = read_{basis}(data, prepared.ancilla)',
                  '        if syndrome == previous:',
                  f'            decoder = comptime(DECODER_{basis.upper()})',
                  f'            {frame} ^= decoder[syndrome ^ frame_syndrome_{basis}({frame})]',
                  '            done = True', '        previous = syndrome']
    lines += ['    return CycleResult(True, frame_x, frame_z)', '', '@guppy',
              'def se_cycle(data: array[qubit, comptime(N_DATA)]) -> CycleResult:',
              '    return se_cycle_with_frame(data, 0, 0)', '',
              'def compile_se_cycle():',
              '    target = se_cycle',
              '    minimal = getattr(target, "with_minimal_opt", None)',
              '    if minimal is not None:', '        target = minimal()',
              '    return target.compile_function()', '']
    return '\n'.join(lines)

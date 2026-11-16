"""Compiled parity evaluation, including record reuse and XOR cancellation."""

from test_sol_logical_measurement import run_guppy_check


def test_measurement_packing_and_detector_parities(run_guppy_check):
    run_guppy_check('''
from itertools import product
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from sol_experiment.measurement_utils import pack_bits, pack_detector_parities

DETECTORS = [(), (-4,), (-1,), (-4, -2), (-4, -4), (-4, -3, -2, -1)]


@guppy.comptime
def evaluate(readout: array[bool, 4]) -> tuple[int, int, int]:
    bits = pack_bits(readout)
    parities = pack_detector_parities(readout, DETECTORS)
    wide = pack_bits([False] * 62 + [readout[0]])
    return bits, parities, wide


@guppy
def entry(readout: array[bool, 4]) -> None:
    bits, parities, wide = evaluate(readout)
    output("bits", bits)
    output("parities", parities)
    output("wide", wide)
    output("readout", readout)


inputs = list(product((False, True), repeat=4))
shots = (entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=1)
         .stabilizer_sim().run_per_shot([dict(readout=list(bits)) for bits in inputs])
         .collated_shots())
assert len(shots) == len(inputs)
for shot, bits in zip(shots, inputs):
    expected_bits = sum(int(bit) << i for i, bit in enumerate(bits))
    expected_parities = sum((sum(bits[ref] for ref in group) % 2) << i
                            for i, group in enumerate(DETECTORS))
    assert shot == dict(bits=[expected_bits], parities=[expected_parities],
                        wide=[int(bits[0]) << 62], readout=[list(bits)])

assert pack_bits([]) == pack_detector_parities([], []) == 0
for invalid in (0, 1, -5, True, False, 1.5):
    try:
        pack_detector_parities([False] * 4, [(invalid,)])
    except ValueError:
        pass
    else:
        raise AssertionError(invalid)
for bits in ([False] * 64, [False] * 65):
    try:
        pack_bits(bits)
    except ValueError:
        pass
    else:
        raise AssertionError("Mask exceeds Guppy's nonnegative integer range")
''')

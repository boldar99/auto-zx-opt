"""Compiled optimized extraction: detector ordering, rejection, and correction."""

from itertools import combinations, product

import stim

from test_sol_logical_measurement import ROOT, run_guppy_check, syndrome
from test_sol_steane_qec import inverse_encoder


def test_optimized_detector_decoding_and_flags(run_guppy_check):
    circuit = stim.Circuit.from_file(
        ROOT / "spiderwarp/assets/circuits/SECircuits/32_20_4_d4.stim"
    )
    detectors = [tuple(t.value for t in op.targets_copy())
                 for op in circuit if op.name == "DETECTOR"]
    singles = {syndrome(1 << q): 1 << q for q in range(32)}

    def expected(raw):
        bits = sum((sum(raw[ref] for ref in group) % 2) << i
                   for i, group in enumerate(detectors))
        flags, checks = bits & 2047, bits >> 11
        accepted = flags == 0 and (checks == 0 or checks in singles)
        return dict(accepted=int(accepted), flags=flags, syndrome=checks,
                    correction=singles.get(checks, 0) if accepted else 0)

    # Solve the independent Stim parity map to exercise every six-bit syndrome
    # without flags, then each flag together with a correctable syndrome.
    basis = {}
    for record in range(41):
        bits = sum((group.count(record - 41) % 2) << i
                   for i, group in enumerate(detectors))
        raw = 1 << record
        while bits:
            pivot = bits.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (bits, raw)
                break
            bits ^= basis[pivot][0]
            raw ^= basis[pivot][1]
    assert len(basis) == 17

    def records(bits):
        raw = 0
        while bits:
            vector, word = basis[bits.bit_length() - 1]
            bits ^= vector
            raw ^= word
        return [bool(raw & (1 << q)) for q in range(41)]

    raw_cases = [[q == i for q in range(41)] for i in range(41)]
    raw_cases += [[True] * 41, [bool(i % 2) for i in range(41)]]
    raw_cases += [records(s << 11) for s in range(64)]
    raw_cases += [records((1 << i) | (syndrome(1 << 31) << 11)) for i in range(11)]
    raw_cases += [records(2047 | (syndrome(1) << 11))]
    cases = [dict(raw=raw, **expected(raw)) for raw in raw_cases]

    run_guppy_check('''
from datetime import timedelta
import json
from pathlib import Path
import sys
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, measure
from selene_sim.backends import ClassicalReplay
from sol_experiment.QEC_optimized_steane import (
    make_optimized_steane_circuit, make_optimized_steane_qec_cycle,
)
from sol_experiment.decoding import decode_syndrome

for invalid in ("Y", "", None):
    try:
        make_optimized_steane_circuit(invalid, 4)
    except ValueError:
        pass
    else:
        raise AssertionError(invalid)

qec_x = make_optimized_steane_circuit("X", 4)
qec_z = make_optimized_steane_circuit("Z", 4)
cycle = make_optimized_steane_qec_cycle()


@guppy
def entry(z_checks: bool) -> None:
    data = array(qubit() for _ in range(32))
    flags, syndrome = qec_z(data) if z_checks else qec_x(data)
    output("flags", flags)
    output("syndrome", syndrome)
    output("data", array(measure(q).read() for q in data))


@guppy
def direct(syndrome: int) -> None:
    result = decode_syndrome(syndrome)
    output("accepted", result.accepted)
    output("syndrome", result.syndrome)
    output("correction", result.correction)


@guppy
def cycle_entry() -> None:
    data = array(qubit() for _ in range(32))
    output("accepted", cycle(data))
    output("data", array(measure(q).read() for q in data))


cases = json.loads(Path(sys.argv[1]).read_text())
tail = [bool(i % 3) for i in range(32)]
records = [case["raw"] + tail for _ in range(2) for case in cases]
result = (entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=60)
          .with_simulator(ClassicalReplay(measurements=records))
          .with_timeout(timedelta(seconds=20))
          .run_per_shot([dict(z_checks=z) for z in (False, True) for _ in cases]))
for shot, case in zip(result.collated_shots(), cases * 2):
    for key in ("flags", "syndrome"):
        assert shot[key] == [case[key]], (key, case, shot)
    assert shot["data"] == [tail], (case, shot)

# A raised first flag skips the second extraction. Ambiguous syndromes are
# decoded only after both extractions, and either terminal ambiguity rejects.
good = next(c["raw"] for c in cases if c["flags"] == 0 and c["syndrome"] == 0)
flagged = next(c["raw"] for c in cases if c["flags"] != 0)
ambiguous = next(c["raw"] for c in cases if c["flags"] == 0 and not c["accepted"])
cycle_records = [flagged + tail, ambiguous + good + tail,
                 good + flagged + tail, good + ambiguous + tail,
                 good + good + tail]
shots = (cycle_entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=60)
         .with_simulator(ClassicalReplay(measurements=cycle_records)).with_shots(5)
         .with_timeout(timedelta(seconds=10)).run().collated_shots())
assert shots == [dict(accepted=[accepted], data=[tail]) for accepted in (0, 0, 0, 0, 1)]

# Test the public packed-syndrome entry point separately from physical readout.
values = list(range(64)) + [-1, 64, 127]
shots = (direct.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=1)
         .stabilizer_sim().with_timeout(timedelta(seconds=10))
         .run_per_shot([dict(syndrome=s) for s in values]).collated_shots())
for shot, value in zip(shots, values):
    case = next((c for c in cases if c["flags"] == 0 and c["syndrome"] == value), None)
    expected = ({key: [case[key]] for key in ("accepted", "syndrome", "correction")}
                if case else {"accepted": [0], "syndrome": [value], "correction": [0]})
    assert shot == expected, (value, shot, expected)
''', cases)


def test_optimized_cycle_preserves_encoded_bells(run_guppy_check):
    singles = [1 << q for q in range(32)]
    doubles = [(1 << a) | (1 << b) for a, b in combinations(range(32), 2)]
    errors = [(x, z) for x in [0] + singles for z in [0] + singles]
    errors += [(e, 0) for e in doubles] + [(0, e) for e in doubles]
    cases = []
    for x_first, (x_error, z_error) in product((False, True), errors):
        accepted = x_error.bit_count() <= 1 and z_error.bit_count() <= 1
        cases.append(dict(
            args=dict(x_first=x_first, x_error=x_error, z_error=z_error,
                      undo_x=0 if accepted else x_error,
                      undo_z=0 if accepted else z_error),
            accepted=int(accepted),
        ))

    run_guppy_check('''
from datetime import timedelta
import json
from pathlib import Path
import sys
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, comptime, output
from guppylang.std.quantum import qubit, h, cx, x, z, measure
from sol_experiment.QEC_optimized_steane import make_optimized_steane_qec_cycle
from sol_experiment.dispatchers import (
    HISTORY_CAPACITY, DispatcherAction, DispatchDecision, zero_flag_dispatcher,
)
from sol_experiment.logical_state_preparation import state_prep_32_20_4_zero_d1

request = json.loads(Path(sys.argv[1]).read_text())
INVERSE = request["inverse"]
cycle = make_optimized_steane_qec_cycle()


@guppy
def x_first_dispatcher(
    history: array[tuple[int, int, int, int], comptime(HISTORY_CAPACITY)], count: int,
) -> DispatchDecision:
    if count == 0:
        return DispatchDecision(comptime(DispatcherAction.X4.value), 0, 0)
    return zero_flag_dispatcher(history, count)


cycle_x_first = make_optimized_steane_qec_cycle(x_first_dispatcher)


@guppy.comptime
def decode(data: array[qubit, 32]) -> None:
    for name, targets in INVERSE:
        if name == "H":
            for q in targets:
                h(data[q])
        else:
            for a, b in zip(targets[::2], targets[1::2]):
                cx(data[a], data[b])


@guppy
def entry(x_first: bool, x_error: int, z_error: int, undo_x: int, undo_z: int) -> None:
    data = array(qubit() for _ in range(32))
    reference = array(qubit() for _ in range(32))
    state_prep_32_20_4_zero_d1(data)
    state_prep_32_20_4_zero_d1(reference)
    for i in range(32):
        h(data[i])
    for i in range(32):
        cx(data[i], reference[i])
    for i in range(32):
        if (x_error & (1 << i)) != 0:
            x(data[i])
        if (z_error & (1 << i)) != 0:
            z(data[i])
    output("accepted", cycle_x_first(data) if x_first else cycle(data))

    # Rejected trajectories must have received no partial correction. Accepted
    # trajectories must be corrected in both components, on all logical qubits.
    for i in range(32):
        if (undo_x & (1 << i)) != 0:
            x(data[i])
        if (undo_z & (1 << i)) != 0:
            z(data[i])
    for i in range(32):
        cx(data[i], reference[i])
    for i in range(32):
        h(data[i])
    decode(data)
    decode(reference)
    output("data", array(measure(q).read() for q in data))
    output("reference", array(measure(q).read() for q in reference))


cases = request["cases"]
shots = (entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=92)
         .stabilizer_sim().with_seed(7).with_timeout(timedelta(seconds=60))
         .run_per_shot([case["args"] for case in cases]).collated_shots())
assert len(shots) == len(cases)
for shot, case in zip(shots, cases):
    assert shot["accepted"] == [case["accepted"]], (case, shot)
    assert shot["data"] == shot["reference"] == [[0] * 32], (case, shot)
print(len(cases), "encoded Bell cases passed")
''', dict(inverse=inverse_encoder(), cases=cases))

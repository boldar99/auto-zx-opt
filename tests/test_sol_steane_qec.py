"""Steane corrections, encoded Bell preservation, and ancilla failure boundaries."""

from itertools import combinations

import stim

from spiderwarp.utils import load_state_prep_circuit
from test_sol_logical_measurement import run_guppy_check, syndrome


def inverse_encoder():
    encoder = stim.Circuit()
    for op in load_state_prep_circuit("misc", "32_20_4_zero_d1"):
        if op.name == "RX":
            encoder.append("H", op.targets_copy())
        elif op.name != "R":
            encoder.append(op)
    return [(op.name, [q.value for q in op.targets_copy()]) for op in encoder.inverse()]


def test_steane_qec_preserves_encoded_bells_and_corrects_paulis(run_guppy_check):
    singles = [1 << q for q in range(32)]
    doubles = [(1 << a) | (1 << b) for a, b in combinations(range(32), 2)]
    cases = []
    for kind in (0, 1):  # Z checks / X checks
        for error in [0] + singles + doubles:
            accepted = error.bit_count() <= 1
            correction = error if accepted else 0
            x_error, z_error = (error, 0) if kind == 0 else (0, error)
            cases.append(dict(
                args=dict(kind=kind, x_error=x_error, z_error=z_error,
                          undo_x=x_error ^ (correction if kind == 0 else 0),
                          undo_z=z_error ^ (correction if kind == 1 else 0)),
                accepted=int(accepted), prep_flags=0,
                syndrome=syndrome(error), correction=correction,
            ))
        # Correct one component of Y, leaving its other component for the caller.
        for error in singles:
            cases.append(dict(
                args=dict(kind=kind, x_error=error, z_error=error,
                          undo_x=error if kind == 1 else 0,
                          undo_z=error if kind == 0 else 0),
                accepted=1, prep_flags=0, syndrome=syndrome(error), correction=error,
            ))

    errors = [(0, 0)] + [(e, 0) for e in singles] + [(0, e) for e in singles]
    errors += [(e, e) for e in singles]
    errors += [(1 << q, 1 << ((q + 1) % 32)) for q in range(32)]
    errors += [(e, 0) for e in doubles[:32]] + [(0, e) for e in doubles[:32]]
    for x_error, z_error in errors:
        z_ok = x_error.bit_count() <= 1
        x_ok = z_error.bit_count() <= 1
        cases.append(dict(
            args=dict(kind=2, x_error=x_error, z_error=z_error,
                      undo_x=0 if z_ok else x_error,
                      undo_z=0 if z_ok and x_ok else z_error),
            accepted=int(z_ok and x_ok), prep_flags=0, syndrome=0, correction=0,
        ))
    # Reject deliberately bad ancillas before touching the entangled data.
    for kind in (3, 4, 5):
        for x_error, z_error in ((0, 0), (1, 0), (0, 1 << 31)):
            cases.append(dict(
                args=dict(kind=kind, x_error=x_error, z_error=z_error,
                          undo_x=x_error, undo_z=z_error),
                accepted=0, prep_flags=7 if kind != 5 else 0, syndrome=0, correction=0,
            ))

    run_guppy_check('''
from datetime import timedelta
import json
from pathlib import Path
import sys
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, h, cx, x, z, measure
from sol_experiment.QEC_Steane import make_steane_qec_from_state_prep, make_steane_qec_cycle
from sol_experiment.logical_state_preparation import state_prep_32_20_4_zero_d1, state_prep_32_20_4_zero_d3

request = json.loads(Path(sys.argv[1]).read_text())
INVERSE = request["inverse"]
qec_z = make_steane_qec_from_state_prep(state_prep_32_20_4_zero_d3, "Z")
qec_x = make_steane_qec_from_state_prep(state_prep_32_20_4_zero_d3, "X")
cycle = make_steane_qec_cycle(state_prep_32_20_4_zero_d3)


@guppy
def failed_preparation(ancilla: array[qubit, 32]) -> int:
    x(ancilla[0])
    output("prep_attempt", 1)
    return 7


failed_z = make_steane_qec_from_state_prep(failed_preparation, "Z")
failed_x = make_steane_qec_from_state_prep(failed_preparation, "X")
failed_cycle = make_steane_qec_cycle(failed_preparation)


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
def bell_roundtrip(kind: int, x_error: int, z_error: int, undo_x: int, undo_z: int) -> None:
    data = array(qubit() for _ in range(32))
    reference = array(qubit() for _ in range(32))
    state_prep_32_20_4_zero_d1(data)
    state_prep_32_20_4_zero_d1(reference)
    # A maximally entangled encoded Bell input exercises all 20 logical qubits.
    for i in range(32):
        h(data[i])
    for i in range(32):
        cx(data[i], reference[i])
    for i in range(32):
        if (x_error & (1 << i)) != 0:
            x(data[i])
        if (z_error & (1 << i)) != 0:
            z(data[i])

    if kind == 2 or kind == 5:
        accepted = cycle(data) if kind == 2 else failed_cycle(data)
        prep_flags, syndrome, correction = 0, 0, 0
    else:
        if kind == 0:
            result = qec_z(data)
        elif kind == 1:
            result = qec_x(data)
        elif kind == 3:
            result = failed_z(data)
        else:
            result = failed_x(data)
        accepted = result.accepted
        prep_flags, syndrome, correction = result.prep_flags, result.syndrome, result.correction
    output("accepted", accepted)
    output("prep_flags", prep_flags)
    output("syndrome", syndrome)
    output("correction", correction)

    # Remove exactly the residual error expected from the specified stage.
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


emulator = (bell_roundtrip.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=98)
            .stabilizer_sim().with_seed(7).with_timeout(timedelta(seconds=60)))
cases = request["cases"]
shots = emulator.run_per_shot([case["args"] for case in cases]).collated_shots()
assert len(shots) == len(cases)
for shot, case in zip(shots, cases):
    for key in ("accepted", "prep_flags", "syndrome", "correction"):
        assert shot[key] == [case[key]], (key, case, shot)
    assert shot["data"] == shot["reference"] == [[0] * 32], (case, shot)
    if case["args"]["kind"] >= 3:
        assert shot["prep_attempt"] == [1], (case, shot)
print(len(cases), "encoded Bell cases passed")
''', dict(inverse=inverse_encoder(), cases=cases))


def test_steane_qec_rus_exhaustion_and_width(run_guppy_check):
    run_guppy_check('''
from datetime import timedelta
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, measure
from selene_sim.backends import ClassicalReplay
from sol_experiment.QEC_Steane import make_steane_qec_from_state_prep, make_steane_qec_cycle
from sol_experiment.logical_state_preparation import state_prep_32_20_4_zero_d1
from sol_experiment.experiment_SPAM import PREPARATION_METHODS

qec_z = make_steane_qec_from_state_prep(PREPARATION_METHODS["d3_RUS"], "Z")
qec_x = make_steane_qec_from_state_prep(PREPARATION_METHODS["d3_RUS"], "X")
cycle = make_steane_qec_cycle(PREPARATION_METHODS["d3_RUS"])


@guppy
def entry(basis_x: bool) -> None:
    data = array(qubit() for _ in range(32))
    state_prep_32_20_4_zero_d1(data)
    result = qec_x(data) if basis_x else qec_z(data)
    output("accepted", result.accepted)
    output("flags", result.prep_flags)
    output("syndrome", result.syndrome)
    output("correction", result.correction)
    output("data", array(measure(q).read() for q in data))


@guppy
def cycle_entry() -> None:
    data = array(qubit() for _ in range(32))
    state_prep_32_20_4_zero_d1(data)
    output("accepted", cycle(data))
    output("data", array(measure(q).read() for q in data))


emulator = (entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=66)
            .with_timeout(timedelta(seconds=10)))
bad = [True] + [False] * 8
for basis_x in (False, True):
    # Exhaustion returns before ancilla readout, so only terminal data is measured.
    result = (emulator.with_simulator(ClassicalReplay(measurements=[bad * 5 + [False] * 32]))
              .run(basis_x=basis_x))
    shot = result.collated_shots()[0]
    assert shot["accepted"] == [0] and shot["flags"] == [1]
    assert shot["syndrome"] == shot["correction"] == [0]

    # One failed preparation then success, followed by syndrome for qubit 0.
    record = bad + [False] * 9 + [True] + [False] * 31 + [False] * 32
    result = (emulator.with_simulator(ClassicalReplay(measurements=[record]))
              .run(basis_x=basis_x))
    shot = result.collated_shots()[0]
    assert shot["accepted"] == [1] and shot["flags"] == [0]
    assert shot["syndrome"] == [63] and shot["correction"] == [1]

# A complete cycle releases its first ancilla before preparing the second.
program = cycle_entry.with_opt_level(OptimizationLevel.Minimal)
program.compile()
shots = program.emulator(n_qubits=66).stabilizer_sim().with_shots(4).run().collated_shots()
assert all(shot["accepted"] == [1] for shot in shots)

for basis in ("Y", "z", "", None):
    try:
        make_steane_qec_from_state_prep(PREPARATION_METHODS["d3"], basis)
    except ValueError:
        pass
    else:
        raise AssertionError(basis)
''')

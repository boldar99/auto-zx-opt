"""Quantum emission stays separate from measurement reads and detector logic."""

import json

import pytest
import stim

from test_sol_logical_measurement import ROOT, CHECKS, run_guppy_check
from test_sol_steane_qec import inverse_encoder


@pytest.fixture
def quantum_calls(monkeypatch):
    pytest.importorskip("guppylang")
    import sol_experiment.circuit_utils as utils

    events = []
    allocated = []

    class LazyMeasurement:
        def __init__(self, q):
            self.q = q

        def read(self):
            raise AssertionError("The circuit emitter must not read measurement results")

    def allocate():
        q = f"ancilla{len(allocated)}"
        allocated.append(q)
        events.append(("R", q))
        return q

    def measure(q):
        events.append(("M", q))
        return LazyMeasurement(q)

    monkeypatch.setattr(utils, "qubit", allocate)
    monkeypatch.setattr(utils, "measure", measure)
    for name in ("reset", "h", "x", "y", "z", "cx", "cz"):
        def gate(*args, name=name):
            events.append((name.upper(), *args))
        monkeypatch.setattr(utils, name, gate)
    return utils, events


def test_emit_circuit_preserves_lifetimes_and_returns_lazy_measurements(quantum_calls):
    utils, events = quantum_calls
    data = ["data0", "data1"]
    measurements = utils.emit_circuit(data, [
        ("R", [0, 1]), ("X", [0]), ("R", [0]),
        ("RX", [2]), ("CX", [2, 0]), ("MX", [2]),
        ("R", [2]), ("H", [2]), ("CZ", [1, 2]), ("M", [2]),
    ], fresh_data=True)
    assert data == ["data0", "data1"]
    assert [m.q for m in measurements] == ["ancilla0", "ancilla1"]
    assert events == [
        ("X", "data0"), ("RESET", "data0"),
        ("R", "ancilla0"), ("H", "ancilla0"),
        ("CX", "ancilla0", "data0"), ("H", "ancilla0"), ("M", "ancilla0"),
        ("R", "ancilla1"), ("H", "ancilla1"),
        ("CZ", "data1", "ancilla1"), ("M", "ancilla1"),
    ]


def test_emit_circuit_resets_existing_qubits_without_reallocating(quantum_calls):
    utils, events = quantum_calls
    utils.emit_circuit(["data0"], [
        ("R", [0]), ("R", [1]), ("RX", [1]), ("M", [1]),
    ])
    assert events == [
        ("RESET", "data0"), ("R", "ancilla0"),
        ("RESET", "ancilla0"), ("H", "ancilla0"), ("M", "ancilla0"),
    ]


@pytest.mark.parametrize("operations", [
    [("T", [0])], [("CX", [0])], [("CX", [0, 0])], [("CX", [0, 1])],
    [("R", [1]), ("M", [1]), ("H", [1])], [("M", [0])],
    [("R", [1])], [("R", [-1])], [("R", [True])],
    [("R", [1]), ("M", [1, 1])],
])
def test_emit_circuit_rejects_invalid_operations(quantum_calls, operations):
    utils, _ = quantum_calls
    with pytest.raises(ValueError):
        utils.emit_circuit(["data0"], operations)


def test_optimized_operations_and_detectors_match_stim(run_guppy_check):
    ops, detectors = json.loads(run_guppy_check('''
import json
from sol_experiment.QEC_optimized_steane import optimized_steane_32_20_4_X_d4
ops, detectors = optimized_steane_32_20_4_X_d4()
print(json.dumps([ops, detectors.flags + detectors.syndrome]))
'''))
    circuit = stim.Circuit()
    for name, targets in ops:
        circuit.append(name, targets)
    for i, group in enumerate(detectors):
        assert all(type(ref) is int and -41 <= ref < 0 for ref in group)
        circuit.append("DETECTOR", [stim.target_rec(ref) for ref in group],
                       [0 if i < 11 else 1, i if i < 11 else i - 11])
    assert circuit == stim.Circuit.from_file(ROOT / "spiderwarp/assets/circuits/SECircuits/32_20_4_d4.stim")
    assert circuit.num_measurements == 41 and circuit.num_detectors == 17


def test_optimized_raw_measurement_order_and_quantum_action(run_guppy_check):
    circuit = stim.Circuit.from_file(ROOT / "spiderwarp/assets/circuits/SECircuits/32_20_4_d4.stim")
    detectors = [tuple(target.value for target in op.targets_copy())
                 for op in circuit if op.name == "DETECTOR"]
    errors = [(0, 0)]
    errors += [(1 << q, 0) for q in range(32)]
    errors += [(0, 1 << q) for q in range(32)]
    errors += [(1 << q, 1 << q) for q in range(32)]
    cases = [dict(x_error=x, z_error=z,
                  syndrome=sum(((z & check).bit_count() % 2) << i for i, check in enumerate(CHECKS)))
             for x, z in errors]
    run_guppy_check('''
from datetime import timedelta
import json
from pathlib import Path
import sys
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import Measurement, qubit, measure, h, cx, x, z
from selene_sim.backends import ClassicalReplay
from sol_experiment.logical_state_preparation import state_prep_32_20_4_zero_d1
from sol_experiment.QEC_optimized_steane import optimized_steane_32_20_4_X_d4
from sol_experiment.circuit_utils import emit_circuit

request = json.loads(Path(sys.argv[1]).read_text())
INVERSE = request["inverse"]


@guppy.comptime
def run_optimized_steane_32_20_4_X_d4(data: array[qubit, 32]) -> array[Measurement, 41]:
    operations, _ = optimized_steane_32_20_4_X_d4()
    return emit_circuit(data, operations)


@guppy
def raw_entry() -> None:
    data = array(qubit() for _ in range(32))
    measurements = run_optimized_steane_32_20_4_X_d4(data)
    output("records", array(m.read() for m in measurements))
    output("data", array(measure(q).read() for q in data))


# Any detector interpretation happens outside the emitter. Single-record
# injections distinguish all 41 returned positions, including rec[-41].
records = [[False] * 41] + [[q == i for q in range(41)] for i in range(41)]
records += [[True] * 41, [bool(i % 2) for i in range(41)]]
tails = [[bool((i + q) % 3) for q in range(32)] for i in range(len(records))]
replay = ClassicalReplay(measurements=[raw + tail for raw, tail in zip(records, tails)])
result = (raw_entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=60)
          .with_simulator(replay).with_shots(len(records))
          .with_timeout(timedelta(seconds=10)).run())
for shot, raw, tail in zip(result.collated_shots(), records, tails):
    assert shot["records"] == [raw]
    assert shot["data"] == [tail]


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
def bell_roundtrip(x_error: int, z_error: int) -> None:
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
    measurements = run_optimized_steane_32_20_4_X_d4(data)
    output("records", array(m.read() for m in measurements))
    for i in range(32):
        if (z_error & (1 << i)) != 0:
            z(data[i])
        if (x_error & (1 << i)) != 0:
            x(data[i])
    for i in range(32):
        cx(data[i], reference[i])
    for i in range(32):
        h(data[i])
    decode(data)
    decode(reference)
    output("data", array(measure(q).read() for q in data))
    output("reference", array(measure(q).read() for q in reference))


cases = request["cases"]
result = (bell_roundtrip.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=92)
          .stabilizer_sim().with_seed(7).with_timeout(timedelta(seconds=30))
          .run_per_shot([dict(x_error=c["x_error"], z_error=c["z_error"]) for c in cases]))
for shot, case in zip(result.collated_shots(), cases):
    raw = shot["records"][0]
    # Stim detector offsets refer to this extraction's local record array.
    mask = sum((sum(raw[ref] for ref in group) % 2) << i
               for i, group in enumerate(request["detectors"]))
    assert mask == case["syndrome"] << 11, (case, shot)
    assert shot["data"] == shot["reference"] == [[0] * 32], (case, shot)
''', dict(inverse=inverse_encoder(), detectors=detectors, cases=cases))

"""Check shared descriptions, compact records, and physical fault effects."""

from collections import Counter
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import pytest
import stim

from sol_experiment import circuit_descriptions as descriptions
from sol_experiment.optimized_steane_fault_catalogue import (
    build_fault_catalogue, extraction_operations, load_catalogues, save_catalogues,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "sol_experiment/assets"
ACTIONS = tuple(product(("X", "Z"), (2, 4)))


@pytest.fixture(scope="module")
def catalogues():
    return {action: build_fault_catalogue(*action) for action in ACTIONS}


@pytest.mark.parametrize("distance", (2, 4))
def test_extraction_descriptions_match_independent_assets(distance):
    operations, detectors = getattr(descriptions, f"optimized_steane_32_20_4_X_d{distance}")()
    circuit = stim.Circuit()
    for gate, qubits in operations:
        circuit.append(gate, qubits)
    for kind, groups in enumerate((detectors.flags, detectors.syndrome)):
        for row, refs in enumerate(groups):
            circuit.append("DETECTOR", [stim.target_rec(ref) for ref in refs], [kind, row])
    assert circuit == stim.Circuit.from_file(ASSETS / f"SECircuits/32_20_4_d{distance}.stim")


@pytest.mark.parametrize("distance", range(1, 5))
def test_preparation_descriptions_match_independent_assets(distance):
    circuit = stim.Circuit()
    for gate, qubits in getattr(descriptions, f"state_prep_ops_32_20_4_zero_d{distance}")():
        circuit.append(gate, qubits)
    assert circuit == stim.Circuit.from_file(ASSETS / f"FTStatePrep/32_20_4_zero_d{distance}.stim")


@pytest.mark.parametrize("action", ACTIONS)
def test_catalogue_keeps_every_physical_alternative(catalogues, action):
    catalogue = catalogues[action]
    basis, distance = action
    operations, detectors = extraction_operations(*action)
    assert len(detectors.flags) == (6 if distance == 2 else 11)
    assert len(detectors.syndrome) == 6
    counts = Counter(op.name for op in operations)
    expected = sum(15 if len(op.targets_copy()) == 2 else 1 if op.name in {"R", "M"} else 3
                   for op in operations)
    assert len(catalogue.faults) == expected
    assert counts["CX"] == (121 if distance == 2 else 156)
    assert counts["H"] == (43 if distance == 2 else 54) + (64 if basis == "Z" else 0)
    per_operation = {}
    for fault in catalogue.faults:
        assert fault.data_x >> 32 == fault.data_z >> 32 == 0
        per_operation.setdefault(fault.operation_index, set()).add(fault.pauli)
    for index, op in enumerate(operations):
        expected_paulis = {"X"} if op.name in {"R", "M"} else {
            "".join(p) for p in product("IXYZ", repeat=len(op.targets_copy())) if any(c != "I" for c in p)
        }
        assert per_operation[index] == expected_paulis

    groups = catalogue.group_by_observation()
    assert sum(map(len, groups.values())) == len(catalogue.faults)
    assert any(len({(f.data_x, f.data_z) for f in group}) > 1 for group in groups.values())
    assert any(not (f.flags or f.syndrome or f.data_x or f.data_z) for f in catalogue.faults)


def encoded_bell():
    """Prepare and unprepare all 20 logical Bell pairs with a reference at 60..91."""
    preparation = descriptions.state_prep_ops_32_20_4_zero_d1()
    prepare = stim.Circuit()
    encoder = stim.Circuit()
    for gate, qubits in preparation:
        prepare.append(gate, qubits)
        prepare.append(gate, [q + 60 for q in qubits])
        if gate == "RX":
            encoder.append("H", qubits)
        elif gate != "R":
            encoder.append(gate, qubits)
    prepare.append("H", range(32))
    pairs = [q for i in range(32) for q in (i, i + 60)]
    prepare.append("CX", pairs)
    undo = stim.Circuit()
    undo.append("CX", pairs)
    undo.append("H", range(32))
    for op in encoder.inverse():
        undo.append(op)
        undo.append(op.name, [t.value + 60 for t in op.targets_copy()])
    return prepare, undo


@pytest.mark.parametrize("action", ACTIONS)
def test_every_fault_matches_tableau_simulation_on_encoded_bells(catalogues, action):
    catalogue = catalogues[action]
    operations, _ = extraction_operations(*action)
    prefixes = [stim.Circuit()]
    for operation in operations:
        prefix = prefixes[-1].copy()
        prefix.append(operation)
        prefixes.append(prefix)
    suffixes = [stim.Circuit() for _ in range(len(operations) + 1)]
    for index in reversed(range(len(operations))):
        suffix = stim.Circuit()
        suffix.append(operations[index])
        suffix += suffixes[index + 1]
        suffixes[index] = suffix

    # Read detector definitions independently from the saved Stim reference.
    annotations = [op for op in stim.Circuit.from_file(ASSETS / f"SECircuits/32_20_4_d{action[1]}.stim")
                   if op.name == "DETECTOR"]
    prepare, undo = encoded_bell()
    initial = stim.TableauSimulator()
    initial.do_circuit(prepare)
    for fault in catalogue.faults:
        operation = operations[fault.operation_index]
        cut = fault.operation_index + (operation.name != "M")
        sim = initial.copy(seed=fault.operation_index)
        sim.do_circuit(prefixes[cut])
        for pauli, target in zip(fault.pauli, operation.targets_copy()):
            if pauli != "I":
                getattr(sim, pauli.lower())(target.value)
        sim.do_circuit(suffixes[cut])

        measured = sim.current_measurement_record()
        observed = [0, 0]
        for annotation in annotations:
            kind, row = map(int, annotation.gate_args_copy())
            parity = sum(measured[t.value] for t in annotation.targets_copy()) % 2
            observed[kind] |= parity << row
        assert observed == [fault.flags, fault.syndrome], (action, fault)

        # Invert the predicted data Pauli and verify the entire entangled state,
        # so a logical Z error cannot hide in an encoded-zero test input.
        for q in range(32):
            if fault.data_x & (1 << q):
                sim.x(q)
            if fault.data_z & (1 << q):
                sim.z(q)
        sim.do_circuit(undo)
        assert all(sim.peek_z(q) == 1 for q in (*range(32), *range(60, 92))), (action, fault)


def test_json_round_trip_contains_only_fault_records(catalogues, tmp_path):
    named = {f"{b}_d{d}": catalogue for (b, d), catalogue in catalogues.items()}
    first = save_catalogues(tmp_path / "first.json", named)
    second = save_catalogues(tmp_path / "second.json", named)
    assert first.read_bytes() == second.read_bytes()
    assert load_catalogues(first) == named
    payload = json.loads(first.read_text())
    assert set(payload) == set(named)
    for rows in payload.values():
        assert all(set(row) == {"operation_index", "pauli", "flags", "syndrome", "data_x", "data_z"}
                   for row in rows)


def test_cli_and_python_api_do_not_import_guppy_or_spiderwarp(tmp_path):
    target = tmp_path / "catalogue.json"
    script = '''
import sys
from sol_experiment.optimized_steane_fault_catalogue import main, load_catalogues
path = main(["--bases", "Z", "--distances", "4", "--output", sys.argv[1]])
assert set(load_catalogues(path)) == {"Z_d4"}
assert not any(name.split(".")[0] in {"guppylang", "spiderwarp"} for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, str(target)], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2776 single-fault alternatives" in result.stdout


@pytest.mark.parametrize("basis,distance", (("Y", 4), ("X", 1), ("Z", 2.0)))
def test_invalid_extraction_labels_raise(basis, distance):
    with pytest.raises(ValueError):
        build_fault_catalogue(basis, distance)

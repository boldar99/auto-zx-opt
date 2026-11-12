"""Exercise the notebook's optimization and record plumbing on its real input."""

import json
from pathlib import Path

import pytest

from spiderwarp.fault_tolerance_verification import verify_fault_order_detect_or_correct


@pytest.fixture(scope="module")
def optimized_notebook():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "32_20_4.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    # These cells build and verify the circuit without writing circuit assets
    # or invoking the optional Guppy export at the end of the notebook.
    for index in (0, 2, 3, 5, 6):
        source = "".join(notebook["cells"][index]["source"])
        source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        exec(compile(source, f"{path}:cell{index}", "exec"), namespace)
    return namespace


def test_notebook_applies_bends_and_reduces_hardware(optimized_notebook):
    ns = optimized_notebook
    assert len(ns["optimised_graph"].paths) < len(ns["covered"].paths)
    assert ns["final_circ"].num_qubits < ns["baseline_hardware_qubits"]
    mapping = ns["final_meas_map"]
    assert set(mapping) == set(range(ns["final_circ"].num_measurements))
    assert len(set(mapping.values())) == len(mapping)
    assert set(range(ns["num_flags"])) <= set(mapping.values())
    absorbed = set(range(ns["se"].num_measurements)) - set(mapping.values())
    assert absorbed
    assert absorbed <= set(range(ns["num_flags"], ns["se"].num_measurements))


def test_notebook_rejects_lost_flags_and_ambiguous_provenance(optimized_notebook):
    ns = optimized_notebook
    mapping = ns["final_meas_map"].copy()
    flag_index = next(index for index, original in mapping.items() if original == 0)
    # Replace a required flag with an absorbed syndrome record, retaining
    # complete keys and unique values so the missing-flag check is exercised.
    absorbed = set(range(ns["se"].num_measurements)) - set(mapping.values())
    mapping[flag_index] = min(absorbed)
    with pytest.raises(ValueError, match="missing flag IDs: \\[0\\]"):
        ns["add_detectors"](ns["final_circ"], mapping)

    mapping[flag_index] = next(value for key, value in mapping.items() if key != flag_index)
    with pytest.raises(ValueError, match="unique original measurement IDs"):
        ns["add_detectors"](ns["final_circ"], mapping)


def test_optimized_notebook_fault_orders_preserve_arbitrary_logical_states(optimized_notebook):
    ns = optimized_notebook
    code = ns["code"]
    circuit = ns["add_detectors"](ns["final_circ"], ns["final_meas_map"])
    assert verify_fault_order_detect_or_correct(
        circuit, code.H_x, code.H_z, code.L_x, code.L_z,
        correctable_faults=1, detectable_faults=2,
        flag_detector_indices=range(ns["num_flags"]),
        prepared_state="arbitrary", verbose=False,
    ) is True

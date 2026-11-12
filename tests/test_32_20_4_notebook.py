"""Exercise the notebook's optimization and record plumbing on its real input."""

import ast
import json
from pathlib import Path

import pytest

from spiderwarp.fault_tolerance_verification import verify_fault_order_detect_or_correct


@pytest.fixture(scope="module", params=(1, 2, 4), ids=lambda d: f"d{d}")
def optimized_notebook(request):
    path = Path(__file__).resolve().parents[1] / "notebooks" / "32_20_4.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    # These cells build and verify the circuit without writing circuit assets
    # or invoking the optional Guppy export at the end of the notebook.
    for index in (0, 2, 3, 5, 6):
        source = "".join(notebook["cells"][index]["source"])
        source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        tree = ast.parse(source)
        if index == 2:
            # Exercise every supported distance regardless of the user's
            # current selection in the interactive notebook.
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "circuit_distance"
                    for target in statement.targets
                ):
                    statement.value = ast.Constant(request.param)
            ast.fix_missing_locations(tree)
        exec(compile(tree, f"{path}:cell{index}", "exec"), namespace)
    return namespace


def test_notebook_uses_the_selected_gadget_fault_budget(optimized_notebook):
    ns = optimized_notebook
    expected = {1: (0, 0), 2: (0, 1), 4: (1, 2)}[ns["circuit_distance"]]
    assert (ns["correctable_faults"], ns["detectable_faults"]) == expected
    assert ns["code_distance"] == 4


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


def test_notebook_refines_reuse_without_losing_the_greedy_incumbent(optimized_notebook):
    ns = optimized_notebook
    assert ns["reuse_plan"].peak_qubits <= ns["greedy_reuse_plan"].peak_qubits
    if ns["circuit_distance"] == 4 and ns["refined_reuse_plan"].solver_status in {"FEASIBLE", "OPTIMAL"}:
        assert ns["final_circ"].num_qubits - ns["code"].n <= 29


def test_notebook_rejects_lost_flags_and_ambiguous_provenance(optimized_notebook):
    ns = optimized_notebook
    mapping = ns["final_meas_map"].copy()
    # Replace a required flag with an absorbed syndrome record, retaining
    # complete keys and unique values so the missing-flag check is exercised.
    if ns["num_flags"]:
        flag_index = next(index for index, original in mapping.items() if original == 0)
        absorbed = set(range(ns["se"].num_measurements)) - set(mapping.values())
        mapping[flag_index] = min(absorbed)
        with pytest.raises(ValueError, match="missing flag IDs: \\[0\\]"):
            ns["add_detectors"](ns["final_circ"], mapping)

    mapping = ns["final_meas_map"].copy()
    first, second = list(mapping)[:2]
    mapping[first] = mapping[second]
    with pytest.raises(ValueError, match="unique original measurement IDs"):
        ns["add_detectors"](ns["final_circ"], mapping)


def test_optimized_notebook_fault_orders_preserve_arbitrary_logical_states(optimized_notebook):
    ns = optimized_notebook
    code = ns["code"]
    circuit = ns["add_detectors"](ns["final_circ"], ns["final_meas_map"])
    assert verify_fault_order_detect_or_correct(
        circuit, code.H_x, code.H_z, code.L_x, code.L_z,
        correctable_faults=ns["correctable_faults"],
        detectable_faults=ns["detectable_faults"],
        flag_detector_indices=range(ns["num_flags"]),
        prepared_state="arbitrary", verbose=False,
    ) is True


def test_final_fault_order_analysis_uses_the_same_budget(optimized_notebook, monkeypatch, tmp_path):
    namespace = optimized_notebook.copy()
    path = Path(__file__).resolve().parents[1] / "notebooks" / "32_20_4.ipynb"
    notebook = json.loads(path.read_text())
    expected = (namespace["correctable_faults"], namespace["detectable_faults"])
    namespace["flagged_syndrome_circuit"] = namespace["add_detectors"](
        namespace["final_circ"], namespace["final_meas_map"]
    )
    # Keep any failure witnesses local to the test if this regresses.
    monkeypatch.chdir(tmp_path)
    source = "".join(notebook["cells"][8]["source"])
    exec(compile(source, f"{path}:cell8", "exec"), namespace)
    assert (namespace["correctable_faults"], namespace["detectable_faults"]) == expected
    assert all(result is True for result in namespace["se_fault_orders"].values())

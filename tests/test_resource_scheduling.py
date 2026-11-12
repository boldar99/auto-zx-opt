import pytest
import stim

from spiderwarp.resource_scheduling import ReuseTarget, plan_resource_aware_reuse


def test_exact_reuse_keeps_data_live_before_their_first_gate():
    pytest.importorskip("ortools.sat.python.cp_model")
    # Each ancilla pair must coexist at its CNOT. Data 0 and 1 already hold
    # quantum information before their first explicit gates in the second pair.
    circuit = stim.Circuit(
        "R 2 3\nCX 2 3\nM 2 3\n"
        "R 4 5\nCX 4 5\nCX 4 0 5 1\nM 4 5"
    )
    # Exercise the CP-SAT backend rather than the <=22-event DP fallback.
    for _ in range(24):
        circuit.append("H", [0])

    plan = plan_resource_aware_reuse(
        circuit, 2, heuristic="decross_exact", max_time_seconds=1,
        target=ReuseTarget.QUBITS,
    )
    assert plan.solver_status == "OPTIMAL"
    assert plan.objective_bound == plan.peak_qubits == plan.circuit.num_qubits == 4
    assert plan.logical_to_physical[0] == 0
    assert plan.logical_to_physical[1] == 1
    assert all(plan.logical_to_physical[q] >= 2 for q in range(2, 6))

    # Record order can change, but each noiseless measured pair stays zero.
    assert not plan.circuit.compile_sampler(seed=17).sample(16).any()

from types import SimpleNamespace
from unittest.mock import patch

import stim

from spiderwarp.path_cover_metrics import metric_hardware_qubits_resource_scheduled
from spiderwarp.resource_scheduling import ReuseTarget


def test_resource_scheduled_metric_uses_requested_heuristic() -> None:
    circuit = stim.Circuit("R 0")
    candidate = SimpleNamespace(
        paths=None,
        _num_qubits=1,
        extract_circuit=lambda: circuit,
    )
    graph = SimpleNamespace(shallow_copy=lambda: candidate)
    paths = {0: (0,)}

    with patch(
        "spiderwarp.resource_scheduling.plan_resource_aware_reuse",
        return_value=SimpleNamespace(peak_qubits=3),
    ) as plan_reuse:
        metric = metric_hardware_qubits_resource_scheduled("decross_greedy")

        assert metric(graph, paths) == 3.0

    assert candidate.paths is paths
    plan_reuse.assert_called_once_with(
        circuit,
        1,
        heuristic="decross_greedy",
        target=ReuseTarget.QUBITS,
    )

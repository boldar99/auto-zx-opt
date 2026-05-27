import networkx as nx

from typing import TYPE_CHECKING, Protocol

from spiderwarp.qubit_reuse import dag_to_circuit
from spiderwarp.stim_utils import get_circuit_depth, get_spacetime_volume

if TYPE_CHECKING:
    from spiderwarp.path_cover import CoveredZXGraph
    


class PathCostFunction(Protocol):
    """Protocol defining the signature for MCTS cost functions."""

    def __call__(self, graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
        ...


class LexicographicCost:
    """Flattens primary and secondary metrics into a UCT-compatible scalar."""

    def __init__(self, primary: PathCostFunction, secondary: PathCostFunction, secondary_weight: float = 1e-3):
        self.primary = primary
        self.secondary = secondary
        self.weight = secondary_weight

    def __call__(self, graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
        return self.primary(graph, paths) + (self.weight * self.secondary(graph, paths))


def metric_parity_measurements(graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
    return graph._num_parity_measurement(paths)


def metric_hardware_qubits_exact(ReuseStrategy):
    def metric_hardware_qubits_exact_with_qubit_reuse(graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
        """
        Exact post-reuse hardware qubit count utilizing the heavy NetworkX injection pipeline.
        WARNING: Do not use inside the MCTS inner loop due to cycle-checking overhead.
        """
        from spiderwarp.qubit_reuse import build_circuit_dag, inject_qubit_reuse

        temp_graph = graph.shallow_copy()
        temp_graph.paths = paths

        dag = build_circuit_dag(temp_graph)
        _, _, total_hw = inject_qubit_reuse(dag, graph._num_qubits, ReuseStrategy())

        return float(total_hw)

    return metric_hardware_qubits_exact_with_qubit_reuse


def metric_depth(graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
    """Calculates depth by finding the longest path in the causal dependency DAG."""
    flow_dag = graph._construct_flow_graph(paths)
    # The number of layers is the longest path length + 1
    return float(nx.dag_longest_path_length(flow_dag) + 1)


def metric_depth_exact(ReuseStrategy):
    def metric_depth_exact_with_qubit_reuse(graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
        """Calculates depth by finding the longest path in the causal dependency DAG."""
        from spiderwarp.qubit_reuse import build_circuit_dag, inject_qubit_reuse, apply_logical_qubit_merge_and_compress

        temp_graph = graph.shallow_copy()
        temp_graph.paths = paths

        dag = build_circuit_dag(temp_graph)
        mod_dag, _, _ = inject_qubit_reuse(dag, graph._num_qubits, ReuseStrategy())
        compressed_dag = apply_logical_qubit_merge_and_compress(mod_dag, graph._num_qubits)
        circ, _ = dag_to_circuit(compressed_dag)

        return float(get_circuit_depth(circ))
    return metric_depth_exact_with_qubit_reuse


def metric_num_paths(graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
    return len(paths)


def metric_spacetime_volume(graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
    """
    Calculates Spacetime Volume: Sum of the active duration of all hardware paths.
    This acts as a highly effective secondary metric to break ties.
    """
    flow_dag = graph._construct_flow_graph(paths)

    asap_times: dict[int, int] = {}
    for node in nx.topological_sort(flow_dag):
        preds = list(flow_dag.predecessors(node))
        asap_times[node] = max((asap_times[p] for p in preds), default=0) + 1

    volume = 0
    for path_nodes in paths.values():
        birth_layer = asap_times[path_nodes[0]]
        death_layer = asap_times[path_nodes[-1]]
        # +1 because inclusive of both the birth and death tick
        volume += (death_layer - birth_layer + 1)

    return float(volume)


def metric_spacetime_volume_exact(ReuseStrategy):
    def metric_spacetime_volume_exact_with_qubit_reuse(graph: "CoveredZXGraph", paths: dict[int, tuple[int, ...]]) -> float:
        """
        Exact spacetime volume calculated after full hardware qubit reuse.

        Volume is defined as the sum of the active lifespans (death layer - birth layer + 1)
        of all hardware qubits. Because aggressive reuse injects new causal dependencies,
        the overall depth and individual lifespans will shift compared to the fast proxy.

        WARNING: Uses the heavy NetworkX injection pipeline. Do not use inside the MCTS inner loop.
        """
        # Local import strictly required to prevent circular dependencies with qubit_reuse.py
        from spiderwarp.qubit_reuse import (
            build_circuit_dag,
            apply_logical_qubit_merge_and_compress,
            inject_qubit_reuse
        )

        # 1. Isolate the candidate state
        temp_graph = graph.shallow_copy()
        temp_graph.paths = paths

        # 2. Run the exact routing pipeline
        dag = build_circuit_dag(temp_graph)
        mod_dag, _, total_hw = inject_qubit_reuse(dag, graph._num_qubits, ReuseStrategy())

        # We MUST compress the DAG so the targets reflect the shared hardware tracks
        compressed_dag = apply_logical_qubit_merge_and_compress(mod_dag, graph._num_qubits)

        circ, _ = dag_to_circuit(compressed_dag)

        return float(get_spacetime_volume(circ))
    return metric_spacetime_volume_exact_with_qubit_reuse

from __future__ import annotations

import networkx as nx
import numpy as np
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.utils import load_state_prep_circuit
from spiderwarp.stim_utils import steane_se_from_stim_state_prep
from spiderwarp.path_cover import CoveredZXGraph

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING



@dataclass
class RoutingState:
    """Holds the live state of the DAG and logical-to-physical tracking pointers."""
    dag: nx.DiGraph
    n_data: int
    ancillas: set[int]
    next_q: dict[int, int]
    prev_q: dict[int, int]
    birth_node: dict[int, int]
    death_node: dict[int, int]
    data_birth: dict[int, int]
    data_death: dict[int, int]


class ReuseStrategy(Protocol):
    """Protocol defining the rules for evaluating a qubit reuse dependency."""

    def setup(self, state: RoutingState) -> None:
        """Initialize any baseline metrics before routing begins."""
        ...

    def evaluate_candidate(self, state: RoutingState) -> float:
        """Evaluate the modified state. Return cost (lower is better) or float('inf') to reject."""
        ...

    def commit_edge(self, state: RoutingState) -> None:
        """Called by the router when an edge is permanently committed."""
        ...


class NoReuseStrategy:
    """
    Evaluates candidates purely by their resulting circuit depth.
    Guarantees minimum hardware qubits while selecting the permutations
    that bloat the depth the least.
    """

    def setup(self, state: RoutingState) -> None:
        pass

    def evaluate_candidate(self, state: RoutingState) -> float:
        return float('inf')

    def commit_edge(self, state: RoutingState) -> None:
        # Stateless evaluation: no cached baselines to update
        pass



class AggressiveDepthAwareStrategy:
    """
    Evaluates candidates purely by their resulting circuit depth.
    Guarantees minimum hardware qubits while selecting the permutations
    that bloat the depth the least.
    """

    def setup(self, state: RoutingState) -> None:
        pass

    def evaluate_candidate(self, state: RoutingState) -> float:
        return float(nx.dag_longest_path_length(state.dag))

    def commit_edge(self, state: RoutingState) -> None:
        # Stateless evaluation: no cached baselines to update
        pass


class PureAggressiveStrategy:
    """Treats all valid edges equally. The router will pick the first valid one."""

    def setup(self, state: RoutingState) -> None:
        pass

    def evaluate_candidate(self, state: RoutingState) -> float:
        return 0.0

    def commit_edge(self, state: RoutingState) -> None:
        # Stateless evaluation: nothing to update
        pass


class DepthPreservingStrategy:
    """Rejects any edge that increases the original baseline depth."""

    def setup(self, state: RoutingState) -> None:
        # The baseline is the original depth before ANY routing occurs
        self.baseline_depth = nx.dag_longest_path_length(state.dag)

    def evaluate_candidate(self, state: RoutingState) -> float:
        current_depth = nx.dag_longest_path_length(state.dag)
        if current_depth > self.baseline_depth:
            return float('inf')  # Outright reject
        return 0.0

    def commit_edge(self, state: RoutingState) -> None:
        # Stateful, but the baseline represents a static original constraint.
        # Do not update self.baseline_depth here.
        pass


class VolumeOptimizingReuseStrategy:
    """
    Evaluates candidates by exact active spacetime volume.
    Strictly rejects any edge that does not result in a net volume decrease.
    """

    def setup(self, state: RoutingState) -> None:
        self.current_vol = self._compute_active_volume(state)

    def evaluate_candidate(self, state: RoutingState) -> float:
        new_vol = self._compute_active_volume(state)

        # Strict constraint: Must strictly decrease volume
        if new_vol < self.current_vol:
            return float(new_vol)

        return float('inf')

    def commit_edge(self, state: RoutingState) -> None:
        # Update the baseline to reflect the newly compressed DAG
        self.current_vol = self._compute_active_volume(state)

    @staticmethod
    def _compute_active_volume(state: RoutingState) -> int:
        asap = {}
        for n in nx.topological_sort(state.dag):
            preds = list(state.dag.predecessors(n))
            asap[n] = max((asap[p] for p in preds), default=0) + 1

        vol = 0
        # 1. Data qubit volume
        for q in range(state.n_data):
            if q in state.data_birth and q in state.data_death:
                vol += (asap[state.data_death[q]] - asap[state.data_birth[q]] + 1)

        # 2. Physical ancilla chain volume
        for q in state.ancillas:
            if q not in state.prev_q:  # Found the root logical qubit of a physical chain
                curr = q
                chain_birth = asap[state.birth_node[curr]]
                chain_death = chain_birth

                while curr is not None:
                    chain_death = asap[state.death_node[curr]]
                    curr = state.next_q.get(curr)

                vol += (chain_death - chain_birth + 1)

        return vol


def build_circuit_dag(cv: CoveredZXGraph) -> nx.DiGraph:
    """
    Converts a sequential list of quantum operations into a dependency DAG.
    Stores the original operation name, targets, and measurement_id in the node attributes.
    """
    dag = nx.DiGraph()
    last_op_on_qubit: dict[int,int] = {}

    for i, circ_op in enumerate(cv.circuit_operations()):
        op_name = circ_op.name
        targets = circ_op.targets
        measurement_id = circ_op.measurement_id

        # Normalize targets to a tuple for consistent storage
        qubits = tuple(targets) if isinstance(targets, (list, tuple)) else (targets,)

        # Rely strictly on the explicit measurement_id from CircuitOperation
        dag.add_node(i, op_name=op_name, targets=qubits, measurement_id=measurement_id)

        # Determine dependencies based on qubit usage
        dependencies = set()
        for q in qubits:
            if q in last_op_on_qubit:
                dependencies.add(last_op_on_qubit[q])
            # Update the tracker: this node 'i' is now the latest operation on qubit 'q'
            last_op_on_qubit[q] = i

        # Draw the edges from dependencies to the current operation
        for dep in dependencies:
            dag.add_edge(dep, i)

    return dag


def dag_to_circuit(dag: nx.DiGraph) -> tuple[stim.Circuit, dict[int, int]]:
    """
    Converts a circuit dependency DAG back into a Stim circuit and a measurement map.
    Extraction MUST be done in topological order to respect causality constraints.
    """
    circuit = stim.Circuit()
    measurement_map: dict[int, int] = {}
    next_measurement_index = 0

    # Ensure operations are ordered correctly respecting the DAG's causal flow
    for node in nx.topological_sort(dag):
        data = dag.nodes[node]
        op_name = data.get("op_name")
        targets = data.get("targets")
        measurement_id = data.get("measurement_id")

        targets_list = list(targets) if isinstance(targets, tuple) else targets
        circuit.append(op_name, targets_list)

        # Dynamically build the measurement map for tracking
        if op_name in CoveredZXGraph.MEASUREMENT_OPS:
            if measurement_id is not None:
                for offset in range(len(targets_list)):
                    measurement_map[next_measurement_index + offset] = measurement_id + offset
            next_measurement_index += len(targets_list)

    return circuit, measurement_map


def inject_qubit_reuse(dag: nx.DiGraph, n_data: int, strategy: ReuseStrategy):
    """
    Best-Fit routing engine. Groups potential reuse dependencies by target,
    tests all valid sources, and commits the edge with the lowest strategy cost.
    """
    mod_dag = dag.copy()
    topo_order = list(nx.topological_sort(mod_dag))

    ancillas = set()
    birth_node, death_node = {}, {}
    data_birth, data_death = {}, {}

    for node in topo_order:
        targets = mod_dag.nodes[node].get("targets", ())
        for q in targets:
            if q >= n_data:
                ancillas.add(q)
                if q not in birth_node: birth_node[q] = node
                death_node[q] = node
            else:
                if q not in data_birth: data_birth[q] = node
                data_death[q] = node

    state = RoutingState(
        dag=mod_dag,
        n_data=n_data,
        ancillas=ancillas,
        next_q={},
        prev_q={},
        birth_node=birth_node,
        death_node=death_node,
        data_birth=data_birth,
        data_death=data_death
    )

    strategy.setup(state)

    # Group candidates by the target Reset node (qB)
    # We want to find the best Qubit A to feed into Qubit B
    candidates_by_target = {qB: [] for qB in ancillas}

    for qA in ancillas:
        for qB in ancillas:
            if qA == qB:
                continue
            dA = death_node[qA]
            bB = birth_node[qB]

            # Fast topological rejection
            if not nx.has_path(mod_dag, bB, dA):
                candidates_by_target[qB].append((qA, dA, bB))

    # Evaluate best-fit for each target qubit
    for qB, sources in candidates_by_target.items():
        if qB in state.prev_q:
            continue

        best_cost = float('inf')
        best_qA = None
        best_dA = None
        best_bB = None

        for qA, dA, bB in sources:
            if qA in state.next_q:
                continue

            # Tentatively apply the edge
            state.dag.add_edge(dA, bB)

            # 1. Hardware Constraint
            if not nx.is_directed_acyclic_graph(state.dag):
                state.dag.remove_edge(dA, bB)
                continue

            # 2. Update tracking state for accurate strategy evaluation
            state.next_q[qA] = qB
            state.prev_q[qB] = qA

            # 3. Strategy Evaluation
            cost = strategy.evaluate_candidate(state)

            if cost < best_cost:
                best_cost = cost
                best_qA = qA
                best_dA = dA
                best_bB = bB

            # Revert the tentative changes to test the next source
            state.dag.remove_edge(dA, bB)
            del state.next_q[qA]
            del state.prev_q[qB]

            # If we found at least one valid, non-rejected source, commit the best one permanently
            if best_qA is not None and best_cost != float('inf'):
                state.dag.add_edge(best_dA, best_bB)
                state.next_q[best_qA] = qB
                state.prev_q[qB] = best_qA

                # ---> ADD THESE TWO LINES <---
                # Notify the strategy that the graph architecture has permanently changed
                strategy.commit_edge(state)

    # Hardware Allocation Mapping
    logical_to_physical = {q: q for q in range(n_data)}
    next_hw = n_data

    for q in ancillas:
        if q not in state.prev_q:
            curr = q
            while curr is not None:
                logical_to_physical[curr] = next_hw
                curr = state.next_q.get(curr)
            next_hw += 1

    return state.dag, logical_to_physical, next_hw


def apply_logical_qubit_merge_and_compress(dag: nx.DiGraph, n_data: int) -> nx.DiGraph:
    """
    1. Merges logical qubits based on M -> R injected edges.
    2. Compresses the remaining active qubit IDs so they are contiguous.
    """
    mod_dag = dag.copy()

    # --- PHASE 1: Union-Find for Merges ---
    parent_map: dict[int, int] = {}

    def get_root(q):
        curr = q
        while curr in parent_map:
            curr = parent_map[curr]
        return curr

    for u, v in mod_dag.edges():
        op_u = mod_dag.nodes[u].get("op_name", "")
        op_v = mod_dag.nodes[v].get("op_name", "")

        # Generalize to match CoveredZXGraph definition
        if op_u in CoveredZXGraph.MEASUREMENT_OPS and op_v in {"R", "RX"}:
            targets_u = mod_dag.nodes[u].get("targets", [])
            targets_v = mod_dag.nodes[v].get("targets", [])

            if len(targets_u) == 1 and len(targets_v) == 1:
                qA = targets_u[0]
                qB = targets_v[0]

                if qA != qB:
                    root_A = get_root(qA)
                    parent_map[qB] = root_A

    # --- PHASE 2: Collect and Compress ---
    active_roots = set()
    for node in mod_dag.nodes():
        targets = mod_dag.nodes[node].get("targets", [])
        for q in targets:
            active_roots.add(get_root(q))

    data_roots = [q for q in active_roots if q < n_data]
    ancilla_roots = sorted([q for q in active_roots if q >= n_data])

    compression_map = {}
    for q in data_roots:
        compression_map[q] = q

    next_dense_id = n_data
    for q in ancilla_roots:
        compression_map[q] = next_dense_id
        next_dense_id += 1

    # --- PHASE 3: Rewrite the DAG ---
    for node in mod_dag.nodes():
        old_targets = mod_dag.nodes[node].get("targets", [])
        new_targets = []
        for q in old_targets:
            root_q = get_root(q)
            compressed_q = compression_map[root_q]
            new_targets.append(compressed_q)

        if isinstance(old_targets, tuple):
            mod_dag.nodes[node]["targets"] = tuple(new_targets)
        else:
            mod_dag.nodes[node]["targets"] = new_targets

    return mod_dag


if __name__ == "__main__":
    from spiderwarp.path_cover import CoveredZXGraph, metric_hardware_qubits_exact

    code_name, circuit_path = "17_1_5", "cc_4_8_8_d5/zero_ft_heuristic_opt"

    code = CSSCode.load_code("FAO", code_name)
    circuit = load_state_prep_circuit("SAT", circuit_path)
    se = steane_se_from_stim_state_prep(circuit, se_basis="Z", n=code.n)

    covered = CoveredZXGraph.from_stim(se)
    covered.basic_FE_rewrites()

    optimised = covered.mcts_boundary_bends(
        cost_func=metric_hardware_qubits_exact,
        max_iterations=500,
        rollout_depth=16,
        seed=0,
    )

    print("Pre-reuse Ancilla qubits:", len(optimised.paths) - code.n)
    _, pre_measurement_map = optimised.extract_circuit_with_measurement_map()
    print("Pre-reuse Measurement map:", pre_measurement_map)

    dag = build_circuit_dag(optimised)
    mod_dag, logical_to_physical, total_hw = inject_qubit_reuse(dag, code.n, AggressiveDepthAwareStrategy())
    compressed_dag = apply_logical_qubit_merge_and_compress(mod_dag, code.n)
    final_circ, final_meas_map = dag_to_circuit(compressed_dag)

    print("\n--- Final Reused Circuit ---")
    print(final_circ)
    print("\nFinal Measurement Map:", final_meas_map)

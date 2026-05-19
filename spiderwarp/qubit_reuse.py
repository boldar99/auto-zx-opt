import networkx as nx
import matplotlib.pyplot as plt
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.utils import load_state_prep_circuit, steane_se_from_stim_state_prep

from spiderwarp.path_cover_opt import CoveredZXGraph, metric_hardware_qubits_exact


def build_circuit_dag(cv: CoveredZXGraph, n_data: int) -> nx.DiGraph:
    """
    Converts a sequential list of quantum operations into a dependency DAG.
    Stores the original operation name, targets, and measurement_id in the node attributes.
    """
    dag = nx.DiGraph()
    last_op_on_qubit = {}

    ordered_operations = cv._find_total_ordering()

    for i, circ_op in enumerate(ordered_operations):
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


def visualize_circuit_dag(di_graph: nx.DiGraph, figsize=(12, 8)):
    """
    Visualizes the circuit DAG using a multipartite layout based on
    topological generations.
    """
    for layer, nodes in enumerate(nx.topological_generations(di_graph)):
        for node in nodes:
            di_graph.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(di_graph, subset_key="layer", align="vertical")

    labels = {}
    for node, data in di_graph.nodes(data=True):
        op = data.get("op_name")
        targs = "-".join([str(t) for t in data.get("targets", ())])
        labels[node] = f"{op}\n{targs}"

        meas_id = data.get("measurement_id")
        if meas_id is not None:
            labels[node] += f"\nm_{meas_id}"

    plt.figure(figsize=figsize)
    NODE_SIZE = 1500

    nx.draw_networkx_nodes(
        di_graph, pos,
        node_color="#ADD8E6",
        node_size=NODE_SIZE,
        edgecolors="black"
    )

    nx.draw_networkx_edges(
        di_graph, pos,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="gray",
        width=1.5,
        node_size=NODE_SIZE
    )

    nx.draw_networkx_labels(
        di_graph, pos,
        labels=labels,
        font_size=8,
        font_weight="bold"
    )

    plt.title("Circuit Dependency DAG (ASAP Generations)", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def inject_aggressive_reuse(dag: nx.DiGraph, n_data: int):
    """
    Maximizes qubit reuse by explicitly injecting dependencies between
    the death (Measurement) of one qubit and the birth (Reset) of another.
    """
    mod_dag = dag.copy()
    topo_order = list(nx.topological_sort(mod_dag))

    ancillas = set()
    birth_node = {}
    death_node = {}

    for node in topo_order:
        targets = mod_dag.nodes[node].get("targets", ())
        for q in targets:
            if q >= n_data:
                ancillas.add(q)
                if q not in birth_node:
                    birth_node[q] = node
                death_node[q] = node

    asap = {}
    for node in topo_order:
        preds = list(mod_dag.predecessors(node))
        asap[node] = max((asap[p] for p in preds), default=0) + 1

    candidates = []
    for qA in ancillas:
        for qB in ancillas:
            if qA == qB:
                continue

            dA = death_node[qA]
            bB = birth_node[qB]

            if nx.has_path(mod_dag, bB, dA):
                continue

            score = asap[dA] - asap[bB]
            candidates.append((score, qA, qB, dA, bB))

    candidates.sort(key=lambda x: x[0])

    next_q = {}
    prev_q = {}

    for score, qA, qB, dA, bB in candidates:
        if qA in next_q or qB in prev_q:
            continue

        mod_dag.add_edge(dA, bB)

        if not nx.is_directed_acyclic_graph(mod_dag):
            mod_dag.remove_edge(dA, bB)
        else:
            next_q[qA] = qB
            prev_q[qB] = qA

    logical_to_physical = {q: q for q in range(n_data)}
    next_hw = n_data

    for q in ancillas:
        if q not in prev_q:
            curr = q
            while curr is not None:
                logical_to_physical[curr] = next_hw
                curr = next_q.get(curr)
            next_hw += 1

    total_hw = next_hw
    return mod_dag, logical_to_physical, total_hw


def inject_depth_preserving_reuse(dag: nx.DiGraph, n_data: int):
    """
    Maximizes qubit reuse ONLY if the injection does not increase
    the total topological depth (critical path) of the circuit.
    """
    mod_dag = dag.copy()

    # Calculate the strict critical path length of the original DAG
    orig_depth = nx.dag_longest_path_length(mod_dag)

    topo_order = list(nx.topological_sort(mod_dag))
    ancillas = set()
    birth_node = {}
    death_node = {}

    for node in topo_order:
        targets = mod_dag.nodes[node].get("targets", ())
        for q in targets:
            if q >= n_data:
                ancillas.add(q)
                if q not in birth_node:
                    birth_node[q] = node
                death_node[q] = node

    asap = {}
    for node in topo_order:
        preds = list(mod_dag.predecessors(node))
        asap[node] = max((asap[p] for p in preds), default=0) + 1

    candidates = []
    for qA in ancillas:
        for qB in ancillas:
            if qA == qB:
                continue
            dA = death_node[qA]
            bB = birth_node[qB]

            # Fast rejection: topological violation
            if nx.has_path(mod_dag, bB, dA):
                continue

            score = asap[dA] - asap[bB]
            candidates.append((score, qA, qB, dA, bB))

    candidates.sort(key=lambda x: x[0])

    next_q = {}
    prev_q = {}

    for score, qA, qB, dA, bB in candidates:
        if qA in next_q or qB in prev_q:
            continue

        mod_dag.add_edge(dA, bB)

        # 1. Check for cycles
        if not nx.is_directed_acyclic_graph(mod_dag):
            mod_dag.remove_edge(dA, bB)
            continue

        # 2. Check for global depth bloating
        new_depth = nx.dag_longest_path_length(mod_dag)
        if new_depth > orig_depth:
            mod_dag.remove_edge(dA, bB)
        else:
            next_q[qA] = qB
            prev_q[qB] = qA

    logical_to_physical = {q: q for q in range(n_data)}
    next_hw = n_data

    for q in ancillas:
        if q not in prev_q:
            curr = q
            while curr is not None:
                logical_to_physical[curr] = next_hw
                curr = next_q.get(curr)
            next_hw += 1

    total_hw = next_hw
    return mod_dag, logical_to_physical, total_hw


def _compute_active_volume(
    dag: nx.DiGraph,
    n_data: int,
    ancillas: set,
    next_q: dict,
    prev_q: dict,
    birth_node: dict,
    death_node: dict,
    data_birth: dict,
    data_death: dict
) -> int:
    """Helper to rapidly calculate exact active volume during the greedy search."""
    asap = {}
    for n in nx.topological_sort(dag):
        preds = list(dag.predecessors(n))
        asap[n] = max((asap[p] for p in preds), default=0) + 1

    vol = 0
    # Add data qubit volume
    for q in range(n_data):
        if q in data_birth and q in data_death:
            vol += (asap[data_death[q]] - asap[data_birth[q]] + 1)

    # Add physical ancilla chain volume
    for q in ancillas:
        if q not in prev_q:  # Found the root logical qubit of a physical chain
            curr = q
            chain_birth = asap[birth_node[curr]]
            chain_death = chain_birth

            while curr is not None:
                chain_death = asap[death_node[curr]]
                curr = next_q.get(curr)

            vol += (chain_death - chain_birth + 1)

    return vol


def inject_volume_optimizing_reuse(dag: nx.DiGraph, n_data: int):
    """
    Injects dependencies greedily, committing the reuse ONLY if it results
    in a net decrease in the total spacetime volume of the circuit.
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

    next_q, prev_q = {}, {}

    # Calculate baseline volume before any routing
    current_vol = _compute_active_volume(
        mod_dag, n_data, ancillas, next_q, prev_q,
        birth_node, death_node, data_birth, data_death
    )

    asap = {}
    for n in topo_order:
        preds = list(mod_dag.predecessors(n))
        asap[n] = max((asap[p] for p in preds), default=0) + 1

    candidates = []
    for qA in ancillas:
        for qB in ancillas:
            if qA == qB:
                continue
            dA = death_node[qA]
            bB = birth_node[qB]
            if not nx.has_path(mod_dag, bB, dA):
                score = asap[dA] - asap[bB]
                candidates.append((score, qA, qB, dA, bB))

    candidates.sort(key=lambda x: x[0])

    for score, qA, qB, dA, bB in candidates:
        if qA in next_q or qB in prev_q:
            continue

        mod_dag.add_edge(dA, bB)

        if not nx.is_directed_acyclic_graph(mod_dag):
            mod_dag.remove_edge(dA, bB)
        else:
            temp_next = next_q.copy()
            temp_next[qA] = qB
            temp_prev = prev_q.copy()
            temp_prev[qB] = qA

            # Evaluate the global volume trade-off
            new_vol = _compute_active_volume(
                mod_dag, n_data, ancillas, temp_next, temp_prev,
                birth_node, death_node, data_birth, data_death
            )

            if new_vol < current_vol:
                # The merge saved more volume than the delay cost
                current_vol = new_vol
                next_q = temp_next
                prev_q = temp_prev
            else:
                # The delay bloated the active lifespan of other qubits; reject it
                mod_dag.remove_edge(dA, bB)

    logical_to_physical = {q: q for q in range(n_data)}
    next_hw = n_data
    for q in ancillas:
        if q not in prev_q:
            curr = q
            while curr is not None:
                logical_to_physical[curr] = next_hw
                curr = next_q.get(curr)
            next_hw += 1

    return mod_dag, logical_to_physical, next_hw


def apply_logical_qubit_merge_and_compress(dag: nx.DiGraph, n_data: int) -> nx.DiGraph:
    """
    1. Merges logical qubits based on M -> R injected edges.
    2. Compresses the remaining active qubit IDs so they are contiguous.
    """
    mod_dag = dag.copy()

    # --- PHASE 1: Union-Find for Merges ---
    parent_map = {}

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

    # 1. Build the DAG
    dag = build_circuit_dag(optimised, code.n)

    # 2. Inject dependency edges for reuse
    mod_dag, logical_to_physical, total_hw = inject_aggressive_reuse(dag, code.n)

    # 3. Compress target IDs based on those edges
    compressed_dag = apply_logical_qubit_merge_and_compress(mod_dag, code.n)

    # 4. Extract topologically
    final_circ, final_meas_map = dag_to_circuit(compressed_dag)

    print("\n--- Final Reused Circuit ---")
    print(final_circ)
    print("\nFinal Measurement Map:", final_meas_map)
import networkx as nx
import matplotlib.pyplot as plt

from spiderwarp.path_cover_opt import CoveredZXGraph


def build_circuit_dag(cv: CoveredZXGraph, n_data: int) -> nx.DiGraph:
    """
    Converts a sequential list of quantum operations into a dependency DAG.
    Stores the original operation name and targets in the node attributes.
    """
    dag = nx.DiGraph()
    last_op_on_qubit = {}

    ordered_operations = cv._find_total_ordering()

    matrix_indices = dict(zip(cv.measurement_qubit_indices(), cv.matrix_transformation_indices()))
    flags = cv.flag_qubit_indices()


    for i, (op_name, targets) in enumerate(ordered_operations):
        # Normalize targets to a tuple for consistent storage and display
        qubits = tuple(targets) if isinstance(targets, (list, tuple)) else (targets,)

        # Explicitly store op_name and targets as node attributes
        dag.add_node(i, op_name=op_name, targets=qubits)
        if op_name in ("M", "MX"):
            [q] = qubits
            q -= n_data
            dag.nodes[i]["is_flag"] = q in flags
            dag.nodes[i]["matrix_index"] = matrix_indices.get(q)

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


def dag_to_circuit(dag: nx.DiGraph) -> tuple[list[tuple[str, list]], list[int], dict[int, int]]:
    """
    Converts a circuit dependency DAG back into a sequential list of operations.

    Args:
        dag: The directed acyclic graph representing the circuit operations.
    """
    ordered_operations = []
    flags = []
    matrix_indices = {}
    meas_id = 0
    for node in dag.nodes():
        data = dag.nodes[node]
        op_name = data.get("op_name")
        targets = data.get("targets")
        if op_name in ("M", "MX"):
            is_flag = data.get("is_flag")
            matrix_index = data.get("matrix_index")
            if is_flag:
                flags.append(meas_id)
            else:
                matrix_indices[meas_id] = matrix_index
            meas_id += 1

        # Convert targets back to a list if stored as a tuple
        targets_list = list(targets) if isinstance(targets, tuple) else targets

        ordered_operations.append([op_name, targets_list])

    print(ordered_operations)

    return ordered_operations, flags, matrix_indices


def visualize_circuit_dag(di_graph: nx.DiGraph, figsize=(12, 8)):
    """
    Visualizes the circuit DAG using a multipartite layout based on
    topological generations.
    """
    # 1. Assign layers using topological generations
    for layer, nodes in enumerate(nx.topological_generations(di_graph)):
        for node in nodes:
            di_graph.nodes[node]["layer"] = layer

    # 2. Use the "layer" attribute for the layout
    pos = nx.multipartite_layout(di_graph, subset_key="layer", align="vertical")

    # 3. Create readable labels
    labels = {}
    for node, data in di_graph.nodes(data=True):
        op = data.get("op_name")
        targs = "-".join([str(t) for t in data.get("targets", ())])
        labels[node] = f"{op}\n{targs}" + str()
        if data.get("is_flag"):
            labels[node] += f"\nFLAG"
        elif data.get("matrix_index") is not None:
            labels[node] += f"\nH({data.get("matrix_index")})"
        elif op in ("M", "MX"):
            raise ValueError(f"Measurement {targs} is neither a flag nor a syndrome measurement.\n{data=}")

    # 4. Plotting
    plt.figure(figsize=figsize)

    # Define a constant node size so both drawing functions align perfectly
    NODE_SIZE = 1500

    nx.draw_networkx_nodes(
        di_graph, pos,
        node_color="#ADD8E6",
        node_size=NODE_SIZE,
        edgecolors="black"
    )

    # FIXED: Passed node_size here so arrows stop at the boundary
    nx.draw_networkx_edges(
        di_graph, pos,
        arrowstyle="-|>",  # Filled arrowhead for better visibility
        arrowsize=20,  # Slightly larger arrow
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
    # Work on a copy so we don't destroy the original causal graph
    mod_dag = dag.copy()

    topo_order = list(nx.topological_sort(mod_dag))

    # 1. Identify the Birth and Death nodes of every logical ancilla
    ancillas = set()
    birth_node = {}
    death_node = {}

    for node in topo_order:
        targets = mod_dag.nodes[node].get("targets", ())
        for q in targets:
            if q >= n_data:
                ancillas.add(q)
                if q not in birth_node:
                    birth_node[q] = node  # First appearance
                death_node[q] = node  # Last appearance (updates until end)

    # 2. Heuristic Scoring (ASAP Layering)
    # We want to chain qubits that naturally align in time to avoid exploding circuit depth.
    asap = {}
    for node in topo_order:
        preds = list(mod_dag.predecessors(node))
        asap[node] = max((asap[p] for p in preds), default=0) + 1

    # 3. Build and Score Candidate Pairs (Death_A -> Birth_B)
    candidates = []
    for qA in ancillas:
        for qB in ancillas:
            if qA == qB:
                continue

            dA = death_node[qA]
            bB = birth_node[qB]

            # Fast rejection: If B MUST happen before A natively, we cannot invert them
            if nx.has_path(mod_dag, bB, dA):
                continue

            # Score: asap[Death_A] - asap[Birth_B]
            # Lower score means A naturally finishes before B starts (ideal for chaining)
            score = asap[dA] - asap[bB]
            candidates.append((score, qA, qB, dA, bB))

    # Sort to try the most natural sequences first
    candidates.sort(key=lambda x: x[0])

    # 4. Greedy Dependency Injection with Cycle Checking
    next_q = {}
    prev_q = {}

    for score, qA, qB, dA, bB in candidates:
        # Skip if A is already mapped to a successor, or B already has a predecessor
        if qA in next_q or qB in prev_q:
            continue

        # Attempt the injection
        mod_dag.add_edge(dA, bB)

        # Rigorous Cycle Check
        if not nx.is_directed_acyclic_graph(mod_dag):
            # Cycle created! Reject the pairing and remove the edge
            mod_dag.remove_edge(dA, bB)
        else:
            # Success! A and B now share a physical track
            next_q[qA] = qB
            prev_q[qB] = qA

    # 5. Hardware Allocation via Chains
    logical_to_physical = {q: q for q in range(n_data)}
    next_hw = n_data

    for q in ancillas:
        if q not in prev_q:  # This ancilla is the start of a new hardware chain
            curr = q
            while curr is not None:
                logical_to_physical[curr] = next_hw
                curr = next_q.get(curr)
            next_hw += 1

    total_hw = next_hw
    return mod_dag, logical_to_physical, total_hw


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

    # Identify the injected reuse edges
    for u, v in mod_dag.edges():
        op_u = mod_dag.nodes[u].get("op_name", "")
        op_v = mod_dag.nodes[v].get("op_name", "")

        if op_u in {"M", "MX"} and op_v in {"R", "RX"}:
            targets_u = mod_dag.nodes[u].get("targets", [])
            targets_v = mod_dag.nodes[v].get("targets", [])

            if len(targets_u) == 1 and len(targets_v) == 1:
                qA = targets_u[0]
                qB = targets_v[0]

                if qA != qB:
                    root_A = get_root(qA)
                    parent_map[qB] = root_A

    # --- PHASE 2: Collect and Compress ---
    # Find all unique surviving root qubits in the graph
    active_roots = set()
    for node in mod_dag.nodes():
        targets = mod_dag.nodes[node].get("targets", [])
        for q in targets:
            active_roots.add(get_root(q))

    # Separate data roots from ancilla roots
    data_roots = [q for q in active_roots if q < n_data]
    ancilla_roots = sorted([q for q in active_roots if q >= n_data])

    # Build the compression map
    compression_map = {}
    # Data qubits map to themselves
    for q in data_roots:
        compression_map[q] = q

    # Ancilla qubits get mapped to a dense, contiguous sequence starting at n_data
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

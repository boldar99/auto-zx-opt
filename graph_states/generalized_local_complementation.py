from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def robust_tableau_to_graph(X_tab, Z_tab):
    """
    Rigorously converts a stabilizer tableau (X|Z) into a Graph State adjacency matrix (I|A).
    Applies local Hadamards (column swaps) when the X block is rank-deficient.
    """
    N = X_tab.shape[0]
    X = X_tab.copy()
    Z = Z_tab.copy()
    hadamards_applied = []

    for i in range(N):
        pivot_row = -1
        for j in range(i, N):
            if X[j, i] == 1:
                pivot_row = j
                break

        # If rank-deficient, apply local Hadamard to qubit i
        if pivot_row == -1:
            X[:, i], Z[:, i] = Z[:, i].copy(), X[:, i].copy()
            hadamards_applied.append(i)

            for j in range(i, N):
                if X[j, i] == 1:
                    pivot_row = j
                    break

        if pivot_row != i:
            X[[i, pivot_row]] = X[[pivot_row, i]]
            Z[[i, pivot_row]] = Z[[pivot_row, i]]

        for j in range(N):
            if i != j and X[j, i] == 1:
                X[j] = (X[j] + X[i]) % 2
                Z[j] = (Z[j] + Z[i]) % 2

    np.fill_diagonal(Z, 0)
    A = np.maximum(Z, Z.T)
    return A, hadamards_applied


def get_joint_tableau(Hx, Hz, Lx, Lz):
    """
    Constructs the (n+k) x (n+k) joint stabilizer tableau representing
    the universal graph state of a CSS code.

    Args:
        Hx, Hz: Parity check matrices for X and Z stabilizers.
        Lx, Lz: Logical X and Z operators (each row is a logical qubit).

    Returns:
        X_tab, Z_tab: The N x N matrices representing the joint state.
    """
    n = Hx.shape[1]
    k = Lx.shape[0]
    N = n + k

    # Verify dimensions (A valid CSS code should have exactly n-k independent stabilizers)
    num_stabilizers = Hx.shape[0] + Hz.shape[0]
    if num_stabilizers + 2 * k != 2 * N:
        # Note: We expect exactly N generators for an N-qubit state.
        # (n-k) stabilizers + 2k logicals = n + k = N.
        pass  # We proceed, but row reduction later will catch dependencies

    X_tab = np.zeros((N, N), dtype=int)
    Z_tab = np.zeros((N, N), dtype=int)

    row = 0

    # 1. Add Physical X-Stabilizers: Sx ⊗ I
    for i in range(Hx.shape[0]):
        X_tab[row, :n] = Hx[i, :]
        row += 1

    # 2. Add Physical Z-Stabilizers: Sz ⊗ I
    for i in range(Hz.shape[0]):
        Z_tab[row, :n] = Hz[i, :]
        row += 1

    # 3. Add Logical X correlations: Lx ⊗ X_log
    for j in range(k):
        X_tab[row, :n] = Lx[j, :]
        X_tab[row, n + j] = 1  # X on the logical ancilla
        row += 1

    # 4. Add Logical Z correlations: Lz ⊗ Z_log
    for j in range(k):
        Z_tab[row, :n] = Lz[j, :]
        Z_tab[row, n + j] = 1  # Z on the logical ancilla
        row += 1

    return X_tab, Z_tab


# --- HELPER FUNCTIONS FOR THE SEARCH ---

def get_s_dot_lambda(G, K_nodes, S_array):
    """
    Calculates S • Lambda_G^K: The sum of the multiplicities in S
    for all vertices that are common neighbors of the set K.
    """
    if not K_nodes:
        return 0

    # Find common neighbors of all nodes in K
    common_neighbors = set(G.neighbors(K_nodes[0]))
    for node in K_nodes[1:]:
        common_neighbors.intersection_update(G.neighbors(node))

    # Sum the multiplicity (S_array) of those common neighbors
    return sum(S_array[v] for v in common_neighbors)


def is_valid_2_incident_multiset(G, S_array):
    return is_valid_r_incident_multiset(G, S_array, 2)


def is_valid_r_incident_multiset(G, S_array, r):
    """
    STRICT r-incidence check enforcing the 2^(r-k) modulo rule.
    """
    supp_S = [i for i, mult in enumerate(S_array) if mult > 0]

    # 1. Independence Check
    for u, v in combinations(supp_S, 2):
        if G.has_edge(u, v):
            return False

    V_minus_S = [i for i in G.nodes() if i not in supp_S]

    # 2. FIXED r-Incidence Checks
    # k goes from 0 to r-1 INCLUSIVE. Therefore, we use range(r).
    for k in range(r):
        subset_size = k + 2
        modulo_base = 2 ** (r - k)  # The correct formula!

        for subset in combinations(V_minus_S, subset_size):
            if get_s_dot_lambda(G, subset, S_array) % modulo_base != 0:
                return False

    return True

def optimized_logical_t_search(adj_matrix, num_physical_qubits, hadamards_applied):
    """
    Optimized branch-and-bound search for LC2 symmetries that yield
    a non-Clifford (T-like) gate on the logical node.
    """
    G = nx.from_numpy_array(adj_matrix)
    N = len(G.nodes())

    # Identify the logical node (assuming a single logical qubit for this search)
    logical_node = num_physical_qubits
    logical_neighbors = list(G.neighbors(logical_node))

    valid_multisets = []

    # Track which nodes are forbidden from being in S (due to independence)
    # forbidden_counts[i] > 0 means node i cannot be added to S.
    forbidden_counts = [0] * num_physical_qubits

    def backtrack(current_qubit, current_S):
        # Base Case: We have assigned a multiplicity to all physical qubits
        if current_qubit == num_physical_qubits:
            # Pad with 0s for logical nodes
            full_S = current_S + [0] * (N - num_physical_qubits)

            # FILTER 1: The Logical Non-Clifford Condition (Your "Odd" Intuition)
            # The sum of S over the logical node's neighbors must be odd.
            logical_kickback_sum = sum(full_S[w] for w in logical_neighbors if w < num_physical_qubits)
            if logical_kickback_sum % 2 == 0:
                return  # Yields a Clifford or Pauli, discard.

            # FILTER 2: Strict 2-Incidence modulo arithmetic (from previous script)
            if is_valid_2_incident_multiset(G, full_S):
                valid_multisets.append(full_S)
            return

        # Recursive Branching
        # Option A: Multiplicity 0 (Always allowed)
        backtrack(current_qubit + 1, current_S + [0])

        # Option B: Multiplicities 1, 2, 3 (Only allowed if node is not forbidden)
        if forbidden_counts[current_qubit] == 0:
            # If we assign > 0, we must forbid all neighbors
            neighbors = [n for n in G.neighbors(current_qubit) if n < num_physical_qubits]
            for neighbor in neighbors:
                forbidden_counts[neighbor] += 1

            for mult in [1, 2, 3]:
                backtrack(current_qubit + 1, current_S + [mult])

            # Backtrack cleanup: un-forbid the neighbors
            for neighbor in neighbors:
                forbidden_counts[neighbor] -= 1

    print(f"Starting optimized recursive search on {num_physical_qubits} physical qubits...")
    backtrack(0, [])

    return valid_multisets


def generalized_logical_rotation_search(adj_matrix, num_physical_qubits, r):
    """
    Optimized branch-and-bound search for LC_r symmetries that yield
    a rotation strictly at the highest fractional level defined by r.
    """
    G = nx.from_numpy_array(adj_matrix)
    N = len(G.nodes())

    # Assume a single logical node for this specific targeted search
    logical_node = num_physical_qubits
    logical_neighbors = list(G.neighbors(logical_node))

    valid_multisets = []
    forbidden_counts = [0] * num_physical_qubits

    # The maximum multiplicity for a node in LC_r is (2^r) - 1
    max_multiplicity = 2 ** r

    def backtrack(current_qubit, current_S):
        # Base Case
        if current_qubit == num_physical_qubits:
            full_S = current_S + [0] * (N - num_physical_qubits)

            # THE UNIVERSAL FILTER:
            # To ensure the phase does not collapse to the (r-1)-th level,
            # the sum of the logical neighbors' multiplicities MUST be odd.
            logical_kickback_sum = sum(full_S[w] for w in logical_neighbors if w < num_physical_qubits)
            if logical_kickback_sum % 2 == 0:
                return  # Fraction reduces; gate drops to a lower Clifford level.

            # Generalized r-incidence check (Requires the updated function from our previous discussion)
            if is_valid_r_incident_multiset(G, full_S, r):
                valid_multisets.append(full_S)
            return

        # Recursive Branching
        # Option A: Multiplicity 0 (Always allowed, does not trigger independence rules)
        backtrack(current_qubit + 1, current_S + [0])

        # Option B: Multiplicities > 0 (Only allowed if node is not forbidden)
        if forbidden_counts[current_qubit] == 0:
            neighbors = [n for n in G.neighbors(current_qubit) if n < num_physical_qubits]

            for neighbor in neighbors:
                forbidden_counts[neighbor] += 1

            # Iterate through all valid non-zero multiplicities for level r
            for mult in range(1, max_multiplicity):
                backtrack(current_qubit + 1, current_S + [mult])

            for neighbor in neighbors:
                forbidden_counts[neighbor] -= 1

    print(f"Starting LC_{r} search on {num_physical_qubits} physical qubits...")
    print(
        f"Branching factor: {max_multiplicity} | Maximum theoretical leaves: {max_multiplicity ** num_physical_qubits}")

    backtrack(0, [])
    return valid_multisets


def apply_lc_r_with_multiplicity(G, S_array, r):
    """
    Applies r-local complementation.
    """
    G_new = G.copy()
    nodes = list(G.nodes())

    for u, v in combinations(nodes, 2):
        common_neighbors = list(set(G.neighbors(u)).intersection(G.neighbors(v)))
        s_dot_lambda = sum(S_array[w] for w in common_neighbors)

        # Definition 9: Toggle if sum mod 2^r == 2^(r-1)
        if s_dot_lambda % (2 ** r) == (2 ** (r - 1)):
            if G_new.has_edge(u, v):
                G_new.remove_edge(u, v)
            else:
                G_new.add_edge(u, v)

    return G_new


def calculate_all_unitaries(G, S_array, hadamards_applied, num_physical_qubits, total_nodes):
    """
    Calculates unitaries for BOTH physical and logical nodes.
    Returns two dictionaries: physical_gates and logical_gates.
    """
    physical_gates = {}
    logical_gates = {}

    def format_phase(axis, multiplier):
        mult = multiplier % 8
        if mult == 0: return ""
        if mult == 4: return f"{axis}"  # pi rotation

        fractions = {1: "π/4", 2: "π/2", 3: "3π/4", 5: "5π/4", 6: "3π/2", 7: "7π/4"}
        return f"{axis}({fractions[mult]})"

    for v in range(total_nodes):
        # 1. Base Multipliers
        x_mult = S_array[v] % 8
        z_sum = sum(S_array[w] for w in G.neighbors(v))
        z_mult = (-z_sum) % 8

        # 2. Hardware / Basis Correction
        if v in hadamards_applied:
            actual_x_mult = z_mult
            actual_z_mult = x_mult
        else:
            actual_x_mult = x_mult
            actual_z_mult = z_mult

        # 3. Format strings
        z_str = format_phase("Z", actual_z_mult)
        x_str = format_phase("X", actual_x_mult)
        gates = [g for g in [x_str, z_str] if g]
        gate_str = " * ".join(gates) if gates else "I"

        # 4. Route to correct dictionary
        if v < num_physical_qubits:
            physical_gates[f"q{v}"] = gate_str
        else:
            logical_gates[f"L{v - num_physical_qubits}"] = gate_str

    return physical_gates, logical_gates


def get_generic_joint_tableau(stab_X, stab_Z, log_X_x, log_X_z, log_Z_x, log_Z_z):
    """
    Constructs the universal joint tableau for ANY generic stabilizer code (CSS or non-CSS).
    Requires full symplectic representations for stabilizers and logicals.
    """
    num_stabs = stab_X.shape[0]
    n = stab_X.shape[1]
    k = log_X_x.shape[0]
    N = n + k

    if num_stabs + 2 * k != 2 * N:
        print(f"Warning: Expected {n - k} stabilizers for an [[{n},{k}]] code, got {num_stabs}.")

    X_tab = np.zeros((N, N), dtype=int)
    Z_tab = np.zeros((N, N), dtype=int)

    row = 0

    # 1. Add Physical Stabilizers (Mixed X and Z)
    for i in range(num_stabs):
        X_tab[row, :n] = stab_X[i, :]
        Z_tab[row, :n] = stab_Z[i, :]
        row += 1

    # 2. Add Logical X correlations (Physical Logical X correlated with Ancilla X)
    for j in range(k):
        X_tab[row, :n] = log_X_x[j, :]
        Z_tab[row, :n] = log_X_z[j, :]
        X_tab[row, n + j] = 1  # The X on the logical ancilla
        row += 1

    # 3. Add Logical Z correlations (Physical Logical Z correlated with Ancilla Z)
    for j in range(k):
        X_tab[row, :n] = log_Z_x[j, :]
        Z_tab[row, :n] = log_Z_z[j, :]
        Z_tab[row, n + j] = 1  # The Z on the logical ancilla
        row += 1

    return X_tab, Z_tab


# ==========================================
# THE [[5, 1, 3]] PERFECT CODE
# ==========================================
def code_5_1_3():
    """
    Generates the symplectic representation of the 5-qubit code.
    Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
    Logicals: Lx = XXXXX, Lz = ZZZZZ
    """
    # Stabilizer X components
    stab_X = np.array([
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0]
    ])

    # Stabilizer Z components
    stab_Z = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1]
    ])

    # Logical X (purely X)
    log_X_x = np.array([[1, 1, 1, 1, 1]])
    log_X_z = np.array([[0, 0, 0, 0, 0]])

    # Logical Z (purely Z)
    log_Z_x = np.array([[0, 0, 0, 0, 0]])
    log_Z_z = np.array([[1, 1, 1, 1, 1]])

    X, Z = get_generic_joint_tableau(stab_X, stab_Z, log_X_x, log_X_z, log_Z_x, log_Z_z)
    return robust_tableau_to_graph(X, Z) + (5, 1)


def shor():
    H_shor_x = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1]
    ])

    H_shor_z = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1]
    ])
    L_shor_x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
    L_shor_z = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0]])

    X, Z = get_joint_tableau(H_shor_x, H_shor_z, L_shor_x, L_shor_z)
    return robust_tableau_to_graph(X, Z) + (9, 1)


def steane():
    H_steane_x = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1]
    ])
    H_steane_z = H_steane_x
    L_steane_x = np.array([[1, 1, 1, 1, 1, 1, 1]])
    L_steane_z = np.array([[1, 1, 1, 1, 1, 1, 1]])

    X, Z = get_joint_tableau(H_steane_x, H_steane_z, L_steane_x, L_steane_z)
    return robust_tableau_to_graph(X, Z) + (7, 1)


def carbon():
    H_x = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    ])
    H_z = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1],
    ])

    L_x = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    ])
    L_z = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ])

    return get_joint_tableau(H_x, H_z, L_x, L_z) + (12, 2)


def dodecahedral_code():
    G = nx.dodecahedral_graph()
    nx.draw(G, with_labels=True)
    L = [14, 4, 11, 1]


def code_15_7_3():
    H_15_7_3_x = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    H_15_7_3_z = H_15_7_3_x
    L_15_7_3_x = np.array([
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    L_15_7_3_z = L_15_7_3_x

    X, Z = get_joint_tableau(H_15_7_3_x, H_15_7_3_z, L_15_7_3_x, L_15_7_3_z)
    return robust_tableau_to_graph(X, Z) + (15, 7)


def code_16_6_4():
    H_16_6_4_x = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    H_16_6_4_z = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    L_16_6_4_x = np.array([
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])
    L_16_6_4_z = L_16_6_4_x

    X, Z = get_joint_tableau(H_16_6_4_x, H_16_6_4_z, L_16_6_4_x, L_16_6_4_z)
    return robust_tableau_to_graph(X, Z) + (16, 6)


def get_graph_data(method):
    adj_matrix, hadamards_applied, n, k = method()
    G = nx.from_numpy_array(adj_matrix)
    color_map = ['lightblue' if i < n else 'lightgreen' for i in range(len(adj_matrix))]

    labels = {}
    for i in range(len(adj_matrix)):
        if i >= n:
            labels[i] = f"L{i - n}"
        elif i in hadamards_applied:
            labels[i] = f"q{i}\n(H)"
        else:
            labels[i] = f"q{i}"

    return G, adj_matrix, hadamards_applied, labels, color_map, n, k


# --- PLAYGROUND EXECUTION ---
def test_graph_gen(method):
    adj_matrix, hadamards_applied, n, k = method()

    print("\n--- TRANSFORMATION COMPLETE ---")
    print(f"Hadamards were forced on physical qubits: {hadamards_applied}")
    print("Final Adjacency Matrix (Graph State):\n", adj_matrix)

    # Visualize
    G = nx.from_numpy_array(adj_matrix)
    plt.figure(figsize=(7, 7))

    # Color coding: Nodes 0-6 are physical, Node 7 is logical
    color_map = ['lightblue' if i < n else 'lightgreen' for i in range(len(adj_matrix))]

    labels = {}
    for i in range(len(adj_matrix)):
        if i >= n:
            labels[i] = f"L{i - n}"
        elif i in hadamards_applied:
            labels[i] = f"q{i}\n(H)"
        else:
            labels[i] = f"q{i}"

    pos = nx.spring_layout(G, seed=10)
    nx.draw(G, pos, labels=labels, node_color=color_map, node_size=1200, font_weight='bold', font_size=10)
    plt.show()


def test_search(method, r):
    G, adj_matrix, hadamards_applied, labels, color_map, n, k = get_graph_data(method)

    results = generalized_logical_rotation_search(adj_matrix, n, r)

    print(f"\nSearch Complete. Found {len(results)} valid {r}-incident multisets.")
    S_arrays = []
    for res in results:
        supp = [i for i, val in enumerate(res) if val > 0]
        print(f"Valid S array: {res} | Support nodes: {supp}")
        S_arrays.append(res)

    if len(S_arrays) == 0:
        print("No valid S arrays found.")
        return

    fig, axes = plt.subplots(nrows=len(S_arrays), ncols=1, figsize=(10, 6 * len(S_arrays)))
    if len(S_arrays) == 1: axes = [axes]

    for i, (ax, res) in enumerate(zip(axes, S_arrays), start=1):
        G_prime = apply_lc_r_with_multiplicity(G, res, r)
        pos = nx.spring_layout(G_prime, seed=10)

        # Calculate separated unitaries
        phys_gates, log_gates = calculate_all_unitaries(G, res, hadamards_applied, n, n + k)

        # Format text cleanly
        phys_text = " | ".join([f"{q}: {op}" for q, op in phys_gates.items() if op != "I"])
        log_text = " | ".join([f"{l}: {op}" for l, op in log_gates.items() if op != "I"])

        if not phys_text: phys_text = "Global Identity"
        if not log_text: log_text = "Identity (No Logical Effect)"

        nx.draw(G_prime, pos, ax=ax, labels=labels, node_color=color_map, node_size=1200, font_weight='bold',
                font_size=10)

        # Display both Physical and Logical unitaries clearly
        title_str = (f"Graph {i} | S_stab = {res[:n]} | S_logicals = {res[n:]}\n"
                     f"Physical Gates: {phys_text}\n"
                     f"EFFECTIVE LOGICAL: {log_text}\n"
                     f"Is CSS: {nx.is_bipartite(G_prime)}")

        ax.set_title(title_str, pad=15)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # G = nx.dodecahedral_graph()
    # nx.draw(G, with_labels=True)
    # L = [14, 4, 11, 1]
    # plt.show()

    test_graph_gen(shor)
    test_search(shor, 1)
    # test_search(steane, 3)

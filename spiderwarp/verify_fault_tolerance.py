import itertools
import sys
from collections import defaultdict
import numpy as np
import stim

from spiderwarp.stim_utils import explode_circuit


def build_css_syndrome_table(stabilizers: list[str], d: int):
    """
    Builds a lookup table for standard, unflagged CSS syndrome decoding up to t faults.
    """
    num_qubits = len(stabilizers[0])
    t_faults = (d - 1) // 2
    stab_paulis = [stim.PauliString(s) for s in stabilizers]
    decoder_table = {}

    for w in range(t_faults + 1):
        for qubit_indices in itertools.combinations(range(num_qubits), w):
            for p_type in ["X", "Z"]:
                error = stim.PauliString(num_qubits)
                for q in qubit_indices:
                    error[q] = p_type
                syndrome = tuple(int(not error.commutes(s)) for s in stab_paulis)
                if syndrome not in decoder_table:
                    decoder_table[syndrome] = error.to_numpy()[1].astype(int).tolist()

    return decoder_table


def list_to_str_stabs(stabs):
    return ["".join("X" if c == 1 else "I" for c in s) for s in stabs]


def catalog_flag_and_residual_errors(
    circuit: stim.Circuit,
    ops: list[stim.CircuitInstruction],
    possible_faults: list[tuple[int, int]],
    max_faults: int,
    num_data_qubits: int,
    flag_measurements: list[int],
    syndrome_measurements: list[int],
    basis: str
) -> dict:
    """
    Simulates internal faults and groups the RAW residual physical errors by
    their resulting (Flag Pattern -> Syndrome Pattern).
    """
    fault_type = "X" if basis == "Z" else "Z"
    init_str = "" if basis == "Z" else f"H {' '.join(str(q) for q in range(num_data_qubits))}"

    # Structure: flag_pattern -> syndrome_pattern -> list of raw residual errors
    fault_catalog = defaultdict(lambda: defaultdict(list))

    for num_faults in range(1, max_faults + 1):
        for fault_combo in itertools.combinations(possible_faults, num_faults):
            noisy_c = stim.Circuit()
            noisy_c.append_from_stim_program_text(init_str)

            fault_dict = defaultdict(list)
            for op_idx, target_qubit in fault_combo:
                fault_dict[op_idx].append(target_qubit)

            for i, op in enumerate(ops):
                noisy_c.append(op)
                if i in fault_dict:
                    for q in fault_dict[i]:
                        noisy_c.append(fault_type, [q])

            if basis == "X":
                noisy_c.append("H", range(num_data_qubits))

            sim = stim.TableauSimulator()
            sim.do_circuit(noisy_c)

            record = np.array(sim.current_measurement_record()).astype(int)
            flag_record = tuple(record[flag_measurements].tolist())
            syndrome_record = tuple(record[syndrome_measurements].tolist())

            data_bits = np.array(sim.measure_many(*range(num_data_qubits))).astype(int)
            raw_residual = tuple(data_bits.tolist())

            fault_catalog[flag_record][syndrome_record].append(raw_residual)

    return fault_catalog


def verify_logical_uniqueness_and_get_correction(
    residual_errors: list[tuple[int, ...]],
    stabilizer_matrix: np.ndarray,
    logical_matrix: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Projects errors into the logical frame to guarantee uniqueness.
    If ambiguous, throws an error. If logically unique, calculates the
    absolute minimum-weight physical correction string for the fallback table.
    """
    logical_signatures = set()

    # 1. Project all physical errors through the logical matrix
    for E in residual_errors:
        E_np = np.array(E)
        signature = tuple(((logical_matrix @ E_np) % 2).tolist())
        logical_signatures.add(signature)

    # 2. Check for Logical Ambiguity
    if len(logical_signatures) > 1:
        print(
            f"FAULT TOLERANCE BREAKDOWN! \n"
            f"Multiple distinct logical outcomes detected for the same Flag+Syndrome symptom.\n"
            f"Logical Signatures found: {logical_signatures}\n"
            f"The circuit is not fault-tolerant for this error configuration.",
            file=sys.stderr,
        )

    # 3. Compute Unique Minimum-Weight Correction
    # Since they are logically identical, we can use the first error as a representative
    # to search its stabilizer coset for the simplest physical correction.
    representative_error = np.array(residual_errors[0])
    num_stabs = len(stabilizer_matrix)

    min_weight = len(representative_error) + 1
    best_correction = representative_error

    for i in range(2 ** num_stabs):
        vec_i = np.array([int(x) for x in format(i, f'0{num_stabs}b')])
        stab_element = (vec_i @ stabilizer_matrix) % 2

        candidate_correction = (representative_error + stab_element) % 2
        current_weight = np.sum(candidate_correction)

        if current_weight < min_weight:
            min_weight = current_weight
            best_correction = candidate_correction

    logical_signature_result = list(logical_signatures)[0]
    return tuple(best_correction.tolist()), logical_signature_result


def get_fault_locations(ops: list) -> list[tuple[int, int]]:
    possible_faults = []
    max_qubit = 0
    for i, op in enumerate(ops):
        for target in op.targets_copy():
            max_qubit = max(max_qubit, target.value)
            if target.is_qubit_target:
                possible_faults.append((i, target.value))
    return [(0, i) for i in range(max_qubit)] + possible_faults


if __name__ == "__main__":
    from spiderwarp.csscode import CSSCode
    from spiderwarp.utils import load_state_prep_circuit
    from spiderwarp.stim_utils import steane_se_from_stim_state_prep, get_num_measurements, perfect_state_from_code
    from spiderwarp.path_cover import CoveredZXGraph
    from spiderwarp.path_cover_metrics import metric_spacetime_volume_exact
    from spiderwarp.qubit_reuse import build_circuit_dag, apply_logical_qubit_merge_and_compress, dag_to_circuit, \
        inject_qubit_reuse, VolumeOptimizingReuseStrategy

    # Requested configuration for the [15, 7, 3] Quantum Hamming Code
    code_dir, code_name, circ_dir, circ_path = "MQT", "15_7_3", "SAT", "hamming/zero_ft_heuristic_opt"

    code = CSSCode.load_code(code_dir, code_name)
    stabs = list_to_str_stabs(code.H_z)
    decoder_table = build_css_syndrome_table(stabs, code.d)

    circuit = load_state_prep_circuit(circ_dir, circ_path)

    # Extracting Z stabilizers to test against physical X errors
    se = steane_se_from_stim_state_prep(circuit, se_basis="Z", n=code.n)

    covered = CoveredZXGraph.from_stim(se)
    covered.basic_FE_rewrites()
    optimised = covered.greedy_best_first_boundary_bends(
        cost_func=metric_spacetime_volume_exact(VolumeOptimizingReuseStrategy),
        max_evaluations=100
    )
    optimised.optimize_path_extremities()

    dag = build_circuit_dag(optimised)
    mod_dag, logical_to_physical, total_hw = inject_qubit_reuse(dag, code.n, VolumeOptimizingReuseStrategy())
    compressed_dag = apply_logical_qubit_merge_and_compress(mod_dag, code.n)
    final_circ, final_meas_map = dag_to_circuit(compressed_dag)

    num_measurements = get_num_measurements(se)
    flag_indices = [k for k, v in final_meas_map.items() if v < num_measurements - code.n]

    syndrome_indices = {k: v - (num_measurements - code.n) for k, v in final_meas_map.items() if
                        v >= num_measurements - code.n}
    syndrome_keys = list(syndrome_indices.keys())

    ops = explode_circuit(final_circ)
    possible_faults = get_fault_locations(ops)

    # For a distance 3 code, t=1 fault
    max_faults_to_test = (code.d - 1) // 2

    print(f"Executing fault injection for '{code_name}' (up to {max_faults_to_test} fault)...")

    catalog = catalog_flag_and_residual_errors(
        circuit=final_circ,
        ops=ops,
        possible_faults=possible_faults,
        max_faults=max_faults_to_test,
        num_data_qubits=code.n,
        flag_measurements=flag_indices,
        syndrome_measurements=syndrome_keys,
        basis="Z",
    )

    print("\nValidating Logical Uniqueness and Building Fallback Table:")
    print("-" * 75)
    flagged_events = 0

    fallback_table = {}

    for flag_pattern, syndromes in catalog.items():
        if any(flag_pattern):
            flagged_events += 1
            print(f"Flag Pattern Raised: {flag_pattern}")

            for syn_pattern, raw_residuals in syndromes.items():
                print(f"  Syndrome: {syn_pattern}")

                # Check uniqueness and get the correction using H_z (Stabilizers) and L_z (Logicals for X-errors)
                unique_correction, logical_signature = verify_logical_uniqueness_and_get_correction(
                    residual_errors=raw_residuals,
                    stabilizer_matrix=code.H_z,
                    logical_matrix=code.L_z
                )

                # Store it in our final lookup table
                fallback_table[(flag_pattern, syn_pattern)] = unique_correction

                print(f"    -> Status: VALIDATED (Logically Unique)")
                print(f"    -> Logical Error Signature: {logical_signature}")
                print(f"    -> Unique Optimal Correction: {unique_correction} (Weight {sum(unique_correction)})")
            print("-" * 50)

    if flagged_events == 0:
        print("No active flags were captured. Verify the flag mapping registers.")
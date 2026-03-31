import itertools

import numpy as np
import stim


def build_css_syndrome_table(stabilizers: list[str], d: int):
    """
    Builds a lookup table for CSS codes.
    Only considers pure X error strings and pure Z error strings up to weight t.
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
    xs = [
        "".join("X" if c == 1 else "I" for c in s)
        for s in stabs
    ]
    zs = [
        "".join("Z" if c == 1 else "I" for c in s)
        for s in stabs
    ]
    return xs


def compute_modified_lookup_table(
        circuit: stim.Circuit,
        H_matrix: np.ndarray,  # Binary Matrix (num_stabilizers x num_data_qubits)
        L_matrix: np.ndarray,  # Binary Matrix (num_logicals x num_data_qubits)
        correction_table: dict,  # Your pre-computed Phase 1 table
        flag_measurements: list[int],
        basis: str,  # "Z" (Measuring Z-stabs) or "X" (Measuring X-stabs)
        d: int,
        verbose=False,
):
    """
    Verifies a syndrome extraction circuit and builds a lookup table for flag corrections.
    """
    num_data_qubits = len(H_matrix[0])
    t_faults = (d - 1) // 2

    init_str = "" if basis == "Z" else f"H {' '.join(str(q) for q in range(num_data_qubits))}"
    fault_type = "X" if basis == "Z" else "Z"

    # Helpers assumed to be defined externally or imported
    ops = explode_circuit(circuit)
    possible_faults = get_fault_locations(ops)

    modified_correction_table = dict()
    modified_correction_table_origin = dict()

    # Iterate fault combinations
    for fault_combo in itertools.combinations(possible_faults, t_faults):
        # 1. Simulate the fault
        noisy_c = build_noisy_circuit(fault_combo, fault_type, init_str, ops)
        sim = stim.TableauSimulator()
        sim.do_circuit(noisy_c)

        # 2. Extract measurements and data
        non_flag_measurements = list(set(range(len(sim.current_measurement_record()))) - set(flag_measurements))
        ancilla_record = np.array(sim.current_measurement_record()).astype(int)[non_flag_measurements]
        flag_record = np.array(sim.current_measurement_record()).astype(int)[flag_measurements]

        if basis == "X":
            sim.do_circuit(stim.Circuit(f"H {' '.join(str(q) for q in range(num_data_qubits))}"))

        data_bits = np.array(sim.measure_many(*range(num_data_qubits))).astype(int)
        actual_syndrome = tuple(((H_matrix @ data_bits) % 2).tolist())

        # 3. Standard Decoding Check
        if actual_syndrome not in correction_table:
            return None  # Should not happen if correction_table is complete for t faults

        standard_correction = correction_table[actual_syndrome]

        # Check if standard decoding fails (Logical Flip)
        final_state_standard = (data_bits + standard_correction) % 2
        logical_flip = (L_matrix @ final_state_standard) % 2

        # 4. If Flag is raised and Logic Fails, calculate the optimal correction
        if np.any(flag_record):
            if verbose:
                print(f"Syndrome: {actual_syndrome} + Flags: {flag_record} -> Logical Flip: {logical_flip}")

            # --- BEGIN FIX ---
            stabs = np.array(H_matrix.tolist() + L_matrix.tolist())
            num_stabs = len(H_matrix)
            min_correction_val = num_data_qubits + 1
            min_correction = np.zeros(num_data_qubits, dtype=int)

            # Brute force search for the simplest error equivalent to data_bits
            for i in range(2 ** len(stabs)):
                # Create binary vector vec_i selecting stabilizers/logicals
                vec_i = np.array(list(map(int, ('0' * len(stabs) + format(i, "b"))[-len(stabs):])))
                vec_l = vec_i[num_stabs:]

                # Calculate residual noise: (Stabilizers + Data)
                # This finds the "class" of the error
                final_state = (vec_i @ stabs + data_bits) % 2

                # Weight of the physical error
                correction_val = final_state.sum()

                if correction_val < min_correction_val:
                    min_correction = (final_state + vec_l @ L_matrix) % 2
                    min_correction_val = correction_val

            min_correction = min_correction.tolist()
            # --- END FIX ---

            full_syndrome = tuple(flag_record.tolist()) + actual_syndrome

            # 5. Update Table or Check Consistency
            if full_syndrome in modified_correction_table and np.any(logical_flip == 1):
                assert not np.any(L_matrix @ ((data_bits + min_correction) % 2) % 2)
                # Ensure the correction is unique for this syndrome+flag combination
                if np.any(modified_correction_table[full_syndrome] != min_correction):
                    if verbose:
                        print(f"NON UNIQUE CORRECTION for {full_syndrome}")
                        print("Saved correction:", modified_correction_table[full_syndrome])
                        print("New correction:", min_correction)
                        print("Fault Combos:", modified_correction_table_origin[full_syndrome], fault_combo)
                    return None
            elif np.any(logical_flip == 1):
                # Save the new correction
                modified_correction_table[full_syndrome] = min_correction
                modified_correction_table_origin[full_syndrome] = fault_combo
            else:
                if verbose:
                    print("Skipping saving measurement (Standard decoding worked despite flag)")

        elif np.any(logical_flip):
            if verbose:
                print(f"LOGICAL FAILURE via Extraction Circuit (No Flag Raised)!")
                print(f"Faults: {fault_combo}")
                print(f"Data has error: {data_bits} (Syndrome {actual_syndrome})")
            return None

    return modified_correction_table


def explode_circuit(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    """
    Decomposes a circuit into a list of atomic instructions.
    E.g., 'CX 0 1 2 3' becomes ['CX 0 1', 'CX 2 3'].
    This allows injecting faults *between* gates that were originally grouped.
    """
    atomized_ops = []

    # Common 2-qubit gates in CSS codes
    TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}

    for op in circuit.flattened():
        # Handle 2-Qubit Gates (Target pairs)
        if op.name in TWO_QUBIT_GATES:
            targets = op.targets_copy()
            # Iterate in steps of 2
            for k in range(0, len(targets), 2):
                atomized_ops.append(
                    stim.CircuitInstruction(op.name, targets[k:k + 2], op.gate_args_copy())
                )

        # Handle Annotations (Don't split, just keep)
        elif op.name in {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}:
            atomized_ops.append(op)

        # Handle 1-Qubit Gates & Measurements (Single targets)
        else:
            # e.g. H, X, Z, M, R, MR
            targets = op.targets_copy()
            for t in targets:
                atomized_ops.append(
                    stim.CircuitInstruction(op.name, [t], op.gate_args_copy())
                )

    return atomized_ops


def get_fault_locations(ops: list) -> list[tuple[int, int]]:
    possible_faults = []
    max_qubit = 0
    for i, op in enumerate(ops):
        for target in op.targets_copy():
            max_qubit = max(max_qubit, target.value)
            if target.is_qubit_target:
                possible_faults.append((i, target.value))
    return [(0, i) for i in range(max_qubit)] + possible_faults


def build_noisy_circuit(fault_combo: tuple[tuple[int, int], ...], fault_type: str, init_str: str,
                        ops: list) -> stim.Circuit:
    # A. Build Noisy Circuit
    noisy_c = stim.Circuit()
    noisy_c.append_from_stim_program_text(init_str)  # Initialize clean state

    current_faults = {f[0]: [] for f in fault_combo}
    for f in fault_combo:
        current_faults[f[0]].append(f[1])

    for i, op in enumerate(ops):
        noisy_c.append(op)
        if i in current_faults:
            for q in current_faults[i]:
                noisy_c.append(fault_type, [q])

    # B. Run Simulation
    return noisy_c


if __name__ == "__main__":
    # Example Usage
    stabs = list_to_str_stabs(steane_code_stabs())
    print(stabs)
    decoder_table = build_css_syndrome_table(stabs, 3)
    print(decoder_table)

    steane_circ = stim.Circuit("""
    H 9
    CX 9 7 6 7 0 10 2 8 7 10 8 7 9 8 3 7 10 8 4 8
    M 8
    CX 5 9 1 10 10 7
    M 7
    CX 10 9
    H 10
    M 10 9
        """)

    compute_modified_lookup_table(steane_circ, steane_code_stabs(), np.array([0, 0, 0, 0, 1, 1, 1]), decoder_table, [3],
                                  "X", 3)

# verify_t_fault_tolerance(steane_code(), list_to_str_stabs(steane_code_stabs()), t_faults=1)

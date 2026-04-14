from collections import defaultdict

import stim

TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}
Z_MEASUREMENTS = {"MR", "M", "MZ"}
X_MEASUREMENTS = {"MX"}
Z_INITIALIZATIONS = {"MR", "R"}
X_INITIALIZATIONS = {"RX"}
SPECIAL_GATES = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}


def layered_ops_to_noisy_stim_circuit(layered_ops: list[list[tuple[str, list[int]]]], num_qubits: int, p_1: float,
                                      p_2: float, p_init: float, p_meas: float, p_mem: float) -> stim.Circuit:
    circuit = stim.Circuit()
    for i, ops in enumerate(layered_ops):
        unused_qubits = set(range(num_qubits))
        for op_name, targets in ops:
            unused_qubits -= set(targets)

            if op_name in Z_MEASUREMENTS:
                p_meas > 0 and circuit.append("X_ERROR", targets, p_meas)
            elif op_name in X_MEASUREMENTS:
                p_meas > 0 and circuit.append("Z_ERROR", targets, p_meas)

            circuit.append(op_name, targets)

            if op_name in X_INITIALIZATIONS:
                p_init > 0 and circuit.append("Z_ERROR", targets, p_init)
            elif op_name in Z_INITIALIZATIONS:
                p_init > 0 and circuit.append("X_ERROR", targets, p_init)
            elif op_name in TWO_QUBIT_GATES:
                p_2 > 0 and circuit.append("DEPOLARIZE2", targets, p_2)
            elif op_name in SPECIAL_GATES:
                pass
            else:
                p_1 > 0 and circuit.append("DEPOLARIZE1", targets, p_1)

        if i != len(layered_ops) - 1:
            p_mem > 0 and circuit.append("DEPOLARIZE1", unused_qubits, p_mem)
            circuit.append("TICK", [])
    return circuit


def _expand_stim_operation_list(operations: list[tuple[str, list[int]]]):
    stim_operations = []
    for op_name, targets in operations:
        if op_name in TWO_QUBIT_GATES:
            for i in range(0, len(targets), 2):
                stim_operations.append((op_name, [targets[i], targets[i + 1]]))
        elif op_name in SPECIAL_GATES:
            stim_operations.append((op_name, targets))
        else:
            for t in targets:
                stim_operations.append((op_name, [t]))
    return stim_operations


def _layer_circuit_ops(operations: list[tuple[str, list[int]]], num_qubits: int):
    # Minor correction: range(num_qubits) avoids creating a ghost qubit tracker
    all_qubits = range(num_qubits)

    # --- PASS 1: ASAP Forward Layering ---
    next_free_layer = {q: 0 for q in all_qubits}
    asap_layers = defaultdict(list)

    for op_name, targets in operations:
        # Note: If you pass "MR" in here, it will be kept as one block.
        # For optimal noise, you should preprocess your operations list
        # to convert ("MR", targets) into ("M", targets) and ("R", targets)
        # before calling this function!

        last_layer = max((next_free_layer[i] for i in targets), default=0)
        asap_layers[last_layer].append((op_name, targets))

        for i in targets:
            next_free_layer[i] = last_layer + 1

    # Convert dict to a dense list of lists
    max_layer = max(asap_layers.keys(), default=-1)
    layers = [asap_layers[i] for i in range(max_layer + 1)]

    # --- PASS 2: ALAP Backward Reset Shifting ---
    # Track the exact layer index where a qubit is NEXT used.
    # Initialize to the length of layers (representing the end of the circuit)
    next_required = {q: len(layers) for q in all_qubits}

    # Iterate backwards through the ASAP layers
    for i in range(len(layers) - 1, -1, -1):
        current_layer_ops = layers[i]
        kept_ops = []

        for op_name, targets in current_layer_ops:
            if op_name in {"R", "RX"}:
                # Splinter the reset: Handle each qubit independently
                for t in targets:
                    target_layer = next_required[t] - 1

                    if target_layer > i:
                        # Push this specific qubit's reset forward in time
                        layers[target_layer].append((op_name, [t]))
                    else:
                        # It's already as late as it can be, keep it here
                        kept_ops.append((op_name, [t]))
            else:
                # Keep normal gates where they are
                kept_ops.append((op_name, targets))
                # Mark these qubits as required at the current layer i
                for t in targets:
                    next_required[t] = i

        # Update the current layer with only the operations that didn't get pushed
        layers[i] = kept_ops

    # --- PASS 3: Cleanup ---
    # Shifting resets out of early layers might leave some layers completely empty.
    # We strip them out to prevent unnecessary DEPOLARIZE1 idle cycles in your noise model.
    return [layer for layer in layers if layer]


def apply_qubit_reuse(layers: list[list[tuple[str, list[int]]]], n_data: int) -> tuple[list[list[tuple[str, list[int]]]], int]:
    """
    Modified to ensure logical data qubits (0 to n_data-1) map 1:1 to physical
    registers and are never reused/overwritten by ancillas.
    """
    # 1. Calculate the lifespan (birth layer, death layer) of every logical qubit
    births = {}
    deaths = {}

    for layer_idx, layer in enumerate(layers):
        for op_name, targets in layer:
            for t in targets:
                if t not in births:
                    births[t] = layer_idx
                deaths[t] = layer_idx

    # 2. Map logical qubits to physical qubits
    logical_to_physical = {}

    # Pre-map data qubits so they are "locked" to their original indices
    for i in range(n_data):
        logical_to_physical[i] = i

    physical_freelist = []
    next_new_physical_qubit = n_data

    free_at_layer = {i: [] for i in range(len(layers))}

    for layer_idx in range(len(layers)):
        # Free up physical qubits whose logical occupants died in the PREVIOUS layer
        if layer_idx > 0:
            for p_q in free_at_layer[layer_idx - 1]:
                # Only return to freelist if it's NOT a data qubit
                if p_q >= n_data:
                    physical_freelist.append(p_q)

        # Allocate logical qubits born in this layer
        for logical_q, birth_layer in births.items():
            if birth_layer == layer_idx:
                # Skip if already pre-mapped (data qubits)
                if logical_q in logical_to_physical:
                    # We still need to track when the data qubit "dies" (ends its role in this gadget)
                    # so we don't accidentally free it if it's not actually the end of the circuit
                    death_layer = deaths[logical_q]
                    free_at_layer[death_layer].append(logical_to_physical[logical_q])
                    continue

                if physical_freelist:
                    # Reuse an available physical qubit
                    assigned_physical = physical_freelist.pop()
                else:
                    # Allocate a brand new physical qubit
                    assigned_physical = next_new_physical_qubit
                    next_new_physical_qubit += 1

                logical_to_physical[logical_q] = assigned_physical

                # Schedule this physical qubit to be freed after the logical qubit dies
                death_layer = deaths[logical_q]
                free_at_layer[death_layer].append(assigned_physical)

    # 3. Rewrite the layers using the new physical mapping
    optimized_layers = []
    for layer in layers:
        new_layer = []
        for op_name, targets in layer:
            mapped_targets = [logical_to_physical[t] for t in targets]
            new_layer.append((op_name, mapped_targets))
        optimized_layers.append(new_layer)

    return optimized_layers, next_new_physical_qubit

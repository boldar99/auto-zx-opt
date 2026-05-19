from collections import defaultdict
import stim

TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}
Z_MEASUREMENTS = {"MR", "M", "MZ"}
X_MEASUREMENTS = {"MX"}
Z_INITIALIZATIONS = {"MR", "R"}
X_INITIALIZATIONS = {"RX"}
SPECIAL_GATES = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}


def _get_target_values(targets: list[stim.GateTarget]) -> list[int]:
    """Safely extracts integer indices from Stim GateTargets, ignoring records."""
    return [t.value for t in targets if t.is_qubit_target]


def _expand_stim_operation_list(operations: list[tuple[str, list[stim.GateTarget]]]):
    stim_operations = []
    for op_name, targets in operations:
        t_vals = _get_target_values(targets)
        if not t_vals:
            continue

        if op_name in TWO_QUBIT_GATES:
            for i in range(0, len(t_vals), 2):
                stim_operations.append((op_name, [t_vals[i], t_vals[i + 1]]))
        elif op_name in SPECIAL_GATES:
            stim_operations.append((op_name, t_vals))
        else:
            for t in t_vals:
                stim_operations.append((op_name, [t]))
    return stim_operations


def _layer_circuit_ops(operations: list[tuple[str, list[int]]], num_qubits: int):
    all_qubits = range(num_qubits)
    next_free_layer = {q: 0 for q in all_qubits}
    asap_layers = defaultdict(list)
    meas_id = 0

    # --- PASS 1: ASAP Forward Layering ---
    for op_name, targets in operations:
        last_layer = max((next_free_layer[i] for i in targets), default=0)

        if op_name in ("M", "MX"):
            asap_layers[last_layer].append(((op_name, meas_id), targets))
            meas_id += 1
        else:
            asap_layers[last_layer].append((op_name, targets))

        for i in targets:
            next_free_layer[i] = last_layer + 1

    max_layer = max(asap_layers.keys(), default=-1)
    if max_layer == -1:
        return []

    layers = [asap_layers[i] for i in range(max_layer + 1)]

    # --- PASS 2: ALAP Backward Reset Shifting ---
    next_required = {q: len(layers) for q in all_qubits}

    for i in range(len(layers) - 1, -1, -1):
        current_layer_ops = layers[i]
        kept_ops = []

        for item in current_layer_ops:
            # Handle tuple unpacking for tagged measurements
            is_tagged_meas = isinstance(item[0], tuple)
            op_name = item[0][0] if is_tagged_meas else item[0]
            targets = item[1]

            if op_name in {"R", "RX"}:
                for t in targets:
                    target_layer = next_required[t] - 1
                    if target_layer > i:
                        layers[target_layer].append((op_name, [t]))
                    else:
                        kept_ops.append((op_name, [t]))
            else:
                kept_ops.append(item)
                for t in targets:
                    next_required[t] = i

        layers[i] = kept_ops

    return [layer for layer in layers if layer]


def layered_ops_to_noisy_stim_circuit(
    layered_ops: list[list[tuple]],
    num_qubits: int,
    p_1: float,
    p_2: float,
    p_init: float,
    p_meas: float,
    p_mem: float
) -> tuple[stim.Circuit, dict[int, int]]:
    circuit = stim.Circuit()
    measurement_mapping = {}
    meas_id = 0

    for i, ops in enumerate(layered_ops):
        # We can now safely subtract ints from ints
        unused_qubits = set(range(num_qubits))

        for item in ops:
            is_tagged_meas = isinstance(item[0], tuple)
            op_name = item[0][0] if is_tagged_meas else item[0]
            targets = item[1]

            unused_qubits -= set(targets)

            if is_tagged_meas:
                _, og_meas_id = item[0]
                measurement_mapping[og_meas_id] = meas_id
                meas_id += 1

            if op_name in Z_MEASUREMENTS and p_meas > 0:
                circuit.append("X_ERROR", targets, p_meas)
            elif op_name in X_MEASUREMENTS and p_meas > 0:
                circuit.append("Z_ERROR", targets, p_meas)

            circuit.append(op_name, targets)

            if op_name in X_INITIALIZATIONS and p_init > 0:
                circuit.append("Z_ERROR", targets, p_init)
            elif op_name in Z_INITIALIZATIONS and p_init > 0:
                circuit.append("X_ERROR", targets, p_init)
            elif op_name in TWO_QUBIT_GATES and p_2 > 0:
                circuit.append("DEPOLARIZE2", targets, p_2)
            elif op_name not in SPECIAL_GATES and p_1 > 0:
                circuit.append("DEPOLARIZE1", targets, p_1)

        if i != len(layered_ops) - 1 and p_mem > 0 and unused_qubits:
            circuit.append("DEPOLARIZE1", sorted(list(unused_qubits)), p_mem)

        circuit.append("TICK", [])

    return circuit, measurement_mapping


def make_stim_circ_noisy(circ: stim.Circuit, p: float) -> tuple[stim.Circuit, dict[int, int]]:
    """Properly utilizes the layer structure to construct the noisy circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op != "DETECTOR"]

    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)

    noisy_circ, mm = layered_ops_to_noisy_stim_circuit(
        layered_ops=layered_ops,
        num_qubits=circ.num_qubits,
        p_1=0,
        p_2=p,
        p_init=(2 / 3) * p,
        p_meas=(2 / 3) * p,
        p_mem=p / 100
    )
    return noisy_circ, mm


def get_circuit_depth(circ: stim.Circuit) -> int:
    """Returns the strict ASAP depth of the circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op not in SPECIAL_GATES]
    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)
    return len(layered_ops)


def get_spacetime_volume(circ: stim.Circuit) -> int:
    """Calculates the sum of active ticks for all qubits across the circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op not in SPECIAL_GATES]
    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)

    volume = 0
    for ops in layered_ops:
        # Count unique qubits targeted in this layer
        active_in_layer = {t for _, targets in ops for t in targets}
        volume += len(active_in_layer)
    return volume
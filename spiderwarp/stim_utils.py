from __future__ import annotations

from collections import defaultdict
from typing import Literal, TYPE_CHECKING

import numpy as np
import pyzx as zx
import stim
import stimcirq
from cirq.contrib.qasm_import import circuit_from_qasm

from spiderwarp.utils import flatten

if TYPE_CHECKING:
    from spiderwarp.csscode import CSSCode


TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "XCX", "SWAP", "CY", "XCZ", "YCX"}
Z_MEASUREMENTS = {"MR", "M", "MZ"}
X_MEASUREMENTS = {"MX"}
Z_INITIALIZATIONS = {"MR", "R"}
X_INITIALIZATIONS = {"RX"}
SPECIAL_GATES = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}
NOISE_GATES = {"X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}


def circuit_for_inspection(circuit: stim.Circuit) -> stim.Circuit:
    """Return a display-only circuit without MPPs, detectors, or observables.

    Repeat blocks are flattened so measurement-record references can be
    updated across iterations. Feedback using a removed MPP result is also
    removed; references to retained measurements are remapped. All other
    operations, including injected faults, are retained. The input is not
    mutated. This view is not an equivalent circuit for execution/verification.
    """
    result = stim.Circuit()
    record_map: list[int | None] = []
    for operation in circuit.flattened():
        if operation.name in {"MPP", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            record_map.extend([None] * operation.num_measurements)
            continue

        next_record = result.num_measurements
        if not any(target.is_measurement_record_target
                   for target in operation.targets_copy()):
            result.append(operation)
        else:
            for group in operation.target_groups():
                targets = []
                for target in group:
                    if not target.is_measurement_record_target:
                        targets.append(target)
                        continue
                    original_record = len(record_map) + target.value
                    if original_record < 0:
                        raise ValueError("measurement reference precedes the input circuit")
                    retained_record = record_map[original_record]
                    if retained_record is None:
                        break
                    targets.append(stim.target_rec(retained_record - next_record))
                else:
                    result.append(stim.CircuitInstruction(
                        operation.name, targets, operation.gate_args_copy(),
                        tag=operation.tag,
                    ))

        record_map.extend(range(next_record, next_record + operation.num_measurements))
    return result


def qasm_str_to_stim_circuit(qasm_str: str) -> stim.Circuit:
    cirq_circuit = circuit_from_qasm(qasm_str)
    return stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit)


def stim_to_pyzx(stim_circuit: stim.Circuit, n_data: int) -> zx.Graph:
    circ = zx.Circuit(n_data)

    for op, targets, _ in stim_circuit.flattened_operations():
        if op in ("R", "RX"):
            for t in targets:
                if t < n_data:
                    continue
                if op == "R":
                    circ.add_gate("InitAncilla", label=t, state="0")
                elif op == "RX":
                    circ.add_gate("InitAncilla", label=t, state="+")

        elif op == "CX":
            for i in range(0, len(targets), 2):
                c, n = targets[i], targets[i + 1]
                circ.add_gate("CNOT", c, n)

        elif op == "CZ":
            for i in range(0, len(targets), 2):
                q1, q2 = targets[i], targets[i + 1]
                circ.add_gate("CZ", q1, q2)

        elif op == "XCX":
            # XCX = (H ⊗ H) CZ (H ⊗ H).  PyZX has no dedicated XCX
            # circuit gate, but this decomposition produces the desired
            # Hadamard edge between X spiders after simplification.
            for i in range(0, len(targets), 2):
                q1, q2 = targets[i], targets[i + 1]
                circ.add_gate("H", q1)
                circ.add_gate("H", q2)
                circ.add_gate("CZ", q1, q2)
                circ.add_gate("H", q1)
                circ.add_gate("H", q2)

        elif op == "H":
            for t in targets:
                circ.add_gate("H", t)

        elif op in ("M", "MX", "MR"):
            for t in targets:
                if op in ("M", "MR"):
                    circ.add_gate("PostSelect", label=t, state="0")
                elif op == "MX":
                    circ.add_gate("PostSelect", label=t, state="+")

    return circ.to_graph()


def explode_circuit(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    """
    Decomposes a circuit into a list of atomic instructions.
    E.g., 'CX 0 1 2 3' becomes ['CX 0 1', 'CX 2 3'].
    This allows injecting faults *between* gates that were originally grouped.
    """
    atomized_ops = []

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


def make_stim_circ_single_type_noisy(
    circ: stim.Circuit, p: float, error_type: str
) -> stim.Circuit:
    """Insert independent Pauli faults without changing measurement records.

    Each atomic unitary or reset gets a fault on each physical target after
    the operation. Measurements (including measure-and-reset) and annotations
    retain the existing verifier model's exclusion from this fault set.
    Keeping operation order preserves detector, observable, and feedback
    references, including across qubit reuse and repeat blocks.
    """
    if error_type not in {"X", "Z"}:
        raise ValueError('error_type must be "X" or "Z"')
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    noisy_circuit = stim.Circuit()
    for operation in circ:
        if isinstance(operation, stim.CircuitRepeatBlock):
            noisy_circuit.append(stim.CircuitRepeatBlock(
                operation.repeat_count,
                make_stim_circ_single_type_noisy(
                    operation.body_copy(), p, error_type
                ),
                tag=operation.tag,
            ))
            continue

        gate = stim.gate_data(operation.name)
        if gate.produces_measurements or not (gate.is_unitary or gate.is_reset):
            noisy_circuit.append(operation)
            continue

        for group in operation.target_groups():
            targets = group
            if gate.takes_pauli_targets:
                # target_groups separates Pauli products and omits combiners.
                targets = []
                for target in group:
                    if targets:
                        targets.append(stim.target_combiner())
                    targets.append(target)
            noisy_circuit.append(stim.CircuitInstruction(
                operation.name, targets, operation.gate_args_copy(),
                tag=operation.tag,
            ))
            qubits = list(dict.fromkeys(
                target.value for target in group
                if target.is_qubit_target or target.is_x_target
                or target.is_y_target or target.is_z_target
            ))
            if qubits:
                noisy_circuit.append(f"{error_type}_ERROR", qubits, p)

    return noisy_circuit


def steane_se_from_stim_state_prep(
    circ: stim.Circuit,
    se_basis: Literal["X", "Z"],
    n: int,
    offset: int = 0,
) -> stim.Circuit:
    ret = stim.Circuit()
    for op in circ:
        targets = [stim.GateTarget(t.value + n + offset) for t in op.targets_copy()]
        if op.name == "DETECTOR":
            continue
        new_op = stim.CircuitInstruction(op.name, targets, op.gate_args_copy())
        ret.append(new_op)
    if se_basis == "X":
        ret.append("CX", flatten(zip(range(n + offset, 2 * n + offset), range(n))))
        ret.append("MX", range(n + offset, 2 * n + offset))
    elif se_basis == "Z":
        ret.append("CX", flatten(zip(range(n), range(n + offset, 2 * n + offset))))
        ret.append("M", range(n + offset, 2 * n + offset))
    else:
        raise ValueError(f"Unknown SE basis: {se_basis!r}")

    return ret


def _get_target_values(targets: list[stim.GateTarget]) -> list[int]:
    """Safely extracts integer indices from Stim GateTargets, ignoring records."""
    return [t.value for t in targets if t.is_qubit_target]


def _expand_stim_operation_list(operations: list[tuple[str, list[stim.GateTarget | int]]]):
    stim_operations = []
    for op_name, targets in operations:
        if isinstance(targets, stim.GateTarget):
            t_vals = _get_target_values(targets)
        else:
            t_vals = targets
        if not targets:
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
                measurement_mapping[meas_id] = og_meas_id
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


def make_stim_circ_noisy(circ: stim.Circuit, p: float, one_cnot_per_layer: bool=False) -> tuple[stim.Circuit, dict[int, int]]:
    """Properly utilizes the layer structure to construct the noisy circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op != "DETECTOR"]

    expanded_ops = _expand_stim_operation_list(operations)
    if one_cnot_per_layer:
        layered_ops = [[op] for op in expanded_ops]
    else:
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


def _projection_feedback_supports(check_rows: np.ndarray) -> list[tuple[int, ...]]:
    """Map measured syndrome bits to Pauli supports that clear their signs.

    Row reduction gives R = E H. Set the pivot correction bits to E s,
    with free bits zero. This solves H correction = s for every physical
    syndrome, including dependent checks whose outcomes satisfy the same
    linear relations as their rows.
    """
    num_checks, num_qubits = check_rows.shape
    augmented = np.hstack([
        check_rows.astype(np.uint8) % 2,
        np.eye(num_checks, dtype=np.uint8),
    ])
    pivot_columns = []
    for column in range(num_qubits):
        pivot_row = len(pivot_columns)
        candidates = np.flatnonzero(augmented[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        augmented[[pivot_row, selected]] = augmented[[selected, pivot_row]]
        for row in range(num_checks):
            if row != pivot_row and augmented[row, column]:
                augmented[row] ^= augmented[pivot_row]
        pivot_columns.append(column)
        if len(pivot_columns) == num_checks:
            break
    return [
        tuple(column for row, column in enumerate(pivot_columns)
              if augmented[row, num_qubits + measurement])
        for measurement in range(num_checks)
    ]


def perfect_state_from_code(code: CSSCode, basis: Literal["X", "Z"]) -> stim.Circuit:
    """Prepare ideal logical plus (X) or zero (Z) using corrected projections.

    Reset in the requested physical basis, project the opposite-basis code
    stabilizers with MPP, and correct their random signs using measurement-
    controlled Paulis. The resulting code stabilizers and requested logical
    operators all have eigenvalue +1. Raw MPP outcomes can still be random;
    callers skip these preparation records with ``circuit.num_measurements``.
    This is noiseless verification input, not an FT preparation gadget.
    """
    if basis not in {"X", "Z"}:
        raise ValueError('basis must be "X" or "Z"')

    perfect_state = stim.Circuit()
    perfect_state.append("RX" if basis == "X" else "R", range(code.n))
    check_rows = np.asarray(code.H_z if basis == "X" else code.H_x).reshape(-1, code.n)
    check_rows = check_rows[np.any(check_rows, axis=1)]
    target_function = stim.target_z if basis == "X" else stim.target_x
    for row in check_rows:
        targets = []
        support = [i for i, r in enumerate(row) if r == 1]
        for q in support:
            targets.append(target_function(q))
            targets.append(stim.target_combiner())
        if targets:
            targets.pop()
            perfect_state.append("MPP", targets)

    correction_gate = "CX" if basis == "X" else "CZ"
    for measurement, support in enumerate(_projection_feedback_supports(check_rows)):
        if support:
            record = stim.target_rec(measurement - perfect_state.num_measurements)
            perfect_state.append(
                correction_gate,
                [target for qubit in support for target in (record, qubit)],
            )
    return perfect_state


def get_circuit_depth(circ: stim.Circuit) -> int:
    """Returns the strict ASAP depth of the circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op not in SPECIAL_GATES]
    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)
    return len(layered_ops)

def get_circuit_width_per_timestep(circ: stim.Circuit) -> list[int]:
    """Returns the strict ASAP depth of the circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op not in SPECIAL_GATES]
    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)
    return [len(ops) for ops in layered_ops]


def get_num_cnots(circ: stim.Circuit) -> int:
    raw_cnots = [l for (name, l, _) in circ.flattened_operations() if name in ("CX", "CNOT")]
    cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
    return len(cnots)


def get_num_measurements(circ: stim.Circuit) -> int:
    return sum(
        len(l) for (name, l, _) in circ.flattened_operations() if name in Z_MEASUREMENTS or name in X_MEASUREMENTS
    )


def get_spacetime_volume(circ: stim.Circuit) -> int:
    """Calculates the sum of active ticks for all qubits across the circuit."""
    return get_circuit_depth(circ) * circ.num_qubits

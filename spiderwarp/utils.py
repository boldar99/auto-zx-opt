import itertools
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pyzx as zx
import stim
import stimcirq

from cirq.contrib.qasm_import import circuit_from_qasm


def _layer_cnot_circuit(cnots):
    num_qubits = max(map(max, cnots))
    all_qubits = range(num_qubits + 1)
    next_free_layer = {q: 0 for q in all_qubits}
    layers = defaultdict(list)
    for c, n in cnots:
        l = max(next_free_layer[c], next_free_layer[n])
        layers[l].append((c, n))
        next_free_layer[c] = l + 1
        next_free_layer[n] = l + 1
    return list(layers.values())


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


def _sorted_pair(v1, v2):
    return (v1, v2) if v1 < v2 else (v2, v1)


def get_project_root() -> Path:
    return Path(__file__).parent


def qpic_to_stim(qpic_str: str) -> stim.Circuit:
    circuit = stim.Circuit()

    wire_to_idx = {}
    next_idx = 0
    prep_queues = {}

    lines = qpic_str.split('\n')

    # --- PASS 1: Map Wires and Build Preparation Queues ---
    for line in lines:
        line = line.split('#')[0].strip()
        if not line: continue
        parts = line.split()

        # Look for Wire definitions
        if len(parts) >= 2 and parts[1] == 'W':
            wire_name = parts[0]
            if wire_name not in wire_to_idx:
                wire_to_idx[wire_name] = next_idx
                next_idx += 1

            # Extract all bracketed labels (e.g., {33 \ket{0}} or {\ket{+}} or {})
            blocks = re.findall(r'\{(.*?)\}', line)
            states = []
            for b in blocks:
                b_clean = b.replace(" ", "")
                # Bypass backslash escape issues by searching for the core string
                if 'ket{+' in b_clean:
                    states.append('+')
                elif 'ket{0' in b_clean:
                    states.append('0')

            if states:
                prep_queues[wire_name] = states

    # --- PASS 2: Build the Circuit Chronologically ---
    needs_reset = {wire: False for wire in wire_to_idx}

    # 1. Pop the initial state for all wires at the very beginning
    for wire in wire_to_idx:
        if wire in prep_queues and prep_queues[wire]:
            state = prep_queues[wire].pop(0)
            circuit.append('RX' if state == '+' else 'R', [wire_to_idx[wire]])

    # 2. Process operations
    for line in lines:
        line = line.split('#')[0].strip()
        if not line: continue
        parts = line.split()

        # Trigger Resets at START blocks
        if 'START' in parts:
            start_idx = parts.index('START')
            for wire in parts[:start_idx]:
                if wire in needs_reset and needs_reset[wire]:
                    if wire in prep_queues and prep_queues[wire]:
                        state = prep_queues[wire].pop(0)
                        circuit.append('RX' if state == '+' else 'R', [wire_to_idx[wire]])
                    needs_reset[wire] = False  # Clear the flag
            continue

        # Ignore non-logical drawing commands
        if parts[0] in ('PREAMBLE', 'TOUCH') or 'END' in parts or parts[1] == 'W':
            continue

        # Parse Measurements
        if len(parts) >= 2 and parts[1] == 'M':
            wire = parts[0]
            if wire not in wire_to_idx: continue

            idx = wire_to_idx[wire]
            label_clean = line.replace(" ", "")

            # Differentiate X vs Z basis
            if '$X$' in label_clean or 'text{X}' in label_clean:
                circuit.append('MX', [idx])
            else:
                circuit.append('M', [idx])

            # CRITICAL: Mark wire as dead. It requires a reset before its next operation.
            needs_reset[wire] = True
            continue

        # Parse CNOT Gates
        if len(parts) == 2 and parts[1].startswith('+'):
            ctrl = parts[0]
            tgt = parts[1][1:]

            if ctrl in wire_to_idx and tgt in wire_to_idx:
                # Fallback: If a gate happens without a START block, check for pending resets
                for wire in (ctrl, tgt):
                    if needs_reset[wire]:
                        if wire in prep_queues and prep_queues[wire]:
                            state = prep_queues[wire].pop(0)
                            circuit.append('RX' if state == '+' else 'R', [wire_to_idx[wire]])
                        needs_reset[wire] = False

                circuit.append('CX', [wire_to_idx[ctrl], wire_to_idx[tgt]])
            continue

    return circuit


def flatten(ls: Iterable[Iterable]) -> list:
    return list(itertools.chain(*ls))


def steane_se_from_stim_state_prep(circ: stim.Circuit, se_basis: str, n: int) -> stim.Circuit:
    ret = stim.Circuit()
    for op in circ:
        targets = [stim.GateTarget(t.value + n) for t in op.targets_copy()]
        new_op = stim.CircuitInstruction(op.name, targets, op.gate_args_copy())
        ret.append(new_op)
    if se_basis == "Z":
        ret.append("CX", flatten(zip(range(n, 2 * n), range(n))))
        ret.append("MX", range(n, 2 * n))
    elif se_basis == "X":
        ret.append("CX", flatten(zip(range(n), range(n, 2 * n))))
        ret.append("M", range(n, 2 * n))
    else:
        raise Exception("Unknown se_basis: {}".format(se_basis))
    return ret


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


import stim


def get_temporal_mapping(circ: stim.Circuit, num_data_qubits: int) -> dict[int, int]:
    """
    Extracts the chronological sequence of measurements from a Stim circuit.

    Args:
        circ: The stim.Circuit containing the extracted syndrome measurements.
        num_data_qubits: The number of data qubits (the offset to subtract).

    Returns:
        dict[int, int]: A mapping where:
            - Keys are the temporal indices (0, 1, 2...) representing the
              position of the measurement in the Stim output array.
            - Values are the relative qubit indices (absolute_qubit - num_data_qubits).
    """
    temporal_to_relative_qubit = {}
    temporal_idx = 0

    for op in circ:
        # Standard single-qubit measurement gates in Stim
        if op.name in ("M", "MX", "MY", "MZ"):
            for target in op.targets_copy():
                if target.is_qubit_target:
                    # Calculate the relative index to match your spatial mapping
                    relative_qubit = target.value - num_data_qubits
                    temporal_to_relative_qubit[temporal_idx] = relative_qubit
                    temporal_idx += 1

    return temporal_to_relative_qubit


def compose_measurement_mappings(
        circ: stim.Circuit,
        spatial_mapping: dict[int, int],
        num_data_qubits: int
) -> dict[int, int]:
    """
    Creates a direct mapping from the Stim output array index to the
    original unoptimized measurement index.
    """
    temporal_mapping = get_temporal_mapping(circ, num_data_qubits)

    composed_mapping = {}
    for temporal_idx, relative_qubit in temporal_mapping.items():
        if relative_qubit in spatial_mapping:
            composed_mapping[temporal_idx] = spatial_mapping[relative_qubit]

    return composed_mapping


def load_state_prep_circuit(directory: str, name: str) -> stim.Circuit:
    root = get_project_root()
    directory = root.joinpath("assets", "circuits", "FTStatePrep", directory)
    if (file := directory.joinpath(f"{name}.stim")).exists():
        circ = stim.Circuit(file.read_text())
    else:
        file = directory.joinpath(f"{name}.qasm")
        cirq_circuit = circuit_from_qasm(file.read_text())
        circ = stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit)
    if circ[0].name not in ('R', "RX"):
        circ.insert(0, stim.CircuitInstruction("R", range(circ.num_qubits)))
    return circ



def qasm_str_to_stim_circuit(qasm_str: str) -> stim.Circuit:
    cirq_circuit = circuit_from_qasm(qasm_str)
    circ = stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit)
    if circ[0].name not in ('R', "RX"):
        circ.insert(0, stim.CircuitInstruction("R", range(circ.num_qubits)))
    return circ


def load_steane_perm_circuits(name: str) -> tuple[stim.Circuit, stim.Circuit, stim.Circuit, stim.Circuit]:
    alt_names = {
        "4_8_8_d5": "cc_4_8_8_d5",
        "17_1_5": "cc_4_8_8_d5",

        "4_8_8_d7": "cc_4_8_8_d7",
        "31_1_7": "cc_4_8_8_d7",

        "6_6_6_d5": "cc_6_6_6_d5",
        "19_1_5": "cc_6_6_6_d5",

        "6_6_6_d7": "cc_6_6_6_d7",
        "39_1_7": "cc_6_6_6_d7",

        "20_2_6": "eve_20_2_6",

        "25_1_5": "rotated_surface_d5"
    }
    name = alt_names.get(name, name)

    root = get_project_root()
    directory = root.joinpath("assets", "circuits", "FTStatePrep", "SteanePerm", name)
    files = os.listdir(directory)
    [c1, c2, c3, c4] = [qasm_str_to_stim_circuit(directory.joinpath(f).read_text()) for f in files]
    return c1, c2, c3, c4


def MQT_STEANE_PERM_QECCS():
    root = get_project_root()
    directory = root.joinpath("assets", "circuits", "FTStatePrep", "SteanePerm")
    return os.listdir(directory)


MQT_STEANE_CODE_NAME = {
    "cc_4_8_8_d5": "17_1_5",
    "cc_4_8_8_d7": "31_1_7",
    "cc_6_6_6_d5": "19_1_5",
    "cc_6_6_6_d7": "39_1_7",
    "eve_20_2_6": "20_2_6",
    "rotated_surface_d5": "25_1_5",
}


if __name__ == "__main__":
    print(MQT_STEANE_PERM_QECCS())
    # circuit = load_state_prep_circuit("misc", "zero_32_20_4")
    # se = steane_se_from_stim_state_prep(circuit, se_basis="Z", n=32)
    # graph = stim_to_pyzx(se, 32)
    # print(graph)

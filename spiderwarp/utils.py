import re
from collections import defaultdict
from pathlib import Path

import stim


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

    # Dictionary to map arbitrary qpic string names to sequential integer Stim indices
    wire_to_idx = {}
    next_idx = 0

    lines = qpic_str.split('\n')

    for line in lines:
        # Strip whitespace and remove inline comments
        line = line.split('#')[0].strip()
        if not line:
            continue

        parts = line.split()

        # Skip purely visual/structural markers
        if parts[0] in ('PREAMBLE', 'TOUCH') or 'START' in parts or 'END' in parts:
            continue

        # 1. Parse Wire Initializations (e.g., "1 W {1 \ket{+}}")
        if len(parts) >= 2 and parts[1] == 'W':
            wire_name = parts[0]
            if wire_name not in wire_to_idx:
                wire_to_idx[wire_name] = next_idx
                next_idx += 1

            idx = wire_to_idx[wire_name]

            # Extract the FIRST bracketed label to determine the initialization state
            first_label_match = re.search(r'\{(.*?)\}', line)
            if first_label_match:
                first_label = first_label_match.group(1).replace(" ", "")
                if 'ket{+' in first_label:
                    circuit.append('RX', [idx])
                elif 'ket{0' in first_label:
                    circuit.append('R', [idx])
            continue

        # 2. Parse Measurements (e.g., "33 M {\scriptsize $Z$}")
        if len(parts) >= 2 and parts[1] == 'M':
            wire_name = parts[0]
            if wire_name not in wire_to_idx:
                continue  # Ignore measuring an undeclared wire

            idx = wire_to_idx[wire_name]
            label_clean = line.replace(" ", "")

            if '$X$' in label_clean:
                circuit.append('MX', [idx])
            else:
                # Default to Z basis if it's explicitly $Z$ or unrecognised
                circuit.append('M', [idx])
            continue

        # 3. Parse Explicit Hadamard Gates (e.g., "33 H")
        if len(parts) >= 2 and parts[1] == 'H':
            wire_name = parts[0]
            if wire_name in wire_to_idx:
                circuit.append('H', [wire_to_idx[wire_name]])
            continue

        # 4. Parse CNOT Gates (e.g., "3 +4")
        if len(parts) == 2 and parts[1].startswith('+'):
            ctrl_name = parts[0]
            tgt_name = parts[1][1:]  # Strip the '+' sign

            if ctrl_name in wire_to_idx and tgt_name in wire_to_idx:
                circuit.append('CX', [wire_to_idx[ctrl_name], wire_to_idx[tgt_name]])
            continue

    return circuit


import stim
import re


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


if __name__ == "__main__":
    # --- Example Usage ---
    root = get_project_root()
    file = root.joinpath("assets", "circuits", "miscellaneous", "zero_32_20_4.qpic")
    with open(file, "r") as f:
        your_qpic_string = f.read()
    circuit = qpic_to_stim(your_qpic_string)
    print(circuit)

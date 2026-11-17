"""Offline single-fault catalogue for the four [[32, 20, 4]] extractions.

Uses the shared Python circuit descriptions and Stim's FlipSimulator. Enumerates
15 Pauli alternatives after each two-qubit gate, X/Y/Z after each one-qubit gate,
X after each reset, and a readout flip before each measurement. RX/MX are lowered
to R,H/H,M to match the Guppy emitter, including the Z variant's data Hadamards.

Flags and syndrome are changes relative to ideal extraction. data_x/data_z are
physical Pauli masks, with bit q identifying data qubit q; phases are ignored.
There is no stabilizer reduction, decoding, probability assignment, or implicit
idle/transport/leakage noise. Duplicate and invisible faults are retained.

    python -m sol_experiment.optimized_steane_fault_catalogue \
        --output out/optimized_steane_fault_catalogue.json
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Mapping

import stim

from sol_experiment.circuit_descriptions import (
    DetectorGroups, optimized_steane_32_20_4_X_d2, optimized_steane_32_20_4_X_d4,
)


@dataclass(frozen=True)
class FaultResponse:
    """One fault at an index in the reconstructed elementary operation list.

    pauli follows the operation's qubit order, including I factors. Faults are
    before M and after every other operation. Each row counts as one fault.
    """

    operation_index: int
    pauli: str
    flags: int
    syndrome: int
    data_x: int
    data_z: int


@dataclass(frozen=True)
class FaultCatalogue:
    basis: str
    distance: int
    faults: tuple[FaultResponse, ...]

    def group_by_observation(self) -> dict[tuple[int, int], list[FaultResponse]]:
        """Group compatible faults while preserving their different data errors."""
        groups = defaultdict(list)
        for fault in self.faults:
            groups[fault.flags, fault.syndrome].append(fault)
        return dict(groups)


def extraction_operations(basis: str, distance: int) -> tuple[list[stim.CircuitInstruction], DetectorGroups]:
    """Reconstruct the elementary operations used to number fault locations."""
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    if type(distance) is not int or distance not in (2, 4):
        raise ValueError("distance must be 2 or 4")
    describe = {2: optimized_steane_32_20_4_X_d2, 4: optimized_steane_32_20_4_X_d4}[distance]
    description, detectors = describe()
    operations = []
    for gate, targets in description:
        width = 2 if gate in {"CX", "CZ"} else 1
        for start in range(0, len(targets), width):
            qubits = targets[start:start + width]
            gates = {"RX": ("R", "H"), "MX": ("H", "M")}.get(gate, (gate,))
            operations.extend(stim.CircuitInstruction(name, qubits) for name in gates)
    if basis == "Z":
        rotations = [stim.CircuitInstruction("H", [q]) for q in range(32)]
        operations = rotations + operations + rotations
    return operations, detectors


def _simulate_fault(operations, prefixes, annotations, index, pauli):
    operation = operations[index]
    cut = index if operation.name == "M" else index + 1
    sim = stim.FlipSimulator(batch_size=1, num_qubits=60,
                             disable_stabilizer_randomization=True)
    sim.do(prefixes[cut])
    for p, target in zip(pauli, operation.targets_copy()):
        if p != "I":
            # A probability-one error channel injects a fault into the frame;
            # an ideal Pauli gate would instead change Stim's reference circuit.
            sim.do(stim.CircuitInstruction(p + "_ERROR", [target], [1]))
    for following in operations[cut:]:
        sim.do(following)
        if following.name == "R":
            # Stim may retain a Z frame on |0>, which is a reset stabilizer.
            # Clear it so each fresh ancilla has the same frame convention.
            sim.set_pauli_flip("I", qubit_index=following.targets_copy()[0].value,
                               instance_index=0)
    sim.do(annotations)
    return sim


def build_fault_catalogue(basis: str, distance: int) -> FaultCatalogue:
    """Return all single-fault alternatives; the no-fault response is all zero."""
    operations, detectors = extraction_operations(basis, distance)
    prefixes = [stim.Circuit()]
    for operation in operations:
        prefix = prefixes[-1].copy()
        prefix.append(operation)
        prefixes.append(prefix)
    annotations = stim.Circuit()
    for groups in (detectors.flags, detectors.syndrome):
        for refs in groups:
            annotations.append("DETECTOR", [stim.target_rec(ref) for ref in refs])

    faults = []
    for index, operation in enumerate(operations):
        width = len(operation.targets_copy())
        alternatives = ("X",) if operation.name in {"R", "M"} else (
            "".join(p) for p in product("IXYZ", repeat=width) if any(c != "I" for c in p)
        )
        for pauli in alternatives:
            sim = _simulate_fault(operations, prefixes, annotations, index, pauli)
            observed = sim.get_detector_flips(instance_index=0)
            n_flags = len(detectors.flags)
            flags = sum(int(bit) << i for i, bit in enumerate(observed[:n_flags]))
            syndrome = sum(int(bit) << i for i, bit in enumerate(observed[n_flags:]))
            data = sim.peek_pauli_flips(instance_index=0)
            x = sum((data[q] in (1, 2)) << q for q in range(32))
            z = sum((data[q] in (2, 3)) << q for q in range(32))
            faults.append(FaultResponse(index, pauli, flags, syndrome, x, z))
    return FaultCatalogue(basis, distance, tuple(faults))


def save_catalogues(path: str | Path, catalogues: Mapping[str, FaultCatalogue]) -> Path:
    """Save only action names and their fault records as deterministic JSON."""
    path = Path(path).expanduser().resolve()
    payload = {name: [asdict(fault) for fault in catalogue.faults]
               for name, catalogue in catalogues.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_catalogues(path: str | Path) -> dict[str, FaultCatalogue]:
    """Load saved records for further offline processing."""
    payload = json.loads(Path(path).expanduser().read_text())
    result = {}
    for name, rows in payload.items():
        basis, distance = name.split("_d")
        result[name] = FaultCatalogue(basis, int(distance), tuple(FaultResponse(**row) for row in rows))
    return result


def main(argv: list[str] | None = None) -> Path:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bases", nargs="+", choices=("X", "Z"), default=("X", "Z"))
    parser.add_argument("--distances", nargs="+", type=int, choices=(2, 4), default=(2, 4))
    parser.add_argument("--output", type=Path, default=Path("out/optimized_steane_fault_catalogue.json"))
    args = parser.parse_args(argv)
    catalogues = {}
    for basis in args.bases:
        for distance in args.distances:
            catalogue = build_fault_catalogue(basis, distance)
            name = f"{basis}_d{distance}"
            catalogues[name] = catalogue
            print(f"{name}: {len(catalogue.faults)} single-fault alternatives, "
                  f"{len(catalogue.group_by_observation())} distinct (flags, syndrome) observations")
    path = save_catalogues(args.output, catalogues)
    print(f"Saved catalogue: {path}")
    return path


if __name__ == "__main__":
    main()

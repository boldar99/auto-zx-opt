from __future__ import annotations

import abc
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import stimcirq
from cirq.contrib.qasm_import import circuit_from_qasm
import numpy as np
import pyzx as zx
import stim

from verify_fault_tolerance import explode_circuit


class Basis(Enum):
    X = 0
    Z = 1

    @classmethod
    def dual(cls, basis: Basis) -> Basis:
        return Basis.Z if basis == Basis.X else Basis.X


@dataclass
class NoiseModel:
    p_1: float = 0.0
    p_2: float = 0.0
    p_init: float = 0.0
    p_meas: float = 0.0
    p_mem: float = 0.0

    @classmethod
    def from_p(cls, p: float) -> NoiseModel:
        return cls(p_1=p / 100, p_2=p, p_init=p, p_meas=p, p_mem=p / 10)

    @classmethod
    def Quantinuum_H2(cls):
        return cls(p_1=3e-5, p_2=1e-3, p_init=1e-3, p_meas=1e-3, p_mem=1e-4)

    @classmethod
    def Quantinuum_Helios(cls):
        return cls(p_1=3e-5, p_2=8e-4, p_init=5e-4, p_meas=5e-4, p_mem=6e-4)

    @classmethod
    def Quantinuum_Sol(cls):
        return cls(p_1=3e-5, p_2=8e-4, p_init=5e-4, p_meas=5e-4, p_mem=6e-4)


@dataclass
class QECC:
    n: int
    k: int
    d: int
    H_x: np.ndarray
    H_z: np.ndarray
    L_x: np.ndarray
    L_z: np.ndarray
    is_self_dual: bool = False

    @classmethod
    def from_dict(cls, data: dict):
        DATA_TYPE = np.uint8
        n = data['n']
        k = data['k']
        d = data['d']
        is_self_dual = data['is_self_dual'] == 1
        if is_self_dual:
            H_x = H_z = np.asarray(data['H_x'], dtype=DATA_TYPE)
            L_x = L_z = np.asarray(data['L_x'], dtype=DATA_TYPE)
        else:
            H_x = np.asarray(data['H_x'], dtype=DATA_TYPE)
            H_z = np.asarray(data['H_z'], dtype=DATA_TYPE)
            L_x = np.asarray(data['L_x'], dtype=DATA_TYPE)
            L_z = np.asarray(data['L_z'], dtype=DATA_TYPE)
        return cls(n, k, d, H_x, H_z, L_x, L_z, is_self_dual=is_self_dual)

    @classmethod
    def from_stabs(cls, stabs: list[str], n, k, d):
        L_x, L_z = [], []
        H_x, H_z = [], []
        for i, stab in enumerate(stabs):

            # Somewhat tricky
            X, Z = (H_x, H_z) if i < n - k else (L_x, L_z)

            stab = stab.strip("+ \n\t")
            if "X" in stab:
                X.append([
                    (1 if v == "X" else 0) for v in stab
                ])
            elif "Z" in stab:
                Z.append([
                    (1 if v == "Z" else 0) for v in stab
                ])
        L_x, L_z = np.asarray(L_x), np.asarray(L_z)
        H_x, H_z = np.asarray(H_x), np.asarray(H_z)
        if len(L_x) == 0:
            L_x = L_z.copy()
        if len(L_z) == 0:
            L_z = L_x.copy()
        is_self_dual = (np.all(L_x == L_z) and np.all(H_x == H_z)).tolist()
        return cls(n, k, d, H_x, H_z, L_x, L_z, is_self_dual=is_self_dual)




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


class Circuit(abc.ABC):
    ket_zero: list[int]
    ket_plus: list[int]
    cnots: list[tuple[int, int]]
    bra_zero: list[int]
    bra_plus: list[int]

    def to_stim(self, noise_model: NoiseModel | None = None, *args, _layer_cnots=True):

        if noise_model is None:
            noise_model = NoiseModel()
        circ = stim.Circuit()

        if noise_model.p_init > 0:
            for i in self.ket_zero + self.ket_plus:
                circ.append("X_ERROR", i, noise_model.p_init)
        for i in self.ket_plus:
            circ.append("H", i)
            if noise_model.p_1 > 0:
                circ.append("DEPOLARIZE1", i, noise_model.p_1)
        circ.append("TICK")

        all_qubits = set(range(max(map(max, self.cnots)) + 1))
        if _layer_cnots:
            cnot_layers = _layer_cnot_circuit(self.cnots)
        else:
            cnot_layers = [self.cnots]

        for cnot_layer in cnot_layers:
            for c, n in cnot_layer:
                circ.append("CNOT", [c, n])
                if noise_model.p_2 > 0:
                    circ.append("DEPOLARIZE2", [c, n], noise_model.p_2)
            if noise_model.p_mem > 0:
                circ.append("Z_ERROR", all_qubits, noise_model.p_mem)
            circ.append("TICK")

        for i in self.bra_plus:
            circ.append("H", i)
            if noise_model.p_1 > 0:
                circ.append("DEPOLARIZE1", i, noise_model.p_1)
        for i in sorted(self.bra_zero + self.bra_plus):
            if noise_model.p_meas > 0:
                circ.append("MR", i, noise_model.p_meas)
            else:
                circ.append("MR", i)

        return circ

    def to_pyzx(self):
        n_data = min(min(self.ket_zero), min(self.bra_zero))
        circ = zx.Circuit(n_data)

        for i in self.ket_zero:
            circ.add_gate("InitAncilla", label=i, basis="Z")
        for i in self.ket_plus:
            circ.add_gate("InitAncilla", label=i, basis="X")

        for c, n in self.cnots:
            circ.add_gate("CNOT", c, n)

        for i in self.bra_zero:
            circ.add_gate("PostSelect", label=i, basis="Z")
        for i in self.bra_plus:
            circ.add_gate("PostSelect", label=i, basis="X")

        return circ.to_graph()

    def to_dict(self) -> dict:
        return {
            "ket_0": self.ket_zero.copy(),
            "ket_+": self.ket_plus.copy(),
            "cnots": [[c, n] for c, n in self.cnots],
            "bra_0": self.bra_zero.copy(),
            "bra_+": self.bra_plus.copy(),
        }

    @property
    def cnot_depth(self):
        return len(_layer_cnot_circuit(self.cnots))


@dataclass
class StatePreparationCircuit(Circuit):
    basis: Basis
    ket_zero: list[int]
    ket_plus: list[int]
    cnots: list[tuple[int, int]]
    bra_zero: list[int]
    bra_plus: list[int]

    @classmethod
    def from_dict(cls, data: dict, basis: Basis):
        return cls(
            basis=basis,
            ket_zero=data["ket_0"],
            ket_plus=data["ket_+"],
            cnots=[(c, n) for [c, n] in data["cnots"]],
            bra_zero=data["bra_0"],
            bra_plus=data["bra_+"]
        )

    @classmethod
    def from_stim(cls, circ: stim.Circuit, basis: Basis):
        ket_plus = []
        cnots = []
        in_cnot_layer = np.zeros(circ.num_qubits, dtype=bool)
        bra = []
        bra_plus = []

        for op in explode_circuit(circ):
            ts = [t.value for t in op.targets_copy()]
            if op.name == "H":
                [t] = ts
                if in_cnot_layer[t]:
                    bra_plus.append(t)
                else:
                    ket_plus.append(t)
            if op.name in ("CNOT", "CX"):
                cnots.append(ts)
                in_cnot_layer[ts[0]] = True
                in_cnot_layer[ts[1]] = True
            if op.name in ("M", "MR"):
                bra.append(ts[0])

        ket_plus.sort()
        bra_plus.sort()
        ket_zero = [q for q in range(circ.num_qubits) if q not in ket_plus]
        bra_zero = [q for q in bra if q not in bra_plus]

        return cls(
            basis,
            ket_zero,
            ket_plus,
            cnots,
            bra_zero,
            bra_plus
        )

    @classmethod
    def from_qasm(cls, qasm_string: str, basis: Basis):
        cirq_circuit = circuit_from_qasm(qasm_string)
        stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit)
        return cls.from_stim(stim_circuit, basis)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["basis"] = self.basis
        return data

    @property
    def n(self):
        return len(self.ket_zero) + len(self.ket_plus) - len(self.bra_zero) - len(self.bra_plus)

    def dual(self) -> StatePreparationCircuit:
        return StatePreparationCircuit(
            basis=Basis.dual(self.basis),
            ket_zero=self.ket_plus,
            ket_plus=self.ket_zero,
            cnots=[(n, c) for c, n in self.cnots],
            bra_zero=self.bra_plus,
            bra_plus=self.bra_zero
        )

    def to_stim(self, noise_model=NoiseModel(), *args, _layer_cnots=True):
        circ = super().to_stim(noise_model, _layer_cnots=_layer_cnots)
        for i in range(len(self.bra_zero) + len(self.bra_plus)):
            circ.append("DETECTOR", stim.target_rec(-i - 1))
        return circ

    def non_ft_version(self):
        qubits_to_keep = (set(self.ket_plus) | set(self.ket_zero)) - (set(self.bra_plus) | set(self.bra_zero))
        return StatePreparationCircuit(
            basis=self.basis,
            ket_zero=[q for q in self.ket_zero if q in qubits_to_keep],
            ket_plus=[q for q in self.ket_plus if q in qubits_to_keep],
            cnots=[(c, n) for c, n in self.cnots if c in qubits_to_keep and n in qubits_to_keep],
            bra_zero=[],
            bra_plus=[]
        )




@dataclass
class SyndromeMeasurementCircuit(Circuit):
    ket_zero: list[int]
    ket_plus: list[int]
    cnots: list[tuple[int, int]]
    bra_zero: list[int]
    bra_plus: list[int]
    flags: list[int]

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            ket_zero=data["ket_0"],
            ket_plus=data["ket_+"],
            cnots=[(c, n) for [c, n] in data["cnots"]],
            bra_zero=data["bra_0"],
            bra_plus=data["bra_+"],
            flags=data["flags"],
        )

    def dual(self) -> SyndromeMeasurementCircuit:
        return SyndromeMeasurementCircuit(
            ket_zero=self.ket_plus.copy(),
            ket_plus=self.ket_zero.copy(),
            cnots=[(n, c) for c, n in self.cnots],
            bra_zero=self.bra_plus.copy(),
            bra_plus=self.bra_zero.copy(),
            flags=self.flags.copy(),
        )

    def to_dict(self) -> dict[str, list]:
        data = super().to_dict()
        data["flags"] = self.flags.copy()
        return data

    @classmethod
    def steane_style(cls, state_prep: StatePreparationCircuit):
        def shift_indices(ops, k):
            ret = []
            for op in ops:
                if isinstance(op, tuple):
                    ret.append((op[0] + k, op[1] + k))
                else:
                    ret.append(op + k)
            return ret

        n = state_prep.n
        cnots = shift_indices(state_prep.cnots, n)
        z_flags = shift_indices(state_prep.bra_zero, n)
        x_flags = shift_indices(state_prep.bra_plus, n)
        if state_prep.basis == Basis.X:
            cnots += [(i, i + n) for i in range(n)]
            bra_zero = [i + n for i in range(n)] + z_flags
            bra_plus = x_flags
        else:
            cnots += [(i + n, i) for i in range(n)]
            bra_zero = z_flags
            bra_plus = [i + n for i in range(n)] + x_flags

        return cls(
            ket_zero=shift_indices(state_prep.ket_zero, n),
            ket_plus=shift_indices(state_prep.ket_plus, n),
            cnots=cnots,
            bra_zero=bra_zero,
            bra_plus=bra_plus,
            flags=x_flags + z_flags,
        )

    def to_stim(self, noise_model=NoiseModel(), *args, _layer_cnots=True):
        circ = super().to_stim(noise_model, _layer_cnots=_layer_cnots)

        num_measurements = len(self.bra_zero) + len(self.bra_plus)
        for i, q in enumerate(self.bra_zero + self.bra_plus):
            if q in self.flags:
                circ.append("DETECTOR", stim.target_rec(i - num_measurements))
        return circ


@dataclass
class QECCGadgets:
    code: QECC
    ft_z_state_prep: StatePreparationCircuit | None
    ft_x_state_prep: StatePreparationCircuit | None
    non_ft_z_state_prep: StatePreparationCircuit | None
    non_ft_x_state_prep: StatePreparationCircuit | None

    @classmethod
    def from_json(cls, filename):
        with open(filename) as json_file:
            data = json.load(json_file)
        code = QECC.from_dict(data)
        ft_z_state_prep = StatePreparationCircuit.from_dict(
            data.get("fault_tolerant_zero_state_prep"), basis=Basis.Z
        )
        non_ft_z_state_prep = StatePreparationCircuit.from_dict(
            data.get("non_fault_tolerant_zero_state_prep"), basis=Basis.Z
        )
        if code.is_self_dual:
            ft_x_state_prep = ft_z_state_prep.dual()
            non_ft_x_state_prep = non_ft_z_state_prep.dual()
        else:
            ft_x_state_prep = StatePreparationCircuit.from_dict(
                data.get("fault_tolerant_plus_state_prep"), basis=Basis.X
            )
            non_ft_x_state_prep = StatePreparationCircuit.from_dict(
                data.get("non_fault_tolerant_plus_state_prep"), basis=Basis.X
            )
        return cls(
            code=code,
            ft_z_state_prep=ft_z_state_prep,
            non_ft_z_state_prep=non_ft_z_state_prep,
            ft_x_state_prep=ft_x_state_prep,
            non_ft_x_state_prep=non_ft_x_state_prep,
        )

    @classmethod
    def load_circuit_data(cls, n, k, d):
        with open(f"circuits_data/{n}_{k}_{d}.qasm") as qasm_file:
            sp = StatePreparationCircuit.from_qasm(qasm_file.read(), basis=Basis.Z)
        with open(f"circuits_data/{n}_{k}_{d}.stabs") as stabs_file:
            qecc = QECC.from_stabs(stabs_file.readlines(), n, k, d)
        return cls(
            code=qecc,
            ft_z_state_prep=sp,
            ft_x_state_prep=sp.dual(),
            non_ft_z_state_prep=sp.non_ft_version(),
            non_ft_x_state_prep=sp.dual().non_ft_version(),
        )



    @property
    def steane_z_syndrome_extraction(self) -> SyndromeMeasurementCircuit:
        return SyndromeMeasurementCircuit.steane_style(self.ft_x_state_prep)

    @property
    def steane_x_syndrome_extraction(self) -> SyndromeMeasurementCircuit:
        return SyndromeMeasurementCircuit.steane_style(self.ft_z_state_prep)

    @property
    def non_ft_steane_z_syndrome_extraction(self) -> SyndromeMeasurementCircuit:
        return SyndromeMeasurementCircuit.steane_style(self.non_ft_x_state_prep)

    @property
    def non_ft_steane_x_syndrome_extraction(self) -> SyndromeMeasurementCircuit:
        return SyndromeMeasurementCircuit.steane_style(self.non_ft_z_state_prep)


if __name__ == '__main__':
    code = QECCGadgets.load_circuit_data(32, 20, 4)
    print(code.ft_z_state_prep.non_ft_version())

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pyzx as zx
import stim
import stimcirq
from cirq.contrib.qasm_import import circuit_from_qasm

from spiderwarp.utils import _layer_cnot_circuit, explode_circuit


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

    def to_dict(self) -> dict:
        return {
            'n': self.n,
            'k': self.k,
            'd': self.d,
            'is_self_dual': self.is_self_dual,
            'H_x': self.H_x.tolist(),
            'H_z': self.H_z.tolist(),
            'L_x': self.L_x.tolist(),
            'L_z': self.L_z.tolist(),
        }

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


class AncillaBlock(abc.ABC):
    ket_zero: list[int]
    ket_plus: list[int]
    cnots: list[tuple[int, int]]
    meas_z: list[int]
    meas_x: list[int]

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

        for i in self.meas_x:
            circ.append("H", i)
            if noise_model.p_1 > 0:
                circ.append("DEPOLARIZE1", i, noise_model.p_1)
        for i in sorted(self.meas_z + self.meas_x):
            if noise_model.p_meas > 0:
                circ.append("MR", i, noise_model.p_meas)
            else:
                circ.append("MR", i)

        return circ

    def to_pyzx(self):
        n_data = min(min(self.ket_zero), min(self.meas_z))
        circ = zx.Circuit(n_data)

        for i in self.ket_zero:
            circ.add_gate("InitAncilla", label=i, basis="Z")
        for i in self.ket_plus:
            circ.add_gate("InitAncilla", label=i, basis="X")

        for c, n in self.cnots:
            circ.add_gate("CNOT", c, n)

        for i in self.meas_z:
            circ.add_gate("PostSelect", label=i, basis="Z")
        for i in self.meas_x:
            circ.add_gate("PostSelect", label=i, basis="X")

        return circ.to_graph()

    def to_dict(self) -> dict:
        return {
            "ket_0": self.ket_zero.copy(),
            "ket_+": self.ket_plus.copy(),
            "cnots": [[c, n] for c, n in self.cnots],
            "bra_0": self.meas_z.copy(),
            "bra_+": self.meas_x.copy(),
        }

    def cnot_depth(self):
        return len(_layer_cnot_circuit(self.cnots))


@dataclass
class StatePrep(AncillaBlock):
    logical_state: list[Literal["0"] | Literal["+"]]
    ket_zero: list[int]
    ket_plus: list[int]
    cnots: list[tuple[int, int]]
    meas_z: list[int]
    meas_x: list[int]

    @classmethod
    def from_dict(cls, data: dict, logical_state: list[Literal["0"] | Literal["+"]]):
        return cls(
            logical_state=logical_state,
            ket_zero=data["ket_0"],
            ket_plus=data["ket_+"],
            cnots=[(c, n) for [c, n] in data["cnots"]],
            meas_z=data["meas_z"],
            meas_x=data["meas_x"]
        )

    @classmethod
    def from_stim(cls, circ: stim.Circuit, logical_state: list[Literal["0"] | Literal["+"]]):
        ket_plus = []
        cnots = []
        in_cnot_layer = np.zeros(circ.num_qubits, dtype=bool)
        bra = []
        meas_x = []

        for op in explode_circuit(circ):
            ts = [t.value for t in op.targets_copy()]
            if op.name == "H":
                [t] = ts
                if in_cnot_layer[t]:
                    meas_x.append(t)
                else:
                    ket_plus.append(t)
            if op.name in ("CNOT", "CX"):
                cnots.append(ts)
                in_cnot_layer[ts[0]] = True
                in_cnot_layer[ts[1]] = True
            if op.name in ("M", "MR"):
                bra.append(ts[0])

        ket_plus.sort()
        meas_x.sort()
        ket_zero = [q for q in range(circ.num_qubits) if q not in ket_plus]
        meas_z = [q for q in bra if q not in meas_x]

        return cls(
            logical_state,
            ket_zero,
            ket_plus,
            cnots,
            meas_z,
            meas_x
        )

    @classmethod
    def from_qasm(cls, qasm_string: str, logical_state: list[Literal["0"] | Literal["+"]]):
        cirq_circuit = circuit_from_qasm(qasm_string)
        stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit)
        return cls.from_stim(stim_circuit, logical_state)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["logical_state"] = self.logical_state
        return data

    @property
    def n(self):
        return len(self.ket_zero) + len(self.ket_plus) - len(self.meas_z) - len(self.meas_x)

    def dual(self) -> StatePrep:
        return StatePrep(
            logical_state=Basis.dual(self.logical_state),
            ket_zero=self.ket_plus,
            ket_plus=self.ket_zero,
            cnots=[(n, c) for c, n in self.cnots],
            meas_z=self.meas_x,
            meas_x=self.meas_z
        )

    def to_stim(self, noise_model=NoiseModel(), *args, _layer_cnots=True):
        circ = super().to_stim(noise_model, _layer_cnots=_layer_cnots)
        for i in range(len(self.meas_z) + len(self.meas_x)):
            circ.append("DETECTOR", stim.target_rec(-i - 1))
        return circ

    def non_ft_version(self):
        qubits_to_keep = (set(self.ket_plus) | set(self.ket_zero)) - (set(self.meas_x) | set(self.meas_z))
        return StatePrep(
            logical_state=self.logical_state,
            ket_zero=[q for q in self.ket_zero if q in qubits_to_keep],
            ket_plus=[q for q in self.ket_plus if q in qubits_to_keep],
            cnots=[(c, n) for c, n in self.cnots if c in qubits_to_keep and n in qubits_to_keep],
            meas_z=[],
            meas_x=[]
        )


@dataclass
class SyndromeGadget(AncillaBlock):
    ket_zero: list[int]
    ket_plus: list[int]
    cnots: list[tuple[int, int]]
    meas_z: list[int]
    meas_x: list[int]
    flags: list[int]

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            ket_zero=data["ket_0"],
            ket_plus=data["ket_+"],
            cnots=[(c, n) for [c, n] in data["cnots"]],
            meas_z=data["bra_0"],
            meas_x=data["bra_+"],
            flags=data["flags"],
        )

    def dual(self) -> SyndromeGadget:
        return SyndromeGadget(
            ket_zero=self.ket_plus.copy(),
            ket_plus=self.ket_zero.copy(),
            cnots=[(n, c) for c, n in self.cnots],
            meas_z=self.meas_x.copy(),
            meas_x=self.meas_z.copy(),
            flags=self.flags.copy(),
        )

    def to_dict(self) -> dict[str, list]:
        data = super().to_dict()
        data["flags"] = self.flags.copy()
        return data

    @classmethod
    def steane_style(cls, state_prep: StatePrep):
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
        z_flags = shift_indices(state_prep.meas_z, n)
        x_flags = shift_indices(state_prep.meas_x, n)
        if state_prep.logical_state == Basis.X:
            cnots += [(i, i + n) for i in range(n)]
            meas_z = [i + n for i in range(n)] + z_flags
            meas_x = x_flags
        else:
            cnots += [(i + n, i) for i in range(n)]
            meas_z = z_flags
            meas_x = [i + n for i in range(n)] + x_flags

        return cls(
            ket_zero=shift_indices(state_prep.ket_zero, n),
            ket_plus=shift_indices(state_prep.ket_plus, n),
            cnots=cnots,
            meas_z=meas_z,
            meas_x=meas_x,
            flags=x_flags + z_flags,
        )

    def to_stim(self, noise_model=NoiseModel(), *args, _layer_cnots=True):
        circ = super().to_stim(noise_model, _layer_cnots=_layer_cnots)

        num_measurements = len(self.meas_z) + len(self.meas_x)
        for i, q in enumerate(self.meas_z + self.meas_x):
            if q in self.flags:
                circ.append("DETECTOR", stim.target_rec(i - num_measurements))
        return circ


@dataclass
class GadgetManager:
    code: QECC
    ft_z_state_prep: StatePrep | None
    ft_x_state_prep: StatePrep | None
    non_ft_z_state_prep: StatePrep | None
    non_ft_x_state_prep: StatePrep | None

    @classmethod
    def from_json(cls, filename):
        with open(filename) as json_file:
            data = json.load(json_file)
        code = QECC.from_dict(data)
        circuit_data = data.get("fault_tolerant_zero_state_prep")
        ft_z_state_prep = circuit_data and StatePrep.from_dict(
            circuit_data, logical_state=["0"]
        )
        circuit_data = data.get("non_fault_tolerant_zero_state_prep")
        non_ft_z_state_prep = circuit_data and StatePrep.from_dict(
            circuit_data, logical_state=["0"]
        )
        if code.is_self_dual:
            ft_x_state_prep = ft_z_state_prep.dual()
            non_ft_x_state_prep = non_ft_z_state_prep.dual()
        else:
            circuit_data = data.get("fault_tolerant_plus_state_prep")
            ft_x_state_prep = circuit_data and StatePrep.from_dict(
                circuit_data, logical_state=["+"]
            )
            circuit_data = data.get("non_fault_tolerant_plus_state_prep")
            non_ft_x_state_prep = circuit_data and StatePrep.from_dict(
                circuit_data, logical_state=["+"]
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
            sp = StatePrep.from_qasm(qasm_file.read(), basis=Basis.Z)
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
    def steane_z_syndrome_extraction(self) -> SyndromeGadget:
        return SyndromeGadget.steane_style(self.ft_x_state_prep)

    @property
    def steane_x_syndrome_extraction(self) -> SyndromeGadget:
        return SyndromeGadget.steane_style(self.ft_z_state_prep)

    @property
    def non_ft_steane_z_syndrome_extraction(self) -> SyndromeGadget:
        return SyndromeGadget.steane_style(self.non_ft_x_state_prep)

    @property
    def non_ft_steane_x_syndrome_extraction(self) -> SyndromeGadget:
        return SyndromeGadget.steane_style(self.non_ft_z_state_prep)


if __name__ == '__main__':
    code = GadgetManager.load_circuit_data(32, 20, 4)
    print(code.ft_z_state_prep.non_ft_version())

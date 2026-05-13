from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pyzx as zx
import stim
import stimcirq
from cirq.contrib.qasm_import import circuit_from_qasm
from mqt.qecc.circuit_synthesis import LutDecoder

from spiderwarp.utils import _layer_cnot_circuit, explode_circuit, get_project_root


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
        return cls(p_1=1e-5, p_2=1e-4, p_init=2e-4, p_meas=2e-4, p_mem=5e-5)


@dataclass
class CSSCode:
    n: int
    k: int
    d: int
    H_x: np.ndarray
    H_z: np.ndarray
    L_x: np.ndarray
    L_z: np.ndarray
    is_self_dual: bool = False
    name: str | None = None

    @classmethod
    def load_code(cls, directory: str, code_name: str) -> CSSCode:
        qecc_dir = {
            "mqt": "MQT",
            "fao": "FAO",
            "MISC": "misc",
        }.get(directory, directory)
        root = get_project_root()
        codes = {
            "steane": "7_1_3",
        }
        filename = codes.get(code_name, code_name) + ".json"
        return cls.load_json(root.joinpath("assets", "qeccs", qecc_dir, filename))

    @classmethod
    def load_json(cls, file: str | Path) -> CSSCode:
        with open(file, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict):
        DATA_TYPE = np.uint8
        name = data.get("name")
        n = data['n']
        k = data['k']
        d = data['d']
        is_self_dual = data['is_self_dual'] == 1
        if "stabs" in data:
            H_x, H_z, L_x, L_z = cls.stabs_to_matrices(data, k, n)
        elif is_self_dual:
            H = data.get('H_x', data.get('H_z'))
            L = data.get('L_x', data.get('L_z'))
            H_x = H_z = np.asarray(H, dtype=DATA_TYPE)
            L_x = L_z = np.asarray(L, dtype=DATA_TYPE)
        else:
            H_x = np.asarray(data['H_x'], dtype=DATA_TYPE)
            H_z = np.asarray(data['H_z'], dtype=DATA_TYPE)
            L_x = np.asarray(data['L_x'], dtype=DATA_TYPE)
            L_z = np.asarray(data['L_z'], dtype=DATA_TYPE)
        return cls(n, k, d, H_x, H_z, L_x, L_z, is_self_dual=is_self_dual, name=name)

    @staticmethod
    def stabs_to_matrices(data: dict, k, n) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H_x, H_z = [], []
        L_x, L_z = [], []
        for i, stab in enumerate(data["stabs"].split("\n")):
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
        return H_x, H_z, L_x, L_z

    def to_dict(self) -> dict:
        ret = {
            'n': self.n,
            'k': self.k,
            'd': self.d,
            'is_self_dual': self.is_self_dual,
            'H_x': self.H_x.tolist(),
            'H_z': self.H_z.tolist(),
            'L_x': self.L_x.tolist(),
            'L_z': self.L_z.tolist(),
        }
        if self.name:
            ret["name"] = self.name
        return ret


if __name__ == '__main__':
    code = CSSCode.load_code("misc", "32_20_4")

    from mqt.qecc import CSSCode as MQTCSSCode

    print(code)

    ccode = MQTCSSCode(Hx = code.H_x, Hz = code.H_z, distance=code.d, n=code.n)
    print(ccode.Lx)
    print(ccode.Lz)
    print(ccode.Hx)
    dec = LutDecoder(ccode)
    print(dec)
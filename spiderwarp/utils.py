import itertools
import os
from pathlib import Path
from typing import Iterable

import stim


def _sorted_pair(v1, v2):
    return (v1, v2) if v1 < v2 else (v2, v1)


def flatten(ls: Iterable[Iterable]) -> list:
    return list(itertools.chain(*ls))


def get_project_root() -> Path:
    return Path(__file__).parent


def load_state_prep_circuit(directory: str, name: str) -> stim.Circuit:
    from spiderwarp.stim_utils import qasm_str_to_stim_circuit

    root = get_project_root()
    directory = root.joinpath("assets", "circuits", "FTStatePrep", directory)
    if (file := directory.joinpath(f"{name}.stim")).exists():
        circ = stim.Circuit(file.read_text())
    else:
        file = directory.joinpath(f"{name}.qasm")
        circ = qasm_str_to_stim_circuit(file.read_text())
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
    [c1, c2, c3, c4] = [load_state_prep_circuit("SteanePerm", f"{name}/{f[:-5]}") for f in files]

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

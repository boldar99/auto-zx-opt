import json
import pickle
import sys

import stim
import pprint

from path_cover_opt import CoveredZXGraph
from qecc import QECCGadgets, _layer_cnot_circuit
from pathlib import Path

from verify_fault_tolerance import compute_modified_lookup_table, list_to_str_stabs, build_css_syndrome_table, \
    explode_circuit

cwd = Path.cwd()


def init_circuits_folder():
    Path(f"{cwd}/simplified_circuits").mkdir(parents=True, exist_ok=True)


def depth_minimal_circuit(cov_graph):
    circuit_data = []
    for cv in cov_graph.min_ancilla_boundary_bends():
        data = {
            "circuit": cv.to_syndrome_measurement_circuit(),
            "H_indices": cv.matrix_transformation_indices(),
            "measurement_indices": cv.measurement_qubit_indices(),
            "flag_indices": cv.flag_qubit_indices(),
        }
        circuit_data.append(data)

    return min(circuit_data, key=lambda d: d["circuit"].cnot_depth)


def make_json_serializable(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(k[0], tuple):
            new_k = "+".join("".join(map(str, e)) for e in k)
        else:
            new_k = "".join(map(str, k))
        new_data[new_k] = v
    return new_data


def generate_simplification(qecc_gadgets):
    diagram = qecc_gadgets.non_ft_steane_z_syndrome_extraction.to_pyzx()
    cov_graph = CoveredZXGraph.from_zx_diagram(diagram)
    cov_graph.basic_FE_rewrites()

    depth_best_circuit_data = depth_minimal_circuit(cov_graph)
    print(depth_best_circuit_data["circuit"].to_stim())

    stabs = list_to_str_stabs(qecc_gadgets.code.H_z)
    depth_best_circuit_data["lookup_table"] = build_css_syndrome_table(stabs, qecc_gadgets.code.d)
    depth_best_circuit_data["modified_lookup_table"] = compute_modified_lookup_table(
        depth_best_circuit_data["circuit"].to_stim(),
        qecc_gadgets.code.H_z,
        qecc_gadgets.code.L_x,
        depth_best_circuit_data["lookup_table"],
        depth_best_circuit_data["flag_indices"],
        "X",
        qecc_gadgets.code.d,
        verbose=False,
    )
    depth_best_circuit_data["lookup_table"] = make_json_serializable(depth_best_circuit_data["lookup_table"])
    depth_best_circuit_data["modified_lookup_table"] = make_json_serializable(depth_best_circuit_data["modified_lookup_table"])
    return depth_best_circuit_data


def save_optimised_se(data, code):
    to_save = data.copy()

    to_save["circuit"] = to_save["circuit"].to_dict()
    to_save["lookup_table"] = make_json_serializable(to_save["lookup_table"])
    to_save["modified_lookup_table"] = make_json_serializable(to_save["modified_lookup_table"])
    pprint(to_save, width=120)

    with open(f"simplified_circuits/{code}.json", "w") as f:
        json.dump(to_save, f, indent=2)



if __name__ == "__main__":
    from pprint import pprint
    init_circuits_folder()
    qecc = "15_7_3"
    # qecc_gadgets = QECCGadgets.from_json(f"circuits/{qecc}.json")
    qecc_gadgets = QECCGadgets.load_circuit_data(32, 20, 4)
    simp_data = generate_simplification(qecc_gadgets)
    save_optimised_se(simp_data, qecc)
    print(simp_data["circuit"].to_stim(_layer_cnots=False))





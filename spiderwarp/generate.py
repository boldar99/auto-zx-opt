import dataclasses
import json
from pathlib import Path

import stim
from matplotlib import pyplot as plt

from spiderwarp.path_cover_opt import CoveredZXGraph
from spiderwarp.csscode import GadgetManager, SyndromeGadget, CSSCode, StatePrep, Basis
from spiderwarp.layer_circuit import _expand_stim_operation_list, _layer_circuit_ops, apply_qubit_reuse, \
    layered_ops_to_noisy_stim_circuit

from spiderwarp.qubit_reuse import build_circuit_dag, visualize_circuit_dag, inject_aggressive_reuse, \
    apply_logical_qubit_merge_and_compress, dag_to_circuit
from spiderwarp.utils import get_project_root, steane_se_from_stim_state_prep, stim_to_pyzx
from spiderwarp.verify_fault_tolerance import compute_modified_lookup_table, list_to_str_stabs, build_css_syndrome_table

cwd = Path.cwd()


def init_circuits_folder():
    Path(f"{cwd}/simplified_circuits").mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass
class SECircuitData:
    circuit: SyndromeGadget
    H_indices: list[int]
    measurement_indices: list[int]
    flag_indices: list[int]

    def cnot_depth(self):
        return self.circuit.cnot_depth()

    def dual(self) -> "SECircuitData":
        return SECircuitData(
            circuit=self.circuit.dual(),
            H_indices=self.H_indices.copy(),
            measurement_indices=self.measurement_indices.copy(),
            flag_indices=self.flag_indices.copy(),
        )

    def to_dict(self):
        return {
            "circuit": self.circuit.to_dict(),
            "H_indices": self.H_indices,
            "measurement_indices": self.measurement_indices,
            "flag_indices": self.flag_indices,
        }

    @classmethod
    def from_dict(cls, json_data):
        return cls(
            circuit=SyndromeGadget.from_dict(json_data["circuit"]),
            H_indices=json_data["H_indices"],
            measurement_indices=json_data["measurement_indices"],
            flag_indices=json_data["flag_indices"],
        )


@dataclasses.dataclass
class OptimisedSteaneData:
    code: CSSCode
    optimized_ft_z_se: SECircuitData
    optimized_ft_x_se: SECircuitData
    optimized_non_ft_z_se: SECircuitData
    optimized_non_ft_x_se: SECircuitData
    standard_lookup_table: dict[tuple[int, ...], list[int]]
    modified_lookup_table: dict[tuple[int, ...], list[int]]

    @classmethod
    def from_dict(cls, json_data):
        return cls(
            code=CSSCode.from_dict(json_data["code"]),
            optimized_ft_z_se=SECircuitData.from_dict(json_data["optimized_ft_z_se"]),
            optimized_ft_x_se=SECircuitData.from_dict(json_data["optimized_ft_x_se"]),
            optimized_non_ft_z_se=SECircuitData.from_dict(json_data["optimized_non_ft_z_se"]),
            optimized_non_ft_x_se=SECircuitData.from_dict(json_data["optimized_non_ft_x_se"]),
            standard_lookup_table=json_data["standard_lookup_table"],
            modified_lookup_table=json_data["modified_lookup_table"],
        )

    @classmethod
    def load_json(cls, filename):
        with open(filename) as f:
            json_data = json.load(f)
        return cls.from_dict(json_data)

    def to_dict(self):
        return {
            "code": self.code.to_dict(),
            "optimized_ft_z_se": self.optimized_ft_z_se.to_dict(),
            "optimized_ft_x_se": self.optimized_ft_x_se.to_dict(),
            "optimized_non_ft_z_se": self.optimized_non_ft_z_se.to_dict(),
            "optimized_non_ft_x_se": self.optimized_non_ft_x_se.to_dict(),
            "standard_lookup_table": {"".join(map(str, k)): v for k, v in self.standard_lookup_table.items()},
            "modified_lookup_table": {"".join(map(str, k)): v for k, v in self.modified_lookup_table.items()},
        }

    def save_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


def depth_minimal_circuit(cov_graph) -> tuple[CoveredZXGraph, SECircuitData]:
    circuit_data: list[SECircuitData] = []
    for cv in cov_graph.min_ancilla_boundary_bends():
        # cv.visualize()
        data = SECircuitData(
            circuit=cv.to_syndrome_measurement_circuit(),
            H_indices=cv.matrix_transformation_indices(),
            measurement_indices=cv.measurement_qubit_indices(),
            flag_indices=cv.flag_qubit_indices(),
        )
        circuit_data.insert(0, (cv, data))

    return min(circuit_data, key=lambda p: p[1].cnot_depth())


def make_json_serializable(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(k[0], tuple):
            new_k = "+".join("".join(map(str, e)) for e in k)
        else:
            new_k = "".join(map(str, k))
        new_data[new_k] = v
    return new_data


def generate_simplification(qecc_gadgets: GadgetManager, verbose=False) -> OptimisedSteaneData:
    ft_z_se = qecc_gadgets.steane_z_syndrome_extraction.to_pyzx()
    ft_z_se_cov_graph = CoveredZXGraph.from_zx_diagram(ft_z_se)
    ft_z_se_cov_graph.basic_FE_rewrites()
    optimized_ft_z_se = depth_minimal_circuit(ft_z_se_cov_graph)

    non_ft_z_se = qecc_gadgets.steane_z_syndrome_extraction.to_pyzx()
    non_ft_z_se_cov_graph = CoveredZXGraph.from_zx_diagram(non_ft_z_se)
    non_ft_z_se_cov_graph.basic_FE_rewrites()
    optimized_non_ft_z_se = depth_minimal_circuit(non_ft_z_se_cov_graph)

    if qecc_gadgets.code.is_self_dual:
        optimized_ft_x_se = optimized_ft_z_se.dual()
        optimized_non_ft_x_se = optimized_non_ft_z_se.dual()
    else:
        ft_x_se = qecc_gadgets.steane_x_syndrome_extraction.to_pyzx()
        ft_x_se_cov_graph = CoveredZXGraph.from_zx_diagram(ft_x_se)
        ft_x_se_cov_graph.basic_FE_rewrites()
        optimized_ft_x_se = depth_minimal_circuit(ft_x_se_cov_graph)

        non_ft_x_se = qecc_gadgets.steane_x_syndrome_extraction.to_pyzx()
        non_ft_x_se_cov_graph = CoveredZXGraph.from_zx_diagram(non_ft_x_se)
        non_ft_x_se_cov_graph.basic_FE_rewrites()
        optimized_non_ft_x_se = depth_minimal_circuit(non_ft_x_se_cov_graph)

    stabs = list_to_str_stabs(qecc_gadgets.code.H_z)
    # TODO: lookup table should deal with X and Z errors separately
    lookup_table = build_css_syndrome_table(stabs, qecc_gadgets.code.d)
    modified_lookup_table = compute_modified_lookup_table(
        optimized_ft_z_se.circuit.to_stim(),
        qecc_gadgets.code.H_z,
        qecc_gadgets.code.L_x,
        lookup_table,
        optimized_ft_z_se.flag_indices,
        "X",
        qecc_gadgets.code.d,
        verbose=False,
    )

    return OptimisedSteaneData(
        optimized_ft_z_se=optimized_ft_z_se,
        optimized_ft_x_se=optimized_ft_x_se,
        optimized_non_ft_z_se=optimized_non_ft_z_se,
        optimized_non_ft_x_se=optimized_non_ft_x_se,
        standard_lookup_table=lookup_table,
        modified_lookup_table=modified_lookup_table,
        code=qecc_gadgets.code,
    )


def generate_code(qecc):
    init_circuits_folder()
    qecc_gadgets = GadgetManager.from_json(f"circuits/{qecc}.json")
    simp_data = generate_simplification(qecc_gadgets)
    simp_data.save_json(f"simplified_circuits/{qecc}.json")


def get_circuit(qecc, folder):
    root = get_project_root()
    file = root.joinpath("assets", "circuits", folder, f"{qecc}.stim")
    root.joinpath()
    with open(file) as f:
        raw_str = f.read()
        return stim.Circuit(raw_str)


if __name__ == "__main__":
    circ, n = get_circuit("zero_32_20_4", "miscellaneous"), 32
    # circ, n = get_circuit("cc_4_8_8_d5/zero_ft_heuristic_opt", "SAT-FTStatePrep"), 17
    # circ, n = get_circuit("hamming/zero_ft_opt_opt", "SAT-FTStatePrep"), 15
    if "R" not in circ[0].name:
        circ.insert(0, stim.CircuitInstruction("R", range(circ.num_qubits)))
    se = steane_se_from_stim_state_prep(circ, se_basis="Z", n=n)
    print(se)
    print("\n\n\n" + "=" * 80 + "\n\n\n")
    ft_z_se = stim_to_pyzx(se, n)
    ft_z_se_cov_graph = CoveredZXGraph.from_zx_diagram(ft_z_se)
    ft_z_se_cov_graph.basic_FE_rewrites()
    ft_z_se_cov_graph.visualize()
    cv, optimized_ft_z_se = depth_minimal_circuit(ft_z_se_cov_graph)
    cv.visualize()
    dag = build_circuit_dag(cv._find_total_ordering())
    visualize_circuit_dag(dag, figsize=(30, 20))
    mod_dag, logical_to_physical, total_hw = inject_aggressive_reuse(dag, n_data=n)
    mod_dag = apply_logical_qubit_merge_and_compress(mod_dag, n_data=n)
    visualize_circuit_dag(mod_dag, figsize=(30, 20))
    operations = dag_to_circuit(mod_dag)
    num_qubits = max(max(ts) for _, ts in operations) + 1
    layered_ops = _layer_circuit_ops(operations, num_qubits)
    # final_ops, num_sim_qubits = apply_qubit_reuse(layered_ops, n)
    p = 0
    noisy_circ = layered_ops_to_noisy_stim_circuit(layered_ops, num_qubits, p, p, p, p, p / 100)
    print(noisy_circ)

    # cv.extract_circuit()
    # stim_circ = optimized_ft_z_se.circuit.to_stim(_layer_cnots=False)



    # operations = [(op, targets) for (op, targets, params) in stim_circ.flattened_operations() if op != "DETECTOR"]
    # detectors = [(op, [stim.target_rec(targets[0][1])]) for (op, targets, params) in stim_circ.flattened_operations() if op == "DETECTOR"]
    # operations = _expand_stim_operation_list(operations)
    # layered_ops = _layer_circuit_ops(operations, stim_circ.num_qubits)
    # print(layered_ops)
    # final_ops, num_sim_qubits = apply_qubit_reuse(layered_ops, n)
    # p = 0
    # noisy_circ = layered_ops_to_noisy_stim_circuit(final_ops + [detectors], circ.num_qubits, p, p, p, p, p / 100)
    # print(noisy_circ)

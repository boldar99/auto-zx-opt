import dataclasses
import json
from pathlib import Path

from spiderwarp.path_cover_opt import CoveredZXGraph
from spiderwarp.qecc import QECCGadgets, SyndromeMeasurementCircuit, QECC
from spiderwarp.verify_fault_tolerance import compute_modified_lookup_table, list_to_str_stabs, build_css_syndrome_table

cwd = Path.cwd()


def init_circuits_folder():
    Path(f"{cwd}/simplified_circuits").mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass
class SECircuitData:
    circuit: SyndromeMeasurementCircuit
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
            circuit=SyndromeMeasurementCircuit.from_dict(json_data["circuit"]),
            H_indices=json_data["H_indices"],
            measurement_indices=json_data["measurement_indices"],
            flag_indices=json_data["flag_indices"],
        )


@dataclasses.dataclass
class OptimisedSteaneData:
    code: QECC
    optimized_ft_z_se: SECircuitData
    optimized_ft_x_se: SECircuitData
    optimized_non_ft_z_se: SECircuitData
    optimized_non_ft_x_se: SECircuitData
    standard_lookup_table: dict[tuple[int, ...], list[int]]
    modified_lookup_table: dict[tuple[int, ...], list[int]]

    @classmethod
    def from_dict(cls, json_data):
        return cls(
            code=QECC.from_dict(json_data["code"]),
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
            "code": self.code.to_json(),
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


def depth_minimal_circuit(cov_graph) -> SECircuitData:
    circuit_data: list[SECircuitData] = []
    for cv in cov_graph.min_ancilla_boundary_bends():
        data = SECircuitData(
            circuit=cv.to_syndrome_measurement_circuit(),
            H_indices=cv.matrix_transformation_indices(),
            measurement_indices=cv.measurement_qubit_indices(),
            flag_indices=cv.flag_qubit_indices(),
        )
        circuit_data.append(data)

    return min(circuit_data, key=SECircuitData.cnot_depth)


def make_json_serializable(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(k[0], tuple):
            new_k = "+".join("".join(map(str, e)) for e in k)
        else:
            new_k = "".join(map(str, k))
        new_data[new_k] = v
    return new_data


def generate_simplification(qecc_gadgets: QECCGadgets, verbose=False) -> OptimisedSteaneData:
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
    qecc_gadgets = QECCGadgets.from_json(f"circuits/{qecc}.json")
    simp_data = generate_simplification(qecc_gadgets)
    simp_data.save_json(f"simplified_circuits/{qecc}.json")


if __name__ == "__main__":
    generate_code("15_7_3")

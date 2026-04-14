from spiderwarp.csscode import *
from spiderwarp.generate import depth_minimal_circuit
from spiderwarp.path_cover_opt import *
import os


def mqt_steane_opt(code_name):
    code = CSSCode.load_code(code_name, "MQT")
    dir = get_project_root().joinpath("assets", "circuits", "Steane-FTStatePrep", code.name)
    files = os.listdir(dir)
    sorted_files = sorted(files)
    qasms = []
    circuits = []
    for i, f in enumerate(sorted_files):
        file = dir.joinpath(f)
        with open(file) as f:
            qasms.append(f.read())
            circuits.append(StatePrep.from_qasm(qasms[-1], state_basis=Basis.Z if i % 2 == 0 else Basis.X))

    cirq_circuit = circuit_from_qasm(qasms[0])
    c1_stim = stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit)

    c2 = SyndromeGadget.steane_style(circuits[1])
    c2 = c2.to_pyzx()
    c2_cov_graph = CoveredZXGraph.from_zx_diagram(c2)
    c2_cov_graph.basic_FE_rewrites()
    c2_opt = c2_cov_graph.best_first_boundary_bends(max_evaluations=2_000)
    print(f"Optimised C_2: {len(c2_cov_graph.paths)} -> {len(c2_opt.paths)}")

    c4 = SyndromeGadget.steane_style(circuits[3])
    c4 = c4.to_pyzx()
    c4_cov_graph = CoveredZXGraph.from_zx_diagram(c4)
    c4_cov_graph.basic_FE_rewrites()
    c4_opt = c4_cov_graph.best_first_boundary_bends(max_evaluations=2_000)
    print(f"Optimised C_4: {len(c4_cov_graph.paths)} -> {len(c4_opt.paths)}")

    circ_3 = circuits[2].to_stim(_layer_cnots=False)
    circ_4_opt = c4_opt.extract_circuit()
    cric_3_4 = circ_3 + circ_4_opt
    sp34 = StatePrep.from_stim(cric_3_4, state_basis=Basis.Z)
    c34 = SyndromeGadget.steane_style(sp34)
    c34 = c34.to_pyzx()
    c34_cov_graph = CoveredZXGraph.from_zx_diagram(c34)
    c34_cov_graph.basic_FE_rewrites()
    c34_opt = c34_cov_graph.best_first_boundary_bends(max_evaluations=2_000)
    print(f"Optimised C_3; C_4: {len(c34_cov_graph.paths)} -> {len(c34_opt.paths)}")

    circ_1_2_opt = c1_stim + c2_opt.extract_circuit()
    circ_3_4_opt = c34_opt.extract_circuit()

    return circ_1_2_opt + circ_3_4_opt


if __name__ == '__main__':
    code_name = "31_1_7"
    # code_name = "17_1_5"
    code_name = "20_2_6"
    print(mqt_steane_opt(code_name))

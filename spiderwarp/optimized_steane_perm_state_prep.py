import numpy as np

from spiderwarp.csscode import CSSCode
from spiderwarp.path_cover_opt import CoveredZXGraph
from spiderwarp.utils import load_steane_perm_circuits, steane_se_from_stim_state_prep, _layer_cnot_circuit


def mqt_steane_opt(code_name, *, optimise_c2: bool = True):
    code = CSSCode.load_code("MQT", code_name)
    circuits = load_steane_perm_circuits(code_name)

    # --- C2 Optimization ---
    se2 = steane_se_from_stim_state_prep(circuits[1], se_basis="X", n=code.n)
    if optimise_c2:
        cov_graph_c2 = CoveredZXGraph.from_stim(se2)
        cov_graph_c2.basic_FE_rewrites()
        c2_opt = cov_graph_c2.best_first_boundary_bends(max_evaluations=100)

        print(f"Optimised C_2: {circuits[1].num_qubits * 2} -> {len(c2_opt.paths)}")
        se2_opt_circ, se2_mm = c2_opt.extract_circuit_with_measurement_map()
        circ_1_2_opt = circuits[0] + se2_opt_circ
    else:
        circ_1_2_opt = circuits[0] + se2
        se2_mm = {i: i for i in range(code.n)}

    # --- C4 Optimization ---
    se4 = steane_se_from_stim_state_prep(circuits[3], se_basis="X", n=code.n)
    cov_graph_c4 = CoveredZXGraph.from_stim(se4)
    cov_graph_c4.offset_measurement_ids_by(code.n)
    # cov_graph_c4.basic_FE_rewrites()
    # cov_graph_c4_opt = cov_graph_c4.mcts_boundary_bends(max_iterations=5, )
    cov_graph_c4_opt = cov_graph_c4

    print(f"Optimised C_4: {circuits[3].num_qubits * 2} -> {len(cov_graph_c4_opt.paths)}")
    cov_graph_c4_opt.visualize()
    se4_opt_circ, se4_mm = cov_graph_c4_opt.extract_circuit_with_measurement_map()

    # --- C3/C4 Combined Optimization ---
    cric_3_4 = circuits[2] + se4_opt_circ
    se34 = steane_se_from_stim_state_prep(cric_3_4, se_basis="Z", n=code.n)
    cov_graph_c34 = CoveredZXGraph.from_stim(se34)
    for v in cov_graph_c34.G.nodes():
        m_id = cov_graph_c34.G.nodes[v]["measurement_id"]
        if m_id is not None:
            cov_graph_c34.set_measurement_id(v, se4_mm.get(m_id, m_id + 2 * code.n - len(se4_mm)))
    # cov_graph_c34_opt = cov_graph_c34
    cov_graph_c34.visualize()
    cov_graph_c34.basic_FE_rewrites()
    # cov_graph_c34_opt = next(cov_graph_c34.bfs_causal_boundary_bends())
    cov_graph_c34_opt = cov_graph_c34
    cov_graph_c34_opt.visualize()

    print(
        f"Optimised C_3; C_4:"
        f"{circuits[2].num_qubits * 2 + len(cov_graph_c4_opt.paths)} -> {len(cov_graph_c34_opt.paths)}"
    )
    se34_opt_circ, se34_mm = cov_graph_c34_opt.extract_circuit_with_measurement_map()
    og_circ = circuits[0] + se2 + steane_se_from_stim_state_prep(circuits[2] + se4, se_basis="Z", n=code.n)
    measurement_mapping = se2_mm | {k + len(se2_mm): v for k, v in se34_mm.items()}

    return  og_circ, circ_1_2_opt + se34_opt_circ, measurement_mapping


if __name__ == '__main__':
    # code_name = "31_1_7"
    # code_name = "20_2_6"
    code_name = "17_1_5"
    code = CSSCode.load_code("MQT", code_name)
    og_circ, circ, M = mqt_steane_opt(code_name, optimise_c2=False)
    circuit = circ.copy()
    circuit.append("M", range(code.H_x.shape[1]))
    smplr = circuit.compile_sampler()
    samples = smplr.sample(25)
    raw_syndrome_measurements = samples[:, :-code.H_x.shape[1]]
    raw_measurements = samples[:, -code.H_x.shape[1]:]

    effective_H_x = np.kron(np.eye(3, dtype=np.uint8), code.H_x)
    effective_H_x = effective_H_x[:, tuple(M.values())].copy()

    distillation_syndromes = raw_syndrome_measurements @ effective_H_x.T % 2
    print(distillation_syndromes)
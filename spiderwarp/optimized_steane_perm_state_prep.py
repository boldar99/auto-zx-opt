import numpy as np
import numpy.typing as npt

import stim

from mqt.qecc.circuit_synthesis import CircuitLevelNoiseIdlingParallel
from mqt.qecc.circuit_synthesis.circuit_utils import measured_qubits, unmeasured_qubits, collect_circuit_layers
from mqt.qecc.circuit_synthesis.noise import NoiseModel
from mqt.qecc.circuit_synthesis.simulation import NoisyNDFTStatePrepSimulator
from mqt.qecc import CSSCode as MQTCSSCode

from spiderwarp.csscode import CSSCode
from spiderwarp.path_cover import CoveredZXGraph
from spiderwarp.path_cover_metrics import metric_num_paths, metric_depth_exact, metric_spacetime_volume_exact, \
    LexicographicCost
from spiderwarp.qubit_reuse import NoReuseStrategy
from spiderwarp.utils import load_steane_perm_circuits
from spiderwarp.stim_utils import steane_se_from_stim_state_prep


def mqt_steane_depth_opt(code_name, *, max_iterations=1000, verbose: bool = False):
    code = CSSCode.load_code("MQT", code_name)
    circuits = load_steane_perm_circuits(code_name)

    # --- C1 Optimization ---
    cov_graph_c1 = CoveredZXGraph.from_stim(circuits[0], num_data_qubits=0)
    # cov_graph_c1.basic_FE_rewrites()
    cov_graph_c1.optimize_path_extremities(max_iterations)
    se1_opt_circ, _ = cov_graph_c1.extract_circuit_with_measurement_map()
    if verbose:
        print(f"Optimised C_1: {len(collect_circuit_layers(circuits[0]))} -> {len(collect_circuit_layers(se1_opt_circ))}")

    # print(circuits[0])
    # print()
    # print(se1_opt_circ)

    # --- C2 Optimization ---
    se2 = steane_se_from_stim_state_prep(circuits[1], se_basis="X", n=code.n)
    cov_graph_c2 = CoveredZXGraph.from_stim(se2)
    # cov_graph_c2.basic_FE_rewrites()
    cov_graph_c2.optimize_path_extremities(max_iterations)
    se2_opt_circ, se2_mm = cov_graph_c2.extract_circuit_with_measurement_map()
    if verbose:
        print(f"Optimised C_2: {len(collect_circuit_layers(se2))} -> {len(collect_circuit_layers(se2_opt_circ))}")
    circ_1_2_opt = se1_opt_circ + se2_opt_circ

    # --- C4 Optimization ---
    se4 = steane_se_from_stim_state_prep(circuits[3], se_basis="X", n=code.n, )
    cov_graph_c4 = CoveredZXGraph.from_stim(se4)
    cov_graph_c4.offset_measurement_ids_by(code.n)
    # cov_graph_c4.basic_FE_rewrites()
    cov_graph_c2.optimize_path_extremities(max_iterations)
    se4_opt_circ, se4_mm = cov_graph_c2.extract_circuit_with_measurement_map()
    if verbose:
        print(f"Optimised C_4: {len(collect_circuit_layers(se4))} -> {len(collect_circuit_layers(se4_opt_circ))}")

    # --- C3/C4 Combined Optimization ---
    cric_3_4 = circuits[2] + se4_opt_circ
    se34 = steane_se_from_stim_state_prep(cric_3_4, se_basis="Z", n=code.n)
    cov_graph_c34 = CoveredZXGraph.from_stim(se34)
    for v in cov_graph_c34.G.nodes():
        m_id = cov_graph_c34.G.nodes[v]["measurement_id"]
        if m_id is not None:
            cov_graph_c34.set_measurement_id(v, se4_mm.get(m_id, m_id + 2 * code.n - len(se4_mm)))
    # cov_graph_c34.basic_FE_rewrites()
    cov_graph_c34.optimize_path_extremities()
    se34_opt_circ, se34_mm = cov_graph_c34.extract_circuit_with_measurement_map()
    if verbose:
        print(
            f"Optimised C_3; C_4:"
            f"{len(collect_circuit_layers(cric_3_4))} -> {len(collect_circuit_layers(se34_opt_circ))}"
        )
    measurement_mapping = se2_mm | {k + len(se2_mm): v for k, v in se34_mm.items()}
    offset_circ = stim.Circuit()
    for (opname, optargs, _) in se34_opt_circ.flattened_operations():
        offset_circ.append(opname, [t if t < code.n else t + code.n for t in optargs])

    og_circ = circuits[0] + se2 + steane_se_from_stim_state_prep(circuits[2] + se4, se_basis="Z", n=code.n, offset=code.n)

    return  og_circ, circ_1_2_opt + offset_circ, measurement_mapping


def mqt_steane_opt(code_name, *, cost_func=metric_num_paths, max_evaluations=1000, optimise_c2: bool = True, verbose: bool = False):
    code = CSSCode.load_code("MQT", code_name)
    circuits = load_steane_perm_circuits(code_name)

    # --- C2 Optimization ---
    se2 = steane_se_from_stim_state_prep(circuits[1], se_basis="X", n=code.n)
    if optimise_c2:
        cov_graph_c2 = CoveredZXGraph.from_stim(se2)
        cov_graph_c2.basic_FE_rewrites()
        c2_opt = cov_graph_c2.greedy_best_first_boundary_bends(cost_func=cost_func, max_evaluations=max_evaluations)

        if verbose:
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
    cov_graph_c4.basic_FE_rewrites()
    cov_graph_c4_opt = cov_graph_c4.greedy_best_first_boundary_bends(cost_func=cost_func, max_evaluations=max_evaluations)

    if verbose:
        print(f"Optimised C_4: {circuits[3].num_qubits * 2} -> {len(cov_graph_c4_opt.paths)}")
    se4_opt_circ, se4_mm = cov_graph_c4_opt.extract_circuit_with_measurement_map()

    # --- C3/C4 Combined Optimization ---
    cric_3_4 = circuits[2] + se4_opt_circ
    se34 = steane_se_from_stim_state_prep(cric_3_4, se_basis="Z", n=code.n)
    cov_graph_c34 = CoveredZXGraph.from_stim(se34)
    for v in cov_graph_c34.G.nodes():
        m_id = cov_graph_c34.G.nodes[v]["measurement_id"]
        if m_id is not None:
            cov_graph_c34.set_measurement_id(v, se4_mm.get(m_id, m_id + 2 * code.n - len(se4_mm)))
    cov_graph_c34.basic_FE_rewrites()
    cov_graph_c34_opt = cov_graph_c34.greedy_best_first_boundary_bends(cost_func=cost_func, max_evaluations=max_evaluations)

    if verbose:
        print(
            f"Optimised C_3; C_4:"
            f"{circuits[2].num_qubits * 3} -> {len(cov_graph_c34_opt.paths)}"
        )
    se34_opt_circ, se34_mm = cov_graph_c34_opt.extract_circuit_with_measurement_map()
    measurement_mapping = se2_mm | {k + len(se2_mm): v for k, v in se34_mm.items()}
    offset_circ = stim.Circuit()
    for (opname, optargs, _) in se34_opt_circ.flattened_operations():
        offset_circ.append(opname, [t if t < code.n else t + code.n for t in optargs])

    og_circ = circuits[0] + se2 + steane_se_from_stim_state_prep(circuits[2] + se4, se_basis="Z", n=code.n, offset=code.n)

    return  og_circ, circ_1_2_opt + offset_circ, measurement_mapping


class OptimisedSteaneNDFTStatePrepSimulator(NoisyNDFTStatePrepSimulator):
    """Class for simulating Steane-type noisy state preparation circuit.

    A state is checked using multiple copies of the state preparation circuit, which are connected using transversal CNOTs.
    """

    def __init__(
        self,
        circ: stim.Circuit,
        measurement_mapping: dict[int, int],
        code: MQTCSSCode,
    ) -> None:
        """Initialize the simulator."""
        matrices = [code.Hx, np.vstack((code.Hz, code.Lz)), code.Hx]
        total_rows = sum(m.shape[0] for m in matrices)
        total_cols = sum(m.shape[1] for m in matrices)
        effective_H_full = np.zeros((total_rows, total_cols), dtype=np.int8)
        r, c = 0, 0
        for m in matrices:
            rows, cols = m.shape
            effective_H_full[r:r + rows, c:c + cols] = m
            r += rows
            c += cols
        self.effective_H_full = effective_H_full
        mqs = measured_qubits(circ)
        self.measurement_order = [(mqs[i], m) for i, m in measurement_mapping.items()]

        super().__init__(circ, code, True)

    def _build_noisy_circuit(self, noise: NoiseModel) -> stim.Circuit:
        _noisy_circ = super()._build_noisy_circuit(noise)
        mqs = measured_qubits(_noisy_circ)[:len(self.measurement_order)]
        indices = []
        for q in mqs:
            for ix, (m, i) in enumerate(self.measurement_order):
                if q == m:
                    indices.append(i)
                    break
            self.measurement_order.pop(ix)
        self.effective_H = self.effective_H_full[:,indices]
        return _noisy_circ


    def _filter_runs(self, samples: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Filter samples based on measurement outcomes.

        Args:
            samples: The samples to filter.

        Returns:
            npt.NDArray[np.int8]: The filtered samples.
        """
        distillation_syndromes = samples[:,:-self.code.Hx.shape[1]] @ self.effective_H.T % 2
        flag_raised = np.any(distillation_syndromes != 0, axis=1)
        return samples[~flag_raised].astype(np.int8)


if __name__ == '__main__':
    # code_name = "17_1_5"
    # code_name = "19_1_5"
    code_name = "17_1_5"
    # code_name = "31_1_7"
    opt_c2 = False


    print(f"Code: {code_name},  {opt_c2=}")

    code = CSSCode.load_code("MQT", code_name)
    mqt_code = MQTCSSCode(Hx=code.H_x, Hz=code.H_z, distance=code.d)
    # og_circ, circ, M = mqt_steane_depth_opt(code_name, max_iterations=100, verbose=True)
    og_circ, circ, M = mqt_steane_opt(code_name, optimise_c2=opt_c2, cost_func=metric_spacetime_volume_exact(NoReuseStrategy), max_evaluations=100, verbose=True)


    p = 0.001
    p_mem_factor = 0.01
    depth = len(collect_circuit_layers(circ))
    og_depth = len(collect_circuit_layers(og_circ))
    print(f"New circuit: #Qubits: {circ.num_qubits},  Depth: {depth},  Circuit Volume: {circ.num_qubits * depth},  p_mem: p*{p_mem_factor}")
    print(f"Orig circuit: #Qubits: {og_circ.num_qubits},  Depth: {og_depth},  Circuit Volume: {og_circ.num_qubits * og_depth},  p_mem: p*{p_mem_factor}")

    sim = OptimisedSteaneNDFTStatePrepSimulator(
        circ=circ,
        code=mqt_code,
        measurement_mapping=M
    )
    noise = CircuitLevelNoiseIdlingParallel(p, 0, p * 2 / 3, p, p * p_mem_factor)
    ler, ar, num_err, num_samples = sim.logical_error_rate(noise=noise, min_errors=50)
    print(f"LER: {ler:.4e},  AR: {ar:.2%},  #Err: {num_err},  #Samples: {num_samples}")

    # Code: cc_4_8_8_d5
    # #Qubits: 68, Depth: 11, p_mem: p*0.1
    # LER: 1.4940e-07, AR: 80.75 %,  # Err: 10, #Samples: 82900000

    # 17_1_5
    # With p_mem = p / 100
    # No C2 opt: LER: 4.34063991136953e-08,  AR: 0.8375929471618616, 50,  1375900000
    # C2 opt:    LER: 3.326459538771059e-08, AR: 0.8501083104247843, 50,  1767900000
    # MQT:       LER: 8.908924288742791e-08, AR: 0.816381803898747,  50,  687400000
    # With p_mem = p / 10
    # No C2 opt: LER: 5.554656040439893e-07, AR: 0.8156263677536237, 50,  110400000
    # C2 opt:    LER: 7.89615398591587e-07,  AR: 0.8117542307692311, 50,  78000000
    # MQT:       LER: 1.833271826560692e-07, AR: 0.8075683979863748, 50,  337700000


    # 19_1_5
    # With p_mem = p / 10
    # No C2 opt: LER: 8.318710496388861e-07,  AR: 0.7826692773437508, 100, 153600000
    # MQT:       LER: 1.7246872672829994e-07, AR: 0.7819220879417323, 50,  370700000
    # With p_mem = p / 100
    # C2 opt:
    # MQT:       LER: 1.0655376608753426e-07, AR: 0.7929108315024583, 50,  591700000


    # 20_2_6
    # With p_mem = p / 10
    # No C2 opt: LER: 1.2630226444271337e-06, AR: 0.7382770335820892, 100, 53600000
    # C2 opt:    LER: 2.822810201888519e-06,  AR: 0.7058635856573711, 100, 25100000
    # MQT:       LER: 2.1890634019219403e-07, AR: 0.7374751000645592, 50,  154900000
    # With p_mem = p / 100
    # No C2 opt: LER:
    # C2 opt:    LER: 6.379902238204728e-08,  AR: 0.7795077844073189, 50,  502800000
    # MQT:       LER: 8.919712516436163e-08,  AR: 0.7513377506702422, 50,  373000000

import itertools
import json

import numpy as np
import stim

from generate import OptimisedSteaneData
from post_selection_simulations import from_json_serializable, load_optimised_se
from qecc import QECCGadgets, NoiseModel, SyndromeMeasurementCircuit


class Simulator:
    def __init__(self, code: str, p: float):
        self.qecc_gadgets = QECCGadgets.from_json(f"circuits/{code}.json")
        self.simp = OptimisedSteaneData.load_json(f"simplified_circuits/{code}.json")
        self.noise_model = NoiseModel.from_p(p)
        self.reset_circuits()

    def reset_circuits(self):
        self.PERFECT_Z_STATE_PREP = self.qecc_gadgets.non_ft_z_state_prep.to_stim(noise_model=None)
        self.PERFECT_X_STATE_PREP = self.qecc_gadgets.non_ft_x_state_prep.to_stim(noise_model=None)

        self.Z_SYNDROME_MEASUREMENT = self.simp.optimized_ft_z_se.circuit.to_stim(self.noise_model)
        self.X_SYNDROME_MEASUREMENT = self.simp.optimized_ft_x_se.circuit.to_stim(self.noise_model)

        self.NON_FT_Z_SYNDROME_MEASUREMENT = self.simp.optimized_non_ft_z_se.circuit.to_stim(self.noise_model)
        self.NON_FT_X_SYNDROME_MEASUREMENT = self.simp.optimized_non_ft_x_se.circuit.to_stim(self.noise_model)

    def run_qec_cycle(self, tableau_simulator: stim.TableauSimulator):
        tableau_simulator.do(self.Z_SYNDROME_MEASUREMENT)

        H = self.qecc_gadgets.code.H_x[:, self.simp.optimized_ft_z_se.H_indices]
        H_panic = self.qecc_gadgets.code.H_x[:, self.simp.optimized_non_ft_z_se.H_indices]

        num_measurements = len(self.simp.optimized_ft_z_se.flag_indices) + len(self.simp.optimized_ft_z_se.measurement_indices)
        flag_indices = tuple(i - num_measurements for i in self.simp.optimized_ft_z_se.flag_indices)
        measurement_indices = (i - num_measurements for i in self.simp.optimized_ft_z_se.measurement_indices)
        flags = tuple(tableau_simulator.current_measurement_record()[f] for f in flag_indices)
        measurements = tuple(tableau_simulator.current_measurement_record()[m] for m in measurement_indices)
        syndromes = tuple((H @ measurements % 2).tolist())

        if np.any(flags):
            tableau_simulator.do(self.NON_FT_X_SYNDROME_MEASUREMENT)
            panic_measurements = tableau_simulator.current_measurement_record()[-9:]
            panic_syndromes = tuple((H_panic @ panic_measurements).tolist())
            for index in self.simp.modified_lookup_table[(flags, panic_syndromes)]:
                tableau_simulator.x(index)
            return
        elif np.any(syndromes):
            for index in self.simp.lookup_table[syndromes]:
                tableau_simulator.x(index)

        tableau_simulator.do(self.X_SYNDROME_MEASUREMENT)
        flags = tuple(tableau_simulator.current_measurement_record()[f] for f in flag_indices)
        measurements = tuple(tableau_simulator.current_measurement_record()[m] for m in measurement_indices)

        if np.any(flags):
            tableau_simulator.do(self.NON_FT_Z_SYNDROME_MEASUREMENT)
            panic_measurements = tableau_simulator.current_measurement_record()[-9:]
            panic_syndromes = tuple((H_panic @ panic_measurements).tolist())
            for index in self.simp.modified_lookup_table[(flags, panic_syndromes)]:
                tableau_simulator.z(index)
            return
        elif np.any(syndromes):
            for index in lookup_table[syndromes]:
                tableau_simulator.z(index)



if __name__ == "__main__":
    qecc = "15_7_3"


    correction_tensor = np.zeros((2,) * 6 + (15,), dtype=np.int16)
    can_correct = np.zeros((2,) * 6, dtype=np.bool)
    for idx in itertools.product([0, 1], repeat=6):
        correction_tensor[idx] = simp["lookup_table"].get(idx, [0] * 15)
        can_correct[idx] = True

    STATE_PREP = qecc_gadgets.non_ft_z_state_prep.to_stim(noise_model=None)

    Z_SYNDROME_MEASUREMENT = simp["circuit"].to_stim(noise_model=NoiseModel.Quantinuum_Sol())
    NON_FT_Z_SYNDROME_MEASUREMENT = ...

    X_SYNDROME_MEASUREMENT = stim.Circuit()
    X_SYNDROME_MEASUREMENT.append("H", range(15))
    X_SYNDROME_MEASUREMENT += Z_SYNDROME_MEASUREMENT
    X_SYNDROME_MEASUREMENT.append("H", range(15))
    NON_FT_X_SYNDROME_MEASUREMENT = ...

    tableau = stim.TableauSimulator()
    tableau.do(STATE_PREP)
    run_qec_cycle(tableau)
    raw_measurements = tableau.measure_many(range(15))
    syndromes = np.array(raw_measurements, dtype=int) @ qecc_gadgets.code.H_z.T % 2
    corrected_measurements = (raw_measurements + correction_tensor[syndromes]) % 2
    logicals = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
    logical_error = np.any(logicals)


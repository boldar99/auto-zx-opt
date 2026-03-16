import json
import pprint
import stim
import numpy as np

from qecc import QECCGadgets, SyndromeMeasurementCircuit, NoiseModel


def from_json_serializable(data):
    new_data = {}
    for k, v in data.items():
        sub_keys = k.split("+")
        if len(sub_keys) == 1:
            new_k = tuple(map(int, k))
        else:
            new_k = tuple(tuple(map(int, sub_key)) for sub_key in sub_keys)
        new_data[new_k] = v
    return new_data


def load_optimised_se(code):
    with open(f"simplified_circuits/{code}.json", "r") as f:
        data = json.load(f)

    data["circuit"] = SyndromeMeasurementCircuit.from_dict(data["circuit"])
    data["lookup_table"] = from_json_serializable(data["lookup_table"])
    data["modified_lookup_table"] = from_json_serializable(data["modified_lookup_table"])

    return data


def get_best_correction(samples, flag_pattern, syndrome_pattern):
    flag_pattern_matched = True if flag_pattern is None else np.all(flags == flag_pattern, axis=1)
    syndrome_pattern_matched = True if syndrome_pattern is None else np.all(syndromes == syndrome_pattern, axis=1)
    filtered_samples = samples[np.logical_and(flag_pattern_matched, syndrome_pattern_matched)]

    min_ler = 2
    min_correction = [0] * 15
    for i in range(2 ** 15):
        vec_i = np.array(list(map(int, ('0' * 15 + format(i, "b"))[-15:])))
        corrected_measurements = (filtered_samples[:, -15:] + vec_i) % 2
        logical_samples = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
        logical_error = np.any(logical_samples, axis=1)
        logical_error_rate = np.average(logical_error)
        if logical_error_rate < min_ler:
            min_ler = logical_error_rate
            min_correction = vec_i
    return min_correction, min_ler


def calculate_post_selected_ler(samples, flags, syndromes, correction_LER, simp, L_z_T, threshold):
    """
    Calculates the logical error rate (LER) by post-selecting (keeping) ONLY the
    measurements where the known LER for the given (flag, syndrome) is below the threshold.
    """
    valid_corrected_measurements = []

    lookup = simp["lookup_table"]
    mod_lookup = simp["modified_lookup_table"]

    for idx, (f, s) in enumerate(zip(flags, syndromes)):
        flag_tup = tuple(f.astype(int).tolist())
        synd_tup = tuple(s.astype(int).tolist())

        # The trivial syndrome (no errors detected) is assumed to have an LER of 0.0
        if sum(synd_tup) == 0:
            ler = 0.0
        else:
            # If the pattern isn't in our dictionary, default to 1.0 so it gets discarded
            ler = correction_LER.get((flag_tup, synd_tup), 1.0)

        # Keep the sample ONLY if its historical LER is strictly below the threshold
        if ler < threshold:
            if any(flag_tup):
                corr = mod_lookup.get((flag_tup, synd_tup), lookup[synd_tup])
            else:
                corr = lookup[synd_tup]

            # Apply the correction to the raw measurement and save it
            corrected = (samples[idx, -15:] + corr) % 2
            valid_corrected_measurements.append(corrected)

    # Edge case: Threshold was too strict and we threw away all data
    if not valid_corrected_measurements:
        print(f"Warning: 0 samples passed the {threshold:.1%} threshold.")
        return 0.0, 0.0

    valid_corrected_measurements = np.array(valid_corrected_measurements)

    # Calculate logical measurements on the surviving subset
    logical_measurements = valid_corrected_measurements @ L_z_T % 2
    measurement_error = np.any(logical_measurements, axis=1)

    final_ler = np.average(measurement_error)
    acceptance_rate = len(valid_corrected_measurements) / len(samples)

    return final_ler, acceptance_rate


def error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern, syndrome_pattern=None, verbose=False):
    flag_pattern_matched = True if flag_pattern is None else np.all(flags == flag_pattern, axis=1)
    syndrome_pattern_matched = True if syndrome_pattern is None else np.all(syndromes == syndrome_pattern, axis=1)
    corrected_measurements = corrected_measurements[np.logical_and(flag_pattern_matched, syndrome_pattern_matched)]
    logical_measurements = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
    measurement_error = np.any(logical_measurements, axis=1)
    error_rate = np.average(measurement_error).tolist()
    if verbose:
        print(f"LER when {flag_pattern=}" +
              (f" and {syndrome_pattern=}" if syndrome_pattern is not None else "") +
              f": {error_rate:.3%}")
        if syndrome_pattern is not None:
            print(f"Correction to be applied {(measurement_error.shape[0])}: ", simp["modified_lookup_table"].get((flag_pattern, syndrome_pattern), simp["lookup_table"][syndrome_pattern]))
    return error_rate


if __name__ == "__main__":
    qecc = "15_7_3"
    qecc_gadgets = QECCGadgets.from_json(f"circuits/{qecc}.json")
    simp = load_optimised_se(qecc)

    circ = qecc_gadgets.ft_x_state_prep.to_stim()
    circ += simp["circuit"].to_stim(noise_model=NoiseModel.Quantinuum_Sol())
    circ.append("H", range(15))
    circ.append("M", range(15))

    NUM_SAMPLES = 1_000_000
    smplr = circ.compile_sampler()
    samples = smplr.sample(NUM_SAMPLES)
    flags = samples[:, -17:-15]
    syndromes = samples[:, -15:] @ qecc_gadgets.code.H_z.T % 2

    correction = []
    modified_correction = []
    for f, s in zip(flags, syndromes):
        syndrome = tuple(s.tolist())
        flag = tuple(f.astype(int).tolist())
        correction.append(simp["lookup_table"][syndrome])
        modified_correction.append(simp["modified_lookup_table"].get((flag, syndrome),  simp["lookup_table"][syndrome]))
    correction = np.array(correction)
    modified_correction = np.array(modified_correction)
    corrected_measurements = samples[:, -15:] + np.where(np.any(flags, axis=1)[:,np.newaxis], modified_correction, correction)


    corrected_measurements = corrected_measurements[~np.any(flags, axis=1)]
    logical_measurements = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
    measurement_error = np.any(logical_measurements, axis=1)
    print(f"Flag post-selected LER: {np.average(measurement_error):.3%}")
    print(f"Acceptance rate: {len(measurement_error) / NUM_SAMPLES:.3%}")
    print()

    corrected_measurements = samples[:, -15:] + np.where(np.any(flags, axis=1)[:,np.newaxis], modified_correction, correction)
    logical_measurements = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
    measurement_error = np.any(logical_measurements, axis=1)
    print(f"With modified decoding LER: {np.average(measurement_error):.3%}")
    print()

    corrected_measurements = (samples[:, -15:] + np.where(np.any(flags, axis=1)[:,np.newaxis], modified_correction, correction)) % 2
    error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(0,0), verbose=True)
    error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(1,1), verbose=True)
    error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(0,1), verbose=True)
    error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(1,0), verbose=True)

    correction_LER = dict()

    for f in range(4):
        vec_f = tuple(map(int, ('0' * 2 + format(f, "b"))[-2:]))
        for i in range(1, 2 ** 4):
            vec_i = tuple(map(int, ('0' * 4 + format(i, "b"))[-4:]))
            error_rate = error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=vec_f, syndrome_pattern=vec_i)
            correction_LER[(vec_f, vec_i)] = error_rate
    print()

    for THRESHOLD in [0.5, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]:
        L_z_T = qecc_gadgets.code.L_z.T

        ps_ler, acc_rate = calculate_post_selected_ler(
            samples=samples,
            flags=flags,
            syndromes=syndromes,
            correction_LER=correction_LER,
            simp=simp,
            L_z_T=L_z_T,
            threshold=THRESHOLD
        )

        print(f"Syndrome post-selection threshold: {THRESHOLD:.0%}")
        print(f"Logical Error Rate: \t{ps_ler:.2%}")
        print(f"Acceptance Rate:    \t{acc_rate:.2%}")
        print()


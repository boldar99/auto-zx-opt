import numpy as np
import pytest
import stim

from spiderwarp.fault_tolerance_verification import verify_ftsp
from spiderwarp.stim_utils import make_stim_circ_single_type_noisy


def _without_inserted_noise(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for operation in circuit.flattened():
        if operation.name not in {"X_ERROR", "Z_ERROR"}:
            result.append(operation)
    return result


@pytest.mark.parametrize("error_type", ["X", "Z"])
@pytest.mark.parametrize("repetitions", [1, 2])
def test_noise_preserves_reused_qubit_measurement_records(
    error_type: str, repetitions: int,
) -> None:
    # M 0 is random; the following M 1 and the reused qubit's M 0 are
    # deterministic. Scheduling M 1 before M 0 invalidates rec[-2].
    body = stim.Circuit(
        "R 0 1\n"
        "H 0\n"
        "M 0 1\n"
        "R 0\n"
        "M 0\n"
        "DETECTOR rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]"
    )
    circuit = body * repetitions
    original = circuit.copy()

    noisy = make_stim_circ_single_type_noisy(circuit, p=0, error_type=error_type)

    assert circuit == original
    assert _without_inserted_noise(noisy) == circuit.flattened()
    assert noisy.num_measurements == 3 * repetitions
    assert noisy.detector_error_model() == circuit.detector_error_model()
    samples = noisy.compile_detector_sampler(seed=1234).sample(
        128, append_observables=True
    )
    assert samples.shape == (128, repetitions + 1)
    assert not samples.any()


@pytest.mark.parametrize("error_type", ["X", "Z"])
def test_batched_two_qubit_gates_get_independent_single_type_faults(
    error_type: str,
) -> None:
    circuit = stim.Circuit(
        "QUBIT_COORDS(1, 2) 0\n"
        "R 0 1 2 3\n"
        "TICK\n"
        "CX 0 1 2 3\n"
        "M 0 1 2 3\n"
        "DETECTOR rec[-4] rec[-3]\n"
        "DETECTOR rec[-2] rec[-1]\n"
        "SHIFT_COORDS(1, 0)"
    )
    p = 0.125

    noisy = make_stim_circ_single_type_noisy(circuit, p=p, error_type=error_type)

    assert _without_inserted_noise(noisy) == circuit.flattened()
    assert [
        operation for operation in noisy
        if operation.name in {"QUBIT_COORDS", "TICK", "DETECTOR", "SHIFT_COORDS"}
    ] == [
        operation for operation in circuit
        if operation.name in {"QUBIT_COORDS", "TICK", "DETECTOR", "SHIFT_COORDS"}
    ]
    noise_operations = [
        operation for operation in noisy if operation.name.endswith("_ERROR")
    ]
    assert all(operation.name == f"{error_type}_ERROR" for operation in noise_operations)
    assert all(operation.gate_args_copy() == [p] for operation in noise_operations)
    assert sum(len(operation.targets_copy()) for operation in noise_operations) == 8
    assert all(
        target.is_qubit_target
        for operation in noise_operations for target in operation.targets_copy()
    )

    dem = noisy.detector_error_model()
    errors = [operation for operation in dem if operation.type == "error"]
    if error_type == "X":
        # For each pair, a reset fault on the target or either output fault
        # flips its parity. A reset fault on the control propagates to both
        # outputs and cancels. Three independent flips give this probability.
        assert [operation.targets_copy() for operation in errors] == [
            [stim.target_relative_detector_id(0)],
            [stim.target_relative_detector_id(1)],
        ]
        expected_probability = (1 - (1 - 2 * p) ** 3) / 2
        assert [operation.args_copy()[0] for operation in errors] == pytest.approx(
            [expected_probability, expected_probability]
        )
    else:
        assert errors == []


@pytest.mark.parametrize("error_type", ["X", "Z"])
def test_feedback_keeps_record_targets_and_noises_only_physical_qubits(
    error_type: str,
) -> None:
    circuit = stim.Circuit(
        "R 0 1\n"
        "H 0\n"
        "M 0\n"
        "CX rec[-1] 1\n"
        "M 1\n"
        "DETECTOR rec[-1] rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]"
    )
    p = 0.125

    noisy = make_stim_circ_single_type_noisy(circuit, p=p, error_type=error_type)

    assert _without_inserted_noise(noisy) == circuit
    operations = list(noisy)
    feedback_index = next(
        index for index, operation in enumerate(operations)
        if operation.name == "CX"
    )
    feedback_noise = operations[feedback_index + 1]
    assert feedback_noise.name == f"{error_type}_ERROR"
    assert feedback_noise.targets_copy() == [stim.GateTarget(1)]
    errors = [
        operation for operation in noisy.detector_error_model()
        if operation.type == "error"
    ]
    if error_type == "X":
        assert len(errors) == 1
        assert errors[0].targets_copy() == [
            stim.target_relative_detector_id(0),
            stim.target_logical_observable_id(0),
        ]
        # The target's reset and the feedback gate are the two fault locations
        # that can break agreement with the recorded control measurement.
        assert errors[0].args_copy() == pytest.approx([2 * p * (1 - p)])
    else:
        assert errors == []


def test_measure_and_reset_remains_excluded_from_the_fault_set() -> None:
    circuit = stim.Circuit("R 0\nMR 0\nM 0\nDETECTOR rec[-1]")

    noisy = make_stim_circ_single_type_noisy(circuit, p=0.125, error_type="X")

    assert noisy.num_measurements == 2
    assert [operation.name for operation in noisy] == [
        "R", "X_ERROR", "MR", "M", "DETECTOR"
    ]
    # MR removes a reset fault before the final readout. Adding a fault after
    # MR would turn this deterministic detector into an error mechanism.
    assert noisy.detector_error_model() == stim.DetectorErrorModel("detector D0")


def test_verify_ftsp_handles_reused_ancillas_and_parallel_measurements() -> None:
    checks = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1],
    ], dtype=np.uint8)
    logical = np.array([[1, 1, 0, 0, 1, 0, 0]], dtype=np.uint8)
    ideal_steane_zero = stim.Circuit(
        "R 0 1\nRX 2\nR 3\nRX 4\nR 5\nRX 6\n"
        "CX 2 1 2 3 6 1 4 3 6 5 4 5\nCX 4 0 1 0"
    )
    gadget = stim.Circuit(
        "R 7 8\n"
        "H 7\n"
        "M 7 8\n"
        "R 7\n"
        "M 7\n"
        "DETECTOR rec[-2]\n"
        "I 0 1"
    )

    assert verify_ftsp(
        gadget,
        H_primary=checks,
        L_primary=logical,
        H_conjugate=checks,
        d=3,
        t=1,
        ideal_input_circuit=ideal_steane_zero,
    ) is True

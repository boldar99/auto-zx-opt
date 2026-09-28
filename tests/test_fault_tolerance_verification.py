from dataclasses import replace

import pytest
import numpy as np
import stim

import spiderwarp.fault_tolerance_verification as ft_verification
from spiderwarp.fault_tolerance_verification import (
    FaultHistory,
    FaultOrderDetectionFailure,
    classify_flag_patterns_by_fault_order,
    extract_deterministic_failure,
    extract_unique_fault_mechanisms,
    find_distinguishable_fault_set_failure,
    find_fault_order_detection_failure,
    prepend_ideal_input,
)


def test_ideal_input_makes_gadget_detectors_deterministic() -> None:
    ideal_input = stim.Circuit("RX 0")
    gadget = stim.Circuit("MX 0\nDETECTOR rec[-1]")

    with pytest.raises(ValueError, match="non-deterministic detectors"):
        gadget.detector_error_model()

    evaluation_circuit = prepend_ideal_input(gadget, ideal_input)
    evaluation_circuit.detector_error_model()


def test_ideal_input_cannot_add_detector_indices() -> None:
    ideal_input = stim.Circuit("M 0\nDETECTOR rec[-1]")

    with pytest.raises(ValueError, match="must not contain detectors"):
        prepend_ideal_input(stim.Circuit(), ideal_input)


def test_concrete_failure_keeps_faults_sharing_a_noise_instruction() -> None:
    noisy_circuit = stim.Circuit(
        "R 0 1\n"
        "X_ERROR(0.001) 0 1\n"
        "M 0 1\n"
        "DETECTOR rec[-2]\n"
        "DETECTOR rec[-1]"
    )

    witness = extract_deterministic_failure(
        noisy_circuit,
        extract_unique_fault_mechanisms(noisy_circuit),
        [],
        "Z",
    )
    samples = witness.compile_sampler().sample(10)

    assert samples.all()


def test_data_completion_fault_is_inserted_before_final_measurement() -> None:
    evaluation_circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")

    witness = extract_deterministic_failure(
        evaluation_circuit, [], [0], "Z"
    )

    assert witness == stim.Circuit("R 0\nX 0\nM 0\nDETECTOR rec[-1]")
    assert witness.compile_sampler().sample(10).all()


def test_flag_classifier_validates_detector_indices() -> None:
    empty_checks = np.zeros((0, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="detectors in the gadget"):
        classify_flag_patterns_by_fault_order(
            stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]"),
            empty_checks,
            empty_checks,
            empty_checks,
            empty_checks,
            flag_detector_indices=[1],
        )


def test_distinguishable_fault_set_rejects_same_flag_syndrome_logical() -> None:
    mechanisms = [
        frozenset({("D", 0), ("D", 1)}),
        frozenset({("D", 0), ("D", 1), ("L", 0)}),
    ]

    failure = find_distinguishable_fault_set_failure(
        mechanisms,
        num_internal_detectors=1,
        num_final_stabilizers=1,
        num_logicals=1,
        flag_detector_indices=[0],
        max_faults=1,
        error_type="X",
    )

    assert failure is not None
    assert failure.flag_pattern == (1,)
    assert failure.syndrome_pattern == (1,)
    assert {
        failure.first_logical_signature,
        failure.second_logical_signature,
    } == {(0,), (1,)}


def test_distinguishable_fault_set_accepts_different_syndromes() -> None:
    mechanisms = [
        frozenset({("D", 0)}),
        frozenset({("D", 0), ("D", 1), ("L", 0)}),
    ]

    failure = find_distinguishable_fault_set_failure(
        mechanisms,
        num_internal_detectors=1,
        num_final_stabilizers=1,
        num_logicals=1,
        flag_detector_indices=[0],
        max_faults=1,
        error_type="X",
    )

    assert failure is None


def test_fault_order_wrapper_returns_stim_counterexample_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combined_circuit = stim.Circuit("X 0\nX 1\nX 2")
    detailed_failure = FaultOrderDetectionFailure(
        error_type="X",
        flag_pattern=(1,),
        syndrome_pattern=(0,),
        correctable_logical_signature=(0,),
        higher_order_logical_signature=(1,),
        correctable_history=FaultHistory((frozenset({("D", 0)}),)),
        higher_order_history=FaultHistory((
            frozenset({("D", 1)}),
            frozenset({("D", 0), ("D", 1), ("L", 0)}),
        )),
        combined_circuit=combined_circuit,
    )
    monkeypatch.setattr(
        ft_verification,
        "verify_distinguishable_fault_set",
        lambda **_: detailed_failure,
    )
    matrix = np.zeros((1, 1), dtype=np.uint8)

    result = ft_verification.verify_fault_order_detect_or_correct(
        stim.Circuit(),
        matrix,
        matrix,
        matrix,
        matrix,
        correctable_faults=1,
        detectable_faults=2,
    )
    assert isinstance(result, stim.Circuit)
    assert str(result) == str(combined_circuit)
    assert detailed_failure.combined_history.fault_count == 3


def test_flag_patterns_are_classified_by_one_or_two_fault_reachability() -> None:
    empty_checks = np.zeros((0, 2), dtype=np.uint8)

    classifications = classify_flag_patterns_by_fault_order(
        stim.Circuit("R 0 1\nM 0 1"),
        empty_checks,
        empty_checks,
        empty_checks,
        empty_checks,
        flag_measurement_indices=range(2),
    )

    assert {
        pattern: analysis.possible_fault_orders
        for pattern, analysis in classifications.items()
    } == {
        (0, 0): frozenset({0}),
        (0, 1): frozenset({1}),
        (1, 0): frozenset({1}),
        (1, 1): frozenset({2}),
    }
    assert all(
        analysis.minimum_distinguishing_stabilizer_sets == ((),)
        for analysis in classifications.values()
    )
    assert all(
        analysis.correction_is_well_defined
        for analysis in classifications.values()
    )


def test_minimum_stabilizers_separate_all_cross_order_syndromes() -> None:
    fault = ft_verification.PhysicalFault(
        error_type="X",
        instruction_offset=0,
        noise_instruction="X_ERROR(0.001) 0",
        pauli_product=(("X", 0),),
        mechanism=frozenset(),
    )

    def history(
        syndrome: tuple[int, ...],
        logical: tuple[int, ...] = (),
    ) -> ft_verification.FlaggedFaultHistory:
        return ft_verification.FlaggedFaultHistory(
            error_type="X",
            faults=(fault,),
            flag_pattern=(1,),
            syndrome_pattern=syndrome,
            logical_signature=logical,
        )

    minimum_sets, witness = ft_verification._minimum_stabilizer_separators(
        (history((0, 0)),),
        (history((1, 0)), history((0, 1))),
        (("Z", 0), ("Z", 1)),
    )
    impossible_sets, impossible_witness = (
        ft_verification._minimum_stabilizer_separators(
            (history((0, 0)),),
            (history((0, 0)),),
            (("Z", 0), ("Z", 1)),
        )
    )
    incompatible_correction = (
        ft_verification._find_incompatible_cross_order_correction(
            (history((0, 0), (0,)),),
            (history((0, 0), (1,)),),
        )
    )

    assert minimum_sets == ((("Z", 0), ("Z", 1)),)
    assert witness is None
    assert impossible_sets == ()
    assert impossible_witness is not None
    assert incompatible_correction is not None


def test_minimal_followup_flow_uses_flag_and_syndrome_information() -> None:
    fault = ft_verification.PhysicalFault(
        error_type="X",
        instruction_offset=0,
        noise_instruction="X_ERROR(0.001) 0",
        pauli_product=(("X", 0),),
        mechanism=frozenset(),
    )
    incoming_fault = replace(fault, origin="incoming")

    def history(
        syndrome: tuple[int, ...],
        logical: tuple[int, ...],
        count: int,
        incoming: bool = False,
    ) -> ft_verification.FlaggedFaultHistory:
        return ft_verification.FlaggedFaultHistory(
            error_type="X",
            faults=((incoming_fault if incoming else fault),) * count,
            flag_pattern=(1,),
            syndrome_pattern=syndrome,
            logical_signature=logical,
        )

    same_class_one = history((1, 0, 0), (0, 0, 0), 1)
    same_class_two = history((1, 0, 0), (0, 0, 0), 2)
    identity_one = history(
        (0, 0, 0), (0, 0, 0), 1, incoming=True
    )
    identity_two = history((0, 0, 0), (0, 0, 0), 2)
    weight_two = history((1, 1, 0), (0, 0, 0), 2)
    weight_three = history((1, 1, 1), (0, 0, 0), 2)
    logical_three = history((0, 0, 0), (1, 1, 1), 2)

    def analysis(
        one: tuple[ft_verification.FlaggedFaultHistory, ...],
        two: tuple[ft_verification.FlaggedFaultHistory, ...],
    ) -> ft_verification.FlagPatternFaultAnalysis:
        return ft_verification.FlagPatternFaultAnalysis(
            flag_pattern=(1,),
            one_fault_histories=one,
            two_fault_histories=two,
            minimum_distinguishing_stabilizer_sets=((),),
        )

    analyses = {
        (0, 0): analysis((same_class_one,), (same_class_two,)),
        (0, 1): analysis((identity_one,), (weight_two,)),
        (1, 0): analysis((), (
            identity_two,
            weight_three,
        )),
        (1, 1): analysis((), (identity_two, logical_three)),
    }
    H_z = np.hstack([
        np.eye(3, dtype=np.uint8),
        np.zeros((3, 3), dtype=np.uint8),
    ])
    L_z = np.hstack([
        np.zeros((3, 3), dtype=np.uint8),
        np.eye(3, dtype=np.uint8),
    ])
    empty = np.zeros((0, 6), dtype=np.uint8)
    plans = ft_verification.plan_minimal_followup_by_flag(
        analyses,
        H_x=empty,
        H_z=H_z,
        L_x=empty,
        L_z=L_z,
        prepared_state="0",
    )

    assert plans[(0, 0)].action == ft_verification.FollowupAction.CORRECT_NOW
    assert plans[(0, 1)].action == ft_verification.FollowupAction.RUN_ONE_FT_SE
    assert plans[(1, 0)].action == ft_verification.FollowupAction.RUN_NON_FT_SE
    assert plans[(1, 0)].ambiguous_syndromes == ()
    assert plans[(1, 1)].action == ft_verification.FollowupAction.DISCARD


def test_two_fault_history_cannot_masquerade_as_one_fault_correction() -> None:
    mechanisms = [
        frozenset({("D", 0)}),
        frozenset({("D", 1), ("L", 0)}),
        frozenset({("D", 0), ("D", 1)}),
    ]

    failure = find_fault_order_detection_failure(
        mechanisms,
        num_internal_detectors=1,
        num_final_stabilizers=1,
        num_logicals=1,
        flag_detector_indices=[0],
        correctable_faults=1,
        detectable_faults=2,
        error_type="X",
    )

    assert isinstance(failure, FaultOrderDetectionFailure)
    assert failure.flag_pattern == (1,)
    assert failure.syndrome_pattern == (1,)
    assert failure.correctable_history.fault_count == 1
    assert failure.higher_order_history.fault_count == 2


def test_two_fault_history_may_reuse_a_compatible_correction() -> None:
    mechanisms = [
        frozenset({("D", 0)}),
        frozenset({("D", 1)}),
        frozenset({("D", 0), ("D", 1)}),
    ]

    failure = find_fault_order_detection_failure(
        mechanisms,
        num_internal_detectors=1,
        num_final_stabilizers=1,
        num_logicals=1,
        flag_detector_indices=[0],
        correctable_faults=1,
        detectable_faults=2,
        error_type="X",
    )

    assert failure is None

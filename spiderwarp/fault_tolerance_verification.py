import itertools
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import stim

from spidercss.utils import NOISE_GATES, make_stim_circ_single_type_noisy


FaultMechanism = frozenset[tuple[str, int]]
StabilizerCheck = tuple[str, int]


class FollowupAction(str, Enum):
    """Minimum next step selected from the observed SE flag pattern."""

    CORRECT_NOW = "correct now; no follow-up SE"
    RUN_ONE_FT_SE = "run 1-FT follow-up SE"
    RUN_NON_FT_SE = "run non-FT follow-up SE"
    DISCARD = "discard"


@dataclass(frozen=True)
class PauliCorrection:
    """A concrete Pauli correction, stored as X and Z support."""

    x_support: tuple[int, ...] = ()
    z_support: tuple[int, ...] = ()

    @property
    def weight(self) -> int:
        return len(set(self.x_support) | set(self.z_support))

    @property
    def pauli_product(self) -> tuple[tuple[str, int], ...]:
        x_support = set(self.x_support)
        z_support = set(self.z_support)
        return tuple(
            (
                "Y" if qubit in x_support and qubit in z_support
                else "X" if qubit in x_support
                else "Z",
                qubit,
            )
            for qubit in sorted(x_support | z_support)
        )


@dataclass(frozen=True)
class CorrectionOutcome:
    """Effective final error left by a correction on one candidate history."""

    history: "FlaggedFaultHistory"
    final_error: PauliCorrection

    @property
    def weight(self) -> int:
        return self.final_error.weight


@dataclass(frozen=True)
class FlagPatternFollowupPlan:
    """Minimal follow-up branch and its syndrome-conditioned decisions."""

    flag_pattern: tuple[int, ...]
    action: FollowupAction
    possible_fault_orders: frozenset[int]
    correction_without_followup: PauliCorrection | None = None
    correction_outcomes: tuple[CorrectionOutcome, ...] = ()
    uniquely_correctable_syndromes: tuple[
        tuple[tuple[int, ...], PauliCorrection], ...
    ] = ()
    ambiguous_syndromes: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class PhysicalFault:
    """One concrete Pauli fault location in a flattened noisy circuit."""

    error_type: str
    instruction_offset: int
    noise_instruction: str
    pauli_product: tuple[tuple[str, int], ...]
    mechanism: FaultMechanism
    origin: str = "se"


@dataclass(frozen=True)
class FlaggedFaultHistory:
    """A concrete one- or two-fault history and its ideal readout."""

    error_type: str
    faults: tuple[PhysicalFault, ...]
    flag_pattern: tuple[int, ...]
    syndrome_pattern: tuple[int, ...]
    logical_signature: tuple[int, ...]

    @property
    def fault_count(self) -> int:
        return len(self.faults)

    @property
    def incoming_fault_count(self) -> int:
        return sum(fault.origin == "incoming" for fault in self.faults)

    @property
    def se_fault_count(self) -> int:
        return sum(fault.origin == "se" for fault in self.faults)

    @property
    def maximum_acceptable_residual_weight(self) -> int:
        if self.fault_count == 0:
            return 0
        if self.incoming_fault_count == 1 and self.se_fault_count == 0:
            return 0
        return 1


@dataclass(frozen=True)
class FlagPatternFaultAnalysis:
    """All candidate histories and ideal order-separating stabilizer sets."""

    flag_pattern: tuple[int, ...]
    one_fault_histories: tuple[FlaggedFaultHistory, ...]
    two_fault_histories: tuple[FlaggedFaultHistory, ...]
    minimum_distinguishing_stabilizer_sets: tuple[
        tuple[StabilizerCheck, ...], ...
    ]
    zero_fault_histories: tuple[FlaggedFaultHistory, ...] = ()
    indistinguishable_order_pair: tuple[
        FlaggedFaultHistory, FlaggedFaultHistory
    ] | None = None
    incompatible_correction_pair: tuple[
        FlaggedFaultHistory, FlaggedFaultHistory
    ] | None = None

    @property
    def possible_fault_orders(self) -> frozenset[int]:
        orders = set()
        if self.zero_fault_histories:
            orders.add(0)
        if self.one_fault_histories:
            orders.add(1)
        if self.two_fault_histories:
            orders.add(2)
        return frozenset(orders)

    @property
    def minimum_number_of_stabilizers(self) -> int | None:
        if not self.minimum_distinguishing_stabilizer_sets:
            return None
        return len(self.minimum_distinguishing_stabilizer_sets[0])

    @property
    def correction_is_well_defined(self) -> bool:
        return self.incompatible_correction_pair is None


@dataclass(frozen=True)
class FaultHistory:
    """One representative combination of DEM fault mechanisms."""

    mechanisms: tuple[FaultMechanism, ...]

    @property
    def fault_count(self) -> int:
        return len(self.mechanisms)


def _combine_fault_histories(
    first: FaultHistory,
    second: FaultHistory,
) -> FaultHistory:
    """Combines two histories modulo two, cancelling shared mechanisms."""
    combined = set(first.mechanisms)
    combined.symmetric_difference_update(second.mechanisms)
    return FaultHistory(tuple(sorted(
        combined,
        key=lambda mechanism: tuple(sorted(mechanism)),
    )))


@dataclass(frozen=True)
class DistinguishableFaultSetFailure:
    """Two fault histories that no flag-and-syndrome decoder can distinguish."""

    error_type: str
    flag_pattern: tuple[int, ...]
    syndrome_pattern: tuple[int, ...]
    first_logical_signature: tuple[int, ...]
    second_logical_signature: tuple[int, ...]
    first_history: FaultHistory
    second_history: FaultHistory
    combined_circuit: stim.Circuit | None = None

    @property
    def combined_history(self) -> FaultHistory:
        return _combine_fault_histories(
            self.first_history, self.second_history
        )

    @property
    def example_circuit(self) -> stim.Circuit | None:
        return self.combined_circuit


@dataclass(frozen=True)
class FaultOrderDetectionFailure:
    """A higher-order history that masquerades as a correctable history."""

    error_type: str
    flag_pattern: tuple[int, ...]
    syndrome_pattern: tuple[int, ...]
    correctable_logical_signature: tuple[int, ...]
    higher_order_logical_signature: tuple[int, ...]
    correctable_history: FaultHistory
    higher_order_history: FaultHistory
    combined_circuit: stim.Circuit | None = None

    @property
    def combined_history(self) -> FaultHistory:
        return _combine_fault_histories(
            self.correctable_history, self.higher_order_history
        )

    @property
    def example_circuit(self) -> stim.Circuit | None:
        return self.combined_circuit


# ==============================================================================
# PHASE 1: Circuit Evaluation Setup & Utilities
# ==============================================================================
def append_ideal_measurements(
    noisy_prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_matrix: np.ndarray | None,
    basis_char: str
) -> stim.Circuit:
    """Appends ideal transversal measurements, detectors, and logical observables to the circuit."""
    eval_circ = noisy_prep_circ.copy()

    num_data_qubits = H_check.shape[1]

    # Transversally measure all data qubits
    eval_circ.append("MX" if basis_char == 'X' else "M", range(num_data_qubits))

    # Append code-space checks as detectors
    for row in H_check:
        detector_targets = []
        for i, val in enumerate(row):
            if val:
                detector_targets.append(stim.target_rec(-num_data_qubits + i))
        if detector_targets:
            eval_circ.append("DETECTOR", detector_targets)

    # Append logical observables
    if L_matrix is not None and L_matrix.size > 0:
        for m, L_row in enumerate(L_matrix):
            obs_targets = []
            for i, val in enumerate(L_row):
                if val:
                    obs_targets.append(stim.target_rec(-num_data_qubits + i))
            if obs_targets:
                eval_circ.append("OBSERVABLE_INCLUDE", obs_targets, m)

    return eval_circ


def prepend_ideal_input(
    noisy_circ: stim.Circuit,
    ideal_input_circuit: stim.Circuit | None,
) -> stim.Circuit:
    """Prepends a noiseless encoded input without adding it to the fault set.

    This is useful when ``noisy_circ`` is a gadget acting on an already encoded
    block instead of a state-preparation circuit. Detectors belong to the gadget
    being verified, so the ideal input circuit must not define any of its own.
    """
    if ideal_input_circuit is None:
        return noisy_circ
    if ideal_input_circuit.num_detectors:
        raise ValueError("ideal_input_circuit must not contain detectors")
    return ideal_input_circuit + noisy_circ


def append_flag_detectors_from_measurements(
    circuit: stim.Circuit,
    flag_measurement_indices: list[int] | tuple[int, ...] | range,
) -> tuple[stim.Circuit, tuple[int, ...]]:
    """Annotates existing measurement results as flag detectors.

    Measurement indices are zero-based in the circuit's measurement record.
    The returned detector indices can be passed to
    :func:`verify_distinguishable_fault_set`.
    """
    measurement_indices = tuple(flag_measurement_indices)
    if len(set(measurement_indices)) != len(measurement_indices):
        raise ValueError("flag_measurement_indices must be unique")
    if any(i < 0 or i >= circuit.num_measurements for i in measurement_indices):
        raise ValueError(
            f"flag measurement indices must be in [0, {circuit.num_measurements})"
        )

    annotated = circuit.copy()
    first_detector = annotated.num_detectors
    for flag_index, measurement_index in enumerate(measurement_indices):
        annotated.append(
            "DETECTOR",
            [stim.target_rec(measurement_index - circuit.num_measurements)],
            [0, flag_index],
        )
    detector_indices = tuple(
        range(first_detector, first_detector + len(measurement_indices))
    )
    return annotated, detector_indices


def _resolve_flag_detectors(
    circuit: stim.Circuit,
    flag_detector_indices: list[int] | tuple[int, ...] | range | None,
    flag_measurement_indices: list[int] | tuple[int, ...] | range | None,
) -> tuple[stim.Circuit, tuple[int, ...]]:
    """Return a circuit and validated indices for its flag detectors."""
    if flag_detector_indices is not None and flag_measurement_indices is not None:
        raise ValueError(
            "specify flag_detector_indices or flag_measurement_indices, not both"
        )

    annotated = circuit.copy()
    if flag_measurement_indices is not None:
        return append_flag_detectors_from_measurements(
            annotated, flag_measurement_indices
        )

    if flag_detector_indices is None:
        coordinates = annotated.get_detector_coordinates()
        detector_indices = tuple(
            index
            for index in range(annotated.num_detectors)
            if coordinates.get(index) and coordinates[index][0] == 0
        )
        if not detector_indices:
            raise ValueError(
                "no flag detectors found; specify flag measurement or detector indices"
            )
    else:
        detector_indices = tuple(flag_detector_indices)

    if len(set(detector_indices)) != len(detector_indices):
        raise ValueError("flag_detector_indices must be unique")
    if any(index < 0 or index >= annotated.num_detectors for index in detector_indices):
        raise ValueError("flag detector indices must refer to detectors in the gadget")
    return annotated, detector_indices


def _bit_tuple(mask: int, width: int) -> tuple[int, ...]:
    return tuple((mask >> i) & 1 for i in range(width))


def _build_fault_signature_layers(
    unique_mechanisms: list[FaultMechanism],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    num_logicals: int,
    flag_indices: tuple[int, ...],
    max_faults: int,
) -> list[dict[tuple[int, int], tuple[FaultMechanism, ...]]]:
    """Enumerates decoder-symptom and residual-class signatures by fault count."""
    mechanism_signatures = []
    for mechanism in unique_mechanisms:
        internal_mask = 0
        syndrome_mask = 0
        logical_mask = 0
        for target_type, target_index in mechanism:
            if target_type == "D":
                if target_index < num_internal_detectors:
                    internal_mask ^= 1 << target_index
                else:
                    syndrome_index = target_index - num_internal_detectors
                    if syndrome_index < num_final_stabilizers:
                        syndrome_mask ^= 1 << syndrome_index
            elif target_type == "L" and target_index < num_logicals:
                logical_mask ^= 1 << target_index

        flag_mask = sum(
            ((internal_mask >> detector_index) & 1) << flag_index
            for flag_index, detector_index in enumerate(flag_indices)
        )
        decoder_mask = flag_mask | (syndrome_mask << len(flag_indices))
        residual_class = syndrome_mask | (
            logical_mask << num_final_stabilizers
        )
        mechanism_signatures.append(
            (decoder_mask, residual_class, mechanism)
        )

    # Subset dynamic programming enumerates every distinct signature reachable
    # at each exact fault count. Repeated identical signatures never need to be
    # selected: a repeated pair cancels and has a lower-weight representative.
    reachable_by_weight: list[
        dict[tuple[int, int], tuple[FaultMechanism, ...]]
    ] = [{} for _ in range(max_faults + 1)]
    reachable_by_weight[0][(0, 0)] = ()

    for decoder_mask, residual_class, mechanism in mechanism_signatures:
        for weight in range(max_faults, 0, -1):
            previous_layer = tuple(reachable_by_weight[weight - 1].items())
            for (old_decoder, old_residual), history in previous_layer:
                state = (
                    old_decoder ^ decoder_mask,
                    old_residual ^ residual_class,
                )
                reachable_by_weight[weight].setdefault(
                    state, history + (mechanism,)
                )

    return reachable_by_weight


def find_distinguishable_fault_set_failure(
    unique_mechanisms: list[FaultMechanism],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    num_logicals: int,
    flag_detector_indices: list[int] | tuple[int, ...] | range,
    max_faults: int,
    error_type: str,
) -> DistinguishableFaultSetFailure | None:
    """Finds a violation of the distinguishable-fault-set condition.

    Two histories are ambiguous when they produce the same cumulative flag
    vector and final data syndrome but residual errors in different stabilizer
    cosets. The logical-observable bits distinguish those cosets.
    """
    if max_faults < 0:
        raise ValueError("max_faults must be non-negative")

    flag_indices = tuple(flag_detector_indices)
    if len(set(flag_indices)) != len(flag_indices):
        raise ValueError("flag_detector_indices must be unique")
    if any(i < 0 or i >= num_internal_detectors for i in flag_indices):
        raise ValueError(
            "flag detector indices must refer to detectors already in the gadget"
        )

    reachable_by_weight = _build_fault_signature_layers(
        unique_mechanisms,
        num_internal_detectors,
        num_final_stabilizers,
        num_logicals,
        flag_indices,
        max_faults,
    )

    first_by_decoder: dict[
        int, tuple[int, tuple[FaultMechanism, ...]]
    ] = {}
    syndrome_mask = (1 << num_final_stabilizers) - 1

    for layer in reachable_by_weight:
        for (decoder_mask, residual_class), history in layer.items():
            previous = first_by_decoder.get(decoder_mask)
            if previous is not None and previous[0] != residual_class:
                first_residual, first_history = previous
                flag_pattern = _bit_tuple(decoder_mask, len(flag_indices))
                measured_syndrome = (
                    decoder_mask >> len(flag_indices)
                ) & syndrome_mask
                return DistinguishableFaultSetFailure(
                    error_type=error_type,
                    flag_pattern=flag_pattern,
                    syndrome_pattern=_bit_tuple(
                        measured_syndrome, num_final_stabilizers
                    ),
                    first_logical_signature=_bit_tuple(
                        first_residual >> num_final_stabilizers, num_logicals
                    ),
                    second_logical_signature=_bit_tuple(
                        residual_class >> num_final_stabilizers, num_logicals
                    ),
                    first_history=FaultHistory(first_history),
                    second_history=FaultHistory(history),
                )
            first_by_decoder.setdefault(
                decoder_mask, (residual_class, history)
            )

    return None


def _classify_higher_order_fault_signatures(
    unique_mechanisms: list[FaultMechanism],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    num_logicals: int,
    flag_detector_indices: list[int] | tuple[int, ...] | range,
    correctable_faults: int,
    detectable_faults: int,
    error_type: str,
) -> tuple[FaultOrderDetectionFailure | None, int, int]:
    """Classifies higher-order signatures as safely decodable or discardable."""
    if correctable_faults < 0:
        raise ValueError("correctable_faults must be non-negative")
    if detectable_faults <= correctable_faults:
        raise ValueError(
            "detectable_faults must be greater than correctable_faults"
        )

    flag_indices = tuple(flag_detector_indices)
    if len(set(flag_indices)) != len(flag_indices):
        raise ValueError("flag_detector_indices must be unique")
    if any(i < 0 or i >= num_internal_detectors for i in flag_indices):
        raise ValueError(
            "flag detector indices must refer to detectors already in the gadget"
        )

    reachable_by_weight = _build_fault_signature_layers(
        unique_mechanisms,
        num_internal_detectors,
        num_final_stabilizers,
        num_logicals,
        flag_indices,
        detectable_faults,
    )

    correction_table: dict[
        int, tuple[int, tuple[FaultMechanism, ...]]
    ] = {}
    for weight in range(correctable_faults + 1):
        for (decoder_mask, residual_class), history in (
            reachable_by_weight[weight].items()
        ):
            existing = correction_table.get(decoder_mask)
            if existing is not None and existing[0] != residual_class:
                raise ValueError(
                    "the correctable fault set is not distinguishable; "
                    "check it before classifying higher-order faults"
                )
            correction_table.setdefault(
                decoder_mask, (residual_class, history)
            )

    safely_decodable = 0
    discardable = 0
    final_syndrome_mask = (1 << num_final_stabilizers) - 1

    for weight in range(correctable_faults + 1, detectable_faults + 1):
        for (decoder_mask, residual_class), history in (
            reachable_by_weight[weight].items()
        ):
            correction = correction_table.get(decoder_mask)
            if correction is None:
                discardable += 1
                continue
            correctable_class, correctable_history = correction
            if correctable_class == residual_class:
                safely_decodable += 1
                continue

            measured_syndrome = (
                decoder_mask >> len(flag_indices)
            ) & final_syndrome_mask
            return (
                FaultOrderDetectionFailure(
                    error_type=error_type,
                    flag_pattern=_bit_tuple(
                        decoder_mask, len(flag_indices)
                    ),
                    syndrome_pattern=_bit_tuple(
                        measured_syndrome, num_final_stabilizers
                    ),
                    correctable_logical_signature=_bit_tuple(
                        correctable_class >> num_final_stabilizers,
                        num_logicals,
                    ),
                    higher_order_logical_signature=_bit_tuple(
                        residual_class >> num_final_stabilizers,
                        num_logicals,
                    ),
                    correctable_history=FaultHistory(correctable_history),
                    higher_order_history=FaultHistory(history),
                ),
                safely_decodable,
                discardable,
            )

    return None, safely_decodable, discardable


def find_fault_order_detection_failure(
    unique_mechanisms: list[FaultMechanism],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    num_logicals: int,
    flag_detector_indices: list[int] | tuple[int, ...] | range,
    correctable_faults: int,
    detectable_faults: int,
    error_type: str,
) -> DistinguishableFaultSetFailure | FaultOrderDetectionFailure | None:
    """Finds an unsafe collision between correctable and higher fault orders.

    A higher-order history is safe when its flag-and-syndrome symptom is absent
    from the correctable table (so it can be discarded), or when it has the
    same residual stabilizer coset as that table entry (so it can be decoded).
    """
    low_order_failure = find_distinguishable_fault_set_failure(
        unique_mechanisms,
        num_internal_detectors,
        num_final_stabilizers,
        num_logicals,
        flag_detector_indices,
        correctable_faults,
        error_type,
    )
    if low_order_failure is not None:
        return low_order_failure

    failure, _, _ = _classify_higher_order_fault_signatures(
        unique_mechanisms,
        num_internal_detectors,
        num_final_stabilizers,
        num_logicals,
        flag_detector_indices,
        correctable_faults,
        detectable_faults,
        error_type,
    )
    return failure


def _pauli_string(
    num_qubits: int,
    support: dict[int, str],
) -> stim.PauliString:
    pauli = stim.PauliString(num_qubits)
    for qubit, pauli_type in support.items():
        pauli[qubit] = pauli_type
    return pauli


def _make_encoded_bell_input(
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    first_reference_qubit: int,
) -> tuple[stim.Circuit, tuple[int, ...], np.ndarray]:
    """Prepares an encoded Bell state that exposes every logical Pauli."""
    L_x = np.atleast_2d(L_x)
    L_z = np.atleast_2d(L_z)
    num_data_qubits = H_x.shape[1]
    if H_z.shape[1] != num_data_qubits:
        raise ValueError("H_x and H_z must act on the same number of qubits")
    if L_x.shape != L_z.shape or L_x.shape[1] != num_data_qubits:
        raise ValueError("L_x and L_z must have matching logical-operator rows")
    if first_reference_qubit < num_data_qubits:
        raise ValueError("reference qubits must follow the data qubits")

    num_logicals = L_x.shape[0]
    reference_qubits = tuple(
        range(first_reference_qubit, first_reference_qubit + num_logicals)
    )
    num_qubits = first_reference_qubit + num_logicals
    logical_pairing = (L_x @ L_z.T) % 2
    stabilizers = []

    for row in H_x:
        stabilizers.append(_pauli_string(
            num_qubits,
            {q: "X" for q, bit in enumerate(row) if bit},
        ))
    for row in H_z:
        stabilizers.append(_pauli_string(
            num_qubits,
            {q: "Z" for q, bit in enumerate(row) if bit},
        ))
    for logical_index, row in enumerate(L_x):
        support = {q: "X" for q, bit in enumerate(row) if bit}
        support[reference_qubits[logical_index]] = "X"
        stabilizers.append(_pauli_string(num_qubits, support))
    for logical_index, row in enumerate(L_z):
        support = {q: "Z" for q, bit in enumerate(row) if bit}
        support.update({
            reference_qubits[reference_index]: "Z"
            for reference_index, bit in enumerate(
                logical_pairing[:, logical_index]
            )
            if bit
        })
        stabilizers.append(_pauli_string(num_qubits, support))

    # Qubits between the data block and references are gadget ancillas. Their
    # initial state is irrelevant because the gadget resets them, but fixing
    # them to |0> makes the tableau fully constrained.
    for qubit in range(num_data_qubits, first_reference_qubit):
        stabilizers.append(_pauli_string(num_qubits, {qubit: "Z"}))

    tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=False,
        allow_underconstrained=False,
    )
    return (
        tableau.to_circuit(method="elimination"),
        reference_qubits,
        logical_pairing,
    )


def _append_bell_sensitive_measurements(
    circuit: stim.Circuit,
    H_check: np.ndarray,
    L_matrix: np.ndarray,
    basis_char: str,
    reference_qubits: tuple[int, ...],
    logical_pairing: np.ndarray,
) -> stim.Circuit:
    """Adds code checks and reference-correlated logical observables."""
    eval_circ = circuit.copy()
    num_data_qubits = H_check.shape[1]
    num_logicals = len(reference_qubits)
    measured_qubits = list(range(num_data_qubits)) + list(reference_qubits)
    num_measurements = len(measured_qubits)
    eval_circ.append(
        "MX" if basis_char == "X" else "M", measured_qubits
    )

    for row in H_check:
        eval_circ.append("DETECTOR", [
            stim.target_rec(-num_measurements + qubit)
            for qubit, bit in enumerate(row)
            if bit
        ])

    for logical_index, row in enumerate(np.atleast_2d(L_matrix)):
        targets = [
            stim.target_rec(-num_measurements + qubit)
            for qubit, bit in enumerate(row)
            if bit
        ]
        if basis_char == "X":
            targets.append(stim.target_rec(-num_logicals + logical_index))
        else:
            targets.extend(
                stim.target_rec(-num_logicals + reference_index)
                for reference_index, bit in enumerate(
                    logical_pairing[:, logical_index]
                )
                if bit
            )
        eval_circ.append("OBSERVABLE_INCLUDE", targets, logical_index)

    return eval_circ


def verify_distinguishable_fault_set(
    circuit: stim.Circuit,
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    max_faults: int,
    *,
    detectable_faults: int | None = None,
    flag_detector_indices: list[int] | tuple[int, ...] | range | None = None,
    flag_measurement_indices: list[int] | tuple[int, ...] | range | None = None,
    ideal_input_circuit: stim.Circuit | None = None,
    prepared_state: str = "0",
    verbose: bool = False,
) -> bool | DistinguishableFaultSetFailure | FaultOrderDetectionFailure:
    """Checks whether flags plus a follow-up syndrome determine a correction.

    This verifies the *distinguishable fault set* condition: for every pair of
    fault histories of size at most ``max_faults``, equal cumulative flags and
    equal residual data syndromes imply stabilizer-equivalent residual errors.

    When ``detectable_faults`` is provided, higher-order histories through that
    count must either have a symptom absent from the correctable table (discard)
    or match the table entry's correction class (safe to decode).

    ``prepared_state`` is ``"0"`` or ``"+"`` for a state-preparation gadget.
    Use ``"arbitrary"`` for a syndrome-extraction gadget; an encoded Bell input
    is then generated automatically so logical errors of either Pauli type are
    observable without fixing the logical data state.

    Faults use the same independent single-Pauli CSS model as ``verify_ftsp``.
    X- and Z-type mechanisms are checked separately.
    """
    if prepared_state not in {"0", "+", "arbitrary"}:
        raise ValueError('prepared_state must be "0", "+", or "arbitrary"')
    if prepared_state == "arbitrary" and ideal_input_circuit is not None:
        raise ValueError(
            "arbitrary-state checking creates its own encoded Bell input"
        )
    if detectable_faults is not None and detectable_faults <= max_faults:
        raise ValueError("detectable_faults must be greater than max_faults")

    annotated_circuit, flag_indices = _resolve_flag_detectors(
        circuit, flag_detector_indices, flag_measurement_indices
    )

    num_internal_detectors = annotated_circuit.num_detectors
    if prepared_state == "arbitrary":
        bell_input, reference_qubits, logical_pairing = _make_encoded_bell_input(
            H_x, H_z, L_x, L_z, annotated_circuit.num_qubits
        )
        basis_checks = (
            ("X", "Z", H_z, np.atleast_2d(L_z)),
            ("Z", "X", H_x, np.atleast_2d(L_x)),
        )
    elif prepared_state == "0":
        basis_checks = (
            ("X", "Z", H_z, np.atleast_2d(L_z)),
            ("Z", "X", H_x, None),
        )
    else:
        basis_checks = (
            ("X", "Z", H_z, None),
            ("Z", "X", H_x, np.atleast_2d(L_x)),
        )

    for error_type, basis_char, H_check, L_matrix in basis_checks:
        if verbose:
            print(
                f"\n  [Distinguishability] {error_type}-error mechanisms "
                f"(up to {max_faults} faults)..."
            )
        noisy_circuit = make_stim_circ_single_type_noisy(
            annotated_circuit.copy(),
            p=0.001,
            error_type=error_type,
        )

        if prepared_state == "arbitrary":
            eval_circuit = _append_bell_sensitive_measurements(
                bell_input + noisy_circuit,
                H_check,
                L_matrix,
                basis_char,
                reference_qubits,
                logical_pairing,
            )
        else:
            eval_input = prepend_ideal_input(
                noisy_circuit, ideal_input_circuit
            )
            eval_circuit = append_ideal_measurements(
                eval_input, H_check, L_matrix, basis_char
            )

        flat_eval_circuit = eval_circuit.flattened()
        try:
            mechanisms = extract_unique_fault_mechanisms(flat_eval_circuit)
        except ValueError as exc:
            if (
                prepared_state == "arbitrary"
                and "non-deterministic observables" in str(exc)
            ):
                raise ValueError(
                    "prepared_state='arbitrary' requires a syndrome-extraction "
                    "gadget that preserves arbitrary encoded states. For a "
                    "state-specific gadget, use prepared_state='0' or '+' and "
                    "supply its ideal_input_circuit."
                ) from exc
            raise
        num_logicals = L_matrix.shape[0] if L_matrix is not None else 0
        failure = find_distinguishable_fault_set_failure(
            mechanisms,
            num_internal_detectors,
            H_check.shape[0],
            num_logicals,
            flag_indices,
            max_faults,
            error_type,
        )
        if failure is not None:
            combined_history = failure.combined_history
            failure = replace(
                failure,
                combined_circuit=extract_deterministic_failure(
                    flat_eval_circuit,
                    list(combined_history.mechanisms),
                    [],
                    basis_char,
                ),
            )
            if verbose:
                print(
                    "    [FAIL] Undetectable logical counterexample: "
                    f"{failure.combined_history.fault_count} faults"
                )
            return failure
        if verbose:
            print("    [PASS] Every flag-and-syndrome symptom has one correction class.")

        if detectable_faults is not None:
            order_failure, safely_decodable, discardable = (
                _classify_higher_order_fault_signatures(
                    mechanisms,
                    num_internal_detectors,
                    H_check.shape[0],
                    num_logicals,
                    flag_indices,
                    max_faults,
                    detectable_faults,
                    error_type,
                )
            )
            if order_failure is not None:
                combined_history = order_failure.combined_history
                order_failure = replace(
                    order_failure,
                    combined_circuit=extract_deterministic_failure(
                        flat_eval_circuit,
                        list(combined_history.mechanisms),
                        [],
                        basis_char,
                    ),
                )
                if verbose:
                    print(
                        "    [FAIL] Undetectable logical counterexample: "
                        f"{order_failure.combined_history.fault_count} faults"
                    )
                return order_failure
            if verbose:
                print(
                    f"    [PASS] Fault orders {max_faults + 1}.."
                    f"{detectable_faults}: {discardable} signatures discard, "
                    f"{safely_decodable} reuse a safe correction."
                )

    return True


def verify_fault_order_detect_or_correct(
    circuit: stim.Circuit,
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    correctable_faults: int,
    detectable_faults: int,
    *,
    flag_detector_indices: list[int] | tuple[int, ...] | range | None = None,
    flag_measurement_indices: list[int] | tuple[int, ...] | range | None = None,
    ideal_input_circuit: stim.Circuit | None = None,
    prepared_state: str = "0",
    verbose: bool = False,
) -> bool | stim.Circuit:
    """Checks low-order correction plus higher-order detection-or-correction.

    For example, ``correctable_faults=1`` and ``detectable_faults=2`` requires
    every zero/one-fault symptom to have a unique correction. Every two-fault
    history must then either produce a new symptom that can be discarded, or
    be compatible with the correction assigned to its zero/one-fault symptom.

    Like :func:`verify_ftsp`, this returns ``True`` on success and a
    deterministic Stim counterexample circuit on failure. The counterexample
    combines the two colliding histories, so their common flag-and-syndrome
    symptom cancels while their incompatible logical residual remains.
    """
    result = verify_distinguishable_fault_set(
        circuit=circuit,
        H_x=H_x,
        H_z=H_z,
        L_x=L_x,
        L_z=L_z,
        max_faults=correctable_faults,
        detectable_faults=detectable_faults,
        flag_detector_indices=flag_detector_indices,
        flag_measurement_indices=flag_measurement_indices,
        ideal_input_circuit=ideal_input_circuit,
        prepared_state=prepared_state,
        verbose=verbose,
    )
    if result is True:
        return result
    if not isinstance(
        result,
        (DistinguishableFaultSetFailure, FaultOrderDetectionFailure),
    ):
        raise AssertionError("verifier returned an invalid result")

    example_circuit = result.example_circuit
    if example_circuit is None:
        raise AssertionError("counterexample circuit was not reconstructed")
    return example_circuit


def _extract_physical_faults(
    flat_eval_circuit: stim.Circuit,
    error_type: str,
) -> tuple[PhysicalFault, ...]:
    """Expands each DEM mechanism into all concrete circuit fault locations."""
    faults = set()
    explanations = flat_eval_circuit.explain_detector_error_model_errors(
        reduce_to_one_representative_error=False
    )
    for explanation in explanations:
        mechanism_targets = []
        for term in explanation.dem_error_terms:
            target = term.dem_target
            if target.is_relative_detector_id():
                mechanism_targets.append(("D", target.val))
            elif target.is_logical_observable_id():
                mechanism_targets.append(("L", target.val))
        mechanism = frozenset(mechanism_targets)
        if not mechanism:
            continue

        for location in explanation.circuit_error_locations:
            instruction_offset = location.stack_frames[0].instruction_offset
            pauli_product = []
            for pauli in location.flipped_pauli_product:
                target = pauli.gate_target
                if target.is_x_target:
                    pauli_type = "X"
                elif target.is_y_target:
                    pauli_type = "Y"
                elif target.is_z_target:
                    pauli_type = "Z"
                else:
                    continue
                pauli_product.append((pauli_type, target.value))
            faults.add(PhysicalFault(
                error_type=error_type,
                instruction_offset=instruction_offset,
                noise_instruction=str(flat_eval_circuit[instruction_offset]),
                pauli_product=tuple(sorted(pauli_product)),
                mechanism=mechanism,
                origin=(
                    "incoming" if location.noise_tag == "incoming" else "se"
                ),
            ))

    return tuple(sorted(
        faults,
        key=lambda fault: (
            fault.instruction_offset,
            fault.pauli_product,
            tuple(sorted(fault.mechanism)),
        ),
    ))


def _make_flagged_fault_history(
    faults: tuple[PhysicalFault, ...],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    num_total_stabilizers: int,
    syndrome_offset: int,
    num_logicals: int,
    flag_indices: tuple[int, ...],
) -> FlaggedFaultHistory:
    internal_mask = 0
    syndrome_mask = 0
    logical_mask = 0
    for fault in faults:
        for target_type, target_index in fault.mechanism:
            if target_type == "D":
                if target_index < num_internal_detectors:
                    internal_mask ^= 1 << target_index
                else:
                    final_index = target_index - num_internal_detectors
                    if final_index < num_final_stabilizers:
                        syndrome_mask ^= 1 << final_index
            elif target_type == "L" and target_index < num_logicals:
                logical_mask ^= 1 << target_index

    combined_syndrome_mask = syndrome_mask << syndrome_offset
    return FlaggedFaultHistory(
        error_type=faults[0].error_type,
        faults=faults,
        flag_pattern=tuple(
            (internal_mask >> detector_index) & 1
            for detector_index in flag_indices
        ),
        syndrome_pattern=_bit_tuple(
            combined_syndrome_mask, num_total_stabilizers
        ),
        logical_signature=_bit_tuple(logical_mask, num_logicals),
    )


def _minimum_stabilizer_separators(
    one_fault_histories: tuple[FlaggedFaultHistory, ...],
    two_fault_histories: tuple[FlaggedFaultHistory, ...],
    stabilizer_checks: tuple[StabilizerCheck, ...],
) -> tuple[
    tuple[tuple[StabilizerCheck, ...], ...],
    tuple[FlaggedFaultHistory, FlaggedFaultHistory] | None,
]:
    """Finds every minimum check subset separating one- and two-fault orders."""
    if not one_fault_histories or not two_fault_histories:
        return ((),), None

    def syndrome_mask(history: FlaggedFaultHistory) -> int:
        return sum(
            bit << index
            for index, bit in enumerate(history.syndrome_pattern)
        )

    one_by_syndrome: dict[int, FlaggedFaultHistory] = {}
    for history in one_fault_histories:
        one_by_syndrome.setdefault(syndrome_mask(history), history)
    two_by_syndrome: dict[int, FlaggedFaultHistory] = {}
    for history in two_fault_histories:
        two_by_syndrome.setdefault(syndrome_mask(history), history)

    shared_syndromes = one_by_syndrome.keys() & two_by_syndrome.keys()
    if shared_syndromes:
        shared = min(shared_syndromes)
        return (), (one_by_syndrome[shared], two_by_syndrome[shared])

    difference_masks = {
        one_syndrome ^ two_syndrome
        for one_syndrome in one_by_syndrome
        for two_syndrome in two_by_syndrome
    }
    for subset_size in range(1, len(stabilizer_checks) + 1):
        solutions = []
        for indices in itertools.combinations(
            range(len(stabilizer_checks)), subset_size
        ):
            selected_mask = sum(1 << index for index in indices)
            if all(selected_mask & difference for difference in difference_masks):
                solutions.append(tuple(
                    stabilizer_checks[index] for index in indices
                ))
        if solutions:
            return tuple(solutions), None

    raise AssertionError("full stabilizer syndrome failed to separate disjoint sets")


def _find_incompatible_cross_order_correction(
    one_fault_histories: tuple[FlaggedFaultHistory, ...],
    two_fault_histories: tuple[FlaggedFaultHistory, ...],
) -> tuple[FlaggedFaultHistory, FlaggedFaultHistory] | None:
    """Finds equal-syndrome histories needing different CSS correction classes."""
    one_by_key: dict[
        tuple[str, tuple[int, ...]],
        dict[tuple[int, ...], FlaggedFaultHistory],
    ] = {}
    for history in one_fault_histories:
        key = (history.error_type, history.syndrome_pattern)
        one_by_key.setdefault(key, {}).setdefault(
            history.logical_signature, history
        )
    two_by_key: dict[
        tuple[str, tuple[int, ...]],
        dict[tuple[int, ...], FlaggedFaultHistory],
    ] = {}
    for history in two_fault_histories:
        key = (history.error_type, history.syndrome_pattern)
        two_by_key.setdefault(key, {}).setdefault(
            history.logical_signature, history
        )

    for key in one_by_key.keys() & two_by_key.keys():
        for one_signature, one_history in one_by_key[key].items():
            for two_signature, two_history in two_by_key[key].items():
                if one_signature != two_signature:
                    return one_history, two_history
    return None


def classify_flag_patterns_by_fault_order(
    circuit: stim.Circuit,
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    *,
    flag_detector_indices: list[int] | tuple[int, ...] | range | None = None,
    flag_measurement_indices: list[int] | tuple[int, ...] | range | None = None,
    ideal_input_circuit: stim.Circuit | None = None,
    prepared_state: str = "0",
) -> dict[tuple[int, ...], FlagPatternFaultAnalysis]:
    """Returns concrete histories and minimum ideal order-discriminating checks.

    Each mapping value contains every concrete zero-, one-, and two-fault
    history, including faults on the incoming data block and faults inside SE,
    found
    in the separate X- and Z-error CSS analyses. Stabilizer checks are named
    ``("Z", i)`` for row ``i`` of ``H_z`` and ``("X", i)`` for row ``i`` of
    ``H_x``. An empty tuple of minimum check sets means that even the complete
    ideal syndrome cannot distinguish one fault from two faults; the included
    ``indistinguishable_order_pair`` is a witness. ``((),)`` means no checks are
    needed because the flag pattern occurs at only one of the two fault orders.
    ``correction_is_well_defined`` separately reports whether equal-syndrome
    histories remain in the same correction class within each CSS channel.
    """
    if prepared_state not in {"0", "+", "arbitrary"}:
        raise ValueError('prepared_state must be "0", "+", or "arbitrary"')
    if prepared_state == "arbitrary" and ideal_input_circuit is not None:
        raise ValueError(
            "arbitrary-state checking creates its own encoded Bell input"
        )

    annotated_circuit, flag_indices = _resolve_flag_detectors(
        circuit, flag_detector_indices, flag_measurement_indices
    )

    num_internal_detectors = annotated_circuit.num_detectors
    num_z_stabilizers = H_z.shape[0]
    num_x_stabilizers = H_x.shape[0]
    num_total_stabilizers = num_z_stabilizers + num_x_stabilizers
    stabilizer_checks = tuple(
        [("Z", index) for index in range(num_z_stabilizers)]
        + [("X", index) for index in range(num_x_stabilizers)]
    )
    if prepared_state == "arbitrary":
        bell_input, reference_qubits, logical_pairing = _make_encoded_bell_input(
            H_x, H_z, L_x, L_z, annotated_circuit.num_qubits
        )
        basis_checks = (
            ("X", "Z", H_z, np.atleast_2d(L_z), 0),
            ("Z", "X", H_x, np.atleast_2d(L_x), num_z_stabilizers),
        )
    elif prepared_state == "0":
        basis_checks = (
            ("X", "Z", H_z, np.atleast_2d(L_z), 0),
            ("Z", "X", H_x, None, num_z_stabilizers),
        )
    else:
        basis_checks = (
            ("X", "Z", H_z, None, 0),
            ("Z", "X", H_x, np.atleast_2d(L_x), num_z_stabilizers),
        )

    histories_by_pattern: dict[
        tuple[int, ...], dict[int, list[FlaggedFaultHistory]]
    ] = {}

    for error_type, basis_char, H_check, L_matrix, syndrome_offset in basis_checks:
        noisy_circuit = make_stim_circ_single_type_noisy(
            annotated_circuit.copy(),
            p=0.001,
            error_type=error_type,
        )
        incoming_noise = stim.Circuit()
        incoming_noise.append(
            f"{error_type}_ERROR",
            range(H_check.shape[1]),
            0.001,
            tag="incoming",
        )
        if prepared_state == "arbitrary":
            eval_circuit = _append_bell_sensitive_measurements(
                bell_input + incoming_noise + noisy_circuit,
                H_check,
                L_matrix,
                basis_char,
                reference_qubits,
                logical_pairing,
            )
        else:
            eval_circuit = append_ideal_measurements(
                prepend_ideal_input(
                    incoming_noise + noisy_circuit, ideal_input_circuit
                ),
                H_check,
                L_matrix,
                basis_char,
            )

        flat_eval_circuit = eval_circuit.flattened()
        physical_faults = _extract_physical_faults(
            flat_eval_circuit, error_type
        )
        num_logicals = L_matrix.shape[0] if L_matrix is not None else 0
        for fault_order in (1, 2):
            for faults in itertools.combinations(physical_faults, fault_order):
                history = _make_flagged_fault_history(
                    faults,
                    num_internal_detectors,
                    H_check.shape[0],
                    num_total_stabilizers,
                    syndrome_offset,
                    num_logicals,
                    tuple(flag_indices),
                )
                histories_by_pattern.setdefault(
                    history.flag_pattern, {0: [], 1: [], 2: []}
                )[fault_order].append(history)

    zero_pattern = (0,) * len(flag_indices)
    zero_history = FlaggedFaultHistory(
        error_type="I",
        faults=(),
        flag_pattern=zero_pattern,
        syndrome_pattern=(0,) * num_total_stabilizers,
        logical_signature=(),
    )
    histories_by_pattern.setdefault(
        zero_pattern, {0: [], 1: [], 2: []}
    )[0].append(zero_history)

    analyses = {}
    for pattern, histories in sorted(histories_by_pattern.items()):
        one_fault_histories = tuple(histories[1])
        two_fault_histories = tuple(histories[2])
        minimum_sets, witness = _minimum_stabilizer_separators(
            one_fault_histories,
            two_fault_histories,
            stabilizer_checks,
        )
        incompatible_correction_pair = (
            _find_incompatible_cross_order_correction(
                one_fault_histories, two_fault_histories
            )
        )
        analyses[pattern] = FlagPatternFaultAnalysis(
            flag_pattern=pattern,
            one_fault_histories=one_fault_histories,
            two_fault_histories=two_fault_histories,
            minimum_distinguishing_stabilizer_sets=minimum_sets,
            zero_fault_histories=tuple(histories[0]),
            indistinguishable_order_pair=witness,
            incompatible_correction_pair=incompatible_correction_pair,
        )
    return analyses


def _correction_class_matrix(
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    prepared_state: str,
) -> tuple[np.ndarray, int, int]:
    n = H_x.shape[1]
    zero_z = np.zeros((H_z.shape[0], n), dtype=np.uint8)
    zero_x = np.zeros((H_x.shape[0], n), dtype=np.uint8)
    rows = [
        np.hstack([H_z % 2, zero_z]),
        np.hstack([zero_x, H_x % 2]),
    ]
    tracked_l_z = (
        np.atleast_2d(L_z) % 2
        if prepared_state in {"0", "arbitrary"}
        else np.zeros((0, n), dtype=np.uint8)
    )
    tracked_l_x = (
        np.atleast_2d(L_x) % 2
        if prepared_state in {"+", "arbitrary"}
        else np.zeros((0, n), dtype=np.uint8)
    )
    if tracked_l_z.size:
        rows.append(np.hstack([
            tracked_l_z,
            np.zeros_like(tracked_l_z),
        ]))
    if tracked_l_x.size:
        rows.append(np.hstack([
            np.zeros_like(tracked_l_x),
            tracked_l_x,
        ]))
    return np.vstack(rows).astype(np.uint8), len(tracked_l_z), len(tracked_l_x)


def _history_class_signature(
    history: FlaggedFaultHistory,
    num_x_logicals: int,
    num_z_logicals: int,
) -> tuple[int, ...]:
    if history.error_type == "X":
        logical = history.logical_signature + (0,) * num_z_logicals
    elif history.error_type == "Z":
        logical = (0,) * num_x_logicals + history.logical_signature
    else:
        logical = (0,) * (num_x_logicals + num_z_logicals)
    return history.syndrome_pattern + logical


def _solve_binary_linear_system(
    matrix: np.ndarray,
    target: tuple[int, ...],
) -> np.ndarray:
    augmented = np.hstack([
        matrix.copy().astype(np.uint8),
        np.asarray(target, dtype=np.uint8).reshape(-1, 1),
    ])
    pivot_columns = []
    pivot_row = 0
    for column in range(matrix.shape[1]):
        candidates = np.flatnonzero(augmented[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        augmented[[pivot_row, selected]] = augmented[[selected, pivot_row]]
        for row in range(matrix.shape[0]):
            if row != pivot_row and augmented[row, column]:
                augmented[row] ^= augmented[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == matrix.shape[0]:
            break
    if any(
        not augmented[row, :-1].any() and augmented[row, -1]
        for row in range(matrix.shape[0])
    ):
        raise ValueError("correction signature is outside the CSS class space")
    solution = np.zeros(matrix.shape[1], dtype=np.uint8)
    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row, -1]
    return solution


def _pauli_from_solution(solution: np.ndarray) -> PauliCorrection:
    n = len(solution) // 2
    return PauliCorrection(
        x_support=tuple(np.flatnonzero(solution[:n])),
        z_support=tuple(np.flatnonzero(solution[n:])),
    )


def _unit_paulis_by_signature(
    class_matrix: np.ndarray,
) -> dict[tuple[int, ...], PauliCorrection]:
    n = class_matrix.shape[1] // 2
    unit_paulis = [PauliCorrection()]
    for qubit in range(n):
        unit_paulis.extend([
            PauliCorrection(x_support=(qubit,)),
            PauliCorrection(z_support=(qubit,)),
            PauliCorrection(x_support=(qubit,), z_support=(qubit,)),
        ])
    result: dict[tuple[int, ...], PauliCorrection] = {}
    for pauli in unit_paulis:
        vector = np.zeros(2 * n, dtype=np.uint8)
        vector[list(pauli.x_support)] = 1
        vector[[n + q for q in pauli.z_support]] = 1
        signature = tuple((class_matrix @ vector) % 2)
        result.setdefault(signature, pauli)
    return result


def _find_acceptable_correction(
    histories: tuple[FlaggedFaultHistory, ...],
    class_matrix: np.ndarray,
    num_x_logicals: int,
    num_z_logicals: int,
    unit_by_signature: dict[tuple[int, ...], PauliCorrection] | None = None,
    correction_cache: dict[tuple[int, ...], PauliCorrection] | None = None,
) -> tuple[PauliCorrection, tuple[CorrectionOutcome, ...]] | None:
    if unit_by_signature is None:
        unit_by_signature = _unit_paulis_by_signature(class_matrix)
    if correction_cache is None:
        correction_cache = {}

    acceptable_signatures = None
    history_signatures = []
    for history in histories:
        signature = _history_class_signature(
            history, num_x_logicals, num_z_logicals
        )
        history_signatures.append(signature)
        residuals = (
            unit_by_signature.keys()
            if history.maximum_acceptable_residual_weight == 1
            else [tuple(0 for _ in signature)]
        )
        allowed = {
            tuple(a ^ b for a, b in zip(signature, residual))
            for residual in residuals
        }
        acceptable_signatures = (
            allowed
            if acceptable_signatures is None
            else acceptable_signatures & allowed
        )
        if not acceptable_signatures:
            return None

    candidates = []
    for signature in acceptable_signatures or ():
        correction = correction_cache.get(signature)
        if correction is None:
            solution = _solve_binary_linear_system(class_matrix, signature)
            correction = _pauli_from_solution(solution)
            correction_cache[signature] = correction
        candidates.append((correction, signature))
    correction, correction_signature = min(
        candidates,
        key=lambda item: (item[0].weight, item[0].pauli_product),
    )
    outcomes = []
    for history, signature in zip(histories, history_signatures):
        residual_signature = tuple(
            a ^ b for a, b in zip(signature, correction_signature)
        )
        outcomes.append(CorrectionOutcome(
            history=history,
            final_error=unit_by_signature[residual_signature],
        ))
    return correction, tuple(outcomes)


def plan_minimal_followup_by_flag(
    analyses: dict[tuple[int, ...], FlagPatternFaultAnalysis],
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    prepared_state: str = "0",
) -> dict[tuple[int, ...], FlagPatternFollowupPlan]:
    """Applies a conservative Gottesman-style follow-up policy per flag.

    A correction is applied immediately when it leaves weight zero for the
    no-fault and lone-incoming-fault cases, and at most weight one for SE-fault
    or total-two-fault cases. Otherwise, any pattern that can arise from one
    fault uses a 1-FT follow-up SE. A two-fault-only pattern uses a non-FT
    follow-up when at least one syndrome outcome admits such a correction,
    with unacceptable outcomes marked for discard; if none do, the state is
    discarded immediately.

    This chooses the required *kind* of branch circuit. The concrete adaptive
    composition must still be verified with incoming errors and branch-circuit
    faults to certify the full ``s``-before/``r``-during Gottesman criterion.
    """
    class_matrix, num_x_logicals, num_z_logicals = _correction_class_matrix(
        H_x, H_z, L_x, L_z, prepared_state
    )
    unit_by_signature = _unit_paulis_by_signature(class_matrix)
    correction_cache: dict[tuple[int, ...], PauliCorrection] = {}
    plans = {}
    for pattern, analysis in sorted(analyses.items()):
        histories = (
            analysis.zero_fault_histories
            + analysis.one_fault_histories
            + analysis.two_fault_histories
        )
        acceptable = _find_acceptable_correction(
            histories,
            class_matrix,
            num_x_logicals,
            num_z_logicals,
            unit_by_signature,
            correction_cache,
        )
        if acceptable is not None:
            correction, outcomes = acceptable
            plans[pattern] = FlagPatternFollowupPlan(
                flag_pattern=pattern,
                action=FollowupAction.CORRECT_NOW,
                possible_fault_orders=analysis.possible_fault_orders,
                correction_without_followup=correction,
                correction_outcomes=outcomes,
            )
            continue

        if 1 in analysis.possible_fault_orders:
            plans[pattern] = FlagPatternFollowupPlan(
                flag_pattern=pattern,
                action=FollowupAction.RUN_ONE_FT_SE,
                possible_fault_orders=analysis.possible_fault_orders,
            )
            continue

        histories_by_syndrome: dict[
            tuple[int, ...], list[FlaggedFaultHistory]
        ] = {}
        for history in histories:
            histories_by_syndrome.setdefault(
                history.syndrome_pattern, []
            ).append(history)
        uniquely_correctable = []
        ambiguous = []
        for syndrome, syndrome_histories in sorted(histories_by_syndrome.items()):
            syndrome_correction = _find_acceptable_correction(
                tuple(syndrome_histories),
                class_matrix,
                num_x_logicals,
                num_z_logicals,
                unit_by_signature,
                correction_cache,
            )
            if syndrome_correction is None:
                ambiguous.append(syndrome)
            else:
                uniquely_correctable.append((syndrome, syndrome_correction[0]))

        if uniquely_correctable:
            action = FollowupAction.RUN_NON_FT_SE
        else:
            action = FollowupAction.DISCARD

        plans[pattern] = FlagPatternFollowupPlan(
            flag_pattern=pattern,
            action=action,
            possible_fault_orders=analysis.possible_fault_orders,
            uniquely_correctable_syndromes=tuple(uniquely_correctable),
            ambiguous_syndromes=tuple(ambiguous),
        )
    return plans


def print_followup_flow(
    plans: dict[tuple[int, ...], FlagPatternFollowupPlan],
    *,
    explain_corrections: bool = False,
) -> None:
    """Prints ``flag_pattern -> action`` with optional correction outcomes."""
    for pattern, plan in plans.items():
        print("".join(map(str, pattern)), "->", plan.action.value)
        if not explain_corrections or plan.correction_without_followup is None:
            continue
        correction = plan.correction_without_followup
        print(
            "  correction:", correction.pauli_product or "I",
            f"(weight {correction.weight})",
        )
        outcome_counts: dict[
            tuple[int, int, tuple[tuple[str, int], ...], int], int
        ] = {}
        for outcome in plan.correction_outcomes:
            key = (
                outcome.history.incoming_fault_count,
                outcome.history.se_fault_count,
                outcome.final_error.pauli_product,
                outcome.weight,
            )
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
        for (
            incoming_faults,
            se_faults,
            final_error,
            weight,
        ), count in sorted(outcome_counts.items()):
            print(
                "  final state:", final_error or "I",
                f"weight={weight}",
                f"incoming={incoming_faults}",
                f"se={se_faults}",
                f"histories={count}",
            )


def extract_unique_fault_mechanisms(
    eval_circ: stim.Circuit
) -> list[FaultMechanism]:
    """Flattens circuit, extracts DEM, and returns a list of unique fault target combinations."""
    flat_circ = eval_circ.flattened()
    dem = flat_circ.detector_error_model(decompose_errors=False)

    fault_mechanisms = set()
    for instruction in dem:
        if instruction.type == "error":
            targets = []
            for tgt in instruction.targets_copy():
                if tgt.is_relative_detector_id():
                    targets.append(("D", tgt.val))
                elif tgt.is_logical_observable_id():
                    targets.append(("L", tgt.val))
            if targets:
                fault_mechanisms.add(frozenset(targets))

    return sorted(fault_mechanisms, key=lambda mechanism: tuple(sorted(mechanism)))

# ==============================================================================
# PHASE 3: Extraction & Rebuild (Direct Noisy Index Mapping)
# ==============================================================================
def extract_deterministic_failure(
    flat_eval_circ: stim.Circuit,
    failed_mechs: list[FaultMechanism],
    failed_y_indices: list[int],
    basis_char: str,
) -> stim.Circuit:
    """Replace selected noise mechanisms with a deterministic witness."""
    dem_filter = stim.DetectorErrorModel()
    for mech in failed_mechs:
        targets = []
        for t_type, t_val in sorted(mech):
            if t_type == "D":
                targets.append(stim.target_relative_detector_id(t_val))
            elif t_type == "L":
                targets.append(stim.target_logical_observable_id(t_val))
        dem_filter.append('error', [1.0], targets)

    explanations = flat_eval_circ.explain_detector_error_model_errors(
        dem_filter=dem_filter,
        reduce_to_one_representative_error=True,
    )

    faults_to_inject: dict[int, list[tuple[str, int]]] = {}
    for exp in explanations:
        loc = exp.circuit_error_locations[0]
        ins_offset = loc.stack_frames[0].instruction_offset

        for p in loc.flipped_pauli_product:
            target = p.gate_target if hasattr(p, "gate_target") else p
            if target.is_x_target:
                pauli_type = "X"
            elif target.is_y_target:
                pauli_type = "Y"
            elif target.is_z_target:
                pauli_type = "Z"
            else:
                continue
            faults_to_inject.setdefault(ins_offset, []).append(
                (pauli_type, target.value)
            )

    completion_index = None
    if failed_y_indices:
        measurement_name = "M" if basis_char == "Z" else "MX"
        required_qubits = set(failed_y_indices)
        for index in range(len(flat_eval_circ) - 1, -1, -1):
            instruction = flat_eval_circ[index]
            measured_qubits = {
                target.value
                for target in instruction.targets_copy()
                if target.is_qubit_target
            }
            if (
                instruction.name == measurement_name
                and required_qubits <= measured_qubits
            ):
                completion_index = index
                break
        if completion_index is None:
            raise ValueError(
                f"could not find the final {measurement_name} data measurement"
            )

    failed_circ = stim.Circuit()
    for idx, inst in enumerate(flat_eval_circ):
        if idx == completion_index:
            completion_pauli = "X" if basis_char == "Z" else "Z"
            failed_circ.append(completion_pauli, failed_y_indices)

        if idx in faults_to_inject:
            for p_type, p_val in faults_to_inject[idx]:
                failed_circ.append(p_type, [p_val])
            del faults_to_inject[idx]
            continue

        if inst.name in NOISE_GATES:
            continue

        failed_circ.append(inst)

    if faults_to_inject:
        raise AssertionError(
            f"failed to place faults at instruction offsets {sorted(faults_to_inject)}"
        )

    return failed_circ


# ==============================================================================
# COMBINATORICAL Z FAULT VERIFIER
# ==============================================================================
def precompute_conjugate_syndromes(H_check: np.ndarray, t: int) -> dict[int, int]:
    """Precomputes the minimum weight data error required to satisfy a given syndrome."""
    valid_syndromes: dict[int, int] = {}
    n_qubits = H_check.shape[1]

    for w in range(t + 1):
        for combo in itertools.combinations(range(n_qubits), w):
            err = np.zeros(n_qubits, dtype=int)
            if w > 0:
                err[list(combo)] = 1

            # Convert binary array syndrome to a fast integer bitmask
            syn_array = (H_check @ err) % 2
            syn_int = sum(int(val) << i for i, val in enumerate(syn_array))

            if syn_int not in valid_syndromes or valid_syndromes[syn_int] > w:
                valid_syndromes[syn_int] = w

    return valid_syndromes


def solve_ftsp_combinatorial_fast(
    unique_mechanisms: list[FaultMechanism],
    num_internal_detectors: int,
    H_check: np.ndarray,
    t: int,
) -> bool | list[FaultMechanism]:
    """Evaluates the conjugate basis using a Bitwise Breadth-First Search (BFS)."""
    valid_syndromes = precompute_conjugate_syndromes(H_check, t)

    # Convert DEM mechanisms into fast integer bitmasks
    mech_masks = []
    for mech in unique_mechanisms:
        int_mask = 0
        ext_mask = 0
        for t_type, t_val in mech:
            if t_type == "D":
                if t_val < num_internal_detectors:
                    int_mask ^= (1 << t_val)
                else:
                    ext_mask ^= (1 << (t_val - num_internal_detectors))
        mech_masks.append((int_mask, ext_mask, mech))

    # BFS State Tracker: {(internal_mask, external_mask): [path_of_mechanisms]}
    reachable_states: dict[
        tuple[int, int], list[FaultMechanism]
    ] = {(0, 0): []}

    for k in range(1, t + 1):
        next_states = {}

        for (curr_int, curr_ext), path in reachable_states.items():
            for m_int, m_ext, mech in mech_masks:
                new_int = curr_int ^ m_int
                new_ext = curr_ext ^ m_ext
                state_key = (new_int, new_ext)

                # Prune degenerate physical faults that produce the same syndrome
                if state_key in reachable_states or state_key in next_states:
                    continue

                new_path = path + [mech]
                next_states[state_key] = new_path

                # Check FTSP Bounds
                if new_int == 0:  # Evades FTSP internal flags
                    if new_ext not in valid_syndromes or valid_syndromes[new_ext] > k:
                        return new_path  # Return the catastrophic cascade

        reachable_states.update(next_states)

    return True


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================
def precompute_primary_syndromes(
    H_check: np.ndarray,
    L_matrix: np.ndarray | None,
    d: int,
) -> dict[int, tuple[int, list[int]]]:
    """Precomputes the minimum weight data error required to produce a combined (syndrome, logical) signature."""
    valid_syndromes: dict[int, tuple[int, list[int]]] = {}
    n_qubits = H_check.shape[1]

    if L_matrix is not None and L_matrix.size > 0:
        combined_matrix = np.vstack([H_check, L_matrix])
    else:
        combined_matrix = H_check

    for w in range(d):
        for combo in itertools.combinations(range(n_qubits), w):
            err = np.zeros(n_qubits, dtype=int)
            if w > 0:
                err[list(combo)] = 1

            syn_array = (combined_matrix @ err) % 2
            syn_int = sum(int(val) << i for i, val in enumerate(syn_array))

            if syn_int not in valid_syndromes or valid_syndromes[syn_int][0] > w:
                valid_syndromes[syn_int] = (w, list(combo))

    return valid_syndromes


def solve_ftsp_combinatorial_primary_fast(
    unique_mechanisms: list[FaultMechanism],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    H_check: np.ndarray,
    L_matrix: np.ndarray | None,
    d: int,
    t: int,
) -> bool | tuple[list[FaultMechanism], list[int]]:
    """Evaluates the primary basis using a Bitwise Breadth-First Search (BFS)."""
    valid_syndromes = precompute_primary_syndromes(H_check, L_matrix, d)

    # Convert DEM mechanisms into fast integer bitmasks
    mech_masks = []
    for mech in unique_mechanisms:
        int_mask = 0
        ext_mask = 0
        for t_type, t_val in mech:
            if t_type == "D":
                if t_val < num_internal_detectors:
                    int_mask ^= (1 << t_val)
                else:
                    ext_mask ^= (1 << (t_val - num_internal_detectors))
            elif t_type == "L":
                ext_mask ^= (1 << (num_final_stabilizers + t_val))
        mech_masks.append((int_mask, ext_mask, mech))

    # BFS State Tracker: {(internal_mask, external_mask): [path_of_mechanisms]}
    reachable_states: dict[
        tuple[int, int], list[FaultMechanism]
    ] = {(0, 0): []}

    for k in range(1, t + 1):
        next_states = {}

        for (curr_int, curr_ext), path in reachable_states.items():
            for m_int, m_ext, mech in mech_masks:
                new_int = curr_int ^ m_int
                new_ext = curr_ext ^ m_ext
                state_key = (new_int, new_ext)

                # Prune degenerate physical faults that produce the same syndrome
                if state_key in reachable_states or state_key in next_states:
                    continue

                new_path = path + [mech]
                next_states[state_key] = new_path

                # Check FTSP Bounds
                if new_int == 0:  # Evades FTSP internal flags
                    stab_mask = (1 << num_final_stabilizers) - 1
                    for syn_data, (w_data, err_indices) in valid_syndromes.items():
                        if k + w_data <= d - 1:
                            if (new_ext ^ syn_data) & stab_mask == 0:
                                if (new_ext ^ syn_data) >> num_final_stabilizers != 0:
                                    return new_path, err_indices

        reachable_states.update(next_states)

    return True


def _make_noisy_gadget(
    circuit: stim.Circuit,
    measurement_basis: str,
) -> stim.Circuit:
    if measurement_basis not in {"X", "Z"}:
        raise ValueError('measurement basis must be "X" or "Z"')
    error_type = "X" if measurement_basis == "Z" else "Z"
    return make_stim_circ_single_type_noisy(
        circuit.copy(), p=0.001, error_type=error_type
    )


def verify_ftsp_primary_exact(
    prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_op: np.ndarray | None,
    d: int,
    t: int,
    basis_char: str,
    verbose: bool = False,
    ideal_input_circuit: stim.Circuit | None = None,
) -> bool | stim.Circuit:
    """Verifies the primary FTSP error basis using the Fast Combinatorial Tracker.

    If supplied, ``ideal_input_circuit`` prepares the encoded input noiselessly;
    faults are inserted only into ``prep_circ``.
    """
    if verbose:
        print(
            f"\n  [Primary] Evaluating {basis_char}-basis via "
            f"Combinatorics (d={d}, t={t})..."
        )

    noisy_prep = _make_noisy_gadget(prep_circ, basis_char)

    num_internal_detectors = noisy_prep.num_detectors
    num_final_stabilizers = H_check.shape[0]

    L_matrix = np.atleast_2d(L_op) if L_op is not None else None

    # Step 1: Prep and Flatten
    eval_input = prepend_ideal_input(noisy_prep, ideal_input_circuit)
    eval_circ = append_ideal_measurements(
        eval_input, H_check, L_matrix, basis_char
    )
    flat_eval_circ = eval_circ.flattened()

    unique_mechanisms = extract_unique_fault_mechanisms(flat_eval_circ)

    # Step 2: Solve the Math
    result = solve_ftsp_combinatorial_primary_fast(
        unique_mechanisms,
        num_internal_detectors,
        num_final_stabilizers,
        H_check,
        L_matrix,
        d,
        t,
    )

    # Step 3: Extract the Verdict
    if isinstance(result, tuple):
        failed_mechs, failed_y_indices = result
        if verbose:
            print("    [FAIL] Catastrophic cascade found!")
            print(
                f"    W(E_init)={len(failed_mechs)} faults bypassed flags "
                f"to require only W(E_data)={len(failed_y_indices)} to fail."
            )
            print("    Extracting unified failure circuit (Prep Faults + Data Faults)...")

        # Pass E_data indices and basis to extraction function
        fault_example = extract_deterministic_failure(
            flat_eval_circ,
            failed_mechs,
            failed_y_indices,
            basis_char,
        )

        return fault_example
    if verbose:
        print("    [PASS] No uncorrectable conjugate cascades found.")
    return True


def verify_ftsp_conjugate_exact(
    prep_circ: stim.Circuit,
    H_check: np.ndarray,
    t: int,
    basis_char: str,
    verbose: bool = False,
    ideal_input_circuit: stim.Circuit | None = None,
) -> bool | stim.Circuit:
    """Verifies the conjugate FTSP error basis using the Fast Combinatorial Tracker.

    If supplied, ``ideal_input_circuit`` prepares the encoded input noiselessly;
    faults are inserted only into ``prep_circ``.
    """
    if verbose:
        print(
            f"\n  [Conjugate] Evaluating {basis_char}-basis via "
            f"Combinatorics (t={t})..."
        )

    noisy_prep = _make_noisy_gadget(prep_circ, basis_char)

    num_internal_detectors = noisy_prep.num_detectors
    # Conjugate tracking does not use L_matrix; we track pure syndrome mass
    eval_input = prepend_ideal_input(noisy_prep, ideal_input_circuit)
    eval_circ = append_ideal_measurements(eval_input, H_check, None, basis_char)
    flat_eval_circ = eval_circ.flattened()

    unique_mechanisms = extract_unique_fault_mechanisms(flat_eval_circ)

    result = solve_ftsp_combinatorial_fast(
        unique_mechanisms, num_internal_detectors, H_check, t
    )

    if isinstance(result, list):
        if verbose:
            print(f"    [FAIL] Bad Conjugate Cascade Detected!")
            print("    Extracting deterministic failure circuit...")

        fault_example = extract_deterministic_failure(
            flat_eval_circ, result, [], basis_char
        )

        return fault_example

    if verbose:
        print("    [PASS] No uncorrectable conjugate cascades found.")
    return True


def verify_ftsp(
    prep_circ: stim.Circuit,
    H_primary: np.ndarray,
    L_primary: np.ndarray,
    H_conjugate: np.ndarray,
    d: int,
    t: int,
    primary_basis: str = "Z",
    conjugate_basis: str = "X",
    verbose: bool = False,
    ideal_input_circuit: stim.Circuit | None = None,
) -> bool | stim.Circuit:
    """Comprehensive FT verification of a preparation or encoded-input gadget.

    ``ideal_input_circuit`` is excluded from the fault set. Use it to prepare a
    deterministic encoded state when ``prep_circ`` is only the gadget under test.
    """
    res_primary = verify_ftsp_primary_exact(
        prep_circ, H_primary, L_primary, d, t,
        basis_char=primary_basis,
        verbose=verbose,
        ideal_input_circuit=ideal_input_circuit,
    )
    if res_primary is not True:
        return res_primary

    res_conjugate = verify_ftsp_conjugate_exact(
        prep_circ, H_conjugate, t,
        basis_char=conjugate_basis,
        verbose=verbose,
        ideal_input_circuit=ideal_input_circuit,
    )
    if res_conjugate is not True:
        return res_conjugate

    return True

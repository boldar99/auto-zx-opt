"""Check data-error and syndrome-information flow for adaptive basis choice.

This is a single-fault diagnostic on the saved, basis-lowered schedules, not
a synthesis or certification of a complete adaptive EC protocol.
"""

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import stim

from spiderwarp.csscode import CSSCode
from spiderwarp._se_circuit import (
    PauliSpace, SEGadget, dual_circuit, extract_readout, lower_to_guppy_gates,
    sample_annotation_parities, validate_gadget,
)
from spiderwarp.fault_tolerance_verification import (
    _append_bell_sensitive_measurements, _make_encoded_bell_input,
)


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    path = root / 'spiderwarp/assets/circuits/SECircuits/32_20_4_d4.stim'
    code = CSSCode.load_code('misc', '32_20_4')
    space = PauliSpace(code)
    raw = stim.Circuit.from_file(path)
    readout = extract_readout(raw, code)
    gadgets = {}
    for basis, circuit in (('Z', dual_circuit(raw)), ('X', raw)):
        gadget = SEGadget(4, basis, lower_to_guppy_gates(readout.annotate(circuit)),
                          readout, space)
        validate_gadget(gadget)
        gadgets[basis] = gadget
    primary = gadgets['Z']
    flag_sources = [set() for _ in range(primary.num_flags)]
    ancilla_x_outputs = set()
    ancilla_syndromes_by_flags = defaultdict(set)
    readout_only_faults = defaultdict(dict)
    examples = {}

    def printed(flags):
        return ''.join(str((flags >> i) & 1) for i in range(primary.num_flags))

    def describe(effect, fault):
        flags, syndrome, residual = primary.unpack(effect)
        return dict(fault=asdict(fault), gate=primary.operations[fault.operation_offset],
                    flags=printed(flags), syndrome=syndrome,
                    output_pauli_mod_stabilizers=space.product(residual),
                    output_X_syndrome=space.component_syndrome(residual, 'X'),
                    output_Z_syndrome=space.component_syndrome(residual, 'Z'),
                    syndrome_record_minus_final_X_syndrome=(
                        syndrome ^ space.component_syndrome(residual, 'X')))

    for effect, faults in primary._fault_witnesses.items():
        flags, syndrome, residual = primary.unpack(effect)
        x, z = space.components(residual)
        for fault in faults:
            if all(q >= code.n for _, q in fault.pauli_product):
                ancilla_x_outputs.add(x)
            if len(fault.pauli_product) != 1:
                continue
            letter, qubit = fault.pauli_product[0]
            if letter not in ('X', 'Z'):
                continue
            for bit in range(primary.num_flags):
                if (flags >> bit) & 1:
                    flag_sources[bit].add(letter)
            if letter != 'X':
                continue
            if qubit >= code.n:
                ancilla_syndromes_by_flags[flags].add(syndrome)
                if not residual:
                    readout_only_faults[flags].setdefault(syndrome, (effect, fault))
                if flags and syndrome and not residual:
                    examples.setdefault('ancilla_X_flagged_corrupt_readout_no_data_error',
                                        (effect, fault))
                if z:
                    examples.setdefault('physical_ancilla_X_produces_data_Z_after_basis_changes',
                                        (effect, fault))
            elif x:
                examples.setdefault('physical_data_X_during_Z_SE_leaves_data_X', (effect, fault))
    assert ancilla_x_outputs == {0}
    assert 'ancilla_X_flagged_corrupt_readout_no_data_error' in examples
    assert 'physical_data_X_during_Z_SE_leaves_data_X' in examples

    # An ideal X-SE cannot distinguish X data errors, whereas Z-SE diagnoses them.
    invisible_to_dual = [q for q in range(code.n)
                         if gadgets['X'].unpack(gadgets['X'].input_effect(1 << q))[:2] == (0, 0)]
    visible_to_repeat = [q for q in range(code.n)
                         if gadgets['Z'].unpack(gadgets['Z'].input_effect(1 << q))[1] != 0]
    assert invisible_to_dual == visible_to_repeat == list(range(code.n))

    # Replay the readout-only witness and independently check the final data.
    effect, fault = examples['ancilla_X_flagged_corrupt_readout_no_data_error']
    bell, refs, pairing = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z, primary.circuit.num_qubits)
    expected_prefix = [(effect >> bit) & 1 for bit in range(primary.num_detectors)]
    for basis, checks, logicals in (('X', code.H_x, code.L_x), ('Z', code.H_z, code.L_z)):
        evaluation = _append_bell_sensitive_measurements(
            bell + primary.with_faults((fault,)), checks, logicals, basis, refs, pairing)
        observed = sample_annotation_parities(evaluation, shots=8)
        assert (observed[:, :primary.num_detectors] == expected_prefix).all()
        assert not observed[:, primary.num_detectors:].any()

    # Same primary readout, but an X error was present before extraction in
    # exactly one history. Both internal faults are single ancilla X errors.
    ambiguous_origin = None
    for flags, shifts in sorted(readout_only_faults.items()):
        if not flags or 0 not in shifts:
            continue
        for q in range(code.n):
            column = space.component_syndrome(1 << q, 'X')
            if not column or column not in shifts:
                continue
            corrupt_effect, corrupt_fault = shifts[column]
            flag_only_effect, flag_only_fault = shifts[0]
            incoming = 1 << q
            with_incoming = primary.input_effect(incoming) ^ flag_only_effect
            assert primary.unpack(corrupt_effect)[:2] == primary.unpack(with_incoming)[:2]
            states = (primary.unpack(corrupt_effect)[2], primary.unpack(with_incoming)[2])
            next_readouts = {
                basis: [gadgets[basis].unpack(gadgets[basis].input_effect(state))[:2]
                        for state in states] for basis in ('X', 'Z')}
            assert next_readouts['X'][0] == next_readouts['X'][1] == (0, 0)
            assert next_readouts['Z'][0] != next_readouts['Z'][1]
            for initial, fault in ((0, corrupt_fault), (incoming, flag_only_fault)):
                evaluation = bell.copy()
                for letter, qubit in space.product(initial):
                    evaluation.append(letter, [qubit])
                evaluation += primary.with_faults((fault,))
                observed = sample_annotation_parities(evaluation, shots=8)
                expected = [(corrupt_effect >> bit) & 1
                            for bit in range(primary.num_detectors)]
                assert (observed == expected).all()
            ambiguous_origin = dict(
                flags=printed(flags), primary_syndrome=column,
                no_incoming_error=describe(corrupt_effect, corrupt_fault),
                incoming_X_qubit=q,
                internal_fault_when_incoming_present=describe(flag_only_effect, flag_only_fault),
                further_fault_free_readouts_by_basis=next_readouts,
                primary_readouts_replayed_shots_per_history=8,
            )
            break
        if ambiguous_origin:
            break
    assert ambiguous_origin is not None

    data_couplings = [offset for offset, (_, qubits) in enumerate(primary.operations)
                      if any(q < code.n for q in qubits) and any(q >= code.n for q in qubits)]
    measurement_offsets = [offset for offset, (gate, _) in enumerate(primary.operations)
                           if gate in ('M', 'MX')]
    flag_completion_offsets = [max(measurement_offsets[i] for i in parity)
                               for parity in primary.readout.flags]
    first_data_contact = min(data_couplings)
    flags_known_before_data = [i for i, offset in enumerate(flag_completion_offsets)
                              if offset < first_data_contact]

    report = dict(
        circuit_sha256=sha256(path.read_bytes()).hexdigest(),
        schedule='Saved Z-SE4 lowered to physical preparation/measurement basis rotations.',
        all_ancilla_only_single_Pauli_faults_have_zero_output_X_component=True,
        flag_bit_single_qubit_physical_Pauli_sources=[sorted(sources) for sources in flag_sources],
        flagged_masks_with_multiple_syndrome_shifts_from_physical_ancilla_X=[
            dict(flags=printed(flags), syndrome_shifts=sorted(syndromes))
            for flags, syndromes in sorted(ancilla_syndromes_by_flags.items())
            if flags and len(syndromes) > 1],
        single_X_data_errors_invisible_to_fault_free_X_SE=invisible_to_dual,
        single_X_data_errors_visible_to_fault_free_Z_SE=visible_to_repeat,
        examples={name: describe(effect, fault) for name, (effect, fault) in examples.items()},
        readout_only_witness_replayed_on_encoded_Bell_input_shots_per_basis=8,
        indistinguishable_incoming_vs_internal_origin=ambiguous_origin,
        preparation_postselection_timing=dict(
            first_data_coupling_operation_offset=first_data_contact,
            flag_completion_operation_offsets=flag_completion_offsets,
            flags_known_before_first_data_coupling=flags_known_before_data,
            all_flags_available_before_data_coupling=(len(flags_known_before_data) == primary.num_flags)),
        scope='Single-fault information flow. Physical Pauli labels can change at Hadamards. '
              'No complete protocol, typed-flag partition, reduced-distance continuation, '
              'or stopping rule is certified here.',
    )
    destination = directory / 'basis_choice_checks.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved', destination, flush=True)
    print('Flags available before first data coupling:', flags_known_before_data,
          'out of', primary.num_flags, flush=True)


if __name__ == '__main__':
    main()

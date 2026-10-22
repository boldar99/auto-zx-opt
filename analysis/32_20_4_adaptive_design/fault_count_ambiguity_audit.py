"""Check whether Z-SE4 flags determine an exact internal fault count.

Uses the saved circuit, the existing exclusive-location Pauli fault model,
and direct Stim replay of one witness for each reported count. This is a
diagnostic for the recursive design note, not a certificate for that policy.
"""

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


def single_channel_examples(primary, bell, references, pairing, code, channels):
    """Find identical nontrivial responses at orders 1 and 2 in each channel.

    Every witness fault is one X or one Z on a single target after a unitary.
    The two-fault history uses different gate locations. Both of its faults
    have nonzero responses, so the ambiguity does not use an invisible extra
    fault, a mixed X/Z history, or correlated two-qubit Pauli faults.
    """
    examples = {}
    flag_mask = (1 << primary.num_flags) - 1
    for channel in channels:
        singles = {}
        for effect, faults in primary._fault_witnesses.items():
            if not effect:
                continue
            selected = [fault for fault in faults
                        if len(fault.pauli_product) == 1
                        and fault.pauli_product[0][0] == channel
                        and not fault.before_measurement
                        and primary.operations[fault.operation_offset][0]
                        not in ('R', 'RX', 'M', 'MX')]
            if selected:
                singles[effect] = selected
        found = None
        for first, first_faults in singles.items():
            for second, second_faults in singles.items():
                combined = first ^ second
                if not combined & flag_mask or combined not in singles:
                    continue
                if not primary.unpack(combined)[2]:
                    continue
                pair = next(((a, b) for a in first_faults for b in second_faults
                             if a.operation_offset != b.operation_offset), None)
                if pair:
                    found = combined, (singles[combined][0],), pair
                    break
            if found:
                break
        assert found is not None, channel
        effect, one, two = found
        replays = {}
        for basis, checks, logicals in (('X', code.H_x, code.L_x),
                                       ('Z', code.H_z, code.L_z)):
            results = []
            for faults in (one, two):
                evaluation = _append_bell_sensitive_measurements(
                    bell + primary.with_faults(faults), checks, logicals,
                    basis, references, pairing)
                results.append(sample_annotation_parities(evaluation, shots=8))
            assert (results[0] == results[1]).all()
            assert (results[0] == results[0][0]).all()
            replays[basis] = results[0][0].tolist()
        flags, syndrome, residual = primary.unpack(effect)

        def describe(faults):
            return [dict(**asdict(fault), gate=primary.operations[fault.operation_offset])
                    for fault in faults]

        examples[channel] = dict(
            extraction_basis=primary.basis,
            flags=''.join(str((flags >> i) & 1) for i in range(primary.num_flags)),
            syndrome=syndrome,
            residual_mod_stabilizers=primary.space.product(residual),
            one_fault=describe(one), two_faults=describe(two),
            each_fault_response_is_nonzero=True,
            identical_full_response=True,
            identical_raw_annotation_parities_by_final_Bell_check_basis=replays,
            direct_replay_shots_per_history_and_basis=8,
        )
    return examples


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    path = root / 'spiderwarp/assets/circuits/SECircuits/32_20_4_d4.stim'
    code = CSSCode.load_code('misc', '32_20_4')
    space = PauliSpace(code)
    raw = stim.Circuit.from_file(path)
    readout = extract_readout(raw, code)
    primary = SEGadget(4, 'Z', lower_to_guppy_gates(readout.annotate(dual_circuit(raw))),
                       readout, space)
    validate_gadget(primary)
    dual_primary = SEGadget(4, 'X', lower_to_guppy_gates(readout.annotate(raw)),
                            readout, space)
    validate_gadget(dual_primary)
    flag_mask = (1 << primary.num_flags) - 1
    incoming_errors_leave_flags_zero = all(
        not primary.unpack(primary.input_effect(1 << q))[0]
        for q in range(2 * code.n))
    assert incoming_errors_leave_flags_zero
    effects = {order: primary.fault_effects(order) for order in (0, 1, 2)}
    flags_by_order = {order: {effect & flag_mask for effect in group}
                      for order, group in effects.items()}
    bell, references, pairing = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z, primary.circuit.num_qubits)
    examples = {}
    for printed in ('00000000000', '10000000000', '00000011011', '00000011100'):
        flags = sum(int(bit) << i for i, bit in enumerate(printed))
        orders = [order for order in (0, 1, 2) if flags in flags_by_order[order]]
        witnesses = {}
        for order in orders:
            effect = next(effect for effect in effects[order]
                          if effect & flag_mask == flags)
            faults = primary.fault_witness(effect, order)
            assert faults is not None and len(faults) == order
            assert len({fault.operation_offset for fault in faults}) == order
            observed = sample_annotation_parities(bell + primary.with_faults(faults), shots=8)
            observed_flags = [sum(int(bit) << i for i, bit in enumerate(row[:primary.num_flags]))
                              for row in observed]
            assert observed_flags == [flags] * 8
            witnesses[str(order)] = dict(
                faults=[asdict(fault) for fault in faults],
                syndrome=primary.unpack(effect)[1],
                direct_replay_shots=8,
            )
        examples[printed] = dict(possible_internal_fault_counts=orders,
                                 minimum=min(orders), witnesses=witnesses)
    overlap = flags_by_order[1] & flags_by_order[2]
    assert 1 in overlap
    assert examples['00000000000']['possible_internal_fault_counts'] == [0, 1, 2]
    report = dict(
        circuit_sha256=sha256(path.read_bytes()).hexdigest(),
        basis='Z', circuit_label=4, max_internal_faults=2,
        model='Nonidentity Pauli gate, reset and measurement faults at distinct locations; no implicit idle faults.',
        printed_bit_order='Flag bit 0 first.',
        incoming_data_paulis_leave_flags_zero=incoming_errors_leave_flags_zero,
        numbers_of_flag_patterns_by_fault_count={str(order): len(flags)
                                                for order, flags in flags_by_order.items()},
        flag_patterns_compatible_with_both_one_and_two_faults=len(overlap),
        examples=examples,
        separate_single_channel_examples={
            **single_channel_examples(primary, bell, references, pairing, code, ('Z',)),
            **single_channel_examples(dual_primary, bell, references, pairing, code, ('X',)),
        },
        scope='Exact compatible orders in this bounded circuit model. No adaptive-policy correctness or minimality claim.',
    )
    destination = directory / 'fault_count_ambiguity_checks.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value['possible_internal_fault_counts']
                      for key, value in examples.items()}, indent=2), flush=True)
    print('Patterns compatible with both one and two faults:', len(overlap), flush=True)
    print('Saved', destination, flush=True)


if __name__ == '__main__':
    main()

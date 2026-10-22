"""Reproduce concrete constraints for redesigning the adaptive SE planner.

This audits the saved circuit and notebook table; it does not synthesize or
certify an adaptive protocol. Run from the repository with its normal Python
environment and PYTHONPATH set to the repository root.
"""

from collections import Counter
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import stim

from spiderwarp.csscode import CSSCode
from spiderwarp._se_circuit import (
    PauliSpace, SEGadget, dual_circuit, extract_readout, lower_to_guppy_gates,
    sample_annotation_parities, validate_gadget,
)
from spiderwarp.fault_tolerance_verification import _make_encoded_bell_input
from spiderwarp._se_policy import PolicyBuilder


def saved_flows(notebook):
    flows = {}
    for cell in notebook['cells']:
        if 'plan_minimal_followup_by_flag(' not in ''.join(cell.get('source', [])):
            continue
        label = None
        for output in cell.get('outputs', []):
            for line in ''.join(output.get('text', [])).splitlines():
                if line.startswith('Minimal SE follow-up flow ('):
                    label = line
                    flows[label] = {}
                elif label is not None and ' -> ' in line:
                    pattern, action = line.split(' -> ', 1)
                    if pattern and set(pattern) <= {'0', '1'}:
                        # Printed tuple order is flag bit 0 first.
                        mask = sum(int(bit) << i for i, bit in enumerate(pattern))
                        flows[label][mask] = action
    return flows


def describe_fault(fault):
    return dict(operation_offset=fault.operation_offset,
                pauli_product=fault.pauli_product,
                before_measurement=fault.before_measurement)


def retained_syndrome_audit(space, primary, recovery):
    """Check one recovery outcome using the current residual-bound contract."""
    builder = PolicyBuilder(space, {('Z', 4): primary, ('X', 2): recovery})
    initial = {(s, 0, space.reduce(error)) for s in range(builder.correctable + 1)
               for error in space.css_errors_of_weight(s)}
    initial.update((-1, 0, error) for error in builder._arbitrary_inputs.values())
    mask = (1 << primary.num_flags) - 1
    faults = {order: [f for f in primary.fault_effects(order) if f & mask == 1]
              for order in (1, 2)}
    first = defaultdict(set)
    for incoming, internal, error in initial:
        base = primary.input_effect(error)
        for order in range(1, builder._remaining((incoming, internal, error)) + 1):
            for fault in faults[order]:
                flags, syndrome, residual = primary.unpack(base ^ fault)
                assert flags == 1
                first[syndrome].add(builder._normalize((incoming, order, residual)))
    counts = Counter()
    combined = set()
    required_examples = []
    for syndrome, contexts in sorted(first.items()):
        results = builder._recovery_outcomes(recovery, None, contexts).get((0, 0), set())
        if not results:
            continue
        combined.update(results)
        required = any(builder._requires_acceptance(c) for c in results)
        correction = next((c for c in builder._candidate_corrections(results)
                           if builder._correction_safe(results, c)), None)
        counts['conditioned_groups'] += 1
        counts['groups_requiring_acceptance'] += required
        counts['groups_with_safe_correction'] += correction is not None
        if required:
            required_examples.append(dict(
                primary_syndrome=syndrome, histories=len(results),
                correction=None if correction is None else space.product(correction)))
    common = next((c for c in builder._candidate_corrections(combined)
                   if builder._correction_safe(combined, c)), None)
    assert common is None
    assert counts['conditioned_groups'] == counts['groups_with_safe_correction']
    return dict(
        prefix=dict(primary_basis='Z', primary_distance=4, primary_flag_mask=1,
                    recovery_basis='X', recovery_distance=2,
                    recovery_flags=0, recovery_syndrome=0),
        **counts, merged_histories=len(combined),
        common_correction_after_discarding_primary_syndrome=None,
        required_examples=required_examples,
        scope=('One recovery outcome, under the current CSS EC A/B plus extended '
               'r-filter contract. This does not certify the entire branch or '
               'prove that cleanup can always be omitted.'))


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    circuit_path = root / 'spiderwarp/assets/circuits/SECircuits/32_20_4_d4.stim'
    notebook_path = root / 'notebooks/32_20_4.ipynb'
    code = CSSCode.load_code('misc', '32_20_4')
    space = PauliSpace(code)
    circuit = stim.Circuit.from_file(circuit_path)
    readout = extract_readout(circuit, code)
    gadgets = {}
    for basis, raw in [('X', circuit), ('Z', dual_circuit(circuit))]:
        gadget = SEGadget(4, basis, lower_to_guppy_gates(readout.annotate(raw)),
                          readout, space)
        validate_gadget(gadget)
        gadgets[basis] = gadget
    x, z = gadgets['X'], gadgets['Z']
    bell, _, _ = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z,
        max(x.circuit.num_qubits, z.circuit.num_qubits))
    flag_mask = (1 << x.num_flags) - 1
    singles = {effect & flag_mask for effect in x.single_fault_effects}
    flows = saved_flows(json.loads(notebook_path.read_text()))
    comparison = {}
    mismatch = None
    for label, flow in flows.items():
        comparison[label] = dict(Counter(
            action for flags, action in flow.items() if flags and flags in singles))
        if mismatch is not None:
            continue
        for flags, action in sorted(flow.items()):
            if action != 'run non-FT follow-up SE' or flags not in singles:
                continue
            effect = next(e for e in x.single_fault_effects if e & flag_mask == flags)
            faults = x.fault_witness(effect, 1)
            replay = bell + x.with_faults(faults)
            parities = sample_annotation_parities(replay, shots=8)
            expected = [(flags >> i) & 1 for i in range(x.num_flags)]
            assert all(list(row[:x.num_flags]) == expected for row in parities)
            path = directory / 'single_fault_with_non_ft_saved_label.stim'
            replay.to_file(path)
            mismatch = dict(
                saved_table=label, printed_flag_pattern=''.join(map(str, expected)),
                saved_action=action, faults=[describe_fault(f) for f in faults],
                replay_shots=8, all_replay_flags_match=True, circuit=path.name,
                interpretation=(
                    'A saved non-FT label is reachable with one full-model gate fault. '
                    'Its old two-fault-only justification cannot be reused. This does '
                    'not establish that every d1 continuation for the prefix is unsafe.'))
            break

    silent = []
    for qubits in [(0, 1), (2, 3)]:
        error = space.reduce(sum(1 << q for q in qubits))
        faults = x.fault_witness(error << x.num_detectors, 2)
        assert faults is not None
        replay = bell + z.circuit + x.with_faults(faults)
        parities = sample_annotation_parities(replay, shots=8)
        assert not parities.any()
        path = directory / ('silent_two_fault_' + '_'.join(f'X{q}' for q in qubits) + '.stim')
        replay.to_file(path)
        silent.append(dict(
            residual=space.product(error), output_syndrome=space.pauli_syndrome(error),
            faults=[describe_fault(f) for f in faults], circuit=path.name,
            replay_shots=8, all_cycle_flags_and_syndromes_zero=True))
    difference = space.reduce(sum(1 << q for q in range(4)))
    assert silent[0]['output_syndrome'] == silent[1]['output_syndrome']
    assert difference != 0 and space.pauli_syndrome(difference) == 0
    recovery_path = root / 'spiderwarp/assets/circuits/SECircuits/32_20_4_d2.stim'
    recovery_circuit = stim.Circuit.from_file(recovery_path)
    recovery_readout = extract_readout(recovery_circuit, code)
    recovery = SEGadget(2, 'X', lower_to_guppy_gates(
        recovery_readout.annotate(recovery_circuit)), recovery_readout, space)
    validate_gadget(recovery)
    retained = retained_syndrome_audit(space, z, recovery)
    report = dict(
        circuit=str(circuit_path.relative_to(root)),
        circuit_sha256=sha256(circuit_path.read_bytes()).hexdigest(),
        notebook_sha256=sha256(notebook_path.read_bytes()).hexdigest(),
        recovery_circuit_sha256=sha256(recovery_path.read_bytes()).hexdigest(),
        model=('All nonidentity Pauli alternatives at a 1Q/2Q gate count as one '
               'fault at that gate; reset and measurement faults included. '
               'These checks introduce no implicit idle faults.'),
        input_data_paulis_do_not_raise_flags=all(
            not (x.input_effect(1 << q) & flag_mask) for q in range(2 * code.n)),
        full_model_single_fault_flag_patterns=len(singles),
        saved_actions_on_single_fault_flag_patterns=comparison,
        single_fault_label_mismatch=mismatch,
        silent_two_fault_histories=silent,
        silent_history_difference=dict(
            pauli=space.product(difference), syndrome=space.pauli_syndrome(difference),
            nontrivial_logical=True),
        retaining_flagged_primary_syndrome=retained,
        scope=(
            'The two silent histories have the same complete measured prefix and '
            'the same final syndrome but differ by a nontrivial logical Pauli. '
            'No fixed final ideal decoder can correct both. The existing clean '
            'path therefore cannot guarantee logical correctness of every accepted '
            'two-fault history. This is not a violation of the t=1 Gottesman contract.'))
    destination = directory / 'diagnostic_checks.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(destination), mismatch=mismatch,
                          silent_histories=2, retained_syndrome_check=retained,
                          replay_checks_passed=True), indent=2))


if __name__ == '__main__':
    main()

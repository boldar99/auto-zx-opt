"""Check the information and fault budget available after a flagged Z primary.

The continuation checks concern primary flag mask 1 on the saved circuits and
the current CSS EC A/B plus extended r-filter contract. They do not certify a
complete protocol or its stochastic logical error rate.
"""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import stim

from spiderwarp.csscode import CSSCode
from spiderwarp._se_circuit import (
    PauliSpace, SEGadget, dual_circuit, extract_readout, lower_to_guppy_gates,
    validate_gadget,
)
from spiderwarp._se_policy import PolicyBuilder


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    code = CSSCode.load_code('misc', '32_20_4')
    space = PauliSpace(code)
    gadgets, fingerprints = {}, {}
    for distance in (1, 2, 4):
        path = root / f'spiderwarp/assets/circuits/SECircuits/32_20_4_d{distance}.stim'
        circuit = stim.Circuit.from_file(path)
        readout = extract_readout(circuit, code)
        fingerprints[str(distance)] = sha256(path.read_bytes()).hexdigest()
        for basis, raw in [('X', circuit), ('Z', dual_circuit(circuit))]:
            gadget = SEGadget(distance, basis,
                              lower_to_guppy_gates(readout.annotate(raw)), readout, space)
            validate_gadget(gadget)
            gadgets[basis, distance] = gadget
    builder = PolicyBuilder(space, gadgets)
    flag_response = {
        f'{basis}{distance}': all(not gadget.unpack(gadget.input_effect(1 << q))[0]
                                for q in range(2 * code.n))
        for (basis, distance), gadget in gadgets.items()
    }
    assert all(flag_response.values())
    primary = gadgets['Z', 4]
    flag_mask = (1 << primary.num_flags) - 1
    order_flags = {order: {effect & flag_mask for effect in primary.fault_effects(order)}
                   for order in (1, 2)}
    certificates = {}
    for printed in ('10000000000', '00000011011', '00000011100', '00000011101'):
        flags = sum(int(bit) << i for i, bit in enumerate(printed))
        certificates[printed] = next((order for order in (1, 2)
                                      if flags in order_flags[order]), None)
    print('FLAG_CERTIFICATES', certificates, flush=True)
    initial = {(s, 0, space.reduce(error)) for s in range(builder.correctable + 1)
               for error in space.css_errors_of_weight(s)}
    initial.update((-1, 0, error) for error in builder._arbitrary_inputs.values())
    matching = {order: [effect for effect in primary.fault_effects(order)
                        if effect & flag_mask == 1] for order in (1, 2)}
    first = defaultdict(set)
    for incoming, internal, error in initial:
        base = primary.input_effect(error)
        for order in range(1, builder._remaining((incoming, internal, error)) + 1):
            for fault in matching[order]:
                flags, syndrome, residual = primary.unpack(base ^ fault)
                assert flags == 1
                first[syndrome].add(builder._normalize((incoming, order, residual)))

    def safe(contexts):
        return next((correction for correction in builder._candidate_corrections(contexts)
                     if builder._correction_safe(contexts, correction)), None)

    continuation = {}
    for distance in (1, 2):
        counts, failures = Counter(), []
        current_only = defaultdict(set)
        for primary_syndrome, contexts in sorted(first.items()):
            groups = builder._run(gadgets['X', distance], contexts)
            for (flags, syndrome), results in groups.items():
                current_only[flags, syndrome].update(results)
                required = any(builder._requires_acceptance(c) for c in results)
                correction = safe(results)
                counts['observed_prefixes'] += 1
                counts['flagged_prefixes'] += bool(flags)
                counts['flagged_prefixes_requiring_acceptance'] += bool(flags) and required
                if flags:
                    continue
                counts['unflagged_prefixes'] += 1
                counts['unflagged_prefixes_with_safe_joint_correction'] += correction is not None
                counts['unflagged_prefixes_requiring_acceptance'] += required
                counts['unflagged_prefixes_with_no_safe_correction_but_required_acceptance'] += (
                    correction is None and required)
                if correction is None and required and len(failures) < 3:
                    failures.append(dict(primary_syndrome=primary_syndrome,
                                         recovery_syndrome=syndrome,
                                         contexts=len(results)))
        for (flags, syndrome), results in current_only.items():
            if flags:
                continue
            if safe(results) is None and any(builder._requires_acceptance(c) for c in results):
                counts['unflagged_current_only_groups_losing_required_correction'] += 1
        continuation[str(distance)] = dict(counts=counts, first_failures=failures)
        print('CONTINUATION', distance, dict(counts), failures, flush=True)

    report = dict(
        circuit_sha256=fingerprints,
        model='Full circuit-level Pauli gate, reset and measurement faults; no implicit idle sites.',
        all_data_paulis_leave_flags_zero=flag_response,
        minimum_internal_fault_order_by_printed_Z_primary_flags=certificates,
        CSS_component_filter_syndrome_counts={
            basis: {str(r): len(space.syndrome_errors(basis, r)) for r in (0, 1, 2)}
            for basis in ('X', 'Z')},
        arbitrary_input_syndrome_sectors=len(builder._arbitrary_inputs),
        primary_flag_mask=1,
        continuations=continuation,
        scope=('Counts refer to distinct observed prefixes, not shot probabilities. '
               'The checks cover one Z-primary flag pattern and retain its syndrome. '
               'Safe corrections include the current checker\'s physical correction-fault model. '
               'Flagged recovery continuations are not synthesized here. The check does not '
               'certify a complete protocol, optimality, idle-noise coverage, or a logical '
               'guarantee on every accepted two-fault history.'))
    destination = directory / 'flag_conditioning_checks.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('SAVED', destination, flush=True)


if __name__ == '__main__':
    main()

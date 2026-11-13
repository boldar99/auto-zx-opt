"""Exhaustive syndrome uniqueness for the exact saved [[32,20,4]] matrices."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from spiderwarp._se_circuit import PauliSpace, row_mask
from spiderwarp.csscode import CSSCode


def main():
    directory = Path(__file__).resolve().parent
    code = CSSCode.load_code('misc', '32_20_4')
    space = PauliSpace(code)
    # Independently enumerate the stabilizer group to check every quotient.
    stabilizers = {0}
    for generator in [*(row_mask(r) for r in code.H_x),
                      *(row_mask(r) << code.n for r in code.H_z)]:
        stabilizers |= {e ^ generator for e in tuple(stabilizers)}
    assert len(stabilizers) == 4096
    assert min(space.weight(e) for e in stabilizers if e) == 16
    components = {}
    summaries = {}
    for basis in ('X', 'Z'):
        groups = defaultdict(list)
        for error in space.component_errors(basis, 2).values():
            groups[space.component_syndrome(error, basis)].append(error)
        components[basis] = groups
        two = [es for es in groups.values() if min(e.bit_count() for e in es) == 2]
        assert len(two) == 31 and all(len(es) == 16 for es in two)
        for es in groups.values():
            for i, a in enumerate(es):
                for b in es[i+1:]:
                    assert (space.reduce(a) == space.reduce(b)) == ((a ^ b) in stabilizers)
        summaries[basis] = dict(syndromes=64, requires_weight_two=31,
            uniquely_correctable_weight_two=0, distinct_logical_classes_per_weight_two_syndrome=16)

    paulis = defaultdict(list)
    for error in space.low_weight(2).values():
        paulis[space.pauli_syndrome(error)].append(error)
    two = {s: es for s, es in paulis.items() if min(space.weight(e) for e in es) == 2}
    counts = Counter(len({space.reduce(e) for e in es}) for es in two.values())
    assert counts == {1: 2976, 16: 93}
    for es in paulis.values():
        for i, a in enumerate(es):
            for b in es[i+1:]:
                assert (space.reduce(a) == space.reduce(b)) == ((a ^ b) in stabilizers)
    unique_mixed = next((s, es[0]) for s, es in two.items()
                       if len(es) == 1 and any(p == 'Y' for p, _ in space.product(es[0])))
    rows = []
    for sz in range(64):
        for sx in range(64):
            syndrome = sx | (sz << 6)
            xs, zs = components['X'][sx], components['Z'][sz]
            physical = paulis.get(syndrome, [])
            minimum_css = max(min(e.bit_count() for e in xs), min(e.bit_count() for e in zs))
            css_classes = len({space.reduce(e) for e in xs}) * len({space.reduce(e) for e in zs})
            rows.append(dict(x_component_syndrome=sx, z_component_syndrome=sz,
                joint_syndrome=syndrome, min_css_component_bound=minimum_css,
                css_bound_two_logical_classes=css_classes, unique_css_bound_two=css_classes == 1,
                min_total_pauli_weight=min((space.weight(e) for e in physical), default=None),
                total_pauli_bound_two_logical_classes=len({space.reduce(e) for e in physical}),
                unique_total_pauli_bound_two=len({space.reduce(e) for e in physical}) == 1))
    assert sum(r['unique_css_bound_two'] for r in rows) == 1089
    assert sum(r['unique_css_bound_two'] for r in rows if r['min_css_component_bound'] == 2) == 0
    with (directory / 'syndromes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Qubit indices and syndrome integers use the repository's zero-based,
    # least-significant-check-first conventions.
    summary = dict(code='misc/32_20_4', n=32, k=20, d=4,
        matrix_sha256=hashlib.sha256(code.H_x.tobytes() + code.H_z.tobytes()).hexdigest(),
        checked_stabilizers=4096, minimum_nonidentity_stabilizer_weight=16,
        components=summaries,
        css_bound_two=dict(input_patterns=529**2, joint_syndromes=4096,
            unique_syndromes=1089, requires_css_order_two=3007,
            uniquely_correctable_css_order_two=0),
        total_pauli_bound_two=dict(input_patterns=1 + 3*32 + 9*496,
            joint_syndromes=len(paulis), requires_weight_two=len(two),
            unique_weight_two=2976, ambiguous_weight_two=93,
            logical_classes_per_ambiguous_syndrome=16,
            unique_example=dict(syndrome=unique_mixed[0], pauli=space.product(unique_mixed[1]))),
        ambiguity_example=dict(errors=[['X0', 'X1'], ['X2', 'X3']],
            common_x_component_syndrome=space.component_syndrome(3, 'X'),
            difference='X0 X1 X2 X3', difference_is_nontrivial_logical=space.reduce(15) != 0),
        selected_sampling_model='CSS component weights <= 2',
        conclusion='No new CSS-order-two input is uniquely correctable from the syndrome alone. Flags or a restricted history can supply additional information.')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

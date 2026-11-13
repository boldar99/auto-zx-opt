"""Compile the user's paired, no-d1, inferred-order-one PS methods."""
import argparse
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from spiderwarp.se_adaptive_experiments import compile_css_candidate
from spiderwarp.se_experiment import emit_protocol
from spiderwarp.se_postselection import postselect_order_one, verify_order_one_guard
from spiderwarp.se_order_two import compile_order_two
from spiderwarp.se_protocol_io import load_protocol, read_manifest
from importlib import import_module

study = import_module('analysis.32_20_4_bounded_comparison.study')
ROOT, OUT, write = study.ROOT, study.OUT, study.write


def inspect(protocol):
    nodes = protocol.policy.nodes

    @lru_cache(None)
    def paths(index):
        node = nodes[index]
        if node.kind in {'accept', 'reject'}:
            return frozenset([()])
        if node.kind == 'frame':
            return paths(node.next)
        return frozenset((node.action,) + tail for child in set(node.children.values()) | {0}
                         for tail in paths(child))

    sequences = sorted(paths(protocol.policy.root))
    allowed = {(4, 4), (4, 2), (4, 4, 2)}
    assert all(tuple(protocol.gadgets[a].distance for a in path) in allowed
               or len(path) == 1 for path in sequences)
    assert not any('_d1' in action for path in sequences for action in path)
    index, zero = protocol.policy.root, []
    while nodes[index].kind not in {'accept', 'reject'}:
        node = nodes[index]
        if node.kind == 'frame':
            index = node.next
        else:
            zero.append(node.action)
            index = node.children.get((0, 0), 0)
    assert zero == ['Z_d4', 'X_d4'] and nodes[index].kind == 'accept'
    return dict(contract=asdict(protocol.policy.contract), paths=sequences,
                zero_record_path=zero, max_d4_extractions=2,
                max_extractions=protocol.verification.max_extractions,
                no_d1=True, worst_case_cnots=protocol.verification.worst_case_cnots)


def build_order_two(parent):
    protocol, provenance = compile_order_two(parent, progress=print)
    path = ROOT / 'build' / study.BUNDLES['adaptive_order_two']
    protocol.save_bundle(path)
    emit_protocol(protocol, path, 'hugr')
    provenance.update(protocol_hash=read_manifest(path)[1]['protocol_hash'],
                      parent_protocol_hash=read_manifest(ROOT / 'build/se_32_20_4')[1]['protocol_hash'],
                      verification=asdict(protocol.verification), inspection=inspect(protocol))
    write(path / 'experiment.json', provenance)
    write(OUT / 'order_two_build_validation.json', provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=('adaptive_history', 'adaptive_syndrome', 'adaptive_no_flags', 'adaptive_order_two',
                                         'adaptive_probability_10', 'adaptive_probability_25'))
    args = parser.parse_args(argv)
    if args.only and args.only.startswith('adaptive_probability_'):
        import_module('analysis.32_20_4_bounded_comparison.build_probability').main([])
        return
    parent = load_protocol(ROOT / 'build/se_32_20_4')
    if args.only == 'adaptive_order_two':
        build_order_two(parent)
        return
    assert parent.policy.contract.input_scope == 'total-order'
    assert parent.policy.contract.correctable == 1
    assert parent.policy.contract.accept_clean_through == 1
    assert parent.policy.contract.accepted_filter_order == 1
    assert set(parent.gadgets) == {'X_d2', 'X_d4', 'Z_d2', 'Z_d4'}
    validation_path = OUT / 'build_validation.json'
    import json
    audits = json.loads(validation_path.read_text())['methods'] if args.only and validation_path.exists() else {}
    for key, compiler in (
            ('adaptive_history', lambda: postselect_order_one(parent, progress=print)),
            ('adaptive_syndrome', lambda: compile_css_candidate(parent, 'css', progress=print)),
            ('adaptive_no_flags', lambda: compile_css_candidate(parent, 'css-no-flags', progress=print))):
        if args.only and args.only != key:
            continue
        path = ROOT / 'build' / study.BUNDLES[key]
        protocol = compiler()
        checks = inspect(protocol)
        checks['order_one_guard'] = verify_order_one_guard(protocol)
        protocol.save_bundle(path)
        emit_protocol(protocol, path, 'hugr')
        checks.update(protocol_hash=read_manifest(path)[1]['protocol_hash'], bundle=str(path.relative_to(ROOT)),
                      verification=asdict(protocol.verification))
        write(path / 'experiment.json', dict(
            input_scope='r+s<=2; r=max(w_X,w_Z)', acceptance='Reject if no compatible r+s<=1 history; reject unsafe ambiguous frames',
            mandatory_acceptance=('All total-order-one histories' if key in {'adaptive_history', 'adaptive_syndrome'}
                                  else 'Clean input only; additional syndrome PS may reject single-fault records'),
            flow=('Z_d4 -> X_d4; reject any flag immediately; no d2 follows'
                  if key == 'adaptive_no_flags' else
                  'Z_d4; first flags zero -> X_d4; raised first flags -> X_d2; second d4 flag or unresolved correction -> one d2 follow-up'),
            accepted_correction_scope='r+s<=1; order-two histories are used for detection, not an extra correction requirement',
            no_d1=True, no_order_two_rescues=True, baseline_excluded=True,
            additional_syndrome_guard=key == 'adaptive_no_flags',
            all_flags_zero=key == 'adaptive_no_flags',
            flag_zero_is_not_actual_fault_zero=key == 'adaptive_no_flags',
            parent_hash=read_manifest(ROOT / 'build/se_32_20_4')[1]['protocol_hash']))
        audits[key] = checks
        print(key, checks['protocol_hash'], checks['paths'], flush=True)
    assert audits['adaptive_syndrome']['protocol_hash'] == audits['adaptive_history']['protocol_hash']
    assert audits['adaptive_no_flags']['protocol_hash'] != audits['adaptive_history']['protocol_hash']
    write(OUT / 'build_validation.json', dict(complete=True, methods=audits,
          user_steering='History/no-flag cohort; no d1 or retired rescues; full-order-two V2 is built and certified separately',
          allowed_distance_paths=[[4, 4, 2], [4, 4], [4, 2]],
          sample_floor=study.SHOT_FLOOR, sample_cap=study.SHOT_CAP, matching_attempted_counts=True))
    if not args.only:
        build_order_two(parent)
        import_module('analysis.32_20_4_bounded_comparison.build_probability').main([])


if __name__ == '__main__':
    main()

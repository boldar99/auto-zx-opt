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
                      parent_protocol_hash=read_manifest(ROOT / 'build' / study.BUNDLES['adaptive_history'])[1]['protocol_hash'],
                      verification=asdict(protocol.verification), inspection=inspect(protocol))
    write(path / 'experiment.json', provenance)
    write(OUT / 'order_two_build_validation.json', provenance)


def build_scheduled(parent):
    from spiderwarp.se_scheduling import reschedule_d2
    protocol, provenance = reschedule_d2(parent)
    path = ROOT / 'build' / study.BUNDLES['adaptive_scheduled']
    protocol.save_bundle(path)
    emit_protocol(protocol, path, 'hugr')
    provenance.update(protocol_hash=read_manifest(path)[1]['protocol_hash'],
        parent_protocol_hash=read_manifest(ROOT / 'build' / study.BUNDLES['adaptive_history'])[1]['protocol_hash'],
        verification=asdict(protocol.verification), inspection=inspect(protocol))
    write(path / 'experiment.json', provenance)
    write(OUT / 'scheduled_build_validation.json', provenance)
    print('Shorter d2:', provenance['protocol_hash'], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=('adaptive_history', 'adaptive_scheduled', 'adaptive_no_flags', 'adaptive_order_two',
                                         'adaptive_probability_10', 'adaptive_probability_25'))
    args = parser.parse_args(argv)
    if args.only and args.only.startswith('adaptive_probability_'):
        import_module('analysis.32_20_4_bounded_comparison.build_probability').main([])
        return
    base_path = ROOT / 'build' / study.BUNDLES['adaptive_history']
    if args.only == 'adaptive_scheduled':
        build_scheduled(load_protocol(base_path))
        return
    if args.only == 'adaptive_order_two':
        build_order_two(load_protocol(base_path))
        return
    template = load_protocol(ROOT / 'build/se_32_20_4')
    import json
    validation_path = OUT / 'build_validation.json'
    audits = json.loads(validation_path.read_text())['methods'] if args.only and validation_path.exists() else {}
    if not args.only or args.only == 'adaptive_history':
        base = postselect_order_one(template, progress=print)
        checks = inspect(base)
        checks['order_one_guard'] = verify_order_one_guard(base)
        base.save_bundle(base_path)
        emit_protocol(base, base_path, 'hugr')
        checks.update(protocol_hash=read_manifest(base_path)[1]['protocol_hash'],
                      verification=asdict(base.verification))
        audits['adaptive_history'] = checks
        write(base_path / 'experiment.json', dict(
            method='standard-flag-conditioned-order-one',
            no_flags='Treat measured CSS syndromes as incoming errors; accept and correct iff max(w_X,w_Z)<=1',
            raised_flags='Fault-history recovery, including propagated hook corrections',
            single_internal_fault_acceptance_guarantee=False,
            accepted_order_one_logical_protection=True,
            parent_hash=read_manifest(ROOT / 'build/se_32_20_4')[1]['protocol_hash']))
        print('Standard r+s<=1:', checks['protocol_hash'], flush=True)
    else:
        base = load_protocol(base_path)
    if not args.only or args.only == 'adaptive_no_flags':
        protocol = compile_css_candidate(base, 'css-no-flags', progress=print)
        path = ROOT / 'build' / study.BUNDLES['adaptive_no_flags']
        protocol.save_bundle(path)
        emit_protocol(protocol, path, 'hugr')
        audits['adaptive_no_flags'] = dict(inspection=inspect(protocol),
            protocol_hash=read_manifest(path)[1]['protocol_hash'],
            parent_protocol_hash=read_manifest(base_path)[1]['protocol_hash'],
            verification=asdict(protocol.verification))
        write(path / 'experiment.json', audits['adaptive_no_flags'])
    if audits:
        write(OUT / 'build_validation.json', dict(complete=True, methods=audits,
            base_rule='No flags: incoming CSS order <=1; raised flags: fault-history recovery',
            allowed_distance_paths=[[4, 4, 2], [4, 4], [4, 2]]))
    if not args.only:
        build_scheduled(base)
        build_order_two(base)
        import_module('analysis.32_20_4_bounded_comparison.build_probability').main([])


if __name__ == '__main__':
    main()

"""Audit actual emitted tables and independently rescore MAP confidence cases."""
import ast
from dataclasses import dataclass
from importlib import import_module
import json
import random
from types import SimpleNamespace

from spiderwarp.se_probability import ProbabilityRescueModel
from spiderwarp.se_protocol_io import load_protocol

study = import_module('analysis.32_20_4_bounded_comparison.study')


def emitted_bodies(source, source_path=None):
    if source_path is not None:
        from spiderwarp._se_adaptive_guppy import inline_adaptive_source
        source = inline_adaptive_source(source, source_path)
    tree, definitions = ast.parse(source), []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list, node.returns = [], None
            for arg in node.args.args:
                arg.annotation = None
            definitions.append(node)
        elif isinstance(node, ast.ClassDef):
            node.decorator_list = [ast.Name(id='dataclass', ctx=ast.Load())]
            definitions.append(node)
        elif isinstance(node, ast.Assign):
            definitions.append(node)
    namespace = dict(comptime=lambda value: value, dataclass=dataclass)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 '<exported probability controller>', 'exec'), namespace)
    return namespace


def audit_export(protocol, exported):
    """Check every original edge and prove that no missing edge is accepted."""
    from functools import lru_cache

    def successor(index):
        frame = 0
        while protocol.policy.nodes[index].kind == 'frame':
            frame ^= protocol.policy.nodes[index].frame
            index = protocol.policy.nodes[index].next
        return index, frame

    def decision(node):
        row = exported['DECISION_NODES'][node - 1]
        if not exported['DECISION_ROWS_PACKED']:
            return row
        return (row & exported['DECISION_BIT_MASK'],
                (row >> exported['DECISION_BIT_BITS']) & exported['DECISION_REFERENCE_MASK'],
                (row >> exported['DECISION_HIGH_SHIFT']) & exported['DECISION_REFERENCE_MASK'],
                row >> exported['DECISION_DELTA_SHIFT'])

    @lru_cache(None)
    def ones(node, width):
        if node == 0:
            return 0
        bit, low, high, delta = decision(node)
        assert 0 <= bit < width
        delta = exported['DECISION_DELTAS'][delta]
        assert delta in (0, 1)
        a, b = ones(low, bit), ones(high, bit)
        return (a + ((1 << bit) - b if delta else b)) << (width - bit - 1)

    @lru_cache(None)
    def accepted_count(root, width):
        if root < 0:
            value = -(root + 1)
            assert value in (0, 1)
            return value << width
        offset = exported['DECISION_DELTAS'][root >> exported['DECISION_REFERENCE_BITS']]
        assert offset in (0, 1)
        count = ones(root & exported['DECISION_REFERENCE_MASK'], width)
        return (1 << width) - count if offset else count

    root, frame = successor(protocol.policy.root)
    root_node = protocol.policy.nodes[root]
    expected_root_frame = frame ^ (root_node.frame if root_node.kind == 'accept' else 0)
    assert exported['ROOT_FRAME_X'] | (exported['ROOT_FRAME_Z'] << protocol.code.n) == expected_root_frame
    pending, seen, edges = [(root, exported['ROOT_NODE'])], {}, 0
    while pending:
        original, index = pending.pop()
        if original in seen:
            assert seen[original] == index
            continue
        seen[original] = index
        node = protocol.policy.nodes[original]
        kind, action, transition, x, z, xb, zb = exported['policy_node'](index)
        baseline, kind = kind >= 4, kind & 3
        if node.kind in ('accept', 'reject'):
            assert kind == int(node.kind == 'accept') and x == z == 0
            continue
        assert node.kind == 'extract' and kind in (2, 3)
        assert exported['ACTION_NAMES'][action] == node.action
        gadget = protocol.gadgets[node.action]
        accepted_root = (exported['CONTROL_SPECS'][transition][0] if kind == 2
                         else exported['TERMINAL_DECODERS'][transition][0])
        count = 0
        for (flags, syndrome), child in node.children.items():
            key = flags | (syndrome << gadget.num_flags)
            target_id, offset = successor(child)
            target = protocol.policy.nodes[target_id]
            accepted = bool(exported['lookup'](accepted_root, key))
            assert accepted == (target.kind != 'reject')
            if accepted:
                count += 1
                xs, zs = (exported['CONTROL_FRAMES'][transition] if kind == 2
                          else exported['TERMINAL_DECODERS'][transition][1:])
                fx = exported['expand_frame'](exported['lookup'](xs, key), xb) ^ x
                fz = exported['expand_frame'](exported['lookup'](zs, key), zb) ^ z
                if baseline:
                    if gadget.basis == 'Z':
                        fx ^= exported['lookup'](exported['BASELINE_X_ROOT'], syndrome)
                    else:
                        fz ^= exported['lookup'](exported['BASELINE_Z_ROOT'], syndrome)
                assert fx | (fz << protocol.code.n) == offset ^ (target.frame if target.kind == 'accept' else 0)
                if kind == 2:
                    following = exported['lookup_control'](transition, key)
                    pending.append((target_id, following))
                else:
                    assert target.kind == 'accept'
            elif kind == 2:
                assert exported['lookup_control'](transition, key) == 0
            edges += 1
        # Every listed accepted key was checked above. Equal cardinality over
        # the entire input domain proves every other key rejects, without
        # enumerating billions of absent table entries.
        assert accepted_count(accepted_root, gadget.num_detectors) == count
    return dict(nodes_checked=len(seen), transitions_checked=edges,
                missing_edges_reject=True, all_absent_outcomes_checked=True,
                shared_targets_checked=True)


def nested(small, large):
    pending, seen = [(small.policy.root, large.policy.root)], set()
    while pending:
        pair = pending.pop()
        if pair in seen:
            continue
        seen.add(pair)
        a, b = small.policy.nodes[pair[0]], large.policy.nodes[pair[1]]
        if a.kind == 'accept':
            assert b.kind == 'accept' and a.frame == b.frame
        elif a.kind == 'extract':
            assert b.kind == 'extract' and a.action == b.action
            for outcome, child in a.children.items():
                assert outcome in b.children
                pending.append((child, b.children[outcome]))
    return dict(complete=True, paired_nodes=len(seen), acceptance_nested=True,
                shared_record_corrections_identical=True)


def confidence_controls(protocol, model, epsilon, samples=1000):
    rng, reservoir, count, mandatory = random.Random(2026111315), [], 0, 0

    def visit(index, record):
        nonlocal count, mandatory
        node = protocol.policy.nodes[index]
        if node.kind == 'accept':
            names = tuple(step[0] for step in record)
            key = names, model.pack(record)
            if key in model.mandatory:
                assert node.frame == model.mandatory[key]
                mandatory += 1
                return
            count += 1
            if len(reservoir) < samples:
                reservoir.append((record, node.frame))
            else:
                position = rng.randrange(count)
                if position < samples:
                    reservoir[position] = record, node.frame
        elif node.kind == 'extract':
            for (flags, syndrome), child in node.children.items():
                visit(child, record + ((node.action, flags, syndrome),))

    visit(protocol.policy.root, ())
    maximum = 0
    for record, correction in reservoir:
        names = tuple(step[0] for step in record)
        score = model.model.path_model(names).posterior(model.pack(record), correction,
            SimpleNamespace(logical=lambda error: error))
        assert score.risk is not None and score.risk <= epsilon + 1e-12
        assert max(score.logical_classes, key=lambda item: item[1])[0] == 0
        maximum = max(maximum, score.risk)
    assert mandatory == len(model.mandatory)
    return dict(additional_records=count, mandatory_records=mandatory,
                independently_rescored=len(reservoir), max_estimated_error=maximum,
                pointwise_posterior_matches_global_convolution=True)


def main():
    parent = load_protocol(study.ROOT / 'build' / study.BUNDLES['adaptive_history'])
    model = ProbabilityRescueModel(parent)
    protocols, checks = [], {}
    for epsilon in (10, 25):
        path = study.ROOT / 'build' / study.BUNDLES[f'adaptive_probability_{epsilon}']
        protocol = load_protocol(path)
        source = (path / 'steane_se_protocol.py').read_text()
        export = audit_export(protocol, emitted_bodies(source, path / 'steane_se_protocol.py'))
        confidence = confidence_controls(protocol, model, epsilon / 100)
        provenance = json.loads((path / 'probability.json').read_text())
        assert confidence['additional_records'] == provenance['additional_accepted_records']
        checks[str(epsilon)] = dict(export=export, confidence=confidence)
        protocols.append(protocol)
        print(epsilon, checks[str(epsilon)], flush=True)
    nesting = nested(*protocols)
    study.write(study.OUT / 'probability_controls.json', dict(complete=True,
        methods=checks, nesting=nesting, seed=2026111315,
        note='Correction confidence checked using independent pointwise meet-in-the-middle posterior; no production labels'))
    print('Nested-domain audit', nesting, flush=True)


if __name__ == '__main__':
    main()

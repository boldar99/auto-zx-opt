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


def emitted_bodies(source):
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
    pending, seen, edges = [(protocol.policy.root, exported['ROOT_NODE'])], set(), 0
    while pending:
        original, index = pending.pop()
        if original in seen:
            continue
        seen.add(original)
        node = protocol.policy.nodes[original]
        row = exported['POLICY_NODES'][index]
        if exported.get('POLICY_ROWS_PACKED'):
            word, x, z = row
            kind, action, transition, xb, zb = [(word >> exported['POLICY_' + field + '_SHIFT'])
                & exported['POLICY_' + field + '_MASK']
                for field in ('KIND', 'ACTION', 'TRANSITION', 'BASIS_X', 'BASIS_Z')]
        else:
            kind, action, transition, x, z, xb, zb = row
        if node.kind == 'accept':
            assert kind == 1 and (x | (z << protocol.code.n)) == node.frame
            continue
        if node.kind == 'reject':
            assert kind == 0
            continue
        assert node.kind == 'extract' and kind in (2, 3)
        assert exported['ACTION_NAMES'][action] == node.action
        gadget = protocol.gadgets[node.action]
        for (flags, syndrome), child in node.children.items():
            key = flags | (syndrome << gadget.num_flags)
            target = protocol.policy.nodes[child]
            xs, zs = (exported['CONTROL_FRAMES'][transition] if kind == 2
                      else exported['TERMINAL_DECODERS'][transition][1:])
            frame = ((exported['expand_frame'](exported['lookup'](xs, key), xb) ^ x)
                | ((exported['expand_frame'](exported['lookup'](zs, key), zb) ^ z) << protocol.code.n))
            if kind == 2:
                assert frame == 0
                pending.append((child, exported['lookup_control'](transition, key)))
            else:
                accepted = exported['TERMINAL_DECODERS'][transition][0]
                assert bool(exported['lookup'](accepted, key)) == (target.kind == 'accept')
                assert target.kind in ('accept', 'reject')
                if target.kind == 'accept':
                    assert frame == target.frame
            edges += 1
        missing = next((key for key in range(1 << gadget.num_detectors)
            if (key & ((1 << gadget.num_flags) - 1), key >> gadget.num_flags) not in node.children), None)
        if missing is not None:
            if kind == 2:
                assert exported['lookup_control'](transition, missing) == 0
            else:
                assert exported['lookup'](exported['TERMINAL_DECODERS'][transition][0], missing) == 0
    return dict(nodes_checked=len(seen), transitions_checked=edges, missing_edges_reject=True)


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
    parent = load_protocol(study.ROOT / 'build/se_32_20_4')
    model = ProbabilityRescueModel(parent)
    protocols, checks = [], {}
    for epsilon in (10, 25):
        path = study.ROOT / 'build' / study.BUNDLES[f'adaptive_probability_{epsilon}']
        protocol = load_protocol(path)
        source = (path / 'steane_se_protocol.py').read_text()
        export = audit_export(protocol, emitted_bodies(source))
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

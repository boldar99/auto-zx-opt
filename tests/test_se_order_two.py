"""Exact history compression, physical uniqueness, and mandatory acceptance."""
from collections import defaultdict
from pathlib import Path

import pytest

from spiderwarp.se_order_two import AMBIGUOUS, OrderTwoEvidence
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment
from spiderwarp.noise_model import CustomErrorModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def parent():
    return load_protocol(ROOT / 'build/se_postselected_32_20_4')


@pytest.fixture(scope='module')
def evidence(parent):
    return OrderTwoEvidence(parent)


def test_initial_retains_all_css_two_classes_and_their_budget(evidence):
    layers = defaultdict(dict)
    for budget, syndrome, error in evidence.initial:
        layers[budget][syndrome] = error
    assert len(layers[2]) == 1 and layers[2][0] == 0
    assert len(layers[1]) == 1088
    assert len(layers[0]) == 3007
    assert all(error == AMBIGUOUS for error in layers[0].values())
    # X0X1 and X2X3 have the same syndrome but inequivalent logical action.
    assert evidence.signature(3) == evidence.signature(12)
    assert evidence.space.reduce(3) != evidence.space.reduce(12)


def test_uniqueness_is_full_pauli_class_not_syndrome(evidence):
    syndrome = evidence.signature(3)
    state = ((0, syndrome, evidence.space.reduce(3)),
             (0, syndrome, evidence.space.reduce(12)))
    assert evidence.correction(state) is None
    assert evidence.correction(((0, syndrome, AMBIGUOUS),)) is None
    assert evidence.correction(((0, syndrome, 3),)) == 3
    assert evidence.correction(()) is None


@pytest.mark.parametrize('name', ('Z_d4', 'X_d2'))
def test_compression_matches_unmerged_two_fault_enumeration(evidence, name):
    # Include two distinct classes with equal syndrome, with one fault still
    # available, and a clean state that can spend both faults in this gadget.
    raw = {(2, 0), (1, evidence.space.reduce(3)),
           (1, evidence.space.reduce(12)), (0, evidence.space.reduce(1))}
    compressed = {}
    for budget, error in raw:
        evidence.add(compressed, (budget, evidence.signature(error)), error)
    state = tuple((b, s, e) for (b, s), e in sorted(compressed.items()))
    expected = defaultdict(dict)
    for budget, error in raw:
        signature = evidence.signature(error)
        readout = evidence.read(signature, name)
        evidence.add(expected[0, readout], (budget, signature), error)
        for order in range(1, budget + 1):
            for flags, syndrome, delta, delta_signature in evidence.fault_responses(name, order):
                evidence.add(expected[flags, syndrome ^ readout],
                             (budget - order, signature ^ delta_signature), error ^ delta)
    expected = {key: tuple((b, s, e) for (b, s), e in sorted(values.items()))
                for key, values in expected.items()}
    assert evidence.transition(state, name) == expected


@pytest.fixture(scope='module')
def compiled():
    path = ROOT / 'build/adaptive_order_two_32_20_4'
    if not (path / 'protocol.json').exists():
        pytest.skip('Build the V2 integration artifact first')
    return load_protocol(path)


def replay(protocol, incoming, effects=()):
    """Algebraic physical execution independent of the compressed evidence."""
    index, frame, position, names = protocol.policy.root, 0, 0, []
    space = protocol.policy.decoder.space
    error = space.reduce(incoming)
    while True:
        node = protocol.policy.nodes[index]
        if node.kind in ('accept', 'reject'):
            return node.kind == 'accept', space.reduce(error ^ frame ^ node.frame), names
        if node.kind == 'frame':
            frame ^= node.frame
            index = node.next
            continue
        gadget = protocol.gadgets[node.action]
        fault = effects[position] if position < len(effects) else 0
        flags, syndrome, delta = gadget.unpack(fault)
        component = 'X' if gadget.basis == 'Z' else 'Z'
        syndrome ^= space.component_syndrome(error ^ frame, component)
        error ^= delta
        names.append(node.action)
        position += 1
        index = node.children.get((flags, syndrome), 0)


def test_all_css_one_and_single_gadget_faults_preserve_parent(compiled, parent):
    for x in (0, *(1 << q for q in range(32))):
        for z in (0, *(1 << (q + 32) for q in range(32))):
            assert replay(compiled, x ^ z) == replay(parent, x ^ z)
    for position, name in enumerate(('Z_d4', 'X_d4')):
        for effect in parent.gadgets[name].single_fault_effects:
            faults = (0,) * position + (effect,)
            new, residual, names = replay(compiled, 0, faults)
            accepted, old_residual, old_names = replay(parent, 0, faults)
            if accepted:
                assert new and residual == old_residual and names == old_names
            elif new:
                assert residual == 0
    assert compiled.verification.verified
    assert compiled.policy.contract == parent.policy.contract
    assert compiled.verification.max_extractions <= 3
    assert compiled.verification.worst_case_cnots <= 433


def test_base_rejected_single_fault_followup_includes_remaining_fault_budget(compiled, parent):
    cases = checked = 0
    for position, name in enumerate(('Z_d4', 'X_d4')):
        for effect in parent.gadgets[name].single_fault_effects:
            faults = [0, 0]
            faults[position] = effect
            if replay(parent, 0, faults)[0]:
                continue
            _, _, names = replay(compiled, 0, faults)
            assert len(names) == 3
            cases += 1
            # A fault in this extra measurement is a second physical location,
            # not an ideal observation after an already exhausted budget.
            for following in compiled.gadgets[names[-1]].single_fault_effects:
                accepted, residual, _ = replay(compiled, 0, faults + [following])
                if accepted:
                    assert residual == 0
                checked += 1
    assert cases == 4 and checked > 0


@pytest.mark.parametrize('basis', ('0', '+'))
def test_noiseless_multiple_cycles_and_external_frame(compiled, basis):
    result = ProtocolSampler(compiled, noise_model=CustomErrorModel(),
                             prepared_state=basis, cycles=10).sample(20, 20261113)
    assert result.accepted == 20 and result.logical_errors == 0
    experiment = _StateExperiment(compiled.code, compiled.required_qubits, basis)
    simulator = experiment.initial.copy(seed=17)
    simulator.x(0)
    simulator.z(17)
    result = compiled.run(simulator, frame_x=1, frame_z=1 << 17)
    assert result.accepted
    assert not experiment.logical_failure(simulator, frame_x=result.frame_x, frame_z=result.frame_z)


def check_compact_export(protocol, exported):
    """Evaluate the actual emitted lookup bodies against every stored edge."""
    pending = [(protocol.policy.root, exported['ROOT_NODE'])]
    visited, edges = set(), 0
    while pending:
        original, index = pending.pop()
        if original in visited:
            continue
        visited.add(original)
        node = protocol.policy.nodes[original]
        word, x, z = exported['POLICY_NODES'][index]
        values = {field: (word >> exported['POLICY_' + field.upper() + '_SHIFT'])
                  & exported['POLICY_' + field.upper() + '_MASK']
                  for field in ('kind', 'action', 'transition', 'basis_x', 'basis_z')}
        kind, transition = values['kind'], values['transition']
        if node.kind == 'accept':
            assert kind == 1 and (x | (z << protocol.code.n)) == node.frame
            continue
        if node.kind == 'reject':
            assert kind == 0
            continue
        assert node.kind == 'extract' and kind in (2, 3)
        assert exported['ACTION_NAMES'][values['action']] == node.action
        gadget = protocol.gadgets[node.action]
        for (flags, syndrome), child in node.children.items():
            key = flags | (syndrome << gadget.num_flags)
            target = protocol.policy.nodes[child]
            xs, zs = (exported['CONTROL_FRAMES'][transition] if kind == 2
                      else exported['TERMINAL_DECODERS'][transition][1:])
            frame = ((exported['expand_frame'](exported['lookup'](xs, key), values['basis_x']) ^ x)
                | ((exported['expand_frame'](exported['lookup'](zs, key), values['basis_z']) ^ z) << protocol.code.n))
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
        # Check a missing edge too: sparse compaction must retain rejection.
        missing = next((s for s in range(1 << gadget.num_detectors)
                        if (s & ((1 << gadget.num_flags) - 1), s >> gadget.num_flags) not in node.children), None)
        if missing is not None:
            if kind == 2:
                assert exported['lookup_control'](transition, missing) == 0
            else:
                assert exported['lookup'](exported['TERMINAL_DECODERS'][transition][0], missing) == 0
    return dict(nodes_checked=len(visited), transitions_checked=edges,
                packed_nodes=exported['N_POLICY_NODES'])


def test_compact_export_preserves_all_transitions_and_rejections(parent, monkeypatch):
    import spiderwarp._se_adaptive_guppy as emitter
    from test_optimized_steane_se import generated_python
    monkeypatch.setattr(emitter, 'COMPACT_POLICY_THRESHOLD', 1)
    source = emitter.generate_adaptive_source(parent)
    exported = generated_python(source)
    assert exported['POLICY_ROWS_PACKED']
    checked = check_compact_export(parent, exported)
    assert checked['transitions_checked'] > 2000


def test_compact_guppy_compiles_and_binary_reloads(parent, monkeypatch):
    pytest.importorskip('guppylang')
    from hugr.package import Package
    import spiderwarp._se_adaptive_guppy as emitter
    monkeypatch.setattr(emitter, 'COMPACT_POLICY_THRESHOLD', 1)
    source = emitter.generate_adaptive_source(parent)
    _, _, _, _, package = emitter.compile_adaptive_source(source)
    assert Package.from_bytes(package.to_bytes())

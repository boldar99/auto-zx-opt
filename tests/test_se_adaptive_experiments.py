"""Compiled CSS candidates, ambiguity rejection and exact rescue frames."""
from pathlib import Path
from collections import defaultdict
import random

import pytest

from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_adaptive_experiments import AdaptiveCSSCandidate, CSSRecoverCandidate, CompactCSSRescueEvidence
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler


@pytest.fixture(scope='module')
def parent():
    path = Path(__file__).resolve().parents[1] / 'build/se_32_20_4'
    if not (path / 'protocol.json').exists():
        pytest.skip('Integration tests require the full-size saved parent')
    return load_protocol(path)


@pytest.fixture(scope='module', params=('css', 'css-recover-exact', 'css-recover-safe'))
def pair(parent, request):
    variant = request.param
    path = Path(__file__).resolve().parents[1] / ('build/adaptive_' + variant.replace('-', '_') + '_32_20_4')
    if not (path / 'protocol.json').exists():
        pytest.skip('Selected full-size candidate must be compiled first')
    compiled = load_protocol(path)
    adapter = CSSRecoverCandidate(parent, variant) if variant.startswith('css-recover') else AdaptiveCSSCandidate(parent, variant)
    return variant, adapter, compiled


def test_exact_rescue_rejects_logically_inequivalent_explanations(parent):
    model = CSSRecoverCandidate(parent, 'css-recover-exact')
    space = parent.policy.decoder.space
    logical = space.reduce(15)  # X0 X1 X2 X3 has zero syndrome, nontrivial logical action.
    assert logical and space.pauli_syndrome(logical) == 0
    assert model.safe_full_frame(frozenset({(0, 1, 0), (1, 1, logical)}), 0) is None
    error = space.reduce(3 | (4 << parent.code.n))
    assert model.safe_full_frame(frozenset({(1, 1, error)}), 0) == error


def test_compact_evidence_matches_full_pauli_histories(parent):
    model = CSSRecoverCandidate(parent, 'css-recover-exact')
    compact = CompactCSSRescueEvidence(model)
    rng = random.Random(20261113)
    for _ in range(25):
        physical = (1 << rng.randrange(32)) ^ (1 << (32 + rng.randrange(32)))
        fault_step = rng.randrange(3)
        raw, packed = model.full_initial, compact.initial
        for step, name in enumerate(('Z_d4', 'X_d4', 'Z_d2')):
            g = model.gadgets[name]
            fault = rng.choice(g.single_fault_effects) if step == fault_step else 0
            flags, syndrome, delta = g.unpack(fault)
            component = 'X' if g.basis == 'Z' else 'Z'
            syndrome ^= model.space.component_syndrome(physical, component)
            physical ^= delta
            raw = model.full_step(raw, name, flags, syndrome)
            packed = compact.step(packed, name, flags, syndrome)
            groups = defaultdict(set)
            clean = set()
            for _, spent, error in raw:
                if spent == 0:
                    clean.add(error)
                else:
                    groups[model.space.pauli_syndrome(error)].add(error)
            expected = (frozenset(clean), frozenset(next(iter(es)) for es in groups.values() if len(es) == 1),
                        frozenset(s for s, es in groups.items() if len(es) > 1))
            assert packed == expected
            assert compact.safe_frame(packed, 0) == model.safe_full_frame(raw, 0)


@pytest.mark.parametrize('basis', ('0', '+'))
@pytest.mark.parametrize('cycles', (1, 3))
def test_compiled_graph_matches_fresh_noisy_adapter_samples(pair, basis, cycles):
    _, adapter, compiled = pair
    expected = ProtocolSampler(adapter, prepared_state=basis, cycles=cycles).sample(1000, 2026111302)
    actual = ProtocolSampler(compiled, prepared_state=basis, cycles=cycles).sample(1000, 2026111302)
    assert (actual.accepted, actual.logical_errors) == (expected.accepted, expected.logical_errors)


def test_all_4096_noiseless_input_syndromes_obey_selected_css_bound(pair):
    variant, _, protocol = pair
    policy, decoder = protocol.policy, protocol.policy.decoder
    space = decoder.space
    kept = 0
    for sx, x in decoder.physical['X'].items():
        for sz, z in decoder.physical['Z'].items():
            error, frame, index = x ^ z, 0, policy.root
            while True:
                node = policy.nodes[index]
                if node.kind in {'accept', 'reject'}:
                    break
                if node.kind == 'frame':
                    frame ^= node.frame
                    index = node.next
                else:
                    gadget = policy.actions[node.action].gadget
                    flags, syndrome, _ = gadget.unpack(gadget.input_effect(error))
                    component = 'X' if gadget.basis == 'Z' else 'Z'
                    index = node.children.get((flags, syndrome ^ space.component_syndrome(frame, component)), 0)
            expected = max(decoder.weights['X'][sx], decoder.weights['Z'][sz]) <= 1
            assert (node.kind == 'accept') == expected
            if expected:
                kept += 1
                assert space.reduce(error ^ frame ^ node.frame) == 0
    assert kept == 1089
    assert protocol.verification.verified and protocol.required_qubits == 60
    assert policy.contract.logical_order == 1
    assert policy.contract.correctable == int(variant.endswith('safe'))


def test_safe_rescue_cannot_change_the_parent_syndrome(parent):
    model = CSSRecoverCandidate(parent, 'css-recover-safe')
    compact = CompactCSSRescueEvidence(model)
    error, logical = parent.policy.decoder.space.reduce(1), parent.policy.decoder.space.reduce(15)
    raw = frozenset({(1, 1, error)})
    packed = frozenset(), frozenset({error}), frozenset()
    assert model.safe_full_frame(raw, 0) is None
    assert compact.safe_frame(packed, 0) is None
    assert model.safe_full_frame(raw, error ^ logical) == error
    assert compact.safe_frame(packed, error ^ logical) == error


def test_clean_repeated_cycles_and_parent_contract_preserved(pair, parent):
    _, _, protocol = pair
    for basis in ('0', '+'):
        result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), prepared_state=basis, cycles=10).sample(20, 17)
        assert result.accepted == 20 and result.logical_errors == 0
    assert parent.policy.contract.correctable == 1 and parent.policy.contract.accept_clean_through == 2

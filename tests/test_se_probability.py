"""Complete posterior support, MAP corrections and unconditional low order."""
from collections import defaultdict
import itertools
from pathlib import Path

import pytest

from spiderwarp.se_likelihood import _PathLikelihood
from spiderwarp.se_probability import ProbabilityRescueModel, ProbabilityRescueCandidate
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler
from test_se_order_two import replay

ROOT = Path(__file__).resolve().parents[1]


def test_all_joint_weights_match_independent_channel_enumeration():
    channels = [{1 | (3 << 3): .13, 2 | (5 << 3): .07},
                {1 | (3 << 3): .11, 3 | (6 << 3): .04},
                {2 | (5 << 3): .09, 7 | (2 << 3): .03},
                {4 | (1 << 3): .06}]
    expected = defaultdict(lambda: defaultdict(float))
    for choices in itertools.product(*(list(c.items()) + [(0, 1 - sum(c.values()))] for c in channels)):
        signature, probability, count = 0, 1.0, 0
        for effect, weight in choices:
            signature ^= effect
            probability *= weight
            count += bool(effect)
        if count <= 2:
            expected[signature & 7][signature >> 3] += probability
    model = _PathLikelihood(channels, 3)
    actual = model.joint_weights()
    assert actual.keys() == expected.keys()
    for observation, weights in expected.items():
        assert {e: w * model.p_zero for e, w in actual[observation].items()} == pytest.approx(dict(weights))
    filtered = model.joint_weights(lambda observation: observation in (2, 5))
    assert filtered == {o: weights for o, weights in actual.items() if o in (2, 5)}


def test_exclusive_alternatives_do_not_create_fictitious_posterior_support():
    model = _PathLikelihood([{1 | (1 << 2): .17, 2 | (2 << 2): .23}], 2)
    assert 3 not in model.joint_weights()
    assert 3 not in model.joint_weights(lambda observation: observation == 3)


@pytest.fixture(scope='module')
def parent():
    return load_protocol(ROOT / 'build/se_postselected_32_20_4')


def test_mandatory_records_and_routes_are_complete(parent):
    model = ProbabilityRescueModel(parent)
    assert len(model.mandatory) == 2371
    assert model.third_routes
    for (names, observation), frame in model.mandatory.items():
        assert model.valid(names, observation)
        assert model.pack(model.unpack(names, observation)) == observation
        result = model.model.replay(model.unpack(names, observation))
        assert result.protected and result.accepted_by_parent
        assert model.model.space.reduce(result.correction) == frame


@pytest.fixture(scope='module', params=(10, 25))
def compiled(request):
    path = ROOT / f'build/adaptive_probability_{request.param}_32_20_4'
    if not (path / 'protocol.json').exists():
        pytest.skip('Build the probability controller first')
    return request.param / 100, load_protocol(path)


def test_every_css_one_and_single_fault_replays_the_parent_exactly(compiled, parent):
    _, compiled = compiled
    for x in (0, *(1 << q for q in range(32))):
        for z in (0, *(1 << (q + 32) for q in range(32))):
            assert replay(compiled, x ^ z) == replay(parent, x ^ z)
    for position, name in enumerate(('Z_d4', 'X_d4')):
        for effect in parent.gadgets[name].single_fault_effects:
            faults = (0,) * position + (effect,)
            old = replay(parent, 0, faults)
            new = replay(compiled, 0, faults)
            if old[0]:
                assert new == old
            elif new[0]:
                # A base-rejected single-fault record is an additional rescue,
                # with its MAP correction checked separately against epsilon.
                assert new[1] == 0
    assert compiled.verification.verified
    assert compiled.policy.contract == parent.policy.contract
    assert compiled.verification.max_extractions <= 3
    assert compiled.verification.worst_case_cnots <= 433


@pytest.mark.parametrize('basis', ('0', '+'))
def test_pointwise_reference_matches_compiled_noisy_repeated_shots(compiled, parent, basis):
    epsilon, compiled = compiled
    reference = ProbabilityRescueCandidate(ProbabilityRescueModel(parent), epsilon)
    a = ProtocolSampler(reference, prepared_state=basis, cycles=3).sample(1000, 2026111312)
    b = ProtocolSampler(compiled, prepared_state=basis, cycles=3).sample(1000, 2026111312)
    assert (a.accepted, a.logical_errors) == (b.accepted, b.logical_errors)


def test_probability_acceptance_domains_are_nested_with_identical_frames():
    paths = [ROOT / f'build/adaptive_probability_{epsilon}_32_20_4' for epsilon in (10, 25)]
    if not all((path / 'protocol.json').exists() for path in paths):
        pytest.skip('Build both probability controllers first')
    small, large = map(load_protocol, paths)
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
    assert len(seen) > 30000

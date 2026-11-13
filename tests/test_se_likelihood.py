"""Physical channel probabilities, conditional likelihoods and frozen PS graphs."""
from collections import Counter, defaultdict
from dataclasses import replace
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import stim

from spiderwarp._se_adaptive import ExtractionAction, PolicyNode
from spiderwarp._se_circuit import Readout, SEGadget
from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol, build_adaptive_se_protocol
from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_likelihood import (
    FullHistoryLikelihood, LikelihoodCandidate, _PathLikelihood,
    collect_likelihood_records, noise_response_channels, postselect_likelihood,
)
from spiderwarp.se_postselection import postselect_order_one
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _noisy_gadget
from test_adaptive_se_policy import bell_actions


def test_weighted_posterior_matches_exhaustive_exclusive_channel_enumeration():
    # Three observation bits and arbitrary final Pauli classes. Equal responses
    # arise at different physical sites; alternatives at one site cannot pair.
    channels = [{1 | (3 << 3): .13, 2 | (5 << 3): .07},
                {1 | (3 << 3): .11, 3 | (6 << 3): .04},
                {2 | (5 << 3): .09, 7 | (2 << 3): .03},
                {4 | (1 << 3): .06}]
    model = _PathLikelihood(channels, 3)
    expected = defaultdict(lambda: defaultdict(float))
    tail = 0
    for choices in itertools.product(*(list(c.items()) + [(0, 1 - sum(c.values()))] for c in channels)):
        effect, probability, faults = 0, 1.0, 0
        for signature, weight in choices:
            effect ^= signature
            probability *= weight
            faults += bool(signature)
        if faults <= 2:
            expected[effect & 7][effect >> 3] += probability
        else:
            tail += probability
    assert model.omitted == pytest.approx(tail, abs=1e-15)
    # Nonlinear decoding/tie breaking must occur after combining physical errors.
    decoder = SimpleNamespace(logical=lambda e: (e ^ (1 if e & 2 else 0)))
    for observation in range(8):
        correction = 3
        score = model.posterior(observation, correction, decoder)
        evidence = sum(expected[observation].values())
        assert score.evidence_probability == pytest.approx(evidence, abs=1e-15)
        classes = defaultdict(float)
        for error, weight in expected[observation].items():
            classes[decoder.logical(error ^ correction)] += weight
        if evidence:
            assert dict(score.logical_classes) == pytest.approx({e: p / evidence for e, p in classes.items()})
            bad = sum(p for e, p in classes.items() if e)
            assert score.risk == pytest.approx(bad / evidence)
            assert score.risk_upper_bound == pytest.approx((bad + tail) / (evidence + tail))
        else:
            assert score.risk is None


def test_one_exclusive_channel_cannot_explain_a_pair_of_its_alternatives():
    model = _PathLikelihood([{1 | (1 << 2): .2, 2 | (2 << 2): .3}], 2)
    score = model.posterior(3, 0, SimpleNamespace(logical=lambda e: e))
    assert score.risk is None and score.evidence_probability == pytest.approx(0, abs=1e-15)
    assert score.omitted_probability == 0


def test_sampler_noise_channels_match_independent_physical_injections():
    code, space, actions = bell_actions()
    g = actions['X']
    noise = CustomErrorModel(p_1q=.013, p_2q=.027, p_init=.031, p_meas=.017,
                             p_linear_mem=.002, p_coherent_mem=.001, p_leakage=.003)
    noisy = list(_noisy_gadget(g.circuit, noise, persistent_data=code.n))
    clean = []
    cases = []
    for op in noisy:
        qs = [t.value for t in op.targets_copy()]
        if op.name in {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Z_ERROR'}:
            if op.name == 'DEPOLARIZE2':
                for a, b in zip(qs[::2], qs[1::2]):
                    cases.append((len(clean), [(tuple((pauli, q) for pauli, q in zip(pair, (a, b)) if pauli != 'I'),
                                               op.gate_args_copy()[0] / 15)
                                              for pair in itertools.product('IXYZ', repeat=2) if pair != ('I', 'I')]))
            else:
                kinds = 'XYZ' if op.name == 'DEPOLARIZE1' else op.name[0]
                for q in qs:
                    cases.append((len(clean), [(((kind, q),), op.gate_args_copy()[0] / len(kinds)) for kind in kinds]))
        elif op.name == 'M':
            for q in qs:
                cases.append((len(clean), [((('X', q),), op.gate_args_copy()[0])]))
                clean.append(stim.CircuitInstruction('M', [q]))
        else:
            clean.append(op)
    expected = []
    for offset, alternatives in cases:
        grouped = defaultdict(float)
        for product, probability in alternatives:
            sim = stim.TableauSimulator(seed=17)
            sim.do_circuit(stim.Circuit('H 0\nCX 0 1'))
            for i in range(len(clean) + 1):
                if i == offset:
                    for pauli, q in product:
                        sim.do(stim.CircuitInstruction(pauli, [q]))
                if i < len(clean):
                    sim.do(clean[i])
            flags, syndrome = g.readout.decode(sim.current_measurement_record())
            sx = int(sim.peek_observable_expectation(stim.PauliString('ZZ')) == -1)
            sz = int(sim.peek_observable_expectation(stim.PauliString('XX')) == -1)
            # For this k=0 Bell code the syndrome identifies the complete class.
            data = space.reduce(sx | (sz << code.n))
            effect = flags | (syndrome << g.num_flags) | (data << g.num_detectors)
            if effect and probability:
                grouped[effect] += probability
        if grouped:
            expected.append(tuple(sorted(grouped.items())))
    actual = noise_response_channels(g, noise)
    def rounded(channels):
        return Counter(tuple((effect, round(p, 14)) for effect, p in sorted(c)) for c in channels)
    assert rounded(expected) == rounded([c.items() for c in actual])
    assert len(noise_response_channels(g, CustomErrorModel())) == 0
    assert len(noise_response_channels(g, noise, include_initial_idles=False)) < len(actual)


@pytest.fixture
def flagged_parent():
    code, space, actions = bell_actions()
    parent = build_adaptive_se_protocol(code, actions, first_action='Z', export=False)
    old = actions['Z']
    # Two independent, harmless flags supply an observable two-fault history.
    circuit = old.circuit + stim.Circuit('R 3 4\nM 3 4')
    readout = Readout(((old.circuit.num_measurements,), (old.circuit.num_measurements + 1,)), old.readout.syndromes)
    gadget = SEGadget(old.distance, old.basis, circuit, readout, space)
    new_actions = dict(parent.policy.actions, Z=ExtractionAction('Z', gadget))
    nodes = list(parent.policy.nodes)
    root = nodes[parent.policy.root]
    nodes[parent.policy.root] = replace(root, children={(f, sy): child for (_, sy), child in root.children.items() for f in range(4)})
    policy = replace(parent.policy, actions=new_actions, nodes=tuple(nodes), report=replace(parent.verification, verified=False))
    parent = AdaptiveSEProtocol(code, policy)
    parent.verify()
    return parent


def test_frozen_rescues_contract_parent_preservation_and_unknown_fallback(flagged_parent, tmp_path):
    parent = flagged_parent
    model = FullHistoryLikelihood(parent, noise_model=CustomErrorModel(p_meas=.02), input_error_probability=0)
    record = (('Z', 3, 0), ('X', 0, 0))
    decision = model.replay(record)
    assert decision.accepted_by_parent and not decision.protected
    assert model.score(record).risk == 0
    old_nodes, old_report = parent.policy.nodes, parent.verification
    for epsilon in (0.01, 0.05):
        compiled, provenance = postselect_likelihood(parent, [record, record], risk_threshold=epsilon, model=model)
        assert compiled.verification.verified
        assert compiled.policy.contract.accept_clean_through == 1
        assert provenance['candidate_records'] == provenance['rescued_records'] == 1
        assert provenance['scores'][0]['estimated_risk'] == 0
        loaded = load_protocol(compiled.save_bundle(tmp_path / str(epsilon)))
        adapter = LikelihoodCandidate(model, risk_threshold=epsilon, rescues=[record])
        for cycles in (1, 3):
            a = ProtocolSampler(adapter, noise_model=CustomErrorModel(p_meas=.1), cycles=cycles).sample(1000, 181)
            b = ProtocolSampler(loaded, noise_model=CustomErrorModel(p_meas=.1), cycles=cycles).sample(1000, 181)
            assert (a.accepted, a.logical_errors) == (b.accepted, b.logical_errors)
        empty, _ = postselect_likelihood(parent, [], model=model)
        strict = postselect_order_one(parent)
        assert empty.policy.nodes == strict.policy.nodes
        assert empty.policy.root == strict.policy.root
    assert parent.policy.nodes is old_nodes and parent.verification is old_report


def test_full_history_and_input_frame_coordinates(flagged_parent):
    model = FullHistoryLikelihood(flagged_parent, noise_model=CustomErrorModel(p_meas=.01), input_error_probability=0)
    first = (('Z', 3, 0), ('X', 0, 0))
    other = (('Z', 3, 1), ('X', 0, 0))
    assert model.score(first).risk == 0
    assert model.score([list(step) for step in first]) == model.score(first)
    assert model.score(other).risk is None  # Same flags; different syndrome needs another event.
    for x, z in itertools.product(range(4), repeat=2):
        frame = x | (z << flagged_parent.code.n)
        measurements = [(name, flagged_parent.gadgets[name].basis, 3, f,
            s ^ model.space.component_syndrome(frame, 'X' if name == 'Z' else 'Z')) for name, f, s in first]
        assert model.normalize_record(measurements, frame_x=x, frame_z=z) == first
    with pytest.raises(ValueError, match='complete'):
        model.score(first[:1])
    with pytest.raises(ValueError, match='action'):
        model.score((('X', 0, 0), ('Z', 0, 0)))
    with pytest.raises(ValueError, match='widths'):
        model.score((('Z', 4, 0), ('X', 0, 0)))


def test_stabilizer_equivalent_physical_frames_have_identical_logical_risk(flagged_parent):
    parent = flagged_parent
    nodes = tuple(replace(node, frame=node.frame ^ 3) if node.kind == 'accept' else node
                  for node in parent.policy.nodes)
    equivalent = AdaptiveSEProtocol(parent.code, replace(parent.policy, nodes=nodes))
    equivalent.verify()
    record = (('Z', 3, 0), ('X', 0, 0))
    assert FullHistoryLikelihood(equivalent).score(record) == FullHistoryLikelihood(parent).score(record)


@pytest.mark.parametrize('compressed', (False, True))
def test_discovery_and_cli_frozen_bundles(flagged_parent, tmp_path, compressed):
    from spiderwarp.se_experiment import main
    model = FullHistoryLikelihood(flagged_parent, noise_model=CustomErrorModel(p_meas=.1))
    discovery = collect_likelihood_records(model, shots=100, cycles=(1,), seed=37)
    assert discovery == collect_likelihood_records(model, shots=100, cycles=(1,), seed=37)
    assert discovery['selection_inputs'].endswith('no failure labels')
    assert all(not model.replay(tuple(r)).protected for r in discovery['records'])
    parent_path = flagged_parent.save_bundle(tmp_path / 'parent')
    import gzip
    raw = json.dumps(dict(records=[(('Z', 3, 0), ('X', 0, 0))])).encode()
    records_path = tmp_path / ('records.json.gz' if compressed else 'records.json')
    records_path.write_bytes(gzip.compress(raw) if compressed else raw)
    main(['build-likelihood', str(parent_path), '--records', str(records_path),
          '--out', str(tmp_path / 'derived'), '--emit', 'none'])
    for path in (tmp_path / 'derived').glob('epsilon_*'):
        protocol = load_protocol(path)
        provenance = json.loads((path / 'likelihood.json').read_text())
        assert protocol.verification.verified
        assert provenance['risk_threshold'] in (0.01, 0.05)
        assert provenance['rescued_records'] == 1
        assert provenance['parent_protocol_hash']
    assert len(list((tmp_path / 'derived').glob('epsilon_*/protocol.json'))) == 2


@pytest.mark.parametrize('threshold', (-1, 1.1, float('nan')))
def test_invalid_risk_thresholds(flagged_parent, threshold):
    with pytest.raises(ValueError, match='risk_threshold'):
        postselect_likelihood(flagged_parent, [], risk_threshold=threshold)


def test_internal_frame_nodes_translate_both_evidence_and_rescue_edges(flagged_parent):
    parent = flagged_parent
    offset = 1 | (2 << parent.code.n)
    nodes = []
    space = parent.policy.decoder.space
    for node in parent.policy.nodes:
        if node.kind == 'extract':
            g = parent.gadgets[node.action]
            syndrome = space.component_syndrome(offset, 'X' if g.basis == 'Z' else 'Z')
            node = replace(node, children={(f, s ^ syndrome): child for (f, s), child in node.children.items()})
        elif node.kind == 'accept':
            node = replace(node, frame=node.frame ^ offset)
        nodes.append(node)
    nodes.append(PolicyNode('frame', frame=offset, next=parent.policy.root))
    shifted = AdaptiveSEProtocol(parent.code, replace(parent.policy, nodes=tuple(nodes), root=len(nodes) - 1))
    shifted.verify()
    record = (('Z', 3, 0), ('X', 0, 0))
    model = FullHistoryLikelihood(shifted)
    assert model.score(record) == FullHistoryLikelihood(parent).score(record)
    compiled, _ = postselect_likelihood(shifted, [record], model=model)
    reference = LikelihoodCandidate(model, rescues=[record])
    for basis in ('0', '+'):
        a = ProtocolSampler(reference, prepared_state=basis, cycles=3).sample(500, 47)
        b = ProtocolSampler(compiled, prepared_state=basis, cycles=3).sample(500, 47)
        assert (a.accepted, a.logical_errors) == (b.accepted, b.logical_errors)


def test_full_size_likelihood_uses_complete_logical_paulis():
    path = Path(__file__).resolve().parents[1] / 'build/se_32_20_4'
    if not (path / 'protocol.json').exists():
        pytest.skip('Full-size integration requires the saved original controller')
    parent = load_protocol(path)
    model = FullHistoryLikelihood(parent)
    if 'Z_d1' in parent.gadgets:
        # Legacy fixture: same early flag, distinguished by later syndromes.
        good = (('Z_d4', 4, 8), ('X_d2', 0, 15), ('Z_d1', 0, 0))
        risky = (('Z_d4', 4, 8), ('X_d2', 0, 0), ('Z_d2', 2, 2))
        assert model.score(good).risk == 0
        assert model.score(risky).risk > .05
        assert model.score(good).risk_upper_bound > .99
        assert not model.replay(good).protected
    else:
        # Current bounded parent removed d1 and its old three-reading records.
        good = (('Z_d4', 0, 0), ('X_d4', 0, 0))
        assert model.replay(good).protected
        assert 0 < model.score(good).evidence_probability < 1
        assert model.score(good).logical_classes
    assert 0 < model.score(good).omitted_probability < .05
    # The probability target uses precisely the final sampler's decoding ties.
    from spiderwarp.lut_decoder import LutDecoder
    import numpy as np
    for component, checks in (('X', parent.code.H_z), ('Z', parent.code.H_x)):
        final = LutDecoder(checks)
        for syndrome, physical in model.decoder.physical[component].items():
            bits = np.array([(syndrome >> i) & 1 for i in range(len(checks))], dtype=np.uint8)
            correction = sum(int(value) << q for q, value in enumerate(final.get_mwr(bits)))
            expected = physical if component == 'X' else physical >> parent.code.n
            assert correction == expected

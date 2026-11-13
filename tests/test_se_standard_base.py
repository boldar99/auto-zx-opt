"""Direct flag-free decoding and unchanged flagged recovery on the real code."""
from functools import lru_cache
from pathlib import Path

import pytest

from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def pair():
    paths = [ROOT / 'build' / name for name in
             ('se_32_20_4', 'se_postselected_32_20_4')]
    if any(not (path / 'protocol.json').exists() for path in paths):
        pytest.skip('Build the full-size flagged-recovery bundle first')
    return tuple(load_protocol(path) for path in paths)


def accepted_records(protocol):
    from spiderwarp.se_postselection import OrderOneEvidence
    evidence = OrderOneEvidence(protocol)
    nodes = protocol.policy.nodes

    @lru_cache(None)
    def visit(index, state):
        if not state:
            return ()
        node = nodes[index]
        if node.kind == 'reject':
            return ()
        if node.kind == 'accept':
            return (((), node.frame),)
        if node.kind == 'frame':
            return tuple((record, frame ^ node.frame) for record, frame in
                         visit(node.next, evidence.translate(state, node.frame)))
        groups = evidence.transition(state, node.action)
        return tuple((((node.action, flags, syndrome),) + record, frame)
                     for (flags, syndrome), child in node.children.items()
                     for record, frame in visit(child, groups.get((flags, syndrome), frozenset())))

    return dict(visit(protocol.policy.root, evidence.initial))


def test_every_flagged_record_keeps_the_exact_recovery_and_frame(pair):
    parent, protocol = pair
    flagged = lambda records: {r: f for r, f in records.items() if any(step[1] for step in r)}
    old, new = flagged(accepted_records(parent)), flagged(accepted_records(protocol))
    assert old and old == new
    assert any(len(record) == 3 for record in new)
    assert any(protocol.policy.decoder.space.weight(frame) > 1 for frame in new.values())


def test_every_flag_free_accept_is_exactly_an_incoming_css_one_syndrome(pair):
    _, protocol = pair
    decoder = protocol.policy.decoder
    expected = { (('Z_d4', 0, sx), ('X_d4', 0, sz)):
                 decoder.physical['X'][sx] ^ decoder.physical['Z'][sz]
                 for sx, wx in decoder.weights['X'].items()
                 for sz, wz in decoder.weights['Z'].items() if max(wx, wz) <= 1 }
    actual = {r: f for r, f in accepted_records(protocol).items() if not any(step[1] for step in r)}
    assert len(expected) == 1089 and expected == actual
    assert protocol.verification.verified
    assert protocol.policy.contract.correctable == 0
    assert protocol.policy.contract.logical_order == 1


@pytest.mark.parametrize('basis', ('0', '+'))
def test_ambiguous_two_input_syndromes_reject_but_single_inputs_correct(pair, basis):
    _, protocol = pair
    experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
    for qubits in ((), (0,), (2, 3), (1, 4)):
        simulator = experiment.initial.copy(seed=17)
        for q in qubits:
            simulator.z(q)
        result = protocol.run(simulator)
        assert result.accepted == (len(qubits) <= 1)
        if result.accepted:
            assert not experiment.logical_failure(simulator, frame_x=result.frame_x, frame_z=result.frame_z)


@pytest.mark.parametrize('basis', ('0', '+'))
def test_external_frame_is_removed_before_interpreting_new_incoming_error(pair, basis):
    _, protocol = pair
    experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
    simulator = experiment.initial.copy(seed=17)
    # A known frame may itself have weight two; only the new incoming error is
    # limited to CSS one. Check both components and different physical sites.
    x, z = 3, (1 << 8) | (1 << 17)
    for q in range(protocol.code.n):
        if x >> q & 1: simulator.x(q)
        if z >> q & 1: simulator.z(q)
    simulator.x(7)
    simulator.z(23)
    result = protocol.run(simulator, frame_x=x, frame_z=z)
    assert result.accepted
    assert result.frame_x == x ^ (1 << 7)
    assert result.frame_z == z ^ (1 << 23)
    assert not experiment.logical_failure(simulator, frame_x=result.frame_x, frame_z=result.frame_z)


@pytest.mark.parametrize('basis', ('0', '+'))
def test_clean_repeated_cycles_and_bundle_roundtrip(pair, tmp_path, basis):
    _, protocol = pair
    loaded = load_protocol(protocol.save_bundle(tmp_path / 'bundle'))
    assert loaded.policy.nodes == protocol.policy.nodes
    result = ProtocolSampler(loaded, noise_model=CustomErrorModel(),
                             prepared_state=basis, cycles=10).sample(20, 17)
    assert result.accepted == 20 and result.logical_errors == 0
    assert result.diagnostics.accepted_cnots == 20 * 10 * 312

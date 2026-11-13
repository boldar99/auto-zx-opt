"""Allocation changes preserve recovery while reducing actual memory channels."""
from collections import Counter
from pathlib import Path

import pytest

from spiderwarp.noise_model import CustomErrorModel, default_noise_model
from spiderwarp.se_likelihood import noise_response_channels
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment
from spiderwarp.se_scheduling import reschedule_d2, scheduled_resources
from test_se_standard_base import accepted_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def pair():
    parent = load_protocol(ROOT / 'build/se_postselected_32_20_4')
    protocol, audit = reschedule_d2(parent)
    return parent, protocol, audit


def test_all_accepted_records_frames_and_paths_are_preserved(pair):
    parent, protocol, audit = pair
    assert protocol.policy.nodes == parent.policy.nodes
    assert protocol.policy.root == parent.policy.root
    assert accepted_records(protocol) == accepted_records(parent)
    assert len(accepted_records(protocol)) == 2371
    assert protocol.verification.verified
    assert protocol.required_qubits == 60
    assert protocol.verification.worst_case_cnots == 433
    assert protocol.verification.max_extractions == 3
    for name in ('X_d4', 'Z_d4'):
        assert protocol.gadgets[name].circuit == parent.gadgets[name].circuit


@pytest.mark.parametrize('name,depth,idles', [('X_d2', 45, 1408), ('Z_d2', 43, 1344)])
def test_idle_noise_is_proportional_to_scheduled_wait_not_gate_count(pair, name, depth, idles):
    parent, protocol, audit = pair
    before, after = (scheduled_resources(p.gadgets[name].circuit, 32) for p in (parent, protocol))
    assert before['cnots'] == after['cnots'] == 121
    assert after['qubits'] == 60
    assert after['depth'] == depth and after['data_idle_units'] == idles
    assert idles == 32 * (depth - 1)
    assert idles < before['data_idle_units']
    assert audit['stages'][name]['after'] == after
    # Compare the production noise responses with every noise source enabled
    # except memory. Their independent channel distributions are identical,
    # retaining exclusivity of alternatives at each physical location.
    noise = default_noise_model()
    noise.p_linear_mem = 0
    channels = lambda g, model: Counter(tuple(sorted(c.items())) for c in noise_response_channels(g, model))
    assert channels(parent.gadgets[name], noise) == channels(protocol.gadgets[name], noise)
    memory = CustomErrorModel(p_linear_mem=1.26e-5)
    mass = lambda g: sum(sum(c.values()) for c in noise_response_channels(g, memory))
    assert mass(protocol.gadgets[name]) < mass(parent.gadgets[name])


@pytest.mark.parametrize('basis', ('0', '+'))
def test_physical_inputs_external_frame_and_repeated_roundtrip(pair, tmp_path, basis):
    _, protocol, _ = pair
    experiment = _StateExperiment(protocol.code, 60, basis)
    incoming_x, incoming_z = 3, (1 << 8) | (1 << 17)
    for x in (0, *(1 << q for q in range(32))):
        for z in (0, *(1 << q for q in range(32))):
            sim = experiment.initial.copy(seed=17)
            for q in range(32):
                if (x ^ incoming_x) >> q & 1: sim.x(q)
                if (z ^ incoming_z) >> q & 1: sim.z(q)
            result = protocol.run(sim, frame_x=incoming_x, frame_z=incoming_z)
            assert result.accepted
            assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
    loaded = load_protocol(protocol.save_bundle(tmp_path))
    assert loaded.required_qubits == 60
    estimate = ProtocolSampler(loaded, noise_model=CustomErrorModel(), prepared_state=basis,
                               cycles=10).sample(20, 17)
    assert estimate.accepted == 20 and estimate.logical_errors == 0
    assert estimate.diagnostics.accepted_cnots == 20 * 10 * 312


def test_qubit_limit_is_enforced(pair):
    with pytest.raises(ValueError, match='Qubit limit'):
        reschedule_d2(pair[0], peak_qubits=59)

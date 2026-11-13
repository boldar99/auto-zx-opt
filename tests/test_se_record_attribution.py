from pathlib import Path

import pytest

from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler
from spiderwarp.se_record_attribution import normalize_record, persistent_onset, sample_records


def test_persistent_onset_handles_transient_failures_and_recovery():
    assert persistent_onset([4]) == 0
    assert persistent_onset([0, 3, 7]) == 1
    assert persistent_onset([1, 0, 2, 4]) == 2
    assert persistent_onset([1, 0, 0, 2]) == 3
    with pytest.raises(ValueError):
        persistent_onset([1, 0])


@pytest.fixture(scope='module')
def protocol():
    root = Path(__file__).resolve().parents[1]
    return load_protocol(root / 'build/se_postselected_32_20_4')


@pytest.mark.parametrize('basis', ('0', '+'))
@pytest.mark.parametrize('cycles', (1, 10))
def test_diagnostic_copies_leave_production_execution_identical(protocol, basis, cycles):
    sampler = ProtocolSampler(protocol, prepared_state=basis, cycles=cycles, collect_paths=False)
    expected = sampler.sample(1000, 2026111317)
    actual = sample_records(sampler, 1000, 2026111317)
    for name, count in dict(accepted=expected.accepted, logical_errors=expected.logical_errors,
            cycles_started=expected.diagnostics.cycles_started,
            cycles_completed=expected.diagnostics.cycles_completed,
            total_cnots=expected.diagnostics.total_cnots,
            accepted_cnots=expected.diagnostics.accepted_cnots).items():
        assert actual['totals'][name] == count
    assert len(actual['failure_traces']) == expected.logical_errors


def test_record_syndromes_remove_the_incoming_cumulative_frame(protocol):
    space = protocol.policy.decoder.space
    x, z = 1, 1 << 17
    raw = [('Z_d4', 'Z', 4, 0, space.component_syndrome(x, 'X')),
           ('X_d4', 'X', 4, 0, space.component_syndrome(z << 32, 'Z'))]
    assert normalize_record(protocol, raw, x, z) == (('Z_d4', 0, 0), ('X_d4', 0, 0))

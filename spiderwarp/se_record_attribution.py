"""Read-only record diagnostics around the unchanged production interpreter.

Only fully accepted trajectories enter record/error statistics. Intermediate
ideal-decoder queries are made on saved simulator copies, after the endpoint
is known to fail. They never influence execution or acceptance.
"""
from collections import Counter

import numpy as np

from spiderwarp.se_protocol_simulation import ProtocolSampler, _NoisySimulator


FIELDS = ('occurrences', 'shots_present', 'failed_shots_present', 'onset_failures',
          'last_shots', 'last_failures')
GRANULARITIES = ('record', 'flags', 'syndromes', 'branch')


def normalize_record(protocol, measurements, frame_x=0, frame_z=0):
    """Remove the incoming cumulative frame, retaining within-cycle history."""
    frame = frame_x | (frame_z << protocol.code.n)
    space = protocol.policy.decoder.space
    return tuple((name, flags, syndrome ^ space.component_syndrome(
        frame, 'X' if basis == 'Z' else 'Z'))
        for name, basis, _, flags, syndrome in measurements)


def group_keys(record):
    if not any(flags for _, flags, _ in record):
        branch = 'all_flags_zero'
    elif record[0][1]:
        branch = 'first_d4_flagged'
    elif len(record) > 1 and record[1][0] == 'X_d4' and record[1][1]:
        branch = 'second_d4_flagged'
    else:
        branch = 'other_flagged'
    return (record, tuple((name, flags) for name, flags, _ in record),
            tuple((name, syndrome) for name, _, syndrome in record), branch)


def logical_mask(experiment, simulator, frame_x=0, frame_z=0):
    """The exact endpoint decoder's 20-bit logical result, without mutation."""
    frame_mask = frame_x if experiment.prepared_state == '0' else frame_z
    frame = np.array([(frame_mask >> q) & 1 for q in range(experiment.code.n)], dtype=np.uint8)
    checks = experiment.code.H_z if experiment.prepared_state == '0' else experiment.code.H_x
    syndrome = experiment._bits(simulator, experiment.checks) ^ (checks @ frame % 2)
    correction = experiment.decoder.get_mwr(syndrome)
    result = (experiment._bits(simulator, experiment.logicals)
              ^ (experiment.logical_rows @ (correction ^ frame) % 2))
    return sum(int(bit) << i for i, bit in enumerate(result))


def persistent_onset(masks):
    """First cycle after the last zero decoded state of an endpoint failure.

    This is a retrospective diagnostic convention, not a causal fault oracle.
    The return value is a zero-based cycle index.
    """
    if not masks or masks[-1] == 0:
        raise ValueError('Persistent onset requires a failing endpoint')
    return max((i + 1 for i, mask in enumerate(masks[:-1]) if mask == 0), default=0)


def sample_records(sampler: ProtocolSampler, shots, seed, *, keep_failures=True):
    rng = np.random.default_rng(seed)
    protocol, experiment = sampler.protocol, sampler.experiment
    groups = {name: {} for name in GRANULARITIES}
    record_frames, failures = {}, []
    totals = Counter(shots=shots)
    onset_cycles, transitions, shot_flags = Counter(), Counter(), {}

    def row(granularity, key):
        return groups[granularity].setdefault(key, [0] * len(FIELDS))

    for shot in range(shots):
        shot_seed = int(rng.integers(0, 2**63))
        simulator = experiment.initial.copy(seed=shot_seed)
        noisy = _NoisySimulator(simulator, sampler.circuits, sampler.noise, sampler.corrections)
        frame_x = frame_z = cost = 0
        records, frames, snapshots = [], [], []
        for cycle in range(sampler.cycles):
            totals['cycles_started'] += 1
            incoming_x, incoming_z = frame_x, frame_z
            result = (protocol.run(noisy) if cycle == 0 else
                      protocol.run(noisy, frame_x=frame_x, frame_z=frame_z))
            cost += sum(sampler.costs[name] for name, _, _, _, _ in result.measurements)
            if not result.accepted:
                break
            totals['cycles_completed'] += 1
            record = normalize_record(protocol, result.measurements, incoming_x, incoming_z)
            correction = ((result.frame_x ^ incoming_x)
                          | ((result.frame_z ^ incoming_z) << protocol.code.n))
            if record in record_frames:
                assert record_frames[record] == correction
            else:
                record_frames[record] = correction
            records.append(record)
            frame_x, frame_z = result.frame_x, result.frame_z
            frames.append((frame_x, frame_z))
            if cycle + 1 < sampler.cycles:
                snapshots.append(simulator.copy(seed=0))
        totals['total_cnots'] += cost
        if len(records) != sampler.cycles:
            continue
        totals['accepted'] += 1
        totals['accepted_cnots'] += cost
        endpoint = logical_mask(experiment, simulator, frame_x, frame_z)
        failed = bool(endpoint)
        totals['logical_errors'] += failed
        keys = [group_keys(record) for record in records]
        for i, granularity in enumerate(GRANULARITIES):
            for key in keys:
                row(granularity, key[i])[0] += 1
            for key in {key[i] for key in keys}:
                counts = row(granularity, key)
                counts[1] += 1
                counts[2] += failed
            last = row(granularity, keys[-1][i])
            last[4] += 1
            last[5] += failed
        flag_bucket = 'any_flags' if any(flags for record in records for _, flags, _ in record) else 'no_flags'
        flags = shot_flags.setdefault(flag_bucket, [0, 0])
        flags[0] += 1
        flags[1] += failed
        if failed:
            masks = [logical_mask(experiment, snapshot, *frame)
                     for snapshot, frame in zip(snapshots, frames)] + [endpoint]
            onset = persistent_onset(masks)
            onset_cycles[onset + 1] += 1
            for i, granularity in enumerate(GRANULARITIES):
                row(granularity, keys[onset][i])[3] += 1
            prior = 0
            for mask in masks:
                transitions['changes'] += mask != prior
                transitions['returns_to_zero'] += bool(prior) and not mask
                prior = mask
            if keep_failures:
                failures.append(dict(shot=shot, simulator_seed=shot_seed, records=records,
                    frames=frames, logical_masks=masks, onset_cycle=onset + 1))
    for name in ('accepted', 'logical_errors', 'cycles_started', 'cycles_completed',
                 'total_cnots', 'accepted_cnots'):
        totals.setdefault(name, 0)
    for values in groups.values():
        assert sum(row[0] for row in values.values()) == totals['accepted'] * sampler.cycles
        assert sum(row[3] for row in values.values()) == totals['logical_errors']
        assert sum(row[4] for row in values.values()) == totals['accepted']
        assert sum(row[5] for row in values.values()) == totals['logical_errors']
    assert sum(onset_cycles.values()) == totals['logical_errors']
    assert sum(values[0] for values in shot_flags.values()) == totals['accepted']
    return dict(totals=dict(totals),
        groups={name: [[key, counts] for key, counts in sorted(values.items())]
                for name, values in groups.items()},
        record_frames=[[key, frame] for key, frame in sorted(record_frames.items())],
        onset_cycles=dict(onset_cycles), transitions=dict(transitions), shot_flags=shot_flags,
        failure_traces=failures)

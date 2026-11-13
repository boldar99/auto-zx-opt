"""Compare compiled candidates with explicit-Pauli decisions on 400k old traces."""
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path

from spiderwarp.se_adaptive_experiments import AdaptiveCSSCandidate, CSSRecoverCandidate
from spiderwarp.se_protocol_io import load_protocol

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_trace_walker', ROOT / 'analysis/32_20_4_postselected_validation/replay_compiled.py')
walker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walker)


if __name__ == '__main__':
    parent = load_protocol('build/se_32_20_4')
    names = sorted(parent.gadgets)
    models = {'css': AdaptiveCSSCandidate(parent), 'css_recover_safe': CSSRecoverCandidate(parent, 'css-recover-safe')}
    protocols = {tag: load_protocol('build/adaptive_' + tag + '_32_20_4') for tag in models}
    space, decoder = parent.policy.decoder.space, parent.policy.decoder
    totals = {}
    for split in ('discovery', 'validation'):
        for suffix in ('zero', 'plus'):
            counts = {tag: Counter() for tag in models}
            path = ROOT / 'analysis/32_20_4_branch_assessment' / (split + '_' + suffix + '.jsonl.gz')
            with gzip.open(path, 'rt') as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    observations = row[4]
                    expected = walker.execute(parent, observations)
                    assert expected[0] == bool(row[2])
                    preferred = space.reduce(expected[1] | (expected[2] << parent.code.n))
                    record = tuple((names[aid], f, s) for aid, f, s in observations)
                    for tag, model in models.items():
                        correction = None
                        if expected[0]:
                            if tag == 'css':
                                signature = space.pauli_syndrome(preferred)
                                if model.unique['Z'][signature & decoder.x_mask] and model.unique['X'][signature >> decoder.x_width]:
                                    correction = preferred
                            else:
                                correction = model.rescue_frame(record, preferred)
                        actual = walker.execute(protocols[tag], observations)
                        assert actual[0] == (correction is not None)
                        assert actual[3] <= expected[3]
                        if actual[0]:
                            frame = actual[1] | (actual[2] << parent.code.n)
                            assert space.reduce(frame) == correction
                            counts[tag]['accepted'] += 1
                            counts[tag]['retargeted'] += correction != preferred
                        counts[tag]['shots'] += 1
                        if index < 1000:
                            shifted = walker.execute(protocols[tag], observations, frame_x=1,
                                frame_z=1 << 17, shift_observations=True)
                            assert shifted[0] == actual[0] and shifted[3] == actual[3]
                            if actual[0]:
                                shifted_frame = shifted[1] | (shifted[2] << parent.code.n)
                                assert space.reduce(shifted_frame ^ (1 | (1 << (17 + parent.code.n)))) == correction
                            counts[tag]['incoming_frame_checks'] += 1
            totals[split + '/' + suffix] = {tag: dict(c) for tag, c in counts.items()}
            print(split, suffix, totals[split + '/' + suffix], flush=True)
    (OUT / 'replay.json').write_text(json.dumps(dict(complete=True, observations_per_method=400000,
        all_decisions_and_frames_match=True, incoming_frame_checks_per_method=4000,
        performance_note='These traces verify controller semantics only; retargeted logical error rates are measured with fresh quantum simulations.',
        totals=totals), indent=2) + '\n')

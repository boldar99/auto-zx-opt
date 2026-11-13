"""Extend the archived quick samples into separate full-budget Steane PS runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from spiderwarp.se_protocol_io import atomic_write, read_manifest
from spiderwarp.se_sweep import engine_identity, num_samples, run_sweep, small_sweep


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_write(Path(path), json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('d4', 'ladder-ancilla'), required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    tag = args.mode.replace('-', '_')
    source = Path('benchmarks') / ('steane_css_ps_' + tag + '_10k')
    results = Path('benchmarks') / ('steane_css_ps_' + tag + '_100k')
    bundle = Path('build') / ('steane_css_ps_' + tag)
    seed_path = results / 'seed_provenance.json'
    old = read(source / 'run.json')
    assert old['status'] == 'complete' and old['actual_attempted_shots'] == 932918
    assert all(read(p)['task']['engine'] == engine_identity() for p in (source / 'tasks').glob('*.json'))
    original_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ('tasks', 'batches') for p in sorted((source / folder).glob('*'))}
    if not seed_path.exists():
        results.mkdir(parents=True, exist_ok=True)
        for relative in original_hashes:
            target = results / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == (source / relative).read_bytes()
            else:
                atomic_write(target, (source / relative).read_bytes())
        write(seed_path, dict(source=str(source), archived_quick_shots=932918,
            archived_quick_run=old, archived_quick_validation=read(source / 'validation.json'),
            source_raw_hashes=original_hashes))
    assert read(seed_path)['source_raw_hashes'] == original_hashes
    configs = small_sweep()
    assert read(source / 'config.json') == configs
    targets = [num_samples(c.get('p', c.get('noise', {}).get('p_2q')), c['cycles'],
                          shots=100000, max_shots=1000000) for c in configs]
    record = dict(old, status='running',
        command=f'python analysis/32_20_4_syndrome_uniqueness/full_steane_ps.py --mode {args.mode} --jobs {args.jobs}',
        protocol_hash=read_manifest(bundle)[1]['protocol_hash'], total_target_shots=sum(targets),
        target_shots=targets, shot_floor=100000, max_shots=1000000, batch_size=10000,
        jobs=args.jobs, budget_scale=1.0, reused_quick_attempts=932918,
        budget_rounding='Exact original full production allocation',
        batch_partition='Original 1k batches preserved; additional batches use up to 10k shots',
        validation='validation.json')
    for key in ('actual_attempted_shots', 'actual_cycles_started', 'actual_cycles_completed', 'wall_seconds', 'real_resume_new_shots'):
        record.pop(key, None)
    write(results / 'config.json', configs)
    write(results / 'run.json', record)
    start = time.monotonic()
    rows = run_sweep(bundle, configs, results, shots=100000, max_shots=1000000,
                     batch_size=10000, jobs=args.jobs, seed=20261112, allocation='inverse-square')
    assert len(rows) == 32 and all(r['shots'] == r['target_shots'] for r in rows)
    for old_path in (source / 'batches').glob('*.csv'):
        prior = list(csv.DictReader(old_path.open()))
        current = list(csv.DictReader((results / 'batches' / old_path.name).open()))
        assert current[:len(prior)] == prior
    assert all(hashlib.sha256((source / name).read_bytes()).hexdigest() == value
               for name, value in original_hashes.items())
    record.update(status='complete', actual_attempted_shots=sum(r['shots'] for r in rows),
        additional_attempted_shots=sum(r['shots'] for r in rows)-932918,
        actual_cycles_started=sum(r['cycles_started'] for r in rows),
        actual_cycles_completed=sum(r['cycles_completed'] for r in rows),
        additional_sampling_wall_seconds=time.monotonic()-start,
        original_quick_batches_preserved=True)
    write(results / 'run.json', record)
    print(args.mode, 'complete:', record['actual_attempted_shots'], 'shots;',
          record['additional_attempted_shots'], 'new; all archived quick batches preserved', flush=True)


if __name__ == '__main__':
    main()

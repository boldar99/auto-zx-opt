"""Sample both retained Steane CSS-PS variants at one tenth production budgets."""
import argparse
import json
from pathlib import Path
import time

from spiderwarp.se_experiment import emit_protocol
from spiderwarp.se_protocol_io import atomic_write, read_manifest
from spiderwarp.se_sweep import num_samples, run_sweep, small_sweep
from spiderwarp.steane_qec import build_steane_qec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    configurations = small_sweep()
    for mode in ('d4', 'ladder-ancilla'):
        name = 'steane_css_ps_' + mode.replace('-', '_')
        bundle, results = Path('build') / name, Path('benchmarks') / (name + '_10k')
        protocol = build_steane_qec(mode, postselection='unique-css-2', output_dir=bundle)
        emit_protocol(protocol, bundle.resolve(), 'hugr')
        targets = [num_samples(c.get('p', c.get('noise', {}).get('p_2q')), c['cycles'],
                               shots=10000, max_shots=100000) for c in configurations]
        record = dict(status='running', mode=mode, postselection='unique-css-2',
            protocol_hash=read_manifest(bundle)[1]['protocol_hash'],
            command='python analysis/32_20_4_syndrome_uniqueness/quick_steane_ps.py --jobs ' + str(args.jobs),
            configurations=32, total_target_shots=sum(targets), target_shots=targets,
            shot_floor=10000, max_shots=100000, batch_size=1000, jobs=args.jobs, seed=20261112,
            allocation='inverse-square', budget_scale=0.1,
            budget_rounding='Recompute the same allocation at a 10k floor / 100k cap; fractional targets round up',
            uniqueness='All data errors with each CSS component weight <=2 must agree modulo stabilizers',
            check_timing='After two identical reads per CSS basis, adjusted for the cumulative Pauli frame',
            peak_qubits=66, initial_data_idles=True, data_waits_during_retries=True,
            preparation='Ideal once per trajectory; noisy serial ancilla attempts',
            final_decoding='Ideal once after the last accepted cycle',
            retry_limit=None, statistic='Conditional 20-logical-qubit block LER, separate bases; 95% Wilson')
        atomic_write(results / 'config.json', json.dumps(configurations, indent=2) + '\n')
        atomic_write(results / 'run.json', json.dumps(record, indent=2) + '\n')
        start = time.monotonic()
        rows = run_sweep(bundle, configurations, results, shots=10000, max_shots=100000,
                         batch_size=1000, jobs=args.jobs, seed=20261112, allocation='inverse-square')
        assert len(rows) == 32 and all(r['shots'] == r['target_shots'] for r in rows)
        record.update(status='complete', actual_attempted_shots=sum(r['shots'] for r in rows),
                      wall_seconds=time.monotonic() - start,
                      actual_cycles_started=sum(r['cycles_started'] for r in rows),
                      actual_cycles_completed=sum(r['cycles_completed'] for r in rows))
        atomic_write(results / 'run.json', json.dumps(record, indent=2) + '\n')
        print(f'Complete: {mode}; {record["actual_attempted_shots"]:,} shots', flush=True)


if __name__ == '__main__':
    main()

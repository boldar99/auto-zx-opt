"""Rebuild and benchmark bounded CSS-order-two adaptive PS controllers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from spiderwarp.se_comparison_names import METHODS
from spiderwarp.se_protocol_io import atomic_write, load_protocol, read_manifest
from spiderwarp.se_sweep import engine_identity, num_samples, run_sweep, small_sweep

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SHOT_FLOOR = 100000
SHOT_CAP = 1000000
BUNDLES = {
    'adaptive_history': 'se_postselected_32_20_4',
    'adaptive_order_two': 'adaptive_order_two_32_20_4',
    'adaptive_probability_10': 'adaptive_probability_10_32_20_4',
    'adaptive_probability_25': 'adaptive_probability_25_32_20_4',
    'adaptive_no_flags': 'adaptive_css_no_flags_32_20_4',
    'steane_fixed_ps': 'steane_single_d3_fixed_ps',
}


def write(path, value):
    atomic_write(Path(path), json.dumps(value, indent=2) + '\n')


def sweep(key, *, jobs=8, shots=SHOT_FLOOR, max_shots=SHOT_CAP):
    method = METHODS[key]
    bundle = ROOT / 'build' / BUNDLES[key]
    results = ROOT / 'benchmarks' / method.results
    _, manifest = read_manifest(bundle)
    configurations = small_sweep()
    targets = [num_samples(c.get('p', c.get('noise', {}).get('p_2q')), c['cycles'],
                           shots=shots, max_shots=max_shots) for c in configurations]
    run_path = results / 'run.json'
    previous = json.loads(run_path.read_text()) if run_path.exists() else {}
    record = dict(status='running', method=key, display_name=method.name,
                  protocol_hash=manifest['protocol_hash'], engine=engine_identity(),
                  seed=20261112, configurations=32, target_shots=targets,
                  total_target_shots=sum(targets), shot_floor=shots, max_shots=max_shots,
                  batch_size=10000, allocation='inverse-square', jobs=jobs,
                  input_scope='r+s<=2' if method.family == 'adaptive' else 'CSS unique correction through weight two',
                  initial_data_idles=True, data_preparation='Ideal once per trajectory',
                  ancilla_preparation='Noisy serial preparations within each executed gadget',
                  peak_qubits=method.qubits,
                  final_decoding='Ideal once after the last accepted cycle',
                  statistic='Conditional block LER and gates on accepted trajectories only; AR=accepted/attempted; 95% Wilson',
                  accepted_gate_statistic='accepted_cnots / accepted / requested_cycles',
                  baseline_excluded=True)
    if method.family == 'steane':
        record.update(preparation_start=3, syndrome_confirmation='single-per-basis',
                      data_postselection='unique-css-2', retry_limit=None,
                      ancilla_preparation='Noisy d3 RUS until all preparation flags are zero')
    else:
        record.update(no_d1=True, max_extractions_per_cycle=3, inferred_order_two_rejected=True,
                      allowed_distance_paths=[[4, 4, 2], [4, 4], [4, 2]], accepted_correction_order=1,
                      mandatory_acceptance='Standard flag-conditioned base; all noiseless CSS-one inputs',
                      base_rule='No flags: incoming CSS syndromes with r_syn<=1; flags: history recovery',
                      single_internal_fault_acceptance_guarantee=False)
        if key == 'adaptive_no_flags':
            record.update(observed_flag_postselection='Reject any nonzero measured flag immediately',
                          flag_zero_is_not_actual_fault_zero=True,
                          max_extractions_per_cycle=2, allowed_distance_paths=[[4, 4]],
                          accepted_cycle_cnots=312,
                          incoming_correction='CSS component weights <=1; unique through CSS component weights <=2')
        elif key == 'adaptive_order_two':
            record.update(inferred_order_two_rejected=False,
                          mandatory_acceptance='Standard flag-conditioned base',
                          additional_acceptance='One final Pauli class modulo stabilizers among all r+s<=2 histories',
                          additional_rescue_orders=[[2, 0], [1, 1], [0, 2]],
                          all_accepted_order_two_correct=False,
                          exact_rescue_certificate_includes_memory=False)
        elif key.startswith('adaptive_probability_'):
            record.update(inferred_order_two_rejected=False,
                          mandatory_acceptance='Standard flag-conditioned base',
                          epsilon=int(key.rsplit('_', 1)[1]) / 100,
                          additional_acceptance='Estimated wrong MAP Pauli-class probability <= epsilon',
                          probability_model='Default hardware noise plus incoming iid Pauli prior, frozen before production',
                          probability_approximation='At most two nonzero response channels; includes scheduled memory noise',
                          all_orders_ler_guarantee=False)
    engine_matches = previous.get('engine') == record['engine']
    if method.family == 'steane' and previous.get('engine'):
        # Steane execution does not import the adaptive interpreter/synthesizer.
        # Its completed run began while that unrelated source was being rebuilt.
        before, after = json.loads(json.dumps(previous['engine'])), json.loads(json.dumps(record['engine']))
        for engine in (before, after):
            for name in ('_se_adaptive.py', 'adaptive_steane_se.py'):
                engine['sources'].pop(name, None)
        engine_matches = before == after
    if previous.get('status') == 'complete' and engine_matches and all(previous.get(k) == record[k] for k in
            ('protocol_hash', 'target_shots', 'seed')):
        for field in ('data_preparation', 'ancilla_preparation', 'peak_qubits', 'preparation_start',
                      'syndrome_confirmation', 'data_postselection', 'retry_limit', 'no_d1',
                      'max_extractions_per_cycle', 'inferred_order_two_rejected', 'mandatory_acceptance'):
            if field in record:
                previous[field] = record[field]
        previous.pop('preparation', None)
        if method.family == 'steane' and previous['engine'] != record['engine']:
            previous['engine_compatibility'] = dict(
                ignored_unexecuted_sources=['_se_adaptive.py', 'adaptive_steane_se.py'],
                reason='Completed Steane execution does not use the adaptive synthesizer/interpreter; all remaining engine fields and fixed program match',
                checked_current_engine=record['engine'])
        write(run_path, previous)
        print(key, 'completed run reused', flush=True)
        return
    write(results / 'config.json', configurations)
    write(run_path, record)
    started = time.monotonic()
    rows = run_sweep(bundle, configurations, results, shots=shots, max_shots=max_shots,
                     batch_size=10000, jobs=jobs, seed=20261112, allocation='inverse-square')
    assert len(rows) == 32 and all(r['shots'] == r['target_shots'] for r in rows)
    record.update(status='complete', attempted_shots=sum(r['shots'] for r in rows),
                  actual_attempted_shots=sum(r['shots'] for r in rows),
                  wall_seconds=time.monotonic() - started)
    write(run_path, record)
    print(key, 'complete:', record['attempted_shots'], 'attempts', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--shots', type=int, default=SHOT_FLOOR)
    parser.add_argument('--max-shots', type=int, default=SHOT_CAP)
    parser.add_argument('--only', choices=tuple(METHODS))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)
    if not args.plot_only:
        for key in ([args.only] if args.only else METHODS):
            sweep(key, jobs=args.jobs, shots=args.shots, max_shots=args.max_shots)
    if not args.only:
        from importlib import import_module
        import_module('analysis.32_20_4_bounded_comparison.present').main()


if __name__ == '__main__':
    main()

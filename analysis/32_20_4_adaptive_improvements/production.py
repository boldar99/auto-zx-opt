"""Full fixed-budget production sweep for a selected compiled CSS candidate."""
import argparse
import json
from pathlib import Path
import time

from spiderwarp.se_adaptive_experiments import compile_css_candidate, compile_css_recovery, VARIANTS
from spiderwarp.se_experiment import emit_protocol
from spiderwarp.se_protocol_io import atomic_write, load_protocol, read_manifest
from spiderwarp.se_sweep import num_samples, run_sweep, small_sweep


def paths(variant):
    tag = variant.replace('-', '_')
    return Path('build') / ('adaptive_' + tag + '_32_20_4'), Path('benchmarks') / ('adaptive_' + tag + '_100k')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=[v for v in VARIANTS if not v.startswith('css-separated')], required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--emit', choices=('none', 'guppy', 'hugr'), default='hugr')
    args = parser.parse_args()
    bundle, results = paths(args.variant)
    parent = load_protocol('build/se_32_20_4')
    if (bundle / 'protocol.json').exists():
        protocol = load_protocol(bundle)
    else:
        builder = compile_css_recovery if args.variant.startswith('css-recover') else compile_css_candidate
        protocol = builder(parent, args.variant, progress=lambda s: print(s, flush=True))
        protocol.save_bundle(bundle)
    provenance = dict(variant=args.variant, parent_protocol_hash=read_manifest('build/se_32_20_4')[1]['protocol_hash'],
        protocol_hash=read_manifest(bundle)[1]['protocol_hash'], css_component_bound=2,
        incoming_correction_uniqueness='Modulo stabilizers; each CSS component independently',
        additional_rule='Flagged histories require r+s<=1' if args.variant != 'css' else 'Original flag handling',
        extra_checks='Z_d2 then X_d2' if args.variant.endswith('check') else None,
        role_reversal=args.variant.endswith('reversed'), original_circuit_assets_unchanged=True,
        certificate='Clean mandatory acceptance; accepted-output logical protection through r+s<=1; no all-single-fault acceptance claim')
    if args.variant.startswith('css-recover'):
        provenance.update(additional_rule='Retain order-one records; rescue only with a common safe frame over incoming CSS<=1 plus at most one internal fault',
            rescue_requires_exact_data_class=args.variant.endswith(('exact', 'safe')),
            rescue_excludes_d1=args.variant.endswith('no-d1'),
            css_input_guard='Reject all stable unflagged weight-two input syndrome records')
        if args.variant.endswith('safe'):
            provenance.update(rescue_preserves_frame_syndrome=True,
                certificate='Original strict-PS contract independently verified: mandatory acceptance and logical preservation through r+s<=1; CSS filters through fault order two')
    atomic_write(bundle / 'experiment.json', json.dumps(provenance, indent=2) + '\n')
    emit_protocol(protocol, bundle.resolve(), args.emit)
    configs = small_sweep()
    targets = [num_samples(c.get('p', c.get('noise', {}).get('p_2q')), c['cycles'],
                          shots=100000, max_shots=1000000) for c in configs]
    record = dict(status='running', variant=args.variant, protocol_hash=provenance['protocol_hash'],
        command=f'python analysis/32_20_4_adaptive_improvements/production.py --variant {args.variant} --jobs {args.jobs}',
        configurations=32, total_target_shots=sum(targets), target_shots=targets,
        shot_floor=100000, max_shots=1000000, batch_size=10000, jobs=args.jobs, seed=20261112,
        allocation='inverse-square', css_component_bound=2, peak_qubits=protocol.required_qubits,
        initial_data_idles=True, final_decoding='Ideal once after the last accepted cycle',
        preparation='Ideal once per trajectory', statistic='Conditional 20-logical-qubit block LER; separate bases; 95% Wilson',
        selection='Rule frozen before full sweep; pilot seed 2026111301, production seed 20261112')
    atomic_write(results / 'config.json', json.dumps(configs, indent=2) + '\n')
    atomic_write(results / 'run.json', json.dumps(record, indent=2) + '\n')
    start = time.monotonic()
    rows = run_sweep(bundle, configs, results, shots=100000, max_shots=1000000,
                     batch_size=10000, jobs=args.jobs, seed=20261112, allocation='inverse-square')
    assert len(rows) == 32 and all(r['shots'] == r['target_shots'] for r in rows)
    record.update(status='complete', actual_attempted_shots=sum(r['shots'] for r in rows),
        actual_cycles_started=sum(r['cycles_started'] for r in rows),
        actual_cycles_completed=sum(r['cycles_completed'] for r in rows), wall_seconds=time.monotonic()-start)
    atomic_write(results / 'run.json', json.dumps(record, indent=2) + '\n')
    print(f'{args.variant}: complete {record["actual_attempted_shots"]:,} shots', flush=True)


if __name__ == '__main__':
    main()

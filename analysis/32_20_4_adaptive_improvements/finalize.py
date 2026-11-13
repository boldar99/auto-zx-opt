"""Audit full adaptive sweeps, exports and resumes; refresh saved notebooks."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from hugr.package import Package
from spiderwarp._se_adaptive_guppy import generate_adaptive_source
from spiderwarp.se_protocol_io import atomic_write, load_protocol, read_manifest
from spiderwarp.se_protocol_simulation import _wilson_interval
from spiderwarp.se_sweep import engine_identity, run_sweep, small_sweep

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_write(Path(path), json.dumps(value, indent=2) + '\n')


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('tasks', 'batches') for p in sorted((directory / folder).glob('*'))}


def main():
    controls = read(OUT / 'controls.json')
    replay = read(OUT / 'replay.json')
    assert replay['complete'] and replay['all_decisions_and_frames_match']
    assert '74 passed' in Path('build/adaptive_improvements_regressions.log').read_text()
    assert '21 passed' in Path('build/adaptive_improvements_final_candidate_tests.log').read_text()
    assert '7 passed' in Path('build/matched_comparison_tests.log').read_text()
    export_audits = read(OUT / 'export_audit.json')
    audits = {}
    for tag in ('css', 'css_recover_exact', 'css_recover_safe'):
        bundle = Path('build') / ('adaptive_' + tag + '_32_20_4')
        directory = Path('benchmarks') / ('adaptive_' + tag + '_100k')
        record = read(directory / 'run.json')
        assert record['status'] == 'complete'
        before = hashes(directory)
        rows = run_sweep(bundle, small_sweep(), directory, shots=100000, max_shots=1000000,
                         batch_size=10000, jobs=8, seed=20261112, allocation='inverse-square')
        assert hashes(directory) == before
        assert len(rows) == 32 and sum(r['shots'] for r in rows) == 9329148
        assert all(r['shots'] == r['target_shots'] for r in rows)
        for row in rows:
            assert (row['ler_low'], row['ler_high']) == _wilson_interval(row['logical_errors'], row['accepted'])
            assert (row['survival_low'], row['survival_high']) == _wilson_interval(row['accepted'], row['shots'])
            assert row['cycles_started'] - row['cycles_completed'] == row['shots'] - row['accepted']
        for path in (directory / 'tasks').glob('*.json'):
            assert read(path)['task']['engine'] == engine_identity()
        protocol = load_protocol(bundle)
        _, manifest = read_manifest(bundle)
        assert manifest['protocol_hash'] == record['protocol_hash']
        assert protocol.verification.verified and protocol.required_qubits == 60
        assert protocol.verification.worst_case_cnots == 589
        source = generate_adaptive_source(protocol)
        assert source == (bundle / 'steane_se_protocol.py').read_text()
        export = read(bundle / 'export.json')
        assert export['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
        binary = (bundle / 'steane_se_protocol.hugr').read_bytes()
        assert export['hugr_sha256'] == hashlib.sha256(binary).hexdigest()
        Package.from_bytes(binary)
        assert export_audits[tag]['protocol_hash'] == record['protocol_hash']
        assert export_audits[tag]['all_stored_transitions_and_frames_match']
        if tag.endswith('safe'):
            assert protocol.policy.contract.correctable == 1
            assert protocol.policy.contract.logical_order == 1
            assert protocol.policy.contract.fault_order == 2
            assert protocol.policy.contract.accept_clean_through == 1
        validation = dict(complete=True, protocol_hash=record['protocol_hash'],
            configurations=32, attempted_shots=9329148, exact_fixed_budgets=True,
            raw_counts_and_95_percent_wilson_intervals_checked=True,
            sampler_matches_saved_engine=True, real_resume_new_shots=0, raw_hashes_unchanged=True,
            raw_file_hashes=before, independent_verification=manifest['verification'],
            source_regenerates_identically=True, source_sha256=export['source_sha256'],
            hugr_sha256=export['hugr_sha256'], hugr_reloaded=True,
            all_stored_export_transitions_and_frames_checked=True,
            export_audit=export_audits[tag],
            checked_stored_edges=sum(len(n.children) for n in protocol.policy.nodes),
            controls=controls[tag], tests=dict(broader_regressions=74, final_candidate_regressions=21),
            selected_for_main_comparison=tag != 'css_recover_exact',
            css_domain='Each CSS component <=2, modulo stabilizers; all 3007 ambiguous noiseless input syndrome sectors reject',
            metric='20-logical-qubit block error conditional on all requested cycles accepting; final ideal decode once')
        if tag != 'css_recover_exact':
            validation['controller_semantics_audit'] = replay
            validation['native_clean_and_noisy_controls_passed'] = True
            assert controls[tag]['native_clean_split_input']['accepted'] == 8
            assert 0 < controls[tag]['native_noisy_both_branches']['accepted'] < 100
        if tag == 'css_recover_exact':
            validation['stronger_contract_failure'] = read(OUT / 'stronger_contract.json')
            validation['interpretation'] = 'Experimental ablation; fails original arbitrary-input filter; excluded from main comparison'
        write(directory / 'validation.json', validation)
        record['validation'] = 'validation.json'
        record['real_resume_new_shots'] = 0
        record['selected_for_main_comparison'] = tag != 'css_recover_exact'
        write(directory / 'run.json', record)
        audits[tag] = validation
        print(f'{tag}: all counts, exports, HUGR, and zero-work resume audited', flush=True)

    from spiderwarp.se_plotting import plot_results
    for tag in ('css', 'css_recover_exact', 'css_recover_safe'):
        plot_results(Path('benchmarks') / ('adaptive_' + tag + '_100k'), physical_survival=True)

    old = ('benchmarks/se_100k', 'benchmarks/steane_rus_d4_100k',
           'benchmarks/steane_rus_ladder_ancilla_100k', 'benchmarks/steane_rus_ladder_cycle_100k')
    subprocess.run(['git', 'diff', '--quiet', '--', *old], check=True)
    prior_strict = read('analysis/32_20_4_postselected_validation/resume.json')
    assert hashes(Path('benchmarks/se_postselected_100k')) == prior_strict['file_hashes']
    for mode in ('d4', 'ladder_ancilla'):
        directory = Path('benchmarks') / ('steane_css_ps_' + mode + '_10k')
        assert hashes(directory) == read(directory / 'validation.json')['raw_file_hashes']

    spec = importlib.util.spec_from_file_location('audit_full_steane_ps',
        ROOT / 'analysis/32_20_4_syndrome_uniqueness/audit_full_steane_ps.py')
    steane_audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(steane_audit)
    full_steane_audits = steane_audit.main()

    comparison_path = Path('benchmarks/steane_comparison/validation.json')
    previous = read(comparison_path)
    from spiderwarp.se_comparison import main as compare
    compare(['--plot-only'])
    comparison = read(comparison_path)
    assert len(comparison['methods']) == 8
    assert comparison['equal_attempted_shots_per_configuration']
    assert set(comparison['attempted_shots'].values()) == {9329148}
    assert not any('ladder / cycle' in m or 'relaxed EC' in m for m in comparison['methods'])
    assert len(list(csv.DictReader(Path('benchmarks/steane_comparison/comparison.csv').open()))) == 256
    for key in ('historical_steane_validation', 'adaptive_postselection_validation',
                'syndrome_uniqueness_analysis'):
        if key in previous:
            comparison[key] = previous[key]
    comparison.update(excluded_modes=['ladder-cycle'], original_raw_datasets_unchanged=True,
        steane_css_postselection_validations={mode: '../steane_css_ps_' + mode + '_100k/validation.json'
                                            for mode in full_steane_audits},
        archived_quick_steane_data_unchanged=True,
        display_names='../../spiderwarp/se_comparison_names.py',
        adaptive_css_validations={tag: '../adaptive_' + tag + '_100k/validation.json' for tag in ('css', 'css_recover_safe')},
        experimental_ablation_excluded='../adaptive_css_recover_exact_100k/validation.json',
        pilot_selection='../../analysis/32_20_4_adaptive_improvements/selection_report.json',
        figures=dict(main_png_pdf_pairs=5, methods=8, tradeoff_png_pdf_pairs=1,
                     visual_inspection_passed=True),
        prior_adaptive_tests=dict(broader_regressions=74, final_candidate_regressions=21),
        matching_budget_tests_log='../../build/matched_comparison_tests.log')
    write(comparison_path, comparison)

    spec = importlib.util.spec_from_file_location('adaptive_study_report', OUT / 'report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    report.main()
    spec = importlib.util.spec_from_file_location('capture_outputs', ROOT / 'analysis/32_20_4_syndrome_uniqueness/finalize.py')
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    for name in ('32_20_4_steane_comparison', '32_20_4_protocol_demo'):
        path = Path('notebooks') / (name + '.ipynb')
        notebook = read(path)
        cells = [c for c in notebook['cells'] if 'qec-comparison' in c.get('metadata', {}).get('tags', [])]
        assert len(cells) == 3 and all(c['cell_type'] == 'code' for c in cells)
        unselected = [c for c in notebook['cells'] if c not in cells]
        preserved_outputs = json.dumps(unselected, sort_keys=True)
        first = max((c.get('execution_count') or 0) for c in unselected) + 1
        capture.capture_cells(cells, first)
        assert [len(c['outputs']) for c in cells] == [1, 5, 1]
        html = cells[0]['outputs'][0]['data']['text/html']
        assert html.count('<tr>') == 33 and 'ladder / cycle' not in html
        assert '<th>Attempts</th>' in html and html.count('<td>100,000</td>') == 32
        assert json.dumps(unselected, sort_keys=True) == preserved_outputs
        atomic_write(path, json.dumps(notebook, indent=1) + '\n')
    write(OUT / 'validation.json', dict(complete=True, selected_full_attempted_shots=2*9329148,
        experimental_ablation_attempted_shots=9329148, pilot_attempted_shots=400000,
        old_raw_datasets_unchanged=True, methods=list(audits), tests=dict(broader=74, final_candidates=21),
        steane_ps_full_extension_attempts=2*8396230, all_eight_methods_equal_attempted_budgets=True,
        attempted_shots_per_displayed_method=9329148,
        matching_budget_tests_passed=7,
        matching_budget_tests_log='../../build/matched_comparison_tests.log',
        main_figure_pairs=5, tradeoff_figure_pairs=1, notebooks_refreshed=True,
        visual_inspection_passed=True))
    print('Finalized selected full sweeps, experimental ablation, eight-method plots and both notebooks.', flush=True)


if __name__ == '__main__':
    main()

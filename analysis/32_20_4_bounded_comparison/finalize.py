"""Audit fresh adaptive reruns, preserve Steane, and regenerate all figures."""
import base64
import csv
from datetime import datetime
import hashlib
from importlib import import_module
import json
import re

study = import_module('analysis.32_20_4_bounded_comparison.study')
FIGURES = ('physical_error', 'cycles', 'survival', 'cnot_cost',
           'physical_survival', 'acceptance_error_tradeoff')


def raw_hashes():
    return {str(path.relative_to(study.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for method in study.METHODS.values() for kind in ('tasks', 'batches')
        for path in sorted((study.ROOT / 'benchmarks' / method.results / kind).glob('*'))}


def main():
    reset = json.loads((study.OUT / 'standard_reset.json').read_text())
    for name, expected in reset['retained_independent_steane'].items():
        assert hashlib.sha256((study.ROOT / name).read_bytes()).hexdigest() == expected
    adaptive = [key for key, method in study.METHODS.items() if method.family == 'adaptive']
    assert set(adaptive) == set(reset['adaptive_methods_to_rerun']) | {'adaptive_scheduled'}
    assert 'adaptive_flag_recovery' not in study.METHODS
    previous_hashes = json.loads((study.OUT / 'standard_raw_hashes.json').read_text())
    for name, expected in previous_hashes.items():
        assert hashlib.sha256((study.ROOT / name).read_bytes()).hexdigest() == expected
    for obsolete in ('benchmarks/adaptive_flag_recovery_100k', 'benchmarks/se_postselected_50k',
                     'benchmarks/adaptive_css_no_flags_50k', 'benchmarks/retired',
                     'analysis/32_20_4_history_attribution', 'spiderwarp/se_flag_recovery.py'):
        assert not (study.ROOT / obsolete).exists()
    since = datetime.fromisoformat(reset['reset_at']).timestamp()
    for key in adaptive:
        directory = study.ROOT / 'benchmarks' / study.METHODS[key].results
        run = json.loads((directory / 'run.json').read_text())
        assert run['status'] == 'complete' and run['attempted_shots'] == 9329148
        assert all(path.stat().st_mtime >= since for path in (directory / 'batches').glob('*.csv'))
    import_module('analysis.32_20_4_bounded_comparison.present').main()
    before = raw_hashes()
    assert len(before) == 64 * len(study.METHODS)
    for key in adaptive:
        study.sweep(key, jobs=8)
    assert before == raw_hashes()
    study.write(study.OUT / 'standard_raw_hashes.json', before)
    comparison = study.ROOT / 'benchmarks/steane_comparison'
    rows = list(csv.DictReader((comparison / 'comparison.csv').open()))
    endpoints = list(csv.DictReader((comparison / 'default_endpoints.csv').open()))
    assert len(rows) == 32 * len(study.METHODS) and len(endpoints) == 4 * len(study.METHODS)
    base_hash = import_module('spiderwarp.se_protocol_io').read_manifest(
        study.ROOT / 'build' / study.BUNDLES['adaptive_history'])[1]['protocol_hash']
    for name in ('order_two_build_validation.json', 'probability_10_build.json',
                 'probability_25_build.json', 'scheduled_build_validation.json'):
        data = json.loads((study.OUT / name).read_text())
        assert data['parent_protocol_hash'] == base_hash
    no_flags = json.loads((study.ROOT / 'build' / study.BUNDLES['adaptive_no_flags'] / 'experiment.json').read_text())
    assert no_flags['parent_protocol_hash'] == base_hash
    books = []
    for name in ('32_20_4_protocol_demo', '32_20_4_steane_comparison'):
        path = study.ROOT / 'notebooks' / (name + '.ipynb')
        book = json.loads(path.read_text())
        images = [output['data']['image/png'] for cell in book['cells']
                  for output in cell.get('outputs', []) if 'image/png' in output.get('data', {})]
        assert len(images) == 6
        assert all(cell.get('execution_count') is not None for cell in book['cells'] if cell['cell_type'] == 'code')
        for figure, encoded in zip(FIGURES, images):
            assert base64.b64decode(encoded) == (comparison / 'figures' / (figure + '.png')).read_bytes()
            assert (comparison / 'figures' / (figure + '.pdf')).stat().st_size > 10000
        books.append(dict(path=str(path.relative_to(study.ROOT)),
                          embedded_current_figures=6, code_cells_executed=True))
    tests = (study.ROOT / 'build/standard_final_tests.log').read_text()
    matched = re.search(r'(\d+) passed(?:, (\d+) skipped)?', tests)
    assert matched and not re.search(r'\d+ failed', tests)
    extra_tests = (study.ROOT / 'build/standard_remaining_budget_test.log').read_text()
    extra_match = re.search(r'(\d+) passed', extra_tests)
    assert extra_match and not re.search(r'\d+ failed', extra_tests)
    scheduled_tests = (study.ROOT / 'build/scheduled_tests.log').read_text()
    scheduled_match = re.search(r'(\d+) passed(?:, (\d+) skipped)?', scheduled_tests)
    assert scheduled_match and not re.search(r'\d+ failed', scheduled_tests)
    validation = dict(complete=True, methods=list(study.METHODS),
        standard_base_protocol_hash=base_hash,
        base_rule='No flags: incoming CSS order <=1; raised flags: fault-history recovery',
        old_history_only_implementation_removed=True,
        previous_adaptive_simulation_data_deleted=True, previous_data_archived=False,
        adaptive_methods_freshly_rerun=reset['adaptive_methods_to_rerun'],
        additional_fresh_methods=['adaptive_scheduled'], configurations_per_method=32,
        attempts_per_method=9329148, new_attempts=9329148 * len(adaptive),
        total_comparison_attempts=9329148 * len(study.METHODS),
        raw_files_verified=len(before), independent_steane_files_preserved=len(reset['retained_independent_steane']),
        resume_additional_shots=0,
        tests=dict(passed=int(matched[1]), skipped=int(matched[2] or 0),
                   log='build/standard_final_tests.log',
                   remaining_budget_regression='build/standard_remaining_budget_test.log'),
        scheduled_tests=dict(passed=int(scheduled_match[1]), skipped=int(scheduled_match[2] or 0),
                             log='build/scheduled_tests.log'),
        notebooks=books, six_figure_pairs_generated=True, visual_inspection_pending=True,
        implementation_sha256={name: hashlib.sha256((study.ROOT / 'spiderwarp' / name).read_bytes()).hexdigest()
            for name in ('se_postselection.py', 'se_order_two.py', 'se_probability.py', 'se_scheduling.py')},
        statistics='LER and cost condition on accepted full trajectories; AR uses all attempts')
    study.write(study.OUT / 'standard_validation.json', validation)
    for path in (comparison / 'validation.json', study.OUT / 'sampling_validation.json'):
        data = json.loads(path.read_text())
        data.update(standard_validation='standard_validation.json', targeted_tests=validation['tests'],
                    scheduled_tests=validation['scheduled_tests'],
                    completed_run_resume=dict(additional_shots=0, raw_files=len(before), all_hashes_preserved=True))
        study.write(path, data)
    print(f'Audited {len(adaptive)} adaptive sweeps, {len(rows)} comparison rows, '
          f'{len(endpoints)} endpoints and {len(before)} raw files; resumes add zero shots.', flush=True)


if __name__ == '__main__':
    main()

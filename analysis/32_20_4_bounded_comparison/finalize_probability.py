"""Rebuild six-method figures/notebooks and audit counts, hashes and resumes."""
import base64
import csv
import hashlib
from importlib import import_module
import json

study = import_module('analysis.32_20_4_bounded_comparison.study')
FIGURES = ('physical_error', 'cycles', 'survival', 'cnot_cost',
           'physical_survival', 'acceptance_error_tradeoff')


def raw_hashes():
    return {str(path.relative_to(study.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for method in study.METHODS.values() for kind in ('tasks', 'batches')
        for path in sorted((study.ROOT / 'benchmarks' / method.results / kind).glob('*'))}


def main():
    import_module('analysis.32_20_4_bounded_comparison.present').main()
    comparison = study.ROOT / 'benchmarks/steane_comparison'
    rows = list(csv.DictReader((comparison / 'comparison.csv').open()))
    endpoints = list(csv.DictReader((comparison / 'default_endpoints.csv').open()))
    assert len(rows) == 192 and len(endpoints) == 24
    before = raw_hashes()
    assert len(before) == 384
    original = json.loads((study.OUT / 'order_two_raw_hashes.json').read_text())
    assert len(original) == 256 and all(before[name] == digest for name, digest in original.items())
    for key in study.METHODS:
        study.sweep(key, jobs=8)
    assert before == raw_hashes()
    study.write(study.OUT / 'probability_raw_hashes.json', before)
    books = []
    for name in ('32_20_4_protocol_demo', '32_20_4_steane_comparison'):
        path = study.ROOT / 'notebooks' / (name + '.ipynb')
        book = json.loads(path.read_text())
        text = ''.join(''.join(cell['source']) for cell in book['cells'])
        assert 'p_err≤10%' in text and 'p_err≤25%' in text
        assert 'two nonzero response channels' in text
        images = [output['data']['image/png'] for cell in book['cells']
                  for output in cell.get('outputs', []) if 'image/png' in output.get('data', {})]
        assert len(images) == 6
        assert all(cell.get('execution_count') is not None for cell in book['cells'] if cell['cell_type'] == 'code')
        for name, encoded in zip(FIGURES, images):
            assert base64.b64decode(encoded) == (comparison / 'figures' / (name + '.png')).read_bytes()
            assert (comparison / 'figures' / (name + '.pdf')).stat().st_size > 10000
        books.append(dict(path=str(path.relative_to(study.ROOT)), embedded_current_figures=6,
                          code_cells_executed=True))
    tests = (study.ROOT / 'build/probability_final_tests.log').read_text()
    assert '79 passed, 12 skipped' in tests
    sources = {name: hashlib.sha256((study.ROOT / 'spiderwarp' / name).read_bytes()).hexdigest()
               for name in ('se_probability.py', 'se_likelihood.py', 'se_postselection.py')}
    validation = dict(complete=True, methods=list(study.METHODS), epsilons=[0.1, 0.25],
        configurations_per_method=32, attempts_per_method=9329148,
        new_attempts=18658296, total_comparison_attempts=55974888,
        raw_files_verified=384, prior_raw_files_preserved=256, resume_additional_shots=0,
        tests=dict(passed=79, skipped=12, log='build/probability_final_tests.log'),
        controls=json.loads((study.OUT / 'probability_controls.json').read_text()),
        implementation_sources_at_audit=sources, notebooks=books,
        six_figure_pairs_generated=True, visual_inspection_pending=True,
        error_definition='Wrong final MAP Pauli class modulo stabilizers',
        approximation='At most two nonzero response channels; default-noise model and incoming iid prior frozen',
        all_orders_ler_guarantee=False, output_labels_used_for_selection=False,
        statistics='LER and cost condition on accepted full trajectories; AR uses all attempts')
    study.write(study.OUT / 'probability_validation.json', validation)
    for path in (comparison / 'validation.json', study.OUT / 'sampling_validation.json'):
        data = json.loads(path.read_text())
        data.update(targeted_tests=validation['tests'], probability_validation='probability_validation.json',
                    completed_run_resume=dict(additional_shots=0, raw_files=384, all_hashes_preserved=True))
        study.write(path, data)
    print('Finalized 192 comparison rows, 24 endpoints, 384 raw files and two notebooks; resumes add zero shots.', flush=True)


if __name__ == '__main__':
    main()

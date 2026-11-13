"""Audit normal-budget extensions without changing the sampled circuits."""
from importlib import import_module
from pathlib import Path
import csv
import hashlib
import json

from spiderwarp.se_comparison_names import METHODS
from spiderwarp.se_protocol_io import read_manifest
from spiderwarp.se_protocol_simulation import _wilson_interval

study = import_module('analysis.32_20_4_bounded_comparison.study')
ROOT, OUT, write = study.ROOT, study.OUT, study.write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    promotion_path = OUT / 'normal_shot_promotion.json'
    promotion = json.loads(promotion_path.read_text())
    # The registry can gain independent experiments while these runs execute.
    # Audit the fixed three-method cohort that was actually promoted.
    selected = {key: METHODS[key] for key in promotion['methods']}
    audits, raw = {}, {}
    for key, method in selected.items():
        before = promotion['methods'][key]
        source, target = ROOT / before['source'], ROOT / before['target']
        assert target == ROOT / 'benchmarks' / method.results
        assert all(sha(ROOT / p) == value for p, value in before['source_raw_sha256'].items())
        run = json.loads((target / 'run.json').read_text())
        assert run['status'] == 'complete'
        assert run['shot_floor'] == 100000 and run['max_shots'] == 1000000
        summary = rows(target / 'summary.csv')
        assert len(summary) == 32
        assert sum(int(r['shots']) for r in summary) == run['total_target_shots'] == 9329148
        for task_path in (source / 'tasks').glob('*.json'):
            old_task = json.loads(task_path.read_text())
            new_task = json.loads((target / 'tasks' / task_path.name).read_text())
            assert old_task['task'] == new_task['task']
            name = task_path.stem + '.csv'
            old_batches = rows(source / 'batches' / name)
            new_batches = rows(target / 'batches' / name)
            assert new_batches[:len(old_batches)] == old_batches
            position = 0
            for batch in new_batches:
                assert int(batch['start']) == position
                position += int(batch['shots'])
            assert position == new_task['target_shots']
        for folder in ('tasks', 'batches'):
            for path in (target / folder).iterdir():
                if path.is_file():
                    raw[str(path.relative_to(ROOT))] = sha(path)
        audits[key] = dict(attempted_shots=9329148, initial_shots=before['initial_attempts'],
                           added_shots=9329148-before['initial_attempts'],
                           original_batches_preserved=True, no_duplicate_shot_ranges=True)
    assert len(raw) == 192
    for key in selected:
        study.sweep(key)
    assert all(sha(ROOT / p) == value for p, value in raw.items())
    assert all(sha(ROOT / p) == value for p, value in promotion['original_fixed_steane_dataset_sha256'].items())
    retained = json.loads((ROOT / 'analysis/32_20_4_steane_single_shot/retained_fixed_artifact_hashes.json').read_text())
    assert all(sha(ROOT / p) == value for p, value in retained.items())
    history, alias = (ROOT / 'build' / name for name in ('se_postselected_32_20_4', 'adaptive_css_32_20_4'))
    assert read_manifest(history)[1]['protocol_hash'] == read_manifest(alias)[1]['protocol_hash']
    for name in ('steane_se_protocol.py', 'steane_se_protocol.hugr'):
        assert (history / name).read_bytes() == (alias / name).read_bytes()
    comparison = ROOT / 'benchmarks/steane_comparison'
    combined = rows(comparison / 'comparison.csv')
    endpoints = rows(comparison / 'default_endpoints.csv')
    assert len(combined) == 96 and len(endpoints) == 12
    assert all(int(r['shots']) == 100000 for r in endpoints)
    assert sum(int(r['shots']) for r in combined) == 27987444
    books = {}
    for name in ('32_20_4_steane_comparison', '32_20_4_protocol_demo'):
        path = ROOT / 'notebooks' / (name + '.ipynb')
        book = json.loads(path.read_text())
        assert (book['nbformat'], book['nbformat_minor']) == (4, 5)
        assert len({c['id'] for c in book['cells']}) == len(book['cells']) == 7
        code = [c for c in book['cells'] if c['cell_type'] == 'code']
        assert [len(c['outputs']) for c in code] == [1, 5, 1]
        table = code[0]['outputs'][0]['data']['text/html']
        assert table.count('<tr>') == 13 and table.count('100,000') == 12
        books[str(path.relative_to(ROOT))] = sha(path)
    # Show actual uncertainty before/after, without using outcomes to choose N.
    before = promotion['methods']['adaptive_no_flags']
    old = {(int(r['cycles']), r['prepared_state']): r
           for r in rows(ROOT / before['source'] / 'summary.csv')
           if 'cycles' in r['sweeps']}
    uncertainty = []
    for row in endpoints:
        if row['method'] != METHODS['adaptive_no_flags'].label:
            continue
        prior = old[int(row['cycles']), row['prepared_state']]
        record = dict(cycles=int(row['cycles']), prepared_state=row['prepared_state'])
        for label, r in (('half', prior), ('normal', row)):
            accepted, failures = int(r['accepted']), int(r['logical_errors'])
            low, high = _wilson_interval(failures, accepted)
            record[label] = dict(shots=int(r['shots']), accepted=accepted, failures=failures,
                                  ler=failures/accepted, interval_95=[low, high], width=high-low)
        record['interval_width_ratio'] = record['normal']['width'] / record['half']['width']
        uncertainty.append(record)
    figures = {str(p.relative_to(ROOT)): sha(p) for p in (comparison / 'figures').iterdir()
               if p.suffix in {'.png', '.pdf'}}
    assert len(figures) == 12
    for path in (comparison / 'validation.json', OUT / 'sampling_validation.json'):
        record = json.loads(path.read_text())
        assert record['equal_attempted_shots_per_configuration']
        record.update(shot_floor=100000, shot_cap=1000000, raw_batch_and_task_files=192,
                      normal_budget_extensions_verified=True, original_batches_preserved=True,
                      no_duplicate_shot_ranges=True, raw_files_unchanged_on_completed_resume=True,
                      notebook_structure_and_saved_outputs_verified=True, notebook_sha256=books,
                      figure_sha256=figures, original_fixed_steane_dataset_preserved=True)
        write(path, record)
    write(OUT / 'raw_hashes.json', raw)
    write(OUT / 'zero_flag_uncertainty.json', uncertainty)
    validation = json.loads((OUT / 'validation.json').read_text())
    validation.update(attempted_shots_per_method=9329148, current_cohort_attempted_shots=27987444,
                      shot_floor=100000, shot_cap=1000000, normal_budget_extensions_verified=True,
                      original_batches_preserved=True, no_duplicate_shot_ranges=True,
                      original_fixed_steane_dataset_preserved=True, notebooks_verified=books,
                      normal_shot_promotion='normal_shot_promotion.json',
                      zero_flag_uncertainty='zero_flag_uncertainty.json')
    write(OUT / 'validation.json', validation)
    promotion.update(status='complete', audits=audits, methods_attempted_shots=9329148,
                     total_attempted_shots=27987444, appended_attempts=13993716,
                     completed_resume_adds_no_shots=True)
    write(promotion_path, promotion)
    print('Normal-budget audit passed: 9,329,148 attempts per method; all original batches preserved.')


if __name__ == '__main__':
    main()

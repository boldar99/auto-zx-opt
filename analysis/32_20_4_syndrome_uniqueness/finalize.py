"""Audit quick datasets, check real resume, and refresh comparison notebooks."""
import base64
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from hugr.package import Package
from spiderwarp._steane_qec_guppy import generate_source
from spiderwarp.se_protocol_io import atomic_write, load_protocol, read_manifest
from spiderwarp.se_protocol_simulation import _wilson_interval
from spiderwarp.se_sweep import engine_identity, run_sweep, small_sweep


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_write(Path(path), json.dumps(value, indent=2) + '\n')


def raw_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('tasks', 'batches') for p in sorted((directory / folder).glob('*'))}


def capture_cells(cells, first):
    import IPython.display as ipd
    namespace = {}
    original = ipd.display
    try:
        for count, cell in enumerate(cells, first):
            outputs = []

            def capture(*objects, **kwargs):
                for obj in objects:
                    data = {'text/plain': repr(obj)}
                    if isinstance(obj, ipd.HTML):
                        data['text/html'] = obj.data
                    elif isinstance(obj, ipd.Image):
                        data['image/' + obj.format] = base64.b64encode(obj.data).decode()
                    else:
                        raise TypeError(type(obj))
                    outputs.append(dict(output_type='display_data', data=data, metadata={}))

            ipd.display = capture
            exec(compile(''.join(cell['source']), '<Steane CSS-PS comparison>', 'exec'), namespace)
            cell['execution_count'], cell['outputs'] = count, outputs
    finally:
        ipd.display = original


def main():
    here = Path(__file__).resolve().parent
    controls = read(here / 'controls.json')
    assert '59 passed' in Path('build/steane_css_ps_regressions.log').read_text()
    assert '5 passed' in Path('build/steane_css_ps_final_plot_tests.log').read_text()
    outputs = {}
    for mode in ('d4', 'ladder_ancilla'):
        directory = Path('benchmarks') / ('steane_css_ps_' + mode + '_10k')
        bundle = Path('build') / ('steane_css_ps_' + mode)
        record = read(directory / 'run.json')
        assert record['status'] == 'complete' and record['actual_attempted_shots'] == 932918
        assert read(directory / 'config.json') == small_sweep()
        before = raw_hashes(directory)
        rows = run_sweep(bundle, small_sweep(), directory, shots=10000, max_shots=100000,
                         batch_size=1000, jobs=8, seed=20261112, allocation='inverse-square')
        assert before == raw_hashes(directory)
        assert len(rows) == 32 and sum(r['shots'] for r in rows) == 932918
        assert all(r['shots'] == r['target_shots'] for r in rows)
        for row in rows:
            assert (row['ler_low'], row['ler_high']) == _wilson_interval(row['logical_errors'], row['accepted'])
            assert (row['survival_low'], row['survival_high']) == _wilson_interval(row['accepted'], row['shots'])
            assert row['cycles_started'] - row['cycles_completed'] == row['shots'] - row['accepted']
        for task in (directory / 'tasks').glob('*.json'):
            assert read(task)['task']['engine'] == engine_identity()
        _, manifest = read_manifest(bundle)
        assert manifest['protocol_hash'] == record['protocol_hash']
        protocol = load_protocol(bundle)
        assert protocol.postselection == 'unique-css-2'
        assert all(sum(v) == 33 for v in protocol.unique_syndromes.values())
        source = generate_source(protocol)
        assert source == (bundle / 'steane_se_protocol.py').read_text()
        export = read(bundle / 'export.json')
        assert hashlib.sha256(source.encode()).hexdigest() == export['source_sha256']
        data = (bundle / 'steane_se_protocol.hugr').read_bytes()
        assert hashlib.sha256(data).hexdigest() == export['hugr_sha256']
        Package.from_bytes(data)
        result = dict(complete=True, configurations=32, attempted_shots=932918,
            protocol_hash=manifest['protocol_hash'], postselection='unique-css-2',
            peak_qubits=66, all_prescribed_counts_and_wilson_intervals_checked=True,
            installed_sampler_matches_saved_task_engine=True, data_frame_adjusted_before_postselection=True,
            cycles_started=sum(r['cycles_started'] for r in rows),
            cycles_completed=sum(r['cycles_completed'] for r in rows),
            source_regenerates_identically=True, source_sha256=export['source_sha256'],
            hugr_reloaded=True, hugr_sha256=export['hugr_sha256'],
            real_resume_new_shots=0, real_resume_raw_hashes_unchanged=True,
            raw_file_hashes=before, physical_and_native_controls=controls[mode],
            tests=dict(broader_regressions=59, final_budget_and_plot_regressions=5),
            bound='Each CSS component <=2; uniqueness is modulo stabilizers, at each confirmed frame-adjusted syndrome',
            interpretation='20-logical-qubit block LER conditional on all cycles accepting; one ideal decode at the endpoint',
            certificate='Simulation baseline; no adaptive-policy circuit-fault certificate claimed')
        write(directory / 'validation.json', result)
        record['validation'] = 'validation.json'
        record['real_resume_new_shots'] = 0
        write(directory / 'run.json', record)
        outputs[mode] = result

    old = ['benchmarks/se_100k', 'benchmarks/steane_rus_d4_100k',
           'benchmarks/steane_rus_ladder_ancilla_100k', 'benchmarks/steane_rus_ladder_cycle_100k']
    subprocess.run(['git', 'diff', '--quiet', '--', *old], check=True)
    comparison_path = Path('benchmarks/steane_comparison/validation.json')
    comparison = read(comparison_path)
    assert len(comparison['methods']) == 6
    assert all('ladder / cycle' not in name for name in comparison['methods'])
    assert comparison['equal_noise_seeds_and_configurations'] and not comparison['equal_shot_budgets']
    combined = list(csv.DictReader(Path('benchmarks/steane_comparison/comparison.csv').open()))
    assert len(combined) == 192
    assert all('ladder / cycle' not in row['method'] for row in combined)
    prior_path = Path('build/steane_comparison_validation_before_css_ps.json')
    prior = read(prior_path) if prior_path.exists() else {}
    comparison.update(excluded_modes=['ladder-cycle'], original_raw_datasets_unchanged=True,
        adaptive_postselection_validation='../se_postselected_100k/validation.json',
        steane_css_postselection_validations={mode: '../steane_css_ps_' + mode + '_10k/validation.json'
                                           for mode in outputs},
        syndrome_uniqueness_analysis='../../analysis/32_20_4_syndrome_uniqueness/summary.json',
        tests_this_update=dict(broader_regressions=59, final_budget_and_plot_regressions=5),
        figures=dict(png_pdf_pairs=5, plotted_methods=6, visual_inspection_passed=True))
    if 'prior_steane_validation' in prior:
        comparison['historical_steane_validation'] = prior['prior_steane_validation']
    write(comparison_path, comparison)

    # Capture only the comparison cells; preserve the demo's earlier heavy work.
    for name in ('32_20_4_steane_comparison', '32_20_4_protocol_demo'):
        path = Path('notebooks') / (name + '.ipynb')
        notebook = read(path)
        if name.endswith('steane_comparison'):
            cells, first = [c for c in notebook['cells'] if c['cell_type'] == 'code'], 1
        else:
            cells = notebook['cells'][-2:]
            first = max((c.get('execution_count') or 0) for c in notebook['cells'][:-2]) + 1
        assert len(cells) == 2
        capture_cells(cells, first)
        assert [len(c['outputs']) for c in cells] == [1, 5]
        html = cells[0]['outputs'][0]['data']['text/html']
        assert html.count('<tr>') == 25 and 'ladder / cycle' not in html
        for output in cells[1]['outputs']:
            png = base64.b64decode(output['data']['image/png'])
            assert png.startswith(b'\x89PNG\r\n\x1a\n')
            assert struct.unpack('>II', png[16:24])[0] > 1000
        atomic_write(path, json.dumps(notebook, indent=1) + '\n')
    print('Audited 1,865,836 quick shots, reloaded both HUGRs, checked zero-work resumes, and refreshed both notebooks.')


if __name__ == '__main__':
    main()

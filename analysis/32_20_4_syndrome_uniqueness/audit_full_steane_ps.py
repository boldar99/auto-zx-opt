"""Audit the full Steane syndrome-PS extensions without resampling any shots."""
import csv
import hashlib
import json
from pathlib import Path

from hugr.package import Package
from spiderwarp._steane_qec_guppy import generate_source
from spiderwarp.se_comparison_names import METHODS
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


def main():
    audits = {}
    for mode, name_key in (('d4', 'steane_d4_ps'), ('ladder_ancilla', 'steane_descending_ps')):
        directory = Path('benchmarks') / METHODS[name_key].results
        archive = directory.with_name(directory.name.replace('_100k', '_10k'))
        bundle = Path('build') / ('steane_css_ps_' + mode)
        record = read(directory / 'run.json')
        assert record['status'] == 'complete' and record['actual_attempted_shots'] == 9329148
        assert read(directory / 'config.json') == small_sweep()
        provenance = read(directory / 'seed_provenance.json')
        prior_validation = read(archive / 'validation.json')
        assert raw_hashes(archive) == provenance['source_raw_hashes'] == prior_validation['raw_file_hashes']
        assert provenance['archived_quick_run'] == read(archive / 'run.json')
        assert provenance['archived_quick_validation'] == prior_validation
        before = raw_hashes(directory)
        rows = run_sweep(bundle, small_sweep(), directory, shots=100000, max_shots=1000000,
                         batch_size=10000, jobs=2, seed=20261112, allocation='inverse-square')
        assert raw_hashes(directory) == before
        assert len(rows) == 32 and sum(r['shots'] for r in rows) == 9329148
        assert all(r['shots'] == r['target_shots'] for r in rows)
        for row in rows:
            assert (row['ler_low'], row['ler_high']) == _wilson_interval(row['logical_errors'], row['accepted'])
            assert (row['survival_low'], row['survival_high']) == _wilson_interval(row['accepted'], row['shots'])
            assert row['cycles_started'] - row['cycles_completed'] == row['shots'] - row['accepted']
        for path in (directory / 'tasks').glob('*.json'):
            assert read(path)['task']['engine'] == engine_identity()
        for path in (archive / 'batches').glob('*.csv'):
            old_rows = list(csv.DictReader(path.open()))
            assert list(csv.DictReader((directory / 'batches' / path.name).open()))[:len(old_rows)] == old_rows
        protocol = load_protocol(bundle)
        _, manifest = read_manifest(bundle)
        assert manifest['protocol_hash'] == record['protocol_hash'] == prior_validation['protocol_hash']
        assert protocol.postselection == 'unique-css-2' and protocol.required_qubits == 66
        assert all(sum(domain) == 33 for domain in protocol.unique_syndromes.values())
        source = generate_source(protocol)
        assert source == (bundle / 'steane_se_protocol.py').read_text()
        export = read(bundle / 'export.json')
        assert hashlib.sha256(source.encode()).hexdigest() == export['source_sha256'] == prior_validation['source_sha256']
        binary = (bundle / 'steane_se_protocol.hugr').read_bytes()
        assert hashlib.sha256(binary).hexdigest() == export['hugr_sha256'] == prior_validation['hugr_sha256']
        Package.from_bytes(binary)
        validation = dict(complete=True, display_name=METHODS[name_key].name,
            configurations=32, attempted_shots=9329148, reused_quick_attempts=932918,
            additional_attempted_shots=8396230, exact_fixed_budgets=True,
            protocol_hash=manifest['protocol_hash'], postselection='unique-css-2', peak_qubits=66,
            all_prescribed_counts_and_wilson_intervals_checked=True,
            installed_sampler_matches_saved_task_engine=True,
            real_resume_new_shots=0, real_resume_raw_hashes_unchanged=True, raw_file_hashes=before,
            archived_quick_data_and_metadata_unchanged=True, original_batch_rows_preserved=True,
            seed_provenance='seed_provenance.json', data_frame_adjusted_before_postselection=True,
            source_regenerates_identically=True, source_sha256=export['source_sha256'],
            hugr_reloaded=True, hugr_sha256=export['hugr_sha256'],
            inherited_unchanged_circuit_controls=prior_validation['physical_and_native_controls'],
            inherited_controls_source='../' + archive.name + '/validation.json',
            bound='Each CSS component <=2; uniqueness modulo stabilizers on confirmed frame-adjusted syndromes',
            interpretation='20-logical-qubit block LER conditional on all requested cycles accepting',
            certificate='Simulation baseline; no adaptive-policy circuit-fault certificate claimed')
        write(directory / 'validation.json', validation)
        record.update(validation='validation.json', display_name=METHODS[name_key].name,
                      real_resume_new_shots=0, selected_for_main_comparison=True)
        write(directory / 'run.json', record)
        audits[mode] = validation
        print(f'{mode}: full counts, archived prefixes, unchanged exports and zero-work resume passed', flush=True)
    return audits


if __name__ == '__main__':
    main()

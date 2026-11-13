"""CSS uniqueness, early rejection and matching standalone Steane control."""
import ast
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from spiderwarp._steane_qec_guppy import generate_source
from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_protocol_io import digest, load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment
from spiderwarp.steane_qec import build_steane_qec, unique_css_syndromes


@pytest.fixture(scope='module', params=('fixed',))
def protocol(request):
    return build_steane_qec(request.param, postselection='unique-css-2')


class ConstantSyndromes:
    def __init__(self, protocol, z, x):
        self.protocol, self.syndromes = protocol, {'Z': z, 'X': x}
        self.record, self.trace = [], []
        self.names = {id(g.circuit): name for name, g in protocol.gadgets.items()}

    def current_measurement_record(self):
        return self.record

    def do_circuit(self, circuit):
        name = self.names[id(circuit)]
        self.trace.append(name)
        values = [False] * circuit.num_measurements
        if name.startswith('read_'):
            basis = name[-1]
            physical = self.protocol.decoders[basis][self.syndromes[basis]]
            values = [bool(physical >> q & 1) for q in range(self.protocol.code.n)]
        self.record.extend(values)


def exported_controller(protocol):
    definitions = []
    for node in ast.parse(generate_source(protocol)).body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list, node.returns = [], None
            for arg in node.args.args:
                arg.annotation = None
            definitions.append(node)
        elif isinstance(node, ast.ClassDef):
            node.decorator_list = [ast.Name(id='dataclass', ctx=ast.Load())]
            definitions.append(node)
        elif isinstance(node, ast.Assign):
            definitions.append(node)
    namespace = dict(comptime=lambda v: v, dataclass=dataclass, array=list,
                     qubit=lambda: 0, measure_bit=lambda q: False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 '<CSS-PS exported controller>', 'exec'), namespace)
    for basis in ('z', 'x'):
        namespace['prepare_' + basis] = lambda level: namespace['Prepared']([0]*32, level)
    return namespace


def test_all_single_syndromes_and_incoming_frames_match_export(protocol):
    assert all(sum(domain) == 33 for domain in unique_css_syndromes(protocol.code).values())
    exported = exported_controller(protocol)
    for basis in ('Z', 'X'):
        for frame in (0, 1, 3, (1 << 17) | (1 << 29)):
            for syndrome in range(64):
                raw = {'Z': 0, 'X': 0}
                raw[basis] = syndrome
                frames = dict(frame_x=frame if basis == 'Z' else 0,
                              frame_z=frame if basis == 'X' else 0)
                reference = ConstantSyndromes(protocol, raw['Z'], raw['X'])
                actual = protocol.run(reference, **frames)
                for b in ('z', 'x'):
                    exported['read_' + b] = lambda data, ancilla, b=b: raw[b.upper()]
                emitted = exported['se_cycle_with_frame']([0]*32, **frames)
                assert (actual.accepted, actual.frame_x, actual.frame_z) == (
                    emitted.accepted, emitted.frame_x, emitted.frame_z)
                shift = exported['frame_syndrome_' + basis.lower()](frame)
                assert actual.accepted == protocol.unique_syndromes[basis][syndrome ^ shift]
                if basis == 'Z' and not actual.accepted:
                    assert reference.trace == ['prep_Z_d3', 'read_Z']
                if not actual.accepted:
                    assert actual.discard_reason == 'ambiguous CSS syndrome through data weight two'


def test_physical_split_inputs_and_ambiguous_pairs(protocol):
    for basis in ('0', '+'):
        experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
        for q in range(32):
            sim = experiment.initial.copy(seed=q)
            sim.x(q)
            sim.z((q + 17) % 32)
            result = protocol.run(sim)
            assert result.accepted
            assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
            again = protocol.run(sim, frame_x=result.frame_x, frame_z=result.frame_z)
            assert again.accepted and (again.frame_x, again.frame_z) == (result.frame_x, result.frame_z)
        for component in ('x', 'z'):
            sim = experiment.initial.copy(seed=17)
            getattr(sim, component)(0)
            getattr(sim, component)(1)
            assert not protocol.run(sim).accepted
            # Those two tracked Paulis are not two new unknown errors.
            tracked = dict(frame_x=3 if component == 'x' else 0,
                           frame_z=3 if component == 'z' else 0)
            assert protocol.run(sim, **tracked).accepted
    result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), cycles=5).sample(20, 17)
    assert result.accepted == 20 and result.logical_errors == 0


def test_postselected_bundle_and_original_artifacts_stay_distinct(protocol, tmp_path):
    saved = protocol.save_bundle(tmp_path / 'postselected')
    loaded = load_protocol(saved)
    assert loaded.postselection == 'unique-css-2' and loaded.unique_syndromes == protocol.unique_syndromes
    raw = json.loads(saved.read_text())
    raw['unique_syndromes']['Z'][0] = False
    raw['protocol_hash'] = digest({k: v for k, v in raw.items() if k not in {'protocol_hash', 'versions'}})
    saved.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='unique-correction domain changed'):
        load_protocol(saved)
    manifest = json.loads(protocol.save_bundle(tmp_path / 'single').read_text())
    assert set(manifest['preparations']) == {'1', '3'}
    assert manifest['preparation_start'] == 3
    assert manifest['syndrome_confirmation'] == 'single-per-basis'
    manifest['syndrome_confirmation'] = 'two-consecutive-per-basis'
    manifest['protocol_hash'] = digest({k: v for k, v in manifest.items() if k not in {'protocol_hash', 'versions'}})
    (tmp_path / 'single/protocol.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Removed repeated-read'):
        load_protocol(tmp_path / 'single')


def test_quick_comparison_requires_explicit_unequal_budget_allowance(tmp_path):
    from spiderwarp.se_comparison import save_comparison
    from spiderwarp.se_protocol_simulation import _wilson_interval
    directories = []
    for name, shots in (('production', 10000), ('quick', 1000)):
        directory = tmp_path / name
        directory.mkdir()
        directories.append(directory)
        low, high = _wilson_interval(1, shots)
        row = dict(shots=shots, target_shots=shots, accepted=shots, logical_errors=1,
                   ler_low=low, ler_high=high, p=.0008, cycles=1, prepared_state='0',
                   noise='{"p_2q":0.0008}', sweeps='["cycles"]', seed=20261112)
        with (directory / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(row))
            writer.writeheader()
            writer.writerow(row)
    with pytest.raises(ValueError, match='budgets differ'):
        save_comparison(directories, ['production', 'quick'], tmp_path / 'out')
    save_comparison(directories, ['production', 'quick'], tmp_path / 'out', require_equal_budgets=False)
    audit = json.loads((tmp_path / 'out/validation.json').read_text())
    assert audit['equal_noise_seeds_and_configurations']
    assert not audit['equal_shot_budgets']
    assert audit['attempted_shots'] == {'production': 10000, 'quick': 1000}


def test_canonical_comparison_rejects_retired_dataset_overrides():
    from spiderwarp.se_comparison import main
    # The canonical comparison reads only the current matched-budget registry;
    # old baseline/quick-result injection switches are no longer supported.
    with pytest.raises(SystemExit) as exc:
        main(['--plot-only', '--modes', 'fixed', '--steane-postselected-results', 'benchmarks/quick'])
    assert exc.value.code == 2

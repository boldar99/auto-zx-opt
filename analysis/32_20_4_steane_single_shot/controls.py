"""Physical CSS-input and native HUGR controls for corrected single-read Steane."""
import json
from pathlib import Path

from spiderwarp.se_protocol_io import atomic_write, load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment
from spiderwarp.noise_model import CustomErrorModel
import spiderwarp.se_native as native


OUT = Path(__file__).resolve().parent


def main():
    records = {}
    for mode in ('fixed',):
        bundle = Path('build') / ('steane_single_d3_' + mode + '_ps')
        protocol = load_protocol(bundle)
        record = dict(physical_css_one_cases=0, native={})
        for basis in ('0', '+'):
            experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
            for x in [0] + [1 << q for q in range(32)]:
                for z in [0] + [1 << q for q in range(32)]:
                    sim = experiment.initial.copy(seed=20261113)
                    for q in range(32):
                        if x >> q & 1: sim.x(q)
                        if z >> q & 1: sim.z(q)
                    result = protocol.run(sim)
                    assert result.accepted
                    assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
                    assert sum(n.startswith('read_') for n, *_ in result.measurements) == 2
                    record['physical_css_one_cases'] += 1
            for name, probability, shots, incoming in (
                    ('clean_split', 0, 4, (1, 1 << 17)),
                    ('noisy', .002, 100, (0, 0)),
                    ('ambiguous_pair', 0, 4, (3, 0))):
                result = native.run_native(bundle,
                    Path('build/steane_single_controls') / mode / (basis.replace('+', 'plus') + '_' + name),
                    prepared_state=basis, cycles=3, incoming_x=incoming[0], incoming_z=incoming[1],
                    p=probability, shots=shots, seed=20261113)
                if name == 'clean_split':
                    assert result['accepted'] == shots and result['logical_errors'] == 0
                elif name == 'ambiguous_pair':
                    assert result['accepted'] == 0
                else:
                    assert 0 < result['accepted'] < shots
                record['native'][basis + '_' + name] = result
                print(mode, basis, name, result['accepted'], '/', shots, flush=True)
        clean = ProtocolSampler(protocol, noise_model=CustomErrorModel(), cycles=10).sample(20, 20261113)
        assert clean.accepted == 20 and clean.logical_errors == 0
        assert clean.diagnostics.total_cnots == 20 * 10 * 288
        record['clean_ten_cycle_shots'] = 20
        records[mode] = record

    # Force only the first reported preparation flag of each fixed-RUS call.
    # Actual d3 preparation, discard, reallocation and subsequent data reads run natively.
    original = native.experiment_source
    def force_first_flag(protocol, **kwargs):
        lines, inside = [], False
        for line in original(protocol, **kwargs).splitlines():
            if line.startswith('def '): inside = line.startswith('def prepare_')
            if inside and line == '    while True:': lines.append('    attempts = 0')
            lines.append(line)
            if inside and line.startswith('        flags = '):
                lines.extend(['        if attempts == 0:', '            flags |= 1', '        attempts += 1'])
        return '\n'.join(lines) + '\n'
    try:
        native.experiment_source = force_first_flag
        for basis in ('0', '+'):
            result = native.run_native('build/steane_single_d3_fixed_ps',
                Path('build/steane_single_controls/fixed_forced_retry') / basis.replace('+', 'plus'),
                prepared_state=basis, cycles=3, incoming_x=1, incoming_z=1 << 17,
                p=0, shots=4, seed=20261113)
            assert result['accepted'] == 4 and result['logical_errors'] == 0
            records['fixed']['native'][basis + '_forced_first_retry'] = result
    finally:
        native.experiment_source = original
    atomic_write(OUT / 'controls.json', json.dumps(dict(complete=True, methods=records,
        physical_css_one_cases=sum(r['physical_css_one_cases'] for r in records.values()),
        native_model='Gate-only test noise with preparation/readout noise, no memory; not production LER',
        forced_flags='Test-only classical flag overrides; native physical retry paths execute'), indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Physical CSS inputs and native accept/reject controls for quick Steane PS."""
import json
from pathlib import Path

from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_native import run_native
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment


def main():
    output = {}
    for mode in ('d4', 'ladder_ancilla'):
        bundle = Path('build') / ('steane_css_ps_' + mode)
        protocol = load_protocol(bundle)
        native, physical = {}, {}
        for basis in ('0', '+'):
            experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
            correct, rejected = 0, 0
            for x in (0, *(1 << q for q in range(32))):
                for z in (0, *(1 << q for q in range(32))):
                    sim = experiment.initial.copy(seed=17)
                    for q in range(32):
                        if x >> q & 1: sim.x(q)
                        if z >> q & 1: sim.z(q)
                    result = protocol.run(sim)
                    assert result.accepted
                    assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
                    correct += 1
            for a in range(32):
                for b in range(a+1, 32):
                    for component in ('x', 'z'):
                        sim = experiment.initial.copy(seed=17)
                        getattr(sim, component)(a)
                        getattr(sim, component)(b)
                        assert not protocol.run(sim).accepted
                        rejected += 1
            assert correct == 1089 and rejected == 992
            physical[basis] = dict(css_bound_one_accepted_and_correct=correct,
                                   pure_weight_two_rejected=rejected)
            for label, x, z, shots, expected in (
                    ('split_css_one', 1, 1 << 17, 4, 4),
                    ('ambiguous_x_pair', 3, 0, 2, 0),
                    ('ambiguous_z_pair', 0, 3, 2, 0)):
                result = run_native(bundle, Path('build') / 'steane_css_ps_controls' / mode / basis / label,
                    prepared_state=basis, cycles=3, p=0, shots=shots, seed=20261112,
                    incoming_x=x, incoming_z=z)
                assert result['accepted'] == expected and result['logical_errors'] == 0
                native[basis + '/' + label] = result
        noiseless = {}
        for basis in ('0', '+'):
            result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), prepared_state=basis,
                                     cycles=5).sample(1000, 20261112)
            assert result.accepted == 1000 and result.logical_errors == 0
            noiseless[basis] = dict(shots=1000, accepted=1000, logical_errors=0, cycles=5)
        output[mode] = dict(physical_inputs=physical, native=native, noiseless_reference=noiseless)
        print(f'All physical/native controls passed: {mode}', flush=True)
    path = Path(__file__).resolve().parent / 'controls.json'
    path.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()

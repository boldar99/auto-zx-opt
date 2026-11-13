"""Physical inputs, first-gadget faults and executable HUGR controls."""
import json
import argparse
from pathlib import Path
import random

from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_adaptive_experiments import CSSRecoverCandidate
from spiderwarp.se_native import run_native
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment


class InjectFirst:
    def __init__(self, simulator, gadget, effect):
        self.sim, self.gadget = simulator, gadget
        self.injected = gadget.with_faults(gadget.fault_witness(effect, 1))
        self.done = False

    def current_measurement_record(self):
        return self.sim.current_measurement_record()

    def do_circuit(self, circuit):
        if not self.done and circuit is self.gadget.circuit:
            self.done = True
            self.sim.do_circuit(self.injected)
        else:
            self.sim.do_circuit(circuit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['css', 'css_recover_safe'])
    parser.add_argument('--native-cycles', type=int, default=1,
                        help='Native smoke-test cycles; repeated reference/frame tests are separate')
    args = parser.parse_args()
    parent = load_protocol('build/se_32_20_4')
    model = CSSRecoverCandidate(parent, 'css-recover-exact')
    target = Path(__file__).resolve().parent / 'controls.json'
    output = json.loads(target.read_text()) if target.exists() else {}
    for variant in args.variants:
        bundle = Path('build') / ('adaptive_' + variant + '_32_20_4')
        protocol = load_protocol(bundle)
        index = protocol.policy.root
        while protocol.policy.nodes[index].kind == 'frame':
            index = protocol.policy.nodes[index].next
        first = protocol.gadgets[protocol.policy.nodes[index].action]
        physical = {}
        for basis in ('0', '+'):
            experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
            inputs = 0
            for x in (0, *(1 << q for q in range(32))):
                for z in (0, *(1 << q for q in range(32))):
                    sim = experiment.initial.copy(seed=17)
                    for q in range(32):
                        if x >> q & 1: sim.x(q)
                        if z >> q & 1: sim.z(q)
                    result = protocol.run(sim)
                    assert result.accepted
                    assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
                    inputs += 1
            accepted, rejected, rescues = 0, 0, 0
            rng = random.Random(20261113)
            for effect in first.single_fault_effects:
                sim = experiment.initial.copy(seed=17)
                wrapped = InjectFirst(sim, first, effect)
                result = protocol.run(wrapped)
                assert wrapped.done
                if result.accepted:
                    accepted += 1
                    assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
                else:
                    rejected += 1
                if variant.startswith('css_recover'):
                    sim = experiment.initial.copy(seed=17)
                    sim.x(rng.randrange(32))
                    sim.z(rng.randrange(32))
                    result = protocol.run(InjectFirst(sim, first, effect))
                    if result.accepted:
                        state = model.initial
                        for name, _, _, flags, syndrome in result.measurements:
                            state = model._step(state, name, flags, syndrome) if state else state
                        if not state:
                            rescues += 1
                            assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
            control = ProtocolSampler(protocol, noise_model=CustomErrorModel(), prepared_state=basis, cycles=5).sample(1000, 20261113)
            assert control.accepted == 1000 and control.logical_errors == 0
            physical[basis] = dict(css_one_inputs_correct=inputs,
                distinct_first_gadget_faults=len(first.single_fault_effects), accepted_correct=accepted,
                postselected=rejected, incoming_css_one_plus_fault_new_rescues_correct=rescues,
                noiseless_five_cycle_shots=1000)
            if variant.endswith('safe'):
                assert rejected == 0 and rescues > 0
        print(variant, physical, flush=True)
        directory = Path('build/adaptive_css_native_controls') / variant
        cycles = args.native_cycles
        clean = run_native(bundle, directory, prepared_state='+', cycles=cycles, p=0, shots=8,
                           seed=20261113, incoming_x=1, incoming_z=1 << 17)
        assert clean['accepted'] == 8 and clean['logical_errors'] == 0
        noisy = run_native(bundle, directory, prepared_state='+', cycles=cycles, p=.001, shots=100,
                           seed=20261113, incoming_x=1, incoming_z=1 << 17)
        assert 0 < noisy['accepted'] < 100
        output[variant] = dict(physical=physical, native_clean_split_input=clean,
                               native_noisy_both_branches=noisy)
        (Path(__file__).resolve().parent / 'controls.json').write_text(json.dumps(output, indent=2) + '\n')
        print(f'{variant}: physical and native controls passed', flush=True)


if __name__ == '__main__':
    main()

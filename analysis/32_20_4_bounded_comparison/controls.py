"""Physical input/fault replays and executable HUGR controls for rebuilt PS."""
import argparse
import json
from dataclasses import asdict
from importlib import import_module

from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_native import run_native
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment

study = import_module('analysis.32_20_4_bounded_comparison.study')


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    keys = ('adaptive_history', 'adaptive_no_flags')
    parser.add_argument('--only', choices=keys)
    args = parser.parse_args(argv)
    path = study.OUT / 'controls.json'
    checks = json.loads(path.read_text())['methods'] if args.only and path.exists() else {}
    for key in keys:
        if args.only and key != args.only:
            continue
        bundle = study.ROOT / 'build' / study.BUNDLES[key]
        protocol = load_protocol(bundle)
        first = protocol.gadgets['Z_d4']
        results = {}
        for basis in ('0', '+'):
            experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
            accepted = rejected = 0
            for x in (0, *(1 << q for q in range(32))):
                for z in (0, *(1 << q for q in range(32))):
                    sim = experiment.initial.copy(seed=17)
                    for q in range(32):
                        if x >> q & 1: sim.x(q)
                        if z >> q & 1: sim.z(q)
                    result = protocol.run(sim)
                    if result.accepted:
                        accepted += 1
                        assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
                    else:
                        rejected += 1
            fault_accept = fault_reject = 0
            for effect in first.single_fault_effects:
                sim = experiment.initial.copy(seed=17)
                wrapped = InjectFirst(sim, first, effect)
                result = protocol.run(wrapped)
                assert wrapped.done
                assert len(result.measurements) <= 3
                assert not any(distance == 1 for _, _, distance, _, _ in result.measurements)
                if result.accepted:
                    fault_accept += 1
                    if key == 'adaptive_no_flags':
                        assert not any(f for _, _, _, f, _ in result.measurements)
                    assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
                else:
                    fault_reject += 1
            second = protocol.gadgets['X_d4']
            followup_accept = followup_reject = second_accept = 0
            for effect in second.single_fault_effects:
                sim = experiment.initial.copy(seed=17)
                wrapped = InjectFirst(sim, second, effect)
                result = protocol.run(wrapped)
                assert wrapped.done
                assert len(result.measurements) <= 3
                if result.accepted:
                    second_accept += 1
                    if key == 'adaptive_no_flags':
                        assert not any(f for _, _, _, f, _ in result.measurements)
                    assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
                    followup_accept += len(result.measurements) == 3
                else:
                    followup_reject += len(result.measurements) == 3
            assert accepted == 1089
            if key == 'adaptive_no_flags':
                assert followup_accept == 0 and followup_reject == 0
                assert fault_reject > 0 and second_accept < len(second.single_fault_effects)
            else:
                assert followup_accept > 0
            if key == 'adaptive_history':
                assert fault_reject == 0 and second_accept == len(second.single_fault_effects)
            clean = ProtocolSampler(protocol, noise_model=CustomErrorModel(), prepared_state=basis,
                                    cycles=10).sample(100, 20261113)
            assert clean.accepted == 100 and clean.logical_errors == 0
            assert clean.diagnostics.accepted_cnots == clean.diagnostics.total_cnots == 100 * 10 * 312
            directory = study.ROOT / 'build/bounded_native_controls' / key / basis.replace('+', 'plus')
            native_clean = run_native(bundle, directory / 'clean', prepared_state=basis,
                                      cycles=3, p=0, shots=8, seed=20261113, incoming_x=1, incoming_z=1 << 17)
            assert native_clean['accepted'] == 8 and native_clean['logical_errors'] == 0
            native_noisy = run_native(bundle, directory / 'noisy', prepared_state=basis,
                                      cycles=1, p=.001, shots=100, seed=20261113)
            assert 0 < native_noisy['accepted'] < 100
            results[basis] = dict(css_one_inputs=1089, accepted_inputs_correct=accepted,
                                  postselected_inputs=rejected, first_fault_classes=len(first.single_fault_effects),
                                  accepted_faults_correct=fault_accept, postselected_faults=fault_reject,
                                  second_fault_classes=len(second.single_fault_effects),
                                  accepted_second_faults_correct=second_accept,
                                  accepted_d2_followup_replays=followup_accept,
                                  rejected_d2_followup_replays=followup_reject,
                                  noiseless_ten_cycle_shots=100, native_clean=native_clean,
                                  native_gate_only_noisy=native_noisy)
            print(key, basis, 'physical and native controls passed', flush=True)
        checks[key] = results
    records = [r for method in checks.values() for r in method.values()]
    study.write(study.OUT / 'controls.json', dict(complete=True, methods=checks,
                physical_css_one_inputs=sum(r['css_one_inputs'] for r in records),
                native_clean_shots=sum(r['native_clean']['shots'] for r in records),
                native_noisy_shots=sum(r['native_gate_only_noisy']['shots'] for r in records),
                native_noise_scope='Gate-only backend controls; differs from production scheduled-memory noise',
                history_method_requires_single_fault_acceptance=True,
                zero_flag_experiment_can_postselect_single_faults=True))


if __name__ == '__main__':
    main()

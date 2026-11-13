"""Independent physical injections and exact residual checks for V2 rescues."""
from collections import Counter
from importlib import import_module
from itertools import combinations
import random

from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import _StateExperiment, _observable
from spiderwarp.se_postselection import OrderOneEvidence

study = import_module('analysis.32_20_4_bounded_comparison.study')


def algebraic_replay(protocol, effects):
    """Propagate physical fault responses without the uniqueness compressor."""
    index, frame, error, position = protocol.policy.root, 0, 0, 0
    space = protocol.policy.decoder.space
    gadgets = protocol.gadgets
    while True:
        node = protocol.policy.nodes[index]
        if node.kind in ('accept', 'reject'):
            return node.kind == 'accept', space.reduce(error ^ frame ^ node.frame)
        if node.kind == 'frame':
            frame ^= node.frame
            index = node.next
            continue
        gadget = gadgets[node.action]
        effect = effects[position] if position < len(effects) else 0
        flags, syndrome, delta = gadget.unpack(effect)
        component = 'X' if gadget.basis == 'Z' else 'Z'
        syndrome ^= space.component_syndrome(error ^ frame, component)
        error ^= delta
        position += 1
        index = node.children.get((flags, syndrome), 0)


def measurement_pair_controls(protocol):
    history = load_protocol(study.ROOT / 'build/se_postselected_32_20_4')
    gadgets = protocol.gadgets
    flips = {}
    for name, gadget in gadgets.items():
        gadget.responses
        flips[name] = {fault.operation_offset: effect
            for effect, faults in gadget._fault_witnesses.items()
            for fault in faults if fault.before_measurement}
        assert all(gadget.unpack(e)[2] == 0 for e in flips[name].values())
    counts = Counter()

    def check(effects, placement):
        accepted, residual = algebraic_replay(protocol, effects)
        old_accepted, _ = algebraic_replay(history, effects)
        counts['physical_pairs'] += 1
        counts[placement + '_pairs'] += 1
        counts['accepted' if accepted else 'rejected'] += 1
        if accepted and not old_accepted:
            assert residual == 0
            counts['additional_exact_rescues'] += 1
            counts[placement + '_rescues'] += 1

    for first, second in combinations(flips['Z_d4'].values(), 2):
        check((first ^ second,), 'both_first')
    for first, second in combinations(flips['X_d4'].values(), 2):
        check((0, first ^ second), 'both_second')
    for first in flips['Z_d4'].values():
        second_name = 'X_d2' if gadgets['Z_d4'].unpack(first)[0] else 'X_d4'
        for second in flips[second_name].values():
            check((first, second), 'across_first_second')
    return dict(counts)


class Inject:
    def __init__(self, simulator, gadgets, plan, rng, incoming):
        self.simulator = simulator
        self.gadgets = {id(g.circuit): g for g in gadgets.values()}
        self.plan, self.rng = plan, rng
        self.position, self.error = 0, incoming
        self.faults = []

    def current_measurement_record(self):
        return self.simulator.current_measurement_record()

    def do_circuit(self, circuit):
        gadget = self.gadgets[id(circuit)]
        order = self.plan[self.position] if self.position < len(self.plan) else 0
        self.position += 1
        if order:
            effect = self.rng.choice(gadget.fault_effects(order))
            faults = gadget.fault_witness(effect, order)
            assert len(faults) == order
            assert len({f.operation_offset for f in faults}) == order
            self.faults.append((gadget.basis, order, faults))
            self.error ^= gadget.unpack(effect)[2]
            self.simulator.do_circuit(gadget.with_faults(faults))
        else:
            self.simulator.do_circuit(circuit)


def main():
    protocol = load_protocol(study.ROOT / 'build/adaptive_order_two_32_20_4')
    evidence = OrderOneEvidence(protocol)
    space = protocol.policy.decoder.space
    rng = random.Random(2026111309)
    counts, examples = Counter(), []
    for basis in ('0', '+'):
        experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
        checks = [(component, _observable(protocol.required_qubits, row, component))
                  for component, rows in (('X', protocol.code.H_x), ('Z', protocol.code.H_z))
                  for row in rows]
        for case in range(3000):
            # Cover incoming-one/internal-one, both faults within either gadget,
            # faults across gadgets, and a fault in an adaptive third reading.
            category = case % 6
            incoming = 0
            if category < 2:
                incoming = (1 << rng.randrange(32)) ^ (1 << (32 + rng.randrange(32)))
                plan = (1, 0, 0) if category == 0 else (0, 1, 0)
            else:
                plan = ((2, 0, 0), (0, 2, 0), (1, 1, 0), (0, 1, 1))[category - 2]
            sim = experiment.initial.copy(seed=case + 17)
            for letter, qubit in space.product(incoming):
                getattr(sim, letter.lower())(qubit)
            wrapped = Inject(sim, protocol.gadgets, plan, rng, space.reduce(incoming))
            result = protocol.run(wrapped)
            counts['attempted'] += 1
            counts[f'category_{category}_attempted'] += 1
            if not result.accepted:
                counts['rejected'] += 1
                continue
            counts['accepted'] += 1
            state = evidence.initial
            for name, _, _, flags, syndrome in result.measurements:
                state = evidence.transition(state, name).get((flags, syndrome), frozenset()) if state else state
            if state:
                counts['protected'] += 1
                continue
            correction = result.frame_x | (result.frame_z << protocol.code.n)
            assert space.reduce(wrapped.error ^ correction) == 0
            # Check codespace without the endpoint decoder, then check logical
            # information. Clearing a syndrome alone would miss a logical error.
            for component, observable in checks:
                sign = sim.peek_observable_expectation(observable)
                parity = 0
                frame = result.frame_z if component == 'X' else result.frame_x
                for qubit in range(protocol.code.n):
                    if observable[qubit] and (frame >> qubit) & 1:
                        parity ^= 1
                assert sign == (-1 if parity else 1)
            assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
            counts['rescued_exactly_correct'] += 1
            counts[f'category_{category}_rescued'] += 1
            if len(examples) < 16:
                examples.append(dict(basis=basis, incoming=space.product(incoming),
                    plan=plan, measurements=result.measurements, frame=[result.frame_x, result.frame_z]))
        print(basis, dict(counts), flush=True)
    assert counts['rescued_exactly_correct'] > 100
    assert counts['attempted'] == 6000
    measurements = measurement_pair_controls(protocol)
    print('Measurement pairs:', measurements, flush=True)
    study.write(study.OUT / 'order_two_controls.json', dict(complete=True,
        counts=dict(counts), measurement_pairs=measurements, examples=examples, seed=2026111309,
        physical_faults='Distinct physical sites; exact stabilizer and logical checks on additional rescues',
        noise_scope='Certificate gate/reset/readout model; production memory noise assessed separately'))


if __name__ == '__main__':
    main()

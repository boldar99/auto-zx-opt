"""Frozen MAP Pauli corrections with mandatory order-one acceptance.

Additional records use the final-data posterior under the scheduled hardware
noise and incoming iid Pauli prior, truncated at two nonzero response channels.
The threshold bounds estimated wrong-Pauli-class probability, a sufficient
condition for logical correction; it is not an all-orders probability bound.
"""
from collections import Counter
from dataclasses import replace
import math
from types import SimpleNamespace

from spiderwarp._se_adaptive import (
    AdaptivePolicy, AdaptiveVerificationReport, FixedCSSDecoder, PolicyNode,
    compact_policy,
)
from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol
from spiderwarp.se_likelihood import FullHistoryLikelihood, _probability
from spiderwarp.optimized_steane_se import ProtocolResult


PATHS = (('Z_d4', 'X_d4'), ('Z_d4', 'X_d2'),
         ('Z_d4', 'X_d4', 'Z_d2'), ('Z_d4', 'X_d4', 'X_d2'))


class ProbabilityRescueModel:
    """Enumerate the entire two-event support, with no discovery sampling.

    Low-order histories retain the parent's exact actions and frames. Other
    records complete two readings, or the parent's already selected third d2.
    Thus both epsilon values share a fixed measurement schedule for common
    records; their acceptance domains are nested.
    """

    def __init__(self, parent, **kwargs):
        if set(parent.gadgets) != {'Z_d4', 'X_d4', 'Z_d2', 'X_d2'}:
            raise ValueError('Probability rescue requires the bounded d4/d2 parent')
        self.parent = parent
        self.model = FullHistoryLikelihood(parent, **kwargs)
        self.mandatory, self.third_routes = {}, {}
        self._collect_parent()

    def pack(self, record):
        observation = offset = 0
        for name, flags, syndrome in record:
            gadget = self.parent.gadgets[name]
            observation |= (flags | (syndrome << gadget.num_flags)) << offset
            offset += gadget.num_detectors
        return observation

    def _collect_parent(self):
        old, evidence = self.parent.policy, self.model.evidence
        space = evidence.space

        def visit(index, frame, state, record):
            node = old.nodes[index]
            if node.kind == 'frame':
                visit(node.next, frame ^ node.frame, evidence.translate(state, node.frame), record)
            elif node.kind == 'accept':
                if state:
                    names = tuple(step[0] for step in record)
                    self.mandatory[names, self.pack(record)] = space.reduce(frame ^ node.frame)
            elif node.kind == 'extract':
                if len(record) == 2:
                    if tuple(step[0] for step in record) != ('Z_d4', 'X_d4'):
                        raise ValueError('Parent uses an unsupported third-reading path')
                    self.third_routes[self.pack(record)] = node.action
                gadget = self.parent.gadgets[node.action]
                offset = space.component_syndrome(frame, 'X' if gadget.basis == 'Z' else 'Z')
                groups = evidence.transition(state, node.action) if state else {}
                for (flags, syndrome), child in node.children.items():
                    following = groups.get((flags, syndrome), frozenset())
                    visit(child, frame, following,
                          record + ((node.action, flags, syndrome ^ offset),))

        visit(old.root, 0, evidence.initial, ())
        if any(names not in PATHS for names, _ in self.mandatory):
            raise ValueError('Parent uses an unsupported path')

    def valid(self, names, observation):
        first = self.parent.gadgets['Z_d4']
        first_flags = observation & ((1 << first.num_flags) - 1)
        if (names[1] == 'X_d2') != bool(first_flags):
            return False
        if names[1] == 'X_d2':
            return len(names) == 2
        width = first.num_detectors + self.parent.gadgets['X_d4'].num_detectors
        continuation = self.third_routes.get(observation & ((1 << width) - 1))
        return continuation is None if len(names) == 2 else names[-1] == continuation

    def records(self, progress=None):
        """Yield MAP error class, probability and correction for every record."""
        for names in PATHS:
            path_model = self.model.path_model(names)
            def keep(observation):
                return self.valid(names, observation) and (names, observation) not in self.mandatory
            groups = path_model.joint_weights(keep)
            if progress:
                progress(f'Probability model {names}: {len(groups):,} additional records')
            for observation, weights in groups.items():
                total = math.fsum(weights.values())
                correction, best = max(weights.items(), key=lambda item: (item[1], -item[0]))
                risk = min(1.0, max(0.0, (total - best) / total))
                evidence = total * path_model.p_zero
                omitted = path_model.omitted
                yield names, observation, correction, risk, evidence, omitted

    def unpack(self, names, observation):
        record = []
        for name in names:
            gadget = self.parent.gadgets[name]
            outcome = observation & ((1 << gadget.num_detectors) - 1)
            record.append((name, outcome & ((1 << gadget.num_flags) - 1), outcome >> gadget.num_flags))
            observation >>= gadget.num_detectors
        return tuple(record)


class ProbabilityRescueCandidate:
    """Pointwise posterior reference, independent of the convolution compiler."""
    def __init__(self, model, epsilon):
        self.model, self.epsilon = model, _probability(epsilon, 'epsilon')
        self.code, self.policy = model.parent.code, model.parent.policy
        self.gadgets, self.required_qubits = model.parent.gadgets, model.parent.required_qubits
        self._decisions = {}

    def decision(self, record):
        names = tuple(step[0] for step in record)
        observation = self.model.pack(record)
        key = names, observation
        if key in self.model.mandatory:
            return True, self.model.mandatory[key]
        if key not in self._decisions:
            path = self.model.model.path_model(names)
            # Exact physical support prevents exclusive alternatives producing a
            # fictitious record through cancellation in the weighted calculation.
            supported = observation == 0 or observation in path.singles
            if not supported:
                for first, values in path.singles.items():
                    second = observation ^ first
                    if second not in path.singles:
                        continue
                    for a in values:
                        for b in path.singles[second]:
                            locations_a = path.locations[first | (a << path.bits)]
                            locations_b = path.locations[second | (b << path.bits)]
                            if len(locations_a) > 1 or len(locations_b) > 1 or locations_a != locations_b:
                                supported = True
                    if supported:
                        break
            score = path.posterior(observation, 0, SimpleNamespace(logical=lambda error: error)) if supported else None
            if score is None or not score.logical_classes:
                decision = False, 0
            else:
                correction, probability = max(score.logical_classes, key=lambda item: (item[1], -item[0]))
                decision = 1 - probability <= self.epsilon, correction
            self._decisions[key] = decision
        return self._decisions[key]

    def run(self, simulator, *, frame_x=0, frame_z=0):
        incoming = frame_x | (frame_z << self.code.n)
        space, record, measurements = self.model.model.space, [], []

        def read(name):
            gadget = self.gadgets[name]
            start = len(simulator.current_measurement_record())
            simulator.do_circuit(gadget.circuit)
            flags, syndrome = gadget.readout.decode(simulator.current_measurement_record()[start:])
            measurements.append((name, gadget.basis, gadget.distance, flags, syndrome))
            relative = syndrome ^ space.component_syndrome(incoming, 'X' if gadget.basis == 'Z' else 'Z')
            record.append((name, flags, relative))
            return flags

        flags = read('Z_d4')
        read('X_d2' if flags else 'X_d4')
        following = self.model.third_routes.get(self.model.pack(record)) if not flags else None
        if following:
            read(following)
        accepted, correction = self.decision(tuple(record))
        frame = incoming ^ correction
        mask = (1 << self.code.n) - 1
        return ProtocolResult(accepted, tuple(measurements), (),
            discard_reason=None if accepted else 'MAP correction confidence',
            frame_x=frame & mask, frame_z=frame >> self.code.n)


def compile_probability_rescues(parent, thresholds=(0.1, 0.25), *, model=None,
                                verify=True, progress=None):
    """Build nested, noise-frozen ordinary controllers for both thresholds."""
    thresholds = tuple(sorted(set(_probability(t, 'epsilon') for t in thresholds)))
    if not thresholds:
        raise ValueError('At least one epsilon is required')
    model = ProbabilityRescueModel(parent) if model is None else model
    if model.parent is not parent:
        raise ValueError('Probability model belongs to a different parent')
    selected = {epsilon: [] for epsilon in thresholds}
    audit = Counter()
    examples = []
    for names, observation, correction, risk, evidence, omitted in model.records(progress):
        audit['scored_records'] += 1
        for epsilon in thresholds:
            if risk <= epsilon:
                selected[epsilon].append((names, observation, correction))
        if risk <= thresholds[-1] and len(examples) < 32:
            examples.append(dict(record=model.unpack(names, observation), correction=correction,
                estimated_error=risk, evidence_probability=evidence, omitted_probability=omitted,
                all_orders_risk_upper_bound=min(1.0, (risk * evidence + omitted) / (evidence + omitted))))
    result = {}
    for epsilon in thresholds:
        trie = [dict(action=None, children={}, frame=None)]

        def add(names, observation, correction):
            index = 0
            for name, flags, syndrome in model.unpack(names, observation):
                node = trie[index]
                if node['frame'] is not None or node['action'] not in (None, name):
                    raise ValueError('Conflicting probability-controller paths')
                node['action'] = name
                if (flags, syndrome) not in node['children']:
                    node['children'][flags, syndrome] = len(trie)
                    trie.append(dict(action=None, children={}, frame=None))
                index = node['children'][flags, syndrome]
            if trie[index]['children'] or trie[index]['frame'] not in (None, correction):
                raise ValueError('Conflicting probability-controller correction')
            trie[index]['frame'] = correction

        for (names, observation), correction in model.mandatory.items():
            add(names, observation, correction)
        for names, observation, correction in selected[epsilon]:
            add(names, observation, correction)
        nodes, interned = [PolicyNode('reject')], {('reject',): 0}

        def intern(index):
            entry = trie[index]
            if entry['frame'] is not None:
                node = PolicyNode('accept', frame=entry['frame'])
                key = 'accept', node.frame
            else:
                children = {outcome: intern(child) for outcome, child in sorted(entry['children'].items())}
                node = PolicyNode('extract', action=entry['action'], children=children)
                key = 'extract', node.action, tuple(children.items())
            if key not in interned:
                interned[key] = len(nodes)
                nodes.append(node)
            return interned[key]

        root = intern(0)
        contract = replace(parent.policy.contract, accept_clean_through=1)
        decoder = FixedCSSDecoder(parent.policy.decoder.space, contract)
        policy = AdaptivePolicy(parent.policy.actions.copy(), contract, decoder, tuple(nodes), root,
                                AdaptiveVerificationReport(contract, decoder.fingerprint),
                                peak_qubit_limit=parent.policy.peak_qubit_limit)
        compact_policy(policy)
        protocol = AdaptiveSEProtocol(parent.code, policy)
        if verify:
            protocol.verify()
        provenance = dict(model.model.metadata(), method='MAP-Pauli-confidence-rescue', model_version=1,
            epsilon=epsilon, acceptance='Order-one compatibility OR estimated MAP correction-error probability <= epsilon',
            error_probability='1 - posterior probability of most likely final Pauli class modulo stabilizers',
            correction='Inverse of MAP final data Pauli class; mandatory order-one frames unchanged',
            scored_records=audit['scored_records'], mandatory_records=len(model.mandatory),
            additional_accepted_records=len(selected[epsilon]), saved_nodes=len(policy.nodes),
            parent_rejected_records_considered=True, discovery_sampling=False, probabilities_used=True,
            noise_profile_frozen=True, output_labels_used=False, all_orders_ler_guarantee=False,
            unknown_record_rule='Reject outside the two-event response support unless order-one-compatible',
            fixed_schedule='Parent third-reading routes retained; otherwise complete two readings',
            threshold_scope='Estimated per-cycle wrong-Pauli-class risk on additional records only',
            risk='Incorrect final data Pauli class after MAP correction', examples=examples)
        if progress:
            progress(f'Perr <= {epsilon:g}: {len(selected[epsilon]):,} additional records; {len(nodes):,} nodes; verified')
        result[epsilon] = protocol, provenance
        del trie
    return result

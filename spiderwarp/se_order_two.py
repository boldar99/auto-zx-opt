"""Exact order-two recovery from observable adaptive syndrome histories.

Keep the existing controller on every order-one-compatible record. Additional
records accept only when all r+s<=2 explanations agree modulo stabilizers.
The circuit model is the existing certificate's physical gate/reset/readout
locations (no implicit idle faults); production sampling also includes memory.
"""
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache

from spiderwarp._se_adaptive import (
    AdaptivePolicy, AdaptiveVerificationReport, FixedCSSDecoder, PolicyNode,
    compact_policy,
)
from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol


AMBIGUOUS = -1


class OrderTwoEvidence:
    """Merge equal-syndrome histories only when their remaining budgets agree.

    A state contains (remaining total order, joint syndrome, Pauli class).
    AMBIGUOUS means at least two stabilizer classes with the same syndrome.
    Those classes have identical future observations under each subsequent
    fault response, so this marker preserves exact correction uniqueness.
    Incoming order is max(w_X,w_Z), rather than total Pauli weight.
    """

    def __init__(self, parent):
        self.parent = parent
        self.space, self.decoder = parent.policy.decoder.space, parent.policy.decoder
        self.signature = lru_cache(maxsize=131072)(self.space.pauli_syndrome)
        self.responses = {}
        self._initial = None

    @staticmethod
    def add(groups, key, error):
        previous = groups.get(key)
        if previous is None:
            groups[key] = error
        elif previous != error:
            groups[key] = AMBIGUOUS

    @property
    def initial(self):
        if self._initial is None:
            groups = {}
            xs = self.space.component_errors('X', 2)
            zs = self.space.component_errors('Z', 2)
            for x, physical_x in xs.items():
                for z, physical_z in zs.items():
                    order = max(physical_x.bit_count(), physical_z.bit_count())
                    error = x ^ z
                    self.add(groups, (2 - order, self.signature(error)), error)
            self._initial = tuple((b, s, e) for (b, s), e in sorted(groups.items()))
        return self._initial

    def read(self, signature, name):
        gadget = self.parent.gadgets[name]
        return (signature & self.decoder.x_mask if gadget.basis == 'Z'
                else signature >> self.decoder.x_width)

    def fault_responses(self, name, order):
        key = name, order
        if key not in self.responses:
            gadget = self.parent.gadgets[name]
            self.responses[key] = tuple((f, s, data, self.signature(data))
                for effect in gadget.fault_effects(order)
                for f, s, data in (gadget.unpack(effect),))
        return self.responses[key]

    def transition(self, state, name):
        groups = defaultdict(dict)
        for budget, signature, error in state:
            expected = self.read(signature, name)
            self.add(groups[0, expected], (budget, signature), error)
            for order in range(1, budget + 1):
                for flags, syndrome, delta, delta_signature in self.fault_responses(name, order):
                    output = AMBIGUOUS if error == AMBIGUOUS else error ^ delta
                    self.add(groups[flags, expected ^ syndrome],
                             (budget - order, signature ^ delta_signature), output)
        return {outcome: tuple((b, s, e) for (b, s), e in sorted(values.items()))
                for outcome, values in groups.items()}

    @staticmethod
    def protected(state):
        return any(budget > 0 for budget, _, _ in state)

    @staticmethod
    def correction(state):
        if not state or any(error == AMBIGUOUS for _, _, error in state):
            return None
        errors = {error for _, _, error in state}
        return next(iter(errors)) if len(errors) == 1 else None


def compile_order_two(parent, *, verify=True, progress=None):
    """Compile all bounded records, including parent-rejected observations.

    On order-one records, quantum actions and corrections are unchanged. Other
    records finish the first two readings and, on a d4/d4 path, may use one d2
    in the basis that separates the most uniquely correctable outcome groups.
    This choice uses exact classes, with no noise probabilities or output labels.
    """
    old = parent.policy
    if (old.contract.input_scope != 'total-order' or old.contract.correctable != 1
            or old.contract.logical_order != 1 or old.contract.fault_order != 2):
        raise ValueError('Order-two rescue requires the total-order-one parent with order-two analysis')
    if set(parent.gadgets) != {'Z_d4', 'X_d4', 'Z_d2', 'X_d2'}:
        raise ValueError('Order-two rescue requires the bounded d4/d2 gadget library')
    evidence = OrderTwoEvidence(parent)
    space = evidence.space
    nodes, interned = [PolicyNode('reject')], {('reject',): 0}
    stats = dict(contexts=0, protected_accepts=0, rescued_contexts=0,
                 ambiguous_rejects=0, extra_followups=0)
    witnesses = []

    def intern(node):
        key = node.kind, node.action, node.frame, tuple(sorted(node.children.items()))
        if key not in interned:
            interned[key] = len(nodes)
            nodes.append(node)
        return interned[key]

    def resolve(index, frame):
        while old.nodes[index].kind == 'frame':
            frame ^= old.nodes[index].frame
            index = old.nodes[index].next
        return old.nodes[index], frame

    @lru_cache(maxsize=None)
    def exhausted(state, names):
        # Every compatible explanation has spent total order two. Further
        # observations in this bounded analysis are ideal, so no pair expansion.
        correction = evidence.correction(state)
        if correction is not None:
            stats['rescued_contexts'] += 1
            return intern(PolicyNode('accept', frame=correction))
        if names != ('Z_d4', 'X_d4'):
            stats['ambiguous_rejects'] += 1
            return 0
        choices = []
        for name in ('Z_d2', 'X_d2'):
            groups = defaultdict(list)
            for item in state:
                groups[evidence.read(item[1], name)].append(item)
            children = {syndrome: evidence.correction(group) for syndrome, group in groups.items()}
            keep = {s: c for s, c in children.items() if c is not None}
            choices.append((len(keep), name, keep))
        count, name, keep = max(choices, key=lambda item: (item[0], item[1] == 'Z_d2'))
        if not count:
            stats['ambiguous_rejects'] += 1
            return 0
        stats['extra_followups'] += 1
        stats['rescued_contexts'] += len(keep)
        return intern(PolicyNode('extract', action=name, children={
            (0, syndrome): intern(PolicyNode('accept', frame=correction))
            for syndrome, correction in sorted(keep.items())}))

    def visit(index, frame, state, names, prefix):
        stats['contexts'] += 1
        if progress and stats['contexts'] % 10000 == 0:
            progress(f"V2: {stats['contexts']:,} contexts, {len(nodes):,} nodes")
        node, frame = resolve(index, frame)
        protected = evidence.protected(state)
        if protected and node.kind == 'accept':
            stats['protected_accepts'] += 1
            return intern(PolicyNode('accept', frame=frame ^ node.frame))
        if protected and node.kind == 'reject':
            raise ValueError(f'Parent rejected an order-one-compatible record: {prefix}')
        if not protected and len(names) >= 2:
            result = exhausted(state, names)
            if result and len(witnesses) < 32:
                witnesses.append(dict(record=prefix, state=state))
            return result
        if protected:
            name = node.action
        else:
            assert names == ('Z_d4',)
            name = 'X_d2' if prefix[0][1] else 'X_d4'
        children = {}
        for outcome, following in sorted(evidence.transition(state, name).items()):
            flags, syndrome = outcome
            relative = syndrome ^ space.component_syndrome(frame,
                'X' if parent.gadgets[name].basis == 'Z' else 'Z')
            child = node.children.get((flags, relative), 0) if node.kind == 'extract' else 0
            target = visit(child, frame, following, names + (name,), prefix + ((name, *outcome),))
            if target:
                children[outcome] = target
        return intern(PolicyNode('extract', action=name, children=children)) if children else 0

    root = visit(old.root, 0, evidence.initial, (), ())
    contract = replace(old.contract, accept_clean_through=1)
    decoder = FixedCSSDecoder(space, contract)
    policy = AdaptivePolicy(old.actions.copy(), contract, decoder, tuple(nodes), root,
                            AdaptiveVerificationReport(contract, decoder.fingerprint),
                            peak_qubit_limit=old.peak_qubit_limit)
    compact_policy(policy)
    derived = AdaptiveSEProtocol(parent.code, policy)
    if verify:
        derived.verify()
    provenance = dict(method='exact-order-two-uniqueness', version=1,
        incoming_order='max(w_X,w_Z)', analyzed_orders=[[2, 0], [1, 1], [0, 2]],
        acceptance='Preserve order-one records; otherwise one final Pauli class modulo stabilizers',
        fault_model=derived.verification.noise_model,
        additional_rescues_certified_through=2, all_accepted_order_two_correct=False,
        memory_noise_in_certificate=False, parent_rejected_records_considered=True,
        probabilities_used=False, output_labels_used=False,
        allowed_paths=[[4, 4], [4, 2], [4, 4, 2]], statistics=stats,
        order_two_states_analyzed=exhausted.cache_info().currsize, witnesses=witnesses)
    if progress:
        progress(f'V2 compiled: {len(policy.nodes):,} nodes; {stats}')
    return derived, provenance

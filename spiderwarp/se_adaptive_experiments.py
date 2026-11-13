"""Observable CSS postselection candidates on the saved adaptive controller.

These sampling adapters are used for discovery before compiling a selected
rule into the ordinary, independently checked decision-graph format.
"""
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache

import numpy as np

from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol
from spiderwarp._se_adaptive import (
    AdaptivePolicy, AdaptiveVerificationReport, FTContract, FixedCSSDecoder,
    PolicyNode, compact_policy,
)
from spiderwarp.optimized_steane_se import ProtocolResult
from spiderwarp.se_postselection import OrderOneEvidence, postselect_order_one
from spiderwarp.steane_qec import unique_css_syndromes

VARIANTS = ('css', 'css-flag', 'css-flag-reversed', 'css-flag-check',
            'css-separated', 'css-separated-retarget', 'css-recover',
            'css-recover-exact', 'css-recover-no-d1', 'css-recover-safe', 'css-no-flags')


def reverse_css_roles(parent):
    """Use the existing dual gadgets, swapping every frame and action role."""
    if not np.array_equal(parent.code.H_x, parent.code.H_z):
        raise ValueError('CSS role reversal requires identical checks')
    n, mask = parent.code.n, (1 << parent.code.n) - 1
    def swap(frame):
        return (frame >> n) | ((frame & mask) << n)
    nodes = tuple(replace(node,
        action=(('X' if node.action[0] == 'Z' else 'Z') + node.action[1:]) if node.action else None,
        frame=swap(node.frame)) for node in parent.policy.nodes)
    report = replace(parent.verification, verified=False, lower_bound_cnots=0)
    policy = replace(parent.policy, nodes=nodes, report=report, optimal=False, lower_bound_cnots=0)
    return AdaptiveSEProtocol(parent.code, policy)


class AdaptiveCSSCandidate:
    def __init__(self, parent, variant='css'):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.parent = reverse_css_roles(parent) if variant.endswith('reversed') else parent
        self.code, self.policy = parent.code, None
        self.required_qubits = parent.required_qubits
        self.gadgets = dict(self.parent.gadgets)
        self.decoder = self.parent.policy.decoder
        self.space = self.decoder.space
        self.unique = unique_css_syndromes(self.code)
        evidence = OrderOneEvidence(self.parent)
        self.initial = evidence.initial
        self.responses = {}
        for name, responses in evidence.responses.items():
            grouped = defaultdict(set)
            for f, s, data in responses:
                grouped[f, s].add(data)
            self.responses[name] = grouped
        if variant.endswith('check'):
            for basis in ('Z', 'X'):
                self.gadgets['check_' + basis + '_d2'] = self.gadgets[basis + '_d2']

    @lru_cache(maxsize=65536)
    def _step(self, state, name, flags, syndrome):
        following = set()
        is_z = self.gadgets[name].basis == 'Z'
        for r, s, data in state:
            expected = data & self.decoder.x_mask if is_z else data >> self.decoder.x_width
            if flags == 0 and syndrome == expected:
                following.add((r, s, data))
            if r + s == 0:
                for delta in self.responses[name].get((flags, syndrome ^ expected), ()):
                    following.add((0, 1, data ^ delta))
        return frozenset(following)

    def run(self, simulator, *, frame_x=0, frame_z=0):
        result = self.parent.run(simulator, frame_x=frame_x, frame_z=frame_z)
        if not result.accepted:
            return result
        measurements = list(result.measurements)
        if self.variant == 'css':
            # r is incoming error order, not the weight of a propagated hook
            # correction. One internal faulty location remains s=1 throughout.
            state = self.initial
            incoming = frame_x | (frame_z << self.code.n)
            for name, basis, _, flags, syndrome in measurements:
                syndrome ^= self.space.component_syndrome(incoming, 'X' if basis == 'Z' else 'Z')
                state = self._step(state, name, flags, syndrome)
                if not state:
                    return replace(result, accepted=False, discard_reason='no order-one incoming/fault explanation')
            return result
        if self.variant == 'css-no-flags' and any(f for _, _, _, f, _ in measurements):
            return replace(result, accepted=False, discard_reason='nonzero preparation flag')
        # Stable flag-free readings directly identify incoming CSS syndromes.
        # Checking only the chosen frame could miss a two-error input that the
        # parent explains as a spurious one-fault readout and leaves uncorrected.
        incoming = frame_x | (frame_z << self.code.n)
        observed = {'X': set(), 'Z': set()}
        if not any(f for _, _, _, f, _ in measurements):
            for _, basis, _, _, raw in measurements:
                component = 'X' if basis == 'Z' else 'Z'
                observed[component].add(raw ^ self.space.component_syndrome(incoming, component))
            if all(len(values) == 1 for values in observed.values()):
                if any(not self.unique['Z' if b == 'X' else 'X'][next(iter(values))]
                       for b, values in observed.items()):
                    return replace(result, accepted=False, discard_reason='ambiguous stable CSS input syndrome')
        delta = (result.frame_x ^ frame_x) | ((result.frame_z ^ frame_z) << self.code.n)
        signature = self.space.pauli_syndrome(delta)
        if (not self.unique['Z'][signature & self.decoder.x_mask]
                or not self.unique['X'][signature >> self.decoder.x_width]):
            return replace(result, accepted=False, discard_reason='ambiguous CSS correction through input weight two')
        if self.variant != 'css' and any(f for _, _, _, f, _ in measurements):
            state = self.initial
            incoming = frame_x | (frame_z << self.code.n)
            for name, basis, _, flags, syndrome in measurements:
                syndrome ^= self.space.component_syndrome(incoming, 'X' if basis == 'Z' else 'Z')
                state = self._step(state, name, flags, syndrome)
                if not state:
                    return replace(result, accepted=False, discard_reason='flagged history has no order-one explanation')
        if self.variant.endswith('check'):
            frame = result.frame_x | (result.frame_z << self.code.n)
            for basis in ('Z', 'X'):
                name = 'check_' + basis + '_d2'
                gadget = self.gadgets[name]
                start = len(simulator.current_measurement_record())
                simulator.do_circuit(gadget.circuit)
                flags, syndrome = gadget.readout.decode(simulator.current_measurement_record()[start:])
                measurements.append((name, basis, 2, flags, syndrome))
                component = 'X' if basis == 'Z' else 'Z'
                syndrome ^= self.space.component_syndrome(frame, component)
                if flags or not self.unique[basis][syndrome]:
                    return ProtocolResult(False, tuple(measurements), (), 'final CSS verification rejected')
                frame ^= self.decoder.physical[component][syndrome]
            mask = (1 << self.code.n) - 1
            return ProtocolResult(True, tuple(measurements), (), frame_x=frame & mask,
                                  frame_z=frame >> self.code.n)
        return result


class SeparatedCSSCandidate(AdaptiveCSSCandidate):
    """Condition separate CSS response histories, retaining Pauli classes.

    Each component gets one incoming-error/fault order independently. Projection
    deliberately forgets cross-component fault-location correlations, producing
    an overapproximation. An accepted frame must work for every surviving class
    in each component. Stable unflagged weight-two input records always reject.
    """
    def __init__(self, parent, variant='css-separated'):
        super().__init__(parent, 'css')
        self.variant = variant
        self.component_initial = {b: frozenset((int(bool(p)), 0, p)
            for p in self.space.component_errors(b, 1)) for b in ('X', 'Z')}
        self.flag_masks, self.component_responses = {}, {}
        for name, g in self.gadgets.items():
            g.responses
            measured = 'X' if g.basis == 'Z' else 'Z'
            assigned = {}
            for effect, witnesses in g._fault_witnesses.items():
                if not any(len(w.pauli_product) == 1 and w.pauli_product[0][0] != 'Y' for w in witnesses):
                    continue
                flags, _, data = g.unpack(effect)
                x, z = self.space.components(data)
                if x and z:
                    raise ValueError('Non-CSS single-Pauli response')
                if not (x or z):
                    continue
                side = 'X' if x else 'Z'
                for bit in range(g.num_flags):
                    if flags >> bit & 1:
                        if bit in assigned and assigned[bit] != side:
                            raise ValueError('A flag mixes CSS response components')
                        assigned[bit] = side
            for component in ('X', 'Z'):
                flag_mask = sum(1 << bit for bit in range(g.num_flags)
                                if assigned.get(bit, measured) == component)
                self.flag_masks[name, component] = flag_mask
                responses = defaultdict(set)
                for effect in g.single_fault_effects:
                    flags, syndrome, data = g.unpack(effect)
                    x, z = self.space.components(data)
                    responses[flags & flag_mask, syndrome if component == measured else 0].add(x if component == 'X' else z)
                self.component_responses[name, component] = responses

    @lru_cache(maxsize=65536)
    def component_step(self, state, name, component, flags, syndrome):
        measured = 'X' if self.gadgets[name].basis == 'Z' else 'Z'
        following = set()
        for r, s, data in state:
            expected = self.space.component_syndrome(data, component) if component == measured else 0
            if flags == 0 and syndrome == expected:
                following.add((r, s, data))
            if r + s == 0:
                for delta in self.component_responses[name, component].get((flags, syndrome ^ expected), ()):
                    following.add((0, 1, data ^ delta))
        return frozenset(following)

    @lru_cache(maxsize=65536)
    def safe_component_frames(self, state, component):
        choices = None
        for _, s, data in state:
            allowed = {data ^ error for error in self.space.component_errors(component, s)}
            choices = allowed if choices is None else choices & allowed
            if not choices:
                break
        return frozenset(choices or ())

    def run(self, simulator, *, frame_x=0, frame_z=0):
        result = self.parent.run(simulator, frame_x=frame_x, frame_z=frame_z)
        if not result.accepted:
            return result
        incoming = frame_x | (frame_z << self.code.n)
        states = dict(self.component_initial)
        records = {'X': set(), 'Z': set()}
        any_flags = False
        for name, basis, _, flags, syndrome in result.measurements:
            measured = 'X' if basis == 'Z' else 'Z'
            syndrome ^= self.space.component_syndrome(incoming, measured)
            records[measured].add(syndrome)
            any_flags |= bool(flags)
            for component in ('X', 'Z'):
                states[component] = self.component_step(states[component], name, component,
                    flags & self.flag_masks[name, component], syndrome if component == measured else 0)
                if not states[component]:
                    return replace(result, accepted=False, discard_reason='no CSS-component order-one explanation')
        if not any_flags and all(len(v) == 1 for v in records.values()):
            if any(not self.unique['Z' if b == 'X' else 'X'][next(iter(records[b]))] for b in ('X', 'Z')):
                return replace(result, accepted=False, discard_reason='ambiguous stable CSS input syndrome')
        delta = self.space.reduce((result.frame_x ^ frame_x) | ((result.frame_z ^ frame_z) << self.code.n))
        correction = 0
        for component, current in zip(('X', 'Z'), self.space.components(delta)):
            choices = self.safe_component_frames(states[component], component)
            if not choices or (current not in choices and not self.variant.endswith('retarget')):
                return replace(result, accepted=False, discard_reason='no common safe CSS-component frame')
            correction ^= current if current in choices else min(choices)
        frame = incoming ^ correction
        mask = (1 << self.code.n) - 1
        return replace(result, frame_x=frame & mask, frame_z=frame >> self.code.n)


class CSSRecoverCandidate(AdaptiveCSSCandidate):
    """Minimum-order decoding plus uniquely safe incoming-one/fault-one rescue.

    Standard order-one records retain the parent's frame. Otherwise, a rescue
    must have a common logical-safe frame for *all* full-Pauli histories with
    incoming CSS component weights <=1 and at most one internal location fault.
    Unflagged, unchanged syndromes requiring CSS weight two always reject.
    """
    def __init__(self, parent, variant='css-recover'):
        super().__init__(parent, 'css')
        self.variant = variant
        self.full_initial = frozenset((int(bool(x or z)), 0, x ^ z)
            for x in self.space.component_errors('X', 1)
            for z in self.space.component_errors('Z', 1))
        self.full_responses = {}
        for name, g in self.gadgets.items():
            groups = defaultdict(set)
            for effect in g.single_fault_effects:
                f, s, data = g.unpack(effect)
                groups[f, s].add(data)
            self.full_responses[name] = groups

    @lru_cache(maxsize=32768)
    def full_step(self, state, name, flags, syndrome):
        component = 'X' if self.gadgets[name].basis == 'Z' else 'Z'
        following = set()
        for r, s, data in state:
            expected = self.space.component_syndrome(data, component)
            if flags == 0 and syndrome == expected:
                following.add((r, s, data))
            if s == 0:
                for delta in self.full_responses[name].get((flags, syndrome ^ expected), ()):
                    following.add((r, 1, data ^ delta))
        return frozenset(following)

    @lru_cache(maxsize=32768)
    def rescue_frame(self, record, current):
        strict, state = self.initial, self.full_initial
        records = {'X': set(), 'Z': set()}
        flagged = False
        for name, flags, syndrome in record:
            measured = 'X' if self.gadgets[name].basis == 'Z' else 'Z'
            records[measured].add(syndrome)
            flagged |= bool(flags)
            strict = self._step(strict, name, flags, syndrome) if strict else strict
        if not flagged and all(len(v) == 1 for v in records.values()):
            if any(not self.unique['Z' if b == 'X' else 'X'][next(iter(records[b]))] for b in ('X', 'Z')):
                return None
        if strict:
            return current
        if self.variant.endswith('no-d1') and any(self.gadgets[name].distance == 1 for name, _, _ in record):
            return None
        for name, flags, syndrome in record:
            state = self.full_step(state, name, flags, syndrome)
            if not state:
                return None
        return self.safe_full_frame(state, current)

    @lru_cache(maxsize=32768)
    def safe_full_frame(self, state, current):
        frame = 0
        for component, preferred in zip(('X', 'Z'), self.space.components(current)):
            choices = None
            for _, s, data in state:
                error = self.space.components(data)[0 if component == 'X' else 1]
                allowed = ({error} if self.variant.endswith(('exact', 'safe')) else
                           {error ^ residual for residual in self.space.component_errors(component, s)})
                choices = allowed if choices is None else choices & allowed
                if not choices:
                    return None
            frame ^= preferred if preferred in choices else min(choices)
        if self.variant.endswith('safe') and self.space.pauli_syndrome(frame ^ current):
            return None
        return frame

    def run(self, simulator, *, frame_x=0, frame_z=0):
        result = self.parent.run(simulator, frame_x=frame_x, frame_z=frame_z)
        if not result.accepted:
            return result
        incoming = frame_x | (frame_z << self.code.n)
        record = tuple((name, flags, syndrome ^ self.space.component_syndrome(incoming, 'X' if basis == 'Z' else 'Z'))
                       for name, basis, _, flags, syndrome in result.measurements)
        current = self.space.reduce((result.frame_x ^ frame_x) | ((result.frame_z ^ frame_z) << self.code.n))
        correction = self.rescue_frame(record, current)
        if correction is None:
            return replace(result, accepted=False, discard_reason='no unique safe CSS rescue frame')
        frame = incoming ^ correction
        mask = (1 << self.code.n) - 1
        return replace(result, frame_x=frame & mask, frame_z=frame >> self.code.n)


class CompactCSSRescueEvidence:
    """Exact uniqueness evidence with ambiguous logical sectors compressed.

    Once a fault is spent, errors with equal syndrome give identical future
    ideal observations. Two distinct stabilizer classes in such a sector can
    never share a residual-CSS-weight<=1 correction for this distance-four code.
    Retain that sector as an ambiguity marker, rather than all logical Paulis.
    """
    def __init__(self, model):
        self.model, self.space = model, model.space
        self.initial = (frozenset(e for _, _, e in model.full_initial), frozenset(), frozenset())

    @lru_cache(maxsize=262144)
    def signature(self, error):
        return self.space.pauli_syndrome(error)

    @lru_cache(maxsize=32768)
    def step(self, state, name, flags, syndrome):
        is_z = self.model.gadgets[name].basis == 'Z'
        def read(signature):
            return signature & self.model.decoder.x_mask if is_z else signature >> self.model.decoder.x_width
        clean, faulted, ambiguous = state
        following = set()
        seen = {}
        bad = {s for s in ambiguous if flags == 0 and read(s) == syndrome}

        def add(error):
            signature = self.signature(error)
            if signature in bad:
                return
            if signature in seen and seen[signature] != error:
                del seen[signature]
                bad.add(signature)
            else:
                seen[signature] = error

        if flags == 0:
            for error in faulted:
                if read(self.signature(error)) == syndrome:
                    add(error)
        responses = self.model.full_responses[name]
        for error in clean:
            expected = read(self.signature(error))
            if flags == 0 and expected == syndrome:
                following.add(error)
            for delta in responses.get((flags, syndrome ^ expected), ()):
                add(error ^ delta)
        return frozenset(following), frozenset(seen.values()), frozenset(bad)

    @lru_cache(maxsize=32768)
    def safe_frame(self, state, preferred):
        clean, faulted, ambiguous = state
        if ambiguous or not (clean or faulted):
            return None
        if self.model.variant.endswith(('exact', 'safe')):
            values = clean | faulted
            if len(values) != 1:
                return None
            result = next(iter(values))
            if self.model.variant.endswith('safe') and self.signature(result ^ preferred):
                return None
            return result
        frame = 0
        for component, current in zip(('X', 'Z'), self.space.components(preferred)):
            choices = None
            for s, errors in ((0, clean), (1, faulted)):
                for data in errors:
                    error = self.space.components(data)[0 if component == 'X' else 1]
                    allowed = {error ^ residual for residual in self.space.component_errors(component, s)}
                    choices = allowed if choices is None else choices & allowed
                    if not choices:
                        return None
            frame ^= current if current in choices else min(choices)
        return frame


def compile_css_candidate(parent, variant='css', *, verify=True, progress=None):
    """Compile the exact sampling rule into a normal, standalone policy DAG.

    The full adaptive CSS mode counts incoming errors and internal locations,
    rather than the weight of their propagated correction. It therefore equals
    the order-one history policy. The stricter zero-flag experiment additionally
    bounds its direct flag-free CSS readings and may postselect single faults.

    For the other exploratory variants the independent contract
    requires clean acceptance and logical preservation of accepted histories
    through incoming CSS order plus circuit fault order one. It does not claim
    unique decoding of arbitrary two-fault circuit histories.
    """
    if variant not in (*VARIANTS[:4], 'css-no-flags'):
        raise ValueError(variant)
    if variant == 'css':
        return postselect_order_one(parent, verify=verify, progress=progress)
    if variant.endswith('reversed'):
        parent = reverse_css_roles(parent)
    old = parent.policy
    evidence = OrderOneEvidence(parent)
    unique = unique_css_syndromes(parent.code)
    nodes = [PolicyNode('reject')]
    interned = {('reject',): 0}

    def intern(node):
        key = (node.kind, node.action, node.frame, node.next, tuple(sorted(node.children.items())))
        if key not in interned:
            interned[key] = len(nodes)
            nodes.append(node)
        return interned[key]

    accept = intern(PolicyNode('accept'))
    checks = accept
    if variant.endswith('check'):
        for basis in ('X', 'Z'):  # Construction runs backwards: execute Z then X.
            component = 'X' if basis == 'Z' else 'Z'
            children = {(0, s): intern(PolicyNode('frame', frame=old.decoder.physical[component][s], next=checks))
                        for s, keep in enumerate(unique[basis]) if keep}
            checks = intern(PolicyNode('extract', action=basis + '_d2', children=children))

    @lru_cache(maxsize=None)
    def visit(index, signature, flagged, state, sx, sz, stable):
        node = old.nodes[index]
        if node.kind == 'reject' or (variant != 'css' and flagged and not state):
            return 0
        if node.kind == 'accept':
            if not flagged and stable and sx >= 0 and sz >= 0:
                if not unique['Z'][sx] or not unique['X'][sz]:
                    return 0
            correction = signature ^ evidence.space.pauli_syndrome(node.frame)
            if not (unique['Z'][correction & evidence.mask] and unique['X'][correction >> evidence.width]):
                return 0
            return intern(PolicyNode('frame', frame=node.frame, next=checks))
        if node.kind == 'frame':
            following = evidence.translate(state, node.frame) if variant != 'css' else state
            child = visit(node.next, signature ^ evidence.space.pauli_syndrome(node.frame), flagged, following,
                          sx, sz, stable)
            return intern(PolicyNode('frame', frame=node.frame, next=child)) if child else 0
        groups = evidence.transition(state, node.action) if variant != 'css' and state else {}
        children = {}
        for outcome, child in node.children.items():
            # f=0 is an observable guard, not an oracle for the actual number
            # of faulty circuit locations. Reject before any follow-up gadget.
            if variant == 'css-no-flags' and outcome[0]:
                continue
            following = groups.get(outcome, frozenset()) if variant != 'css' else state
            next_flagged = flagged or bool(outcome[0])
            ax, az, unchanged = sx, sz, stable
            if next_flagged:
                ax, az, unchanged = -1, -1, False
            else:
                component = 'X' if old.actions[node.action].gadget.basis == 'Z' else 'Z'
                offset = signature & evidence.mask if component == 'X' else signature >> evidence.width
                raw = outcome[1] ^ offset
                if component == 'X':
                    unchanged &= sx < 0 or sx == raw
                    ax = raw
                else:
                    unchanged &= sz < 0 or sz == raw
                    az = raw
            target = visit(child, signature, next_flagged, following, ax, az, unchanged)
            if target:
                children[outcome] = target
        return intern(PolicyNode('extract', action=node.action, children=children))

    root = visit(old.root, 0, False, evidence.initial if variant != 'css' else frozenset(), -1, -1, True)
    contract = FTContract(0, 1, fault_order=2, accept_clean_through=0,
                          input_scope=old.contract.input_scope,
                          accepted_filter_order=old.contract.accepted_filter_order)
    decoder = FixedCSSDecoder(evidence.space, contract)
    report = AdaptiveVerificationReport(contract, decoder.fingerprint)
    policy = AdaptivePolicy(old.actions.copy(), contract, decoder, tuple(nodes), root, report,
                            peak_qubit_limit=old.peak_qubit_limit)
    compact_policy(policy)
    derived = AdaptiveSEProtocol(parent.code, policy)
    if progress:
        progress(f'{variant}: compiled {len(policy.nodes):,} nodes; {visit.cache_info()}')
    if verify:
        derived.verify()
        if progress:
            progress(f'{variant}: independent accepted-order-one verification passed')
    return derived


def compile_css_recovery(parent, variant='css-recover-exact', *, verify=True, progress=None):
    """Compile strict decoding plus an observation-conditioned CSS rescue."""
    if not variant.startswith('css-recover') or variant not in VARIANTS:
        raise ValueError(variant)
    model = CSSRecoverCandidate(parent, variant)
    compact = CompactCSSRescueEvidence(model)
    old, space = parent.policy, model.space
    nodes, interned = [PolicyNode('reject')], {}
    calls = 0

    def intern(node):
        key = node.kind, node.action, node.frame, node.next, tuple(sorted(node.children.items()))
        if key not in interned:
            interned[key] = len(nodes)
            nodes.append(node)
        return interned[key]

    @lru_cache(maxsize=None)
    def visit(index, frame, strict, full, prefix, sx, sz, stable, flagged, used_d1):
        nonlocal calls
        calls += 1
        if progress and calls % 10000 == 0:
            progress(f'{variant}: {calls:,} contexts, {len(nodes):,} interned nodes')
        node = old.nodes[index]
        if node.kind == 'reject':
            return 0
        if node.kind == 'accept':
            if not flagged and stable and sx >= 0 and sz >= 0:
                if not model.unique['Z'][sx] or not model.unique['X'][sz]:
                    return 0
            if strict:
                return intern(node)
            if variant.endswith('no-d1') and used_d1:
                return 0
            if full is None or not any(full):
                return 0
            preferred = space.reduce(frame ^ node.frame)
            correction = compact.safe_frame(full, preferred)
            return 0 if correction is None else intern(PolicyNode('accept', frame=correction ^ frame))
        if node.kind == 'frame':
            following = visit(node.next, frame ^ space.reduce(node.frame), strict, full,
                              prefix, sx, sz, stable, flagged, used_d1)
            return intern(replace(node, next=following)) if following else 0
        g = old.actions[node.action].gadget
        component = 'X' if g.basis == 'Z' else 'Z'
        offset = space.component_syndrome(frame, component)
        children = {}
        for outcome, child in node.children.items():
            f, observed = outcome
            raw = observed ^ offset
            next_flagged = flagged or bool(f)
            ax, az, unchanged = sx, sz, stable
            if next_flagged:
                ax, az, unchanged = -1, -1, False
            elif component == 'X':
                unchanged &= sx < 0 or sx == raw
                ax = raw
            else:
                unchanged &= sz < 0 or sz == raw
                az = raw
            next_strict = model._step(strict, node.action, f, raw) if strict else strict
            next_prefix = prefix + ((node.action, f, raw),) if full is None else ()
            next_full = full
            if not next_strict:
                if full is None:
                    next_full = compact.initial
                    for name, flags, syndrome in next_prefix:
                        next_full = compact.step(next_full, name, flags, syndrome)
                        if not any(next_full):
                            break
                else:
                    next_full = compact.step(full, node.action, f, raw)
                if not any(next_full):
                    continue
                next_prefix = ()
            target = visit(child, frame, next_strict, next_full, next_prefix,
                           ax, az, unchanged, next_flagged, used_d1 or g.distance == 1)
            if target:
                children[outcome] = target
        return intern(replace(node, children=children))

    root = visit(old.root, 0, model.initial, None, (), -1, -1, True, False, False)
    contract = (FTContract(1, 1, fault_order=2, accept_clean_through=1,
                           input_scope=old.contract.input_scope,
                           accepted_filter_order=old.contract.accepted_filter_order)
                if variant.endswith('safe') else FTContract(0, 1, fault_order=2, accept_clean_through=0,
                                                           input_scope=old.contract.input_scope,
                                                           accepted_filter_order=old.contract.accepted_filter_order))
    decoder = FixedCSSDecoder(space, contract)
    policy = AdaptivePolicy(old.actions.copy(), contract, decoder, tuple(nodes), root,
                            AdaptiveVerificationReport(contract, decoder.fingerprint),
                            peak_qubit_limit=old.peak_qubit_limit)
    compact_policy(policy)
    derived = AdaptiveSEProtocol(parent.code, policy)
    if progress:
        progress(f'{variant}: compiled {len(policy.nodes):,} nodes from {calls:,} contexts')
    if verify:
        derived.verify()
        if progress:
            progress(f'{variant}: independent accepted-order-one verification passed')
    return derived

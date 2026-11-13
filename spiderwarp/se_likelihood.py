"""Noise-weighted postselection conditioned on a complete adaptive-cycle record.

The likelihood expansion retains zero, one and two *nonzero response channels*.
It is exact within that expansion, including exclusive Pauli alternatives,
invisible faults, physical-location multiplicity and stabilizer degeneracy.
The omitted all-orders probability is reported, not silently renormalized away.
An estimated risk threshold is consequently not an all-orders LER guarantee.

Compilation freezes a finite set of additional accepted records into an ordinary
policy DAG. Unseen records retain the exact order-one guard. No probabilities,
fault counter, state-preparation label or final logical readout enter execution.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import itertools
import math
from pathlib import Path

from spiderwarp._se_adaptive import (
    AdaptivePolicy, AdaptiveVerificationReport, FixedCSSDecoder, PolicyNode,
    compact_policy,
)
from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol
from spiderwarp.noise_model import CustomErrorModel, default_noise_model
from spiderwarp.se_postselection import OrderOneEvidence
from spiderwarp.se_protocol_simulation import _noisy_gadget


DEFAULT_RISK_THRESHOLDS = (0.01, 0.05)


def _probability(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return value


def noise_response_channels(gadget, noise, *, include_initial_idles=True):
    """Propagate the sampler's actual stochastic channels backwards.

    Each returned mapping is one independent channel's exclusive nonidentity
    alternatives, with probabilities summed when physical Paulis have identical
    responses. A zero response is folded into that channel's identity branch.
    Unlike SEGadget.single_fault_effects, these include scheduled data/ancilla
    memory, probabilities and physical-location multiplicity.
    """
    for name, value in vars(noise).items():
        _probability(value, name)
    circuit = _noisy_gadget(gadget.circuit, noise,
        persistent_data=gadget.space.n if include_initial_idles else 0)
    n, shift = gadget.space.n, gadget.num_detectors
    xs = [gadget.space.reduce(1 << q) << shift if q < n else 0
          for q in range(circuit.num_qubits)]
    zs = [gadget.space.reduce(1 << (q + n)) << shift if q < n else 0
          for q in range(circuit.num_qubits)]
    columns = [0] * circuit.num_measurements
    for bit, parity in enumerate(gadget.readout.flags + gadget.readout.syndromes):
        for index in parity:
            columns[index] ^= 1 << bit
    record, channels = len(columns), []

    def channel(alternatives):
        grouped = defaultdict(float)
        for effect, probability in alternatives:
            if effect and probability:
                grouped[effect] += probability
        if grouped:
            channels.append(dict(grouped))

    for op in reversed(list(circuit.flattened())):
        name, args = op.name, op.gate_args_copy()
        qs = [t.value for t in op.targets_copy()]
        if name == 'DEPOLARIZE1':
            for q in qs:
                channel((effect, args[0] / 3) for effect in (xs[q], zs[q], xs[q] ^ zs[q]))
        elif name == 'DEPOLARIZE2':
            for a, b in zip(qs[::2], qs[1::2]):
                first, second = (0, xs[a], zs[a], xs[a] ^ zs[a]), (0, xs[b], zs[b], xs[b] ^ zs[b])
                channel((first[i] ^ second[j], args[0] / 15)
                        for i, j in itertools.product(range(4), repeat=2) if i or j)
        elif name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR'}:
            for q in qs:
                effect = xs[q] if name == 'X_ERROR' else zs[q] if name == 'Z_ERROR' else xs[q] ^ zs[q]
                channel(((effect, args[0]),))
        elif name in {'M', 'MX'}:
            for q in reversed(qs):
                record -= 1
                channel(((columns[record], args[0] if args else 0.0),))
                if name == 'M':
                    xs[q] ^= columns[record]
                else:
                    zs[q] ^= columns[record]
        elif name in {'R', 'RX'}:
            for q in qs:
                xs[q] = zs[q] = 0
        elif name == 'H':
            for q in qs:
                xs[q], zs[q] = zs[q], xs[q]
        elif name in {'S', 'S_DAG'}:
            for q in qs:
                xs[q] ^= zs[q]
        elif name == 'CX':
            for a, b in reversed(list(zip(qs[::2], qs[1::2]))):
                xs[a] ^= xs[b]
                zs[b] ^= zs[a]
        elif name == 'CZ':
            for a, b in reversed(list(zip(qs[::2], qs[1::2]))):
                xs[a] ^= zs[b]
                xs[b] ^= zs[a]
        elif name in {'X', 'Y', 'Z', 'I', 'TICK', 'QUBIT_COORDS', 'OBSERVABLE_INCLUDE'}:
            pass  # Deterministic Paulis do not change relative error responses.
        else:
            raise ValueError(f"Unsupported likelihood noise/circuit instruction: {name}")
    if record:
        raise ValueError('Likelihood readout does not match the circuit measurements')
    # Lifting local responses through later extractions relies on this identity.
    for bit, effect in enumerate(xs[:n] + zs[:n]):
        flags, syndrome, data = gadget.unpack(effect)
        error = 1 << bit
        component = 'X' if gadget.basis == 'Z' else 'Z'
        if (flags or syndrome != gadget.space.component_syndrome(error, component)
                or data != gadget.space.reduce(error)):
            raise ValueError('Likelihood inference requires ideal Pauli-preserving extractions')
    return tuple(channels)


@dataclass(frozen=True)
class LogicalRisk:
    """Conditional joint X/Z block risk under the declared two-channel model."""
    risk: float | None
    evidence_probability: float
    omitted_probability: float
    risk_upper_bound: float
    logical_classes: tuple[tuple[int, float], ...]


class _PathLikelihood:
    """Meet in the middle on observation signatures, excluding same-site pairs."""

    def __init__(self, channels, observation_bits):
        self.bits = observation_bits
        self.mask = (1 << observation_bits) - 1
        self.singles = defaultdict(lambda: defaultdict(float))
        self.exclusive = defaultdict(lambda: defaultdict(float))
        locations = defaultdict(set)
        self.channel_odds = []
        counts = [1.0, 0.0, 0.0]
        for channel_id, channel in enumerate(channels):
            # Zero effects are already folded into the identity probability.
            active = math.fsum(channel.values())
            identity = 1 - active
            if identity <= 0:
                raise ValueError('Likelihood expansion requires positive identity probability at every channel')
            counts = [counts[0] * identity,
                      counts[1] * identity + counts[0] * active,
                      counts[2] * identity + counts[1] * active]
            alternatives = [(effect, p / identity) for effect, p in channel.items()]
            self.channel_odds.append(tuple(alternatives))
            for effect, odds in alternatives:
                self.singles[effect & self.mask][effect >> self.bits] += odds
                locations[effect].add(channel_id)
            # Half the ordered self-convolution is subtracted below. Alternatives
            # at one physical channel are exclusive, even if their XOR is useful.
            for i, (first, a) in enumerate(alternatives):
                for j in range(i, len(alternatives)):
                    second, b = alternatives[j]
                    effect = first ^ second
                    self.exclusive[effect & self.mask][effect >> self.bits] += a * b * (0.5 if i == j else 1)
        self.p_zero = counts[0]
        self.omitted = max(0.0, 1 - math.fsum(counts))
        # Plain dicts prevent unsuccessful observation lookups growing the model.
        self.singles = {o: dict(values) for o, values in self.singles.items()}
        self.exclusive = {o: dict(values) for o, values in self.exclusive.items()}
        self.locations = {effect: frozenset(sites) for effect, sites in locations.items()}

    def joint_weights(self, keep=None):
        """All final Pauli weights for selected observations through two events.

        Aggregate equal responses before convolution, retaining exact channel
        supports so exclusive alternatives cannot create spurious tiny positive
        probabilities through floating-point cancellation. Values are odds;
        multiply by p_zero for their unconditional expansion probabilities.
        """
        keep = (lambda observation: True) if keep is None else keep
        groups = defaultdict(lambda: defaultdict(float))
        if keep(0):
            groups[0][0] = 1.0
        items = [(observation, error, odds, self.locations[observation | (error << self.bits)])
                 for observation, values in self.singles.items() for error, odds in values.items()]
        for observation, error, odds, _ in items:
            if keep(observation):
                groups[observation][error] += odds

        def can_pair(first, second):
            return len(first) > 1 or len(second) > 1 or first != second

        for i, (first, a, wa, sites_a) in enumerate(items):
            for j in range(i, len(items)):
                second, b, wb, sites_b = items[j]
                observation = first ^ second
                if keep(observation) and can_pair(sites_a, sites_b):
                    groups[observation][a ^ b] += wa * wb * (0.5 if i == j else 1)
        for alternatives in self.channel_odds:
            for i, (first, a) in enumerate(alternatives):
                for j in range(i, len(alternatives)):
                    second, b = alternatives[j]
                    observation = (first ^ second) & self.mask
                    if keep(observation) and can_pair(self.locations[first], self.locations[second]):
                        groups[observation][(first ^ second) >> self.bits] -= a * b * (0.5 if i == j else 1)
        result = {}
        for observation, weights in groups.items():
            if any(weight < -1e-12 for weight in weights.values()):
                raise ArithmeticError('Negative joint probability after exclusive-channel subtraction')
            positive = {error: weight for error, weight in weights.items() if weight > 0}
            if positive:
                result[observation] = positive
        return result

    def posterior(self, observation, correction, decoder):
        weights = defaultdict(float)
        if observation == 0:
            weights[0] = 1.0
        for error, weight in self.singles.get(observation, {}).items():
            weights[error] += weight
        for first, values in self.singles.items():
            second = observation ^ first
            if first > second or second not in self.singles:
                continue
            other = self.singles[second]
            if first != second:
                for a, wa in values.items():
                    for b, wb in other.items():
                        weights[a ^ b] += wa * wb
            else:
                items = list(values.items())
                for i, (a, wa) in enumerate(items):
                    for j in range(i, len(items)):
                        b, wb = items[j]
                        weights[a ^ b] += wa * wb * (0.5 if i == j else 1)
        for error, weight in self.exclusive.get(observation, {}).items():
            weights[error] -= weight
        classes = defaultdict(float)
        for error, weight in weights.items():
            if weight < -1e-10 * max(1.0, abs(weight)):
                raise ArithmeticError('Negative likelihood after exclusive-channel subtraction')
            if weight > 0:
                classes[decoder.logical(error ^ correction)] += weight
        total = math.fsum(classes.values())
        evidence = total * self.p_zero
        if not total:
            return LogicalRisk(None, 0.0, self.omitted, 1.0, ())
        bad = math.fsum(weight for logical, weight in classes.items() if logical)
        risk = min(1.0, bad / total)
        # All omitted mass could match this observation and be logically wrong.
        denominator = evidence + self.omitted
        upper = min(1.0, (bad * self.p_zero + self.omitted) / denominator)
        return LogicalRisk(risk, evidence, self.omitted, upper,
                           tuple(sorted((logical, weight / total) for logical, weight in classes.items())))


@dataclass(frozen=True)
class RecordDecision:
    accepted_by_parent: bool
    protected: bool
    correction: int
    edges: tuple[tuple[str, int, int], ...]


class FullHistoryLikelihood:
    """Offline likelihoods for complete, input-frame-relative raw records.

    ``record`` is a sequence of (action name, raw flags, raw syndrome). For a
    nonzero incoming frame use ``normalize_record`` first. The incoming-error
    prior is independent depolarization on each data qubit, defaulting to p_2q;
    it is a modeling prior, not an additional physical operation. It is reset
    each cycle; this does not implement a belief propagated between cycles.
    """

    def __init__(self, parent, *, noise_model=None, input_error_probability=None,
                 include_initial_idles=True):
        self.parent = parent
        noise = default_noise_model() if noise_model is None else noise_model
        self.noise = CustomErrorModel(**vars(noise))
        for name, value in vars(self.noise).items():
            _probability(value, name)
        self.input_error_probability = _probability(
            noise.p_2q if input_error_probability is None else input_error_probability,
            'input_error_probability')
        self.include_initial_idles = include_initial_idles
        self.evidence = OrderOneEvidence(parent)
        self.decoder, self.space = parent.policy.decoder, parent.policy.decoder.space
        self.channels = {name: noise_response_channels(action.gadget, self.noise,
                            include_initial_idles=include_initial_idles)
                         for name, action in parent.policy.actions.items()}
        # Per-instance bounded caches do not retain other models/noise profiles.
        self.path_model = lru_cache(maxsize=128)(self._path_model)
        self._score_cached = lru_cache(maxsize=32768)(self._score)

    def score(self, record):
        """Accept tuples or JSON-style lists, caching by their complete contents."""
        return self._score_cached(tuple(tuple(step) for step in record))

    def normalize_record(self, measurements, *, frame_x=0, frame_z=0):
        n = self.parent.code.n
        if any(type(v) is not int or v < 0 or v >> n for v in (frame_x, frame_z)):
            raise ValueError('Input Pauli-frame masks must fit the data block')
        frame = frame_x | (frame_z << n)
        return tuple((name, flags, syndrome ^ self.space.component_syndrome(
            frame, 'X' if basis == 'Z' else 'Z'))
            for name, basis, _, flags, syndrome in measurements)

    def replay(self, record):
        """Validate the whole path and retain the exact mandatory-acceptance guard."""
        policy = self.parent.policy
        index, position, frame = policy.root, 0, 0
        state, edges = self.evidence.initial, []
        while True:
            node = policy.nodes[index]
            if node.kind in {'accept', 'reject'}:
                if position != len(record):
                    raise ValueError('Record continues beyond the parent terminal node')
                return RecordDecision(node.kind == 'accept', bool(state), frame ^ node.frame, tuple(edges))
            if node.kind == 'frame':
                frame ^= node.frame
                state = self.evidence.translate(state, node.frame)
                index = node.next
                continue
            if position == len(record):
                raise ValueError('Likelihood requires a complete parent record')
            name, flags, syndrome = record[position]
            if name != node.action:
                raise ValueError('Record action does not match the adaptive path')
            gadget = policy.actions[name].gadget
            if (type(flags) is not int or type(syndrome) is not int or flags < 0 or syndrome < 0
                    or flags >> gadget.num_flags or syndrome >> len(gadget.readout.syndromes)):
                raise ValueError('Record flags/syndrome exceed their readout widths')
            relative = syndrome ^ self.space.component_syndrome(frame, 'X' if gadget.basis == 'Z' else 'Z')
            outcome = flags, relative
            edges.append((name, *outcome))
            state = self.evidence.transition(state, name).get(outcome, frozenset()) if state else state
            index = node.children.get(outcome, 0)
            position += 1

    def _path_model(self, names):
        gadgets = [self.parent.policy.actions[name].gadget for name in names]
        offsets, width = [], 0
        for gadget in gadgets:
            offsets.append(width)
            width += gadget.num_detectors

        def lift(error, start, observation=0):
            for j in range(start, len(gadgets)):
                g = gadgets[j]
                syndrome = self.space.component_syndrome(error, 'X' if g.basis == 'Z' else 'Z')
                observation ^= syndrome << (offsets[j] + g.num_flags)
            return observation | (error << width)

        channels = []
        p = self.input_error_probability
        if p:
            for q in range(self.parent.code.n):
                grouped = defaultdict(float)
                x, z = self.space.reduce(1 << q), self.space.reduce(1 << (q + self.parent.code.n))
                for error in (x, z, x ^ z):
                    effect = lift(error, 0)
                    if effect:
                        grouped[effect] += p / 3
                if grouped:
                    channels.append(dict(grouped))
        for i, name in enumerate(names):
            g = gadgets[i]
            mask = (1 << g.num_detectors) - 1
            for channel in self.channels[name]:
                grouped = defaultdict(float)
                for effect, probability in channel.items():
                    lifted = lift(effect >> g.num_detectors, i + 1, (effect & mask) << offsets[i])
                    if lifted:
                        grouped[lifted] += probability
                if grouped:
                    channels.append(dict(grouped))
        return _PathLikelihood(channels, width)

    def _score(self, record):
        decision = self.replay(record)
        if not decision.accepted_by_parent:
            raise ValueError('Likelihood postselection cannot rescue parent-rejected records')
        names, observation, offset = [], 0, 0
        for name, flags, syndrome in record:
            g = self.parent.policy.actions[name].gadget
            names.append(name)
            observation |= (flags | (syndrome << g.num_flags)) << offset
            offset += g.num_detectors
        return self.path_model(tuple(names)).posterior(
            observation, self.space.reduce(decision.correction), self.decoder)

    def metadata(self):
        here = Path(__file__).parent
        sources = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
                   ('se_likelihood.py', 'noise_model.py', 'se_protocol_simulation.py',
                    '_se_circuit.py', '_se_adaptive.py', 'se_postselection.py')}
        return dict(method='full-history-likelihood', model_version=1, model_sources=sources,
                    expansion_nonzero_channels=2,
                    noise=vars(self.noise), input_error_probability=self.input_error_probability,
                    input_prior='independent depolarizing data qubits, reset each cycle',
                    include_initial_idles=self.include_initial_idles,
                    risk='any nontrivial X or Z logical Pauli after the parent frame and fixed ideal decoder',
                    threshold_scope='estimated risk for additional rescues; mandatory r+s<=1 always retained',
                    approximation='condition on at most two nonzero response channels; omitted mass is reported',
                    all_orders_ler_guarantee=False)


class LikelihoodCandidate:
    """Sampling reference; only measured records and the incoming frame are read.

    With ``rescues=None`` score every encountered record. With a frozen rescue
    set, match the compiled policy (including its conservative unknown-record
    fallback). Discovery collection and held-out evaluation must use different
    seeds, and the rescue set must be frozen before evaluation.
    """

    def __init__(self, model, *, risk_threshold=0.01, rescues=None):
        self.model, self.parent = model, model.parent
        self.risk_threshold = _probability(risk_threshold, 'risk_threshold')
        self.rescues = None if rescues is None else frozenset(
            tuple(tuple(step) for step in record) for record in rescues)
        self.code, self.policy = self.parent.code, self.parent.policy
        self.gadgets, self.required_qubits = self.parent.gadgets, self.parent.required_qubits

    def run(self, simulator, *, frame_x=0, frame_z=0):
        result = self.parent.run(simulator, frame_x=frame_x, frame_z=frame_z)
        if not result.accepted:
            return result
        record = self.model.normalize_record(result.measurements, frame_x=frame_x, frame_z=frame_z)
        if self.model.replay(record).protected:
            return result
        if self.rescues is not None:
            keep = record in self.rescues
        else:
            score = self.model.score(record)
            keep = score.risk is not None and score.risk <= self.risk_threshold
        return result if keep else replace(result, accepted=False, discard_reason='full-history likelihood risk')


def collect_likelihood_records(model, *, shots=10000, cycles=(1, 10), seed=2026111304,
                               progress=None):
    """Discover candidate records without accessing any ideal output labels."""
    from spiderwarp.se_protocol_simulation import ProtocolSampler
    if type(shots) is not int or shots <= 0 or type(seed) is not int or seed < 0:
        raise ValueError('Discovery shots must be positive and seed nonnegative integers')
    cycles = tuple(cycles)
    if not cycles or any(type(n) is not int or n <= 0 for n in cycles):
        raise ValueError('Discovery cycles must be positive integers')
    records = set()

    class Collector:
        code, policy = model.parent.code, model.parent.policy
        gadgets, required_qubits = model.parent.gadgets, model.parent.required_qubits

        def run(self, simulator, *, frame_x=0, frame_z=0):
            result = model.parent.run(simulator, frame_x=frame_x, frame_z=frame_z)
            if result.accepted:
                record = model.normalize_record(result.measurements, frame_x=frame_x, frame_z=frame_z)
                if not model.replay(record).protected:
                    records.add(record)
            return result

    tasks = []
    for n in cycles:
        for basis in ('0', '+'):
            task_seed = seed + len(tasks)
            sampler = ProtocolSampler(Collector(), noise_model=model.noise, prepared_state=basis,
                cycles=n, include_initial_idles=model.include_initial_idles, collect_paths=False)
            sampler.sample(shots, task_seed)
            tasks.append(dict(cycles=n, prepared_state=basis, shots=shots, seed=task_seed))
            if progress:
                progress(f'Discovery N={n}, basis={basis}: {len(records):,} distinct rescue candidates')
    return dict(records=sorted(records), discovery_tasks=tasks,
                selection_inputs='observations and incoming Pauli frame only; no failure labels',
                evaluation='freeze before independently seeded holdout sampling')


def postselect_likelihood(parent, records, *, risk_threshold=0.01, model=None,
                         verify=True, progress=None):
    """Freeze likelihood-selected rescues plus all order-one records into a DAG.

    ``records`` enumerates candidate complete histories (e.g. from discovery
    sampling). It contains observations only. Unknown histories fall back to
    strict history PS. Parent actions and terminal frames are never modified,
    so accepted arbitrary-input filters are inherited as well as rechecked.
    Returns (protocol, provenance), ready for save_bundle/export_guppy.
    """
    _probability(risk_threshold, 'risk_threshold')
    model = FullHistoryLikelihood(parent) if model is None else model
    if model.parent is not parent:
        raise ValueError('Likelihood model belongs to a different parent protocol')
    old, evidence = parent.policy, model.evidence
    trie = [dict()]
    terminal = set()
    scores = []
    unique = sorted({tuple(tuple(step) for step in record) for record in records})
    for record in unique:
        decision = model.replay(record)
        if not decision.accepted_by_parent or decision.protected:
            continue
        risk = model.score(record)
        keep = risk.risk is not None and risk.risk <= risk_threshold
        scores.append(dict(record=record, estimated_risk=risk.risk,
                           evidence_probability=risk.evidence_probability,
                           omitted_probability=risk.omitted_probability,
                           risk_upper_bound=risk.risk_upper_bound, rescued=keep))
        if keep:
            index = 0
            for edge in decision.edges:
                if edge not in trie[index]:
                    trie[index][edge] = len(trie)
                    trie.append({})
                index = trie[index][edge]
            terminal.add(index)
    nodes, interned = [PolicyNode('reject')], {('reject',): 0}

    def intern(key, node):
        if key not in interned:
            interned[key] = len(nodes)
            nodes.append(node)
        return interned[key]

    @lru_cache(maxsize=None)
    def visit(index, state, prefix):
        node = old.nodes[index]
        if node.kind == 'reject' or (not state and prefix < 0):
            return 0
        if node.kind == 'accept':
            if not state and prefix not in terminal:
                return 0
            return intern(('accept', node.frame), PolicyNode('accept', frame=node.frame))
        if node.kind == 'frame':
            child = visit(node.next, evidence.translate(state, node.frame), prefix)
            return intern(('frame', node.frame, child), PolicyNode('frame', frame=node.frame, next=child)) if child else 0
        groups = evidence.transition(state, node.action) if state else {}
        extra = {(f, s): child for (name, f, s), child in trie[prefix].items()
                 if name == node.action} if prefix >= 0 else {}
        children = {}
        for outcome in sorted(groups.keys() | extra.keys()):
            child = visit(node.children.get(outcome, 0), groups.get(outcome, frozenset()), extra.get(outcome, -1))
            if child:
                children[outcome] = child
        return intern(('extract', node.action, tuple(children.items())),
                      PolicyNode('extract', action=node.action, children=children)) if children else 0

    root = visit(old.root, evidence.initial, 0)
    contract = replace(old.contract, accept_clean_through=1)
    decoder = FixedCSSDecoder(evidence.space, contract)
    report = AdaptiveVerificationReport(contract, decoder.fingerprint)
    policy = AdaptivePolicy(old.actions.copy(), contract, decoder, tuple(nodes), root, report,
                            peak_qubit_limit=old.peak_qubit_limit)
    compact_policy(policy)
    derived = AdaptiveSEProtocol(parent.code, policy)
    if progress:
        progress(f'Likelihood PS epsilon={risk_threshold:g}: {len(terminal):,} rescues; {len(policy.nodes):,} nodes')
    if verify:
        derived.verify()
    provenance = dict(model.metadata(), risk_threshold=risk_threshold,
                      candidate_records=len(unique), rescued_records=len(terminal), scores=scores,
                      unknown_record_rule='exact order-one history guard',
                      quantum_circuits_unchanged=True, surviving_frames_unchanged=True,
                      required_acceptance='r+s<=1; clean-input s=2 acceptance relaxed')
    return derived, provenance

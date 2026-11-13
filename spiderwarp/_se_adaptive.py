"""Exact, postselected CSS Steane policies with terminal Pauli frames.

Only physical locations consume fault budget. X and Z constraints use the same
joint histories. Distances are circuit labels, never assumptions about a noisy
response. The ideal decoder is fixed before synthesis, including its tie breaks.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
import hashlib
import heapq
import time
from typing import Callable, Mapping

from spiderwarp._se_circuit import CircuitFault, PauliSpace, SEGadget, validate_gadget


# (minimum incoming CSS order, internal location count, current error).
# For EC A, incoming=-1 and error is a *syndrome*, with logical action forgotten.
# Otherwise error is modulo stabilizers, relative to the ideal-decoded input.
History = tuple[int, int, int]
Outcome = tuple[int, int]
Record = tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class CompatibleSet:
    """Joint logical histories plus bitsets of arbitrary-input syndrome sectors.

    A bitset retains the *joint* X/Z syndrome, rather than its independent
    projections. This makes the arbitrary-input domain compact without losing
    correlations when subsequent observations condition it.
    """

    logical: frozenset[History]
    filters: tuple[int, ...]
    sectors: tuple[int, ...] = field(compare=False, hash=False, repr=False)

    def __len__(self):
        return len(self.logical) + sum(mask.bit_count() for mask in self.filters)

    def __iter__(self):
        yield from self.logical
        for s, mask in enumerate(self.filters):
            while mask:
                low = mask & -mask
                yield -1, s, self.sectors[low.bit_length() - 1]
                mask ^= low

    def __le__(self, other):
        return self.logical <= other.logical and all(
            (a & b) == a for a, b in zip(self.filters, other.filters))


@dataclass(frozen=True)
class FTContract:
    """Explicit acceptance, logical-correctness, and residual-filter orders.

    The default domain retains incoming CSS order r and internal fault count s
    with r+s<=fault_order. Accepted histories through accepted_filter_order
    pass their s-filter; higher analyzed orders need not be correctable.
    input_scope='arbitrary' explicitly selects the legacy broader input domain.
    Corrections are noiseless classical frames. Mandatory acceptance is separate
    from logical protection of accepted histories, allowing ambiguous-record PS.
    """

    correctable: int
    logical_order: int
    max_syndrome_sectors: int = 65536
    fault_order: int | None = None
    accept_clean_through: int | None = None
    input_scope: str = "total-order"
    accepted_filter_order: int | None = None

    def __post_init__(self):
        if self.fault_order is None:
            object.__setattr__(self, "fault_order", self.logical_order)
        if any(type(value) is not int for value in
               (self.correctable, self.logical_order, self.fault_order, self.max_syndrome_sectors)):
            raise ValueError("Fault orders and syndrome-sector limit must be integers")
        if not 0 <= self.correctable <= self.logical_order <= self.fault_order <= 2:
            raise ValueError("Exact adaptive policies support 0 <= t <= logical <= fault order <= 2")
        if self.accept_clean_through is None:
            object.__setattr__(self, "accept_clean_through", self.correctable)
        if type(self.accept_clean_through) is not int:
            raise ValueError("Clean-input acceptance order must be an integer")
        if not self.correctable <= self.accept_clean_through <= self.fault_order:
            raise ValueError("Clean-input acceptance order must lie between t and the fault order")
        if self.max_syndrome_sectors < 1:
            raise ValueError("max_syndrome_sectors must be positive")
        if self.input_scope not in {"total-order", "arbitrary"}:
            raise ValueError("Input scope must be total-order or arbitrary")
        if self.accepted_filter_order is None:
            object.__setattr__(self, 'accepted_filter_order', self.fault_order)
        if not self.logical_order <= self.accepted_filter_order <= self.fault_order:
            raise ValueError("Accepted filter order must lie between logical and analyzed fault orders")

    @classmethod
    def for_code(cls, code):
        t, m = (code.d - 1) // 2, code.d // 2
        return cls(t, t, fault_order=m, accept_clean_through=m)

    @classmethod
    def correct_or_reject(cls, code):
        """Stronger logical target, which may admit a terminal-fault no-go proof."""
        return cls((code.d - 1) // 2, code.d // 2)

    @classmethod
    def css_gottesman(cls, code):
        """Logical guarantees through t, with the additional CSS filter through m."""
        t = (code.d - 1) // 2
        return cls(t, t, fault_order=code.d // 2)

    def remaining(self, history: History) -> int:
        r, s, _ = history
        return self.correctable - s if r < 0 else self.fault_order - r - s

    def must_accept(self, history: History) -> bool:
        r, s, _ = history
        return r >= 0 and (r + s <= self.correctable or (r == 0 and s <= self.accept_clean_through))


class FixedCSSDecoder:
    """Deterministic minimum-weight component decoders on attainable syndromes.

    Increasing qubit index breaks ties. Store both physical representatives and
    their stabilizer classes; the same tables are used for input and output.
    """

    def __init__(self, space: PauliSpace, contract: FTContract):
        self.space = space
        self.physical = {}
        self.classes = {}
        self.weights = {}
        for basis in ("X", "Z"):
            shift = 0 if basis == "X" else space.n
            columns = [space.component_syndrome(1 << (q + shift), basis)
                       for q in range(space.n)]
            table = {0: 0}
            pending = deque([0])
            while pending:
                syndrome = pending.popleft()
                error = table[syndrome]
                for q, column in enumerate(columns):
                    following = syndrome ^ column
                    if following not in table:
                        table[following] = error ^ (1 << (q + shift))
                        pending.append(following)
                        if len(table) > contract.max_syndrome_sectors:
                            raise NotImplementedError("Too many attainable CSS syndromes")
            self.physical[basis] = table
            self.classes[basis] = {s: space.reduce(p) for s, p in table.items()}
            self.weights[basis] = {s: p.bit_count() for s, p in table.items()}
            # A distance label must not silently grant an invalid decoder.
            for error in space.component_errors(basis, contract.correctable):
                syndrome = space.component_syndrome(error, basis)
                if self.classes[basis][syndrome] != error:
                    raise ValueError("Code does not correct the requested CSS radius")
        self.x_width = len(space._checks["X"])
        self.x_mask = (1 << self.x_width) - 1
        if len(self.physical["X"]) * len(self.physical["Z"]) > contract.max_syndrome_sectors:
            raise NotImplementedError("Too many arbitrary-input syndrome sectors")

    def correction(self, syndrome: int) -> int:
        return (self.classes["X"][syndrome & self.x_mask]
                ^ self.classes["Z"][syndrome >> self.x_width])

    def logical(self, error: int) -> int:
        return error ^ self.correction(self.space.pauli_syndrome(error))

    @cached_property
    def fingerprint(self):
        identity = self.space.n, self.space._checks, self.space.pivots, self.physical
        return hashlib.sha256(repr(identity).encode()).hexdigest()


@dataclass(frozen=True)
class ExtractionAction:
    name: str
    gadget: SEGadget = field(compare=False, repr=False)

    @cached_property
    def cnot_cost(self):
        # CZ = H CX H and XCX likewise needs one entangling gate. The basis
        # rotations remain explicit physical fault locations in the gadget.
        return sum(name in {"CX", "CZ", "XCX"} for name, _ in self.gadget.operations)

    @property
    def qubits(self):
        return self.gadget.circuit.num_qubits


@dataclass(frozen=True)
class PolicyNode:
    kind: str                         # accept, reject, extract, frame
    action: str | None = None
    frame: int = 0
    children: Mapping[Outcome, int] = field(default_factory=dict)
    next: int | None = None


@dataclass
class AdaptiveVerificationReport:
    contract: FTContract
    decoder_hash: str
    verified: bool = False
    checked_transitions: int = 0
    accepted_contexts: int = 0
    discarded_contexts: int = 0
    ec_a_input_syndromes: int = 0
    worst_case_cnots: int = 0
    max_extractions: int = 0
    peak_qubits: int = 0
    lower_bound_cnots: int = 0
    noise_model: str = (
        "All 15 nonidentity Paulis per two-qubit gate; X/Y/Z per one-qubit "
        "gate; reset and measurement flips; no implicit idles; terminal Pauli frames."
    )

    @property
    def correctable_faults(self):
        return self.contract.correctable

    @property
    def detectable_faults(self):
        return self.contract.fault_order


@dataclass
class AdaptivePolicy:
    actions: dict[str, ExtractionAction]
    contract: FTContract
    decoder: FixedCSSDecoder
    nodes: tuple[PolicyNode, ...]
    root: int
    report: AdaptiveVerificationReport
    optimal: bool = False
    search_states: int = 0
    peak_qubit_limit: int | None = None
    lower_bound_cnots: int = 0
    search_limit_reached: bool = False


def compact_policy(policy: AdaptivePolicy):
    """Merge identical behavior, treating absent and explicit rejection equally.

    Unknown outcomes already go to node zero. Removing explicit zero edges is
    therefore exact for *all* records, including records outside the fault bound.
    """
    nodes = [PolicyNode("reject")]
    interned = {("reject",): 0}
    mapping, active = {}, set()

    def visit(index):
        if index in mapping:
            return mapping[index]
        if index in active:
            raise ValueError("Cannot compact a cyclic policy")
        active.add(index)
        old = policy.nodes[index]
        if old.kind == "reject":
            key, node = ("reject",), PolicyNode("reject")
        elif old.kind == "accept":
            key, node = ("accept", old.frame), PolicyNode("accept", frame=old.frame)
        elif old.kind == "frame":
            following = visit(old.next)
            key = "frame", old.frame, following
            node = PolicyNode("frame", frame=old.frame, next=following)
        else:
            children = {outcome: visit(child) for outcome, child in old.children.items()}
            children = {outcome: child for outcome, child in children.items() if child}
            key = "extract", old.action, tuple(sorted(children.items()))
            node = PolicyNode("extract", action=old.action, children=children)
        if key not in interned:
            interned[key] = len(nodes)
            nodes.append(node)
        mapping[index] = interned[key]
        active.remove(index)
        return mapping[index]

    root = visit(policy.root)
    policy.nodes, policy.root = tuple(nodes), root


class AdaptiveVerificationError(ValueError):
    def __init__(self, reason: str, record: Record, history: History | None = None):
        self.reason, self.record, self.history = reason, record, history
        super().__init__(f"{reason}; record={record}; history={history}")


class PolicySearchError(ValueError):
    """No certified policy was found within the stated search budget."""


@dataclass(frozen=True)
class TerminalFaultObstruction:
    action: str
    faults: tuple[CircuitFault, ...]
    output_error: int
    decoded_logical: int


class FTContractInfeasibleError(ValueError):
    """Every possible final action admits an invisible, uncorrectable fault path."""

    def __init__(self, obstructions: tuple[TerminalFaultObstruction, ...]):
        self.obstructions = obstructions
        super().__init__(
            "The accepted-output logical guarantee is impossible for this library: "
            "every possible final extraction has an uncorrectable data error, "
            "within the promised fault order, introduced after those data qubits' "
            "last gates with zero flag and syndrome changes. Its record is also "
            "produced by a fault-free input that must accept. "
            "Use the code's correction radius for logical guarantees, or change "
            "the code/output-boundary fault model."
        )


def terminal_fault_obstructions(actions, decoder, order):
    """Construct a no-go certificate, without attempting a continuation search.

    Local data Paulis after each qubit's last operation commute with everything
    that remains. They cannot influence any classical outcome. A terminal
    correction on the fault-free record must be a stabilizer. Thus an error
    miscorrected by the fixed ideal decoder remains miscorrected on that record,
    and rejection would also reject the mandatory fault-free history. A witness
    for *every* available final action rules out all finite policies on the
    library, rather than just a particular schedule or extraction cap.
    """
    space = decoder.space
    bad = [(error, physical) for basis in ("X", "Z")
           for error, physical in space.component_errors(basis, order).items()
           if decoder.logical(error)]
    witnesses = []
    for name, action in actions.items():
        g = action.gadget
        last = {q: offset for offset, (_, qs) in enumerate(g.operations)
                for q in qs if q < space.n}
        _ = g.responses
        effects = {fault: effect for effect, faults in g._fault_witnesses.items()
                   for fault in faults}
        for error, physical in bad:
            sites = defaultdict(list)
            for letter, q in space.product(physical):
                if q not in last:
                    break
                sites[last[q]].append((letter, q))
            else:
                faults = tuple(CircuitFault(offset, tuple(product))
                               for offset, product in sorted(sites.items()))
                if len(faults) > order or any(f not in effects for f in faults):
                    continue
                response = 0
                for fault in faults:
                    response ^= effects[fault]
                if g.unpack(response) != (0, 0, error):
                    continue
                witnesses.append(TerminalFaultObstruction(
                    name, faults, physical, decoder.logical(error)))
                break
        else:
            return ()
    return tuple(witnesses)


class HistoryEngine:
    def __init__(self, space: PauliSpace, contract: FTContract, decoder=None):
        self.space, self.contract = space, contract
        self.decoder = decoder or FixedCSSDecoder(space, contract)
        self._syndromes = {}
        self._layers = {}
        # Choose linear coordinates: index(a XOR b) = index(a) XOR index(b).
        sectors, seen = [0], {0}
        for sx in self.decoder.classes["X"]:
            for sz in self.decoder.classes["Z"]:
                signature = sx | (sz << self.decoder.x_width)
                if signature not in seen:
                    following = [p ^ signature for p in sectors]
                    sectors.extend(following)
                    seen.update(following)
        self.sectors = tuple(sectors)
        self.indices = {p: i for i, p in enumerate(sectors)}
        self.selectors = {"X": defaultdict(int), "Z": defaultdict(int)}
        for i, p in enumerate(sectors):
            self.selectors["X"][p & self.decoder.x_mask] |= 1 << i
            self.selectors["Z"][p >> self.decoder.x_width] |= 1 << i
        self._swap_masks = []
        for bit in range((len(sectors) - 1).bit_length()):
            stride = 1 << bit
            block = (1 << stride) - 1
            mask = sum(block << start for start in range(0, len(sectors), 2 * stride))
            self._swap_masks.append((stride, mask))
        self.universal_filter = tuple(all(
            max(self.decoder.weights[basis].values()) <= s for basis in ("X", "Z"))
            for s in range(contract.fault_order + 1))
        self._filter_constraints = {}

    def syndrome(self, error):
        if error not in self._syndromes:
            self._syndromes[error] = self.space.pauli_syndrome(error)
        return self._syndromes[error]

    def remaining_max(self, histories):
        remaining = max((self.contract.remaining(h) for h in histories.logical), default=0)
        return max(remaining, max((max(0, self.contract.correctable - s)
                                   for s, mask in enumerate(histories.filters) if mask), default=0))

    def translate(self, histories, frame):
        """Change Pauli-frame coordinates without a physical operation."""
        if not frame:
            return histories
        frame = self.space.reduce(frame)
        if not frame:
            return histories
        signature = self.syndrome(frame)
        logical = frozenset((r, s, error ^ frame if r + s <= self.contract.logical_order
                             else self.decoder.correction(self.syndrome(error) ^ signature))
                            for r, s, error in histories.logical)
        filters = tuple(self.permute(mask, signature) if mask else 0 for mask in histories.filters)
        return CompatibleSet(logical, filters, self.sectors)

    def normalize(self, histories):
        """Remove a known offset so equivalent decision states can be shared."""
        anchored = [h for h in histories.logical if h[0] + h[1] <= self.contract.logical_order]
        if anchored:
            frame = min(anchored, key=lambda h: (h[1], h[0], h[2]))[2]
        else:
            frame = 0
            for s, mask in enumerate(histories.filters):
                if mask and not self.universal_filter[s]:
                    signature = self.sectors[(mask & -mask).bit_length() - 1]
                    frame = self.decoder.correction(signature)
                    break
        return self.translate(histories, frame), frame

    def initial(self) -> CompatibleSet:
        histories = set()
        decoder = self.decoder
        for sx, px in decoder.classes["X"].items():
            for sz, pz in decoder.classes["Z"].items():
                signature = sx | (sz << decoder.x_width)
                r = max(decoder.weights["X"][sx], decoder.weights["Z"][sz])
                maximum = (self.contract.fault_order if self.contract.input_scope == "total-order"
                           else self.contract.logical_order)
                if r <= maximum:
                    # If physical inputs differ by a logical Pauli, their
                    # ideal-decoded inputs differ by that same Pauli. Relative
                    # to that input, both start at this decoder representative.
                    histories.add((r, 0, px ^ pz))
        filters = (((1 << len(self.sectors)) - 1,) + (0,) * self.contract.fault_order
                   if self.contract.input_scope == "arbitrary"
                   else (0,) * (self.contract.fault_order + 1))
        return CompatibleSet(frozenset(histories), filters, self.sectors)

    def layers(self, action: ExtractionAction, order):
        key = action.name, order
        if key not in self._layers:
            g = action.gadget
            self._layers[key] = tuple(
                (*g.unpack(effect), self.syndrome(g.unpack(effect)[2]))
                for effect in g.fault_effects(order))
        return self._layers[key]

    @lru_cache(maxsize=8192)
    def permute(self, mask, signature):
        index = self.indices[signature]
        while index:
            low = index & -index
            stride, selector = self._swap_masks[low.bit_length() - 1]
            mask = ((mask & selector) << stride) | ((mask >> stride) & selector)
            index ^= low
        return mask

    def transition(self, histories, action: ExtractionAction, observed: Outcome | None = None):
        """Extend the *joint* response, including faults during this action."""
        logical = defaultdict(set)
        filters = defaultdict(lambda: [0] * (self.contract.fault_order + 1))
        decoder = self.decoder
        is_z = action.gadget.basis == "Z"
        for history in histories.logical:
            r, s, error = history
            signature = self.syndrome(error)
            readout = signature & decoder.x_mask if is_z else signature >> decoder.x_width
            for order in range(self.contract.remaining(history) + 1):
                spent = s + order
                needs_logical = r + spent <= self.contract.logical_order
                mandatory = self.contract.must_accept((r, spent, 0))
                bounded = self.contract.input_scope == "total-order"
                if not bounded and not needs_logical and not mandatory and (spent <= self.contract.correctable or self.universal_filter[spent]):
                    # The same syndrome constraint is already covered by the
                    # arbitrary-input EC A domain, or is the universal filter.
                    continue
                for flags, syndrome, data, data_syndrome in self.layers(action, order):
                    outcome = flags, readout ^ syndrome
                    if observed is not None and outcome != observed:
                        continue
                    if bounded or needs_logical or mandatory:
                        residual = error ^ data if needs_logical else decoder.correction(signature ^ data_syndrome)
                        logical[outcome].add((r, spent, residual))
                    else:
                        filters[outcome][spent] |= 1 << self.indices[signature ^ data_syndrome]
        component = "X" if is_z else "Z"
        for s, mask in enumerate(histories.filters):
            if not mask:
                continue
            partitions = [(readout, mask & selector)
                          for readout, selector in self.selectors[component].items() if mask & selector]
            for order in range(max(0, self.contract.correctable - s) + 1):
                if self.universal_filter[s + order]:
                    continue
                responses = {(f, sy, ds) for f, sy, _, ds in self.layers(action, order)}
                for flags, syndrome, data_syndrome in responses:
                    if observed is not None:
                        if flags != observed[0]:
                            continue
                        selector = self.selectors[component].get(observed[1] ^ syndrome, 0)
                        selected = [(observed[1] ^ syndrome, mask & selector)] if mask & selector else []
                    else:
                        selected = partitions
                    for readout, partition in selected:
                        filters[flags, readout ^ syndrome][s + order] |= self.permute(partition, data_syndrome)
        return {outcome: CompatibleSet(frozenset(logical[outcome]), tuple(filters[outcome]), self.sectors)
                for outcome in logical.keys() | filters.keys()}

    def safe_frame(self, histories) -> int | None:
        """Intersect decoder-trivial residual classes and CSS filter constraints."""
        components = []
        if not isinstance(histories, CompatibleSet):
            masks = [0] * (self.contract.fault_order + 1)
            logical = set()
            for r, s, p in histories:
                if r < 0:
                    masks[s] |= 1 << self.indices[p]
                else:
                    logical.add((r, s, p))
            histories = CompatibleSet(frozenset(logical), tuple(masks), self.sectors)
        for basis in ("X", "Z"):
            choices = None
            signatures = None
            shift = 0 if basis == "X" else self.space.n
            mask = ((1 << self.space.n) - 1) << shift
            for r, s, error in sorted(histories.logical, key=lambda h: h[1]):
                if r >= 0 and r + s > self.contract.accepted_filter_order:
                    continue
                allowed_syndromes = self.space.syndrome_errors(basis, s)
                if r < 0 or r + s > self.contract.logical_order:
                    full_syndrome = error if r < 0 else self.syndrome(error)
                    syndrome = (full_syndrome & self.decoder.x_mask if basis == "X"
                                else full_syndrome >> self.decoder.x_width)
                    allowed = {syndrome ^ sig for sig in allowed_syndromes}
                    signatures = allowed if signatures is None else signatures & allowed
                    if not signatures:
                        return None
                else:
                    component = error & mask
                    allowed = {component ^ self.decoder.classes[basis][sig]
                               for sig in allowed_syndromes}
                    choices = allowed if choices is None else choices & allowed
                    if not choices:
                        return None
            for s, mask_value in enumerate(histories.filters):
                if not mask_value or self.universal_filter[s]:
                    continue
                key = basis, s, mask_value
                if key not in self._filter_constraints:
                    allowed = None
                    for syndrome, selector in self.selectors[basis].items():
                        if mask_value & selector:
                            current = {syndrome ^ sig for sig in self.space.syndrome_errors(basis, s)}
                            allowed = current if allowed is None else allowed & current
                            if not allowed:
                                break
                    self._filter_constraints[key] = frozenset(allowed or ())
                    if len(self._filter_constraints) > 16384:
                        self._filter_constraints.pop(next(iter(self._filter_constraints)))
                allowed = self._filter_constraints[key]
                signatures = allowed if signatures is None else signatures & allowed
                if not signatures:
                    return None
            if choices is None:
                choices = ({0} if signatures is None else
                           {self.decoder.classes[basis][sig] for sig in signatures})
            elif signatures is not None:
                choices = {p for p in choices
                           if self.space.component_syndrome(p, basis) in signatures}
            if not choices:
                return None
            components.append(min(choices, key=lambda p: (p.bit_count(), p)))
        return components[0] ^ components[1]


class AdaptivePolicyBuilder:
    """Exact minimax search over a fixed action library and CNOT budget."""

    def __init__(self, space: PauliSpace, actions: Mapping[str, SEGadget], *,
                 contract: FTContract | None = None, peak_qubits: int | None = None,
                 max_search_states: int = 100000, progress: Callable[[str], None] | None = None):
        self.space = space
        if type(max_search_states) is not int or max_search_states <= 0:
            raise ValueError("max_search_states must be a positive integer")
        if peak_qubits is not None and (type(peak_qubits) is not int or peak_qubits < space.n):
            raise ValueError("peak_qubits must be an integer at least as large as the data block")
        self.contract = contract or FTContract.for_code(space.code)
        self.engine = HistoryEngine(space, self.contract)
        self.all_actions = {name: ExtractionAction(name, g) for name, g in actions.items()}
        if not self.all_actions or any(not isinstance(n, str) or not n for n in actions):
            raise ValueError("Supply named extraction actions")
        for action in self.all_actions.values():
            if action.gadget.space.code is not space.code:
                raise ValueError("All actions must use the same code object")
            validate_gadget(action.gadget)
            if action.cnot_cost <= 0:
                raise ValueError("Extraction actions must have positive CNOT cost")
        self.actions = {n: a for n, a in self.all_actions.items()
                        if peak_qubits is None or a.qubits <= peak_qubits}
        self.peak_qubits = peak_qubits
        self.max_search_states = max_search_states
        self.progress = progress or (lambda message: None)
        self._transitions = {}
        self._terminal = {}
        self._solutions = {}
        self._failures = {}
        self.nodes = [PolicyNode("reject")]
        self._accept_ids = {}
        self._node_ids = {}
        self._frame_ids = {}
        self.search_states = 0

    def transition(self, histories, name):
        key = histories, name
        if key not in self._transitions:
            self._transitions[key] = self.engine.transition(histories, self.actions[name])
            if len(self._transitions) > 2048:
                self._transitions.pop(next(iter(self._transitions)))
        return self._transitions[key]

    def terminal(self, histories):
        if histories not in self._terminal:
            frame = self.engine.safe_frame(histories)
            if frame is not None:
                if frame not in self._accept_ids:
                    self._accept_ids[frame] = len(self.nodes)
                    self.nodes.append(PolicyNode("accept", frame=frame))
                result = self._accept_ids[frame]
            elif not any(self.contract.must_accept(h) for h in histories.logical):
                result = 0
            else:
                result = None
                # Ideal extractions cannot distinguish a logical Pauli. If two
                # histories requiring logical preservation have the same full
                # data syndrome and different logical classes, and one must
                # accept, their fault-free continuations can never finish.
                by_syndrome = defaultdict(list)
                for h in histories.logical:
                    r, s, error = h
                    if r + s > self.contract.logical_order:
                        continue
                    signature = self.engine.syndrome(error)
                    for previous in by_syndrome[signature]:
                        if (previous[2] != error and
                                (self.contract.must_accept(previous) or self.contract.must_accept(h))):
                            result = -1
                            break
                    if result == -1:
                        break
                    by_syndrome[signature].append(h)
            self._terminal[histories] = result
            if len(self._terminal) > 100000:
                self._terminal.pop(next(iter(self._terminal)))
        return self._terminal[histories]

    def extraction_node(self, name, children):
        key = name, tuple(sorted(children.items()))
        if key not in self._node_ids:
            self._node_ids[key] = len(self.nodes)
            self.nodes.append(PolicyNode("extract", action=name, children=dict(children)))
        return self._node_ids[key]

    def frame_node(self, frame, child):
        if not frame or self.nodes[child].kind == "reject":
            return child
        if self.nodes[child].kind == "accept":
            value = frame ^ self.nodes[child].frame
            if value not in self._accept_ids:
                self._accept_ids[value] = len(self.nodes)
                self.nodes.append(PolicyNode("accept", frame=value))
            return self._accept_ids[value]
        key = frame, child
        if key not in self._frame_ids:
            self._frame_ids[key] = len(self.nodes)
            self.nodes.append(PolicyNode("frame", frame=frame, next=child))
        return self._frame_ids[key]

    def readout_frame(self, histories, action, outcome):
        """Use an ideal readout as a consistent Pauli-frame coordinate choice.

        With no future faults the extracted component syndrome is exact. Its
        fixed decoder offset may be removed immediately, including on records
        with only the weaker high-order filter obligation. The subsequent
        stopping test still checks the complete histories and logical classes.
        """
        if self.engine.remaining_max(histories):
            return 0
        basis = "X" if action.gadget.basis == "Z" else "Z"
        return self.engine.decoder.classes[basis][outcome[1]]

    def seed(self, histories, budget, steps, basis):
        normalized, frame = self.engine.normalize(histories)
        result = self._seed_normalized(normalized, budget, steps, basis)
        return None if result is None else (result[0], self.frame_node(frame, result[1]))

    def _seed_normalized(self, histories, budget, steps, basis):
        """Try an alternating ladder as a certified incumbent, before optimizing.

        This schedule is only a search heuristic: every leaf still passes the
        same exact stopping test. A failed seed does not prove infeasibility.
        """
        terminal = self.terminal(histories)
        if terminal is not None:
            return None if terminal < 0 else (0, terminal)
        known = self._solutions.get(histories)
        if known is not None and known[0] <= budget:
            return known
        if steps == 0:
            return None
        label = max(1, 2 * self.engine.remaining_max(histories))
        actions = [a for a in self.actions.values() if a.gadget.basis == basis
                   and a.gadget.distance >= label and a.cnot_cost <= budget]
        if not actions:
            return None
        # Obtain an incumbent using the best-protected gadgets; exact search
        # subsequently replaces them by cheaper eligible d2/d1 actions.
        unspent = bool(histories.filters[0]) or any(s == 0 for _, s, _ in histories.logical)
        action = min(actions, key=lambda a: (
            -a.gadget.distance if label > 1 and unspent else a.cnot_cost, a.cnot_cost, a.name))
        children, worst = {}, 0
        for outcome, child in sorted(self.transition(histories, action.name).items(), key=lambda item: -len(item[1])):
            frame = self.readout_frame(histories, action, outcome)
            result = self.seed(self.engine.translate(child, frame),
                               budget - action.cnot_cost, steps - 1,
                               "X" if basis == "Z" else "Z")
            if result is None:
                return None
            cost, node = result
            worst = max(worst, cost)
            children[outcome] = self.frame_node(frame, node)
        result = action.cnot_cost + worst, self.extraction_node(action.name, children)
        if known is None or result[0] < known[0]:
            self._solutions[histories] = result
        return result

    def zero_record_lower_bound(self, first_action, upper):
        """Shortest possible safe stopping on the mandatory fault-free record.

        We relax all other outcomes. Every real policy must still handle this
        record, so its shortest-path cost is a lower bound on minimax cost.
        """
        first = self.actions[first_action]
        state = self.engine.transition(self.engine.initial(), first, (0, 0))[(0, 0)]
        pending = [(first.cnot_cost, 0, state)]
        costs = {state: first.cnot_cost}
        serial = 0
        while pending:
            cost, _, histories = heapq.heappop(pending)
            if costs[histories] != cost:
                continue
            terminal = self.terminal(histories)
            if terminal is not None:
                if terminal >= 0:
                    return cost
                continue
            label = max(1, 2 * self.engine.remaining_max(histories))
            for action in self.actions.values():
                following_cost = cost + action.cnot_cost
                if action.gadget.distance < label or following_cost > upper:
                    continue
                following = self.engine.transition(histories, action, (0, 0)).get((0, 0))
                if following is None:
                    raise ValueError("A validated extraction lost the ideal zero record")
                if histories <= following or costs.get(following, upper + 1) <= following_cost:
                    continue
                costs[following] = following_cost
                serial += 1
                heapq.heappush(pending, (following_cost, serial, following))
        raise ValueError("A feasible policy has no safe fault-free path within its cost")

    def solve(self, histories, budget):
        normalized, frame = self.engine.normalize(histories)
        result = self._solve_normalized(normalized, budget)
        return None if result is None else (result[0], self.frame_node(frame, result[1]))

    def _solve_normalized(self, histories, budget):
        terminal = self.terminal(histories)
        if terminal is not None:
            return None if terminal < 0 else (0, terminal)
        known = self._solutions.get(histories)
        if known is not None and known[0] <= budget:
            return known
        if self._failures.get(histories, -1) >= budget:
            return None
        self.search_states += 1
        if self.search_states > self.max_search_states:
            raise PolicySearchError(f"Search state limit {self.max_search_states} reached")
        if self.search_states % 2000 == 0:
            self.progress(f"Explored {self.search_states} bounded search states")
        candidates = []
        for action in self.actions.values():
            if action.cnot_cost > budget:
                continue
            minimum_label = max(1, 2 * self.engine.remaining_max(histories))
            if action.gadget.distance < minimum_label:
                continue
            groups = self.transition(histories, action.name)
            # A fault-free repeated readout can leave the entire state unchanged.
            if any(histories <= child for child in groups.values()):
                continue
            if any(self.terminal(h) == -1 for h in groups.values()):
                continue
            unresolved = [h for h in groups.values() if self.terminal(h) is None]
            # Order the exact search by how much uncertainty remains, rather
            # than repeatedly selecting a cheap noisy circuit that amplifies
            # it. This changes search order only, not the feasible action set.
            score = max((len(h) for h in unresolved), default=0), len(unresolved), action.cnot_cost, action.name
            candidates.append((score, action, groups))
        for _, action, groups in sorted(candidates, key=lambda candidate: candidate[0]):
            children = {}
            worst = 0
            for outcome, child in sorted(groups.items(), key=lambda item: -len(item[1])):
                frame = self.readout_frame(histories, action, outcome)
                result = self.solve(self.engine.translate(child, frame), budget - action.cnot_cost)
                if result is None:
                    break
                cost, node = result
                children[outcome] = self.frame_node(frame, node)
                worst = max(worst, cost)
            else:
                result = action.cnot_cost + worst, self.extraction_node(action.name, children)
                # This is an upper bound, not necessarily the optimum for H.
                if known is None or result[0] < known[0]:
                    self._solutions[histories] = result
                return result
        self._failures[histories] = max(budget, self._failures.get(histories, -1))
        return None

    def build_paired(self, *, first_action: str, verify: bool = True) -> AdaptivePolicy:
        """Prescribed d4/d4, d4/d2, or d4/d4/d2 postselected flow.

        First flags zero selects the opposite d4 extraction; otherwise the
        opposite d2 extraction. Inferred total order two rejects immediately.
        A second d4 flag or unresolved endpoint can trigger one d2 follow-up.
        Ambiguous terminal records may reject, including single-fault records.
        """
        if self.contract.input_scope != "total-order" or self.contract.fault_order != 2:
            raise ValueError("Paired d4 flow requires total-order-two histories")
        actions = {name: action for name, action in self.actions.items()
                   if action.gadget.distance in {2, 4}}
        first = actions[first_action]
        if first.gadget.distance != 4:
            raise ValueError("Paired flow starts at d4")
        dual = "X" if first.gadget.basis == "Z" else "Z"
        choices = {d: next(a for a in actions.values() if a.gadget.basis == dual
                            and a.gadget.distance == d) for d in (2, 4)}
        self.progress("Enumerating bounded first-primary outcomes")
        groups = self.engine.transition(self.engine.initial(), first)
        children = {}
        completed = {}

        def low_order(state):
            return sum(r + s <= 1 for r, s, _ in state.logical)

        def terminal(state):
            if not low_order(state):
                return 0
            frame = self.engine.safe_frame(state)
            if frame is None:
                return None
            if frame not in self._accept_ids:
                self._accept_ids[frame] = len(self.nodes)
                self.nodes.append(PolicyNode("accept", frame=frame))
            return self._accept_ids[frame]

        @lru_cache(None)
        def followup(state):
            candidates = []
            for action in actions.values():
                if action.gadget.distance != 2:
                    continue
                routes, covered, rejected, mandatory_rejected = {}, 0, 0, 0
                for observed, following in self.engine.transition(state, action).items():
                    node = terminal(following)
                    if node is None:
                        mandatory_rejected += sum(self.contract.must_accept(h) for h in following.logical)
                        node = 0
                    routes[observed] = node
                    if node:
                        covered += low_order(following)
                    else:
                        rejected += low_order(following)
                # Choose a basis from compatible histories, never from a
                # prepared-state label or sampled logical outcomes.
                candidates.append((mandatory_rejected, -covered, rejected, action.name, routes))
            required, _, _, name, routes = min(candidates, key=lambda item: item[:4])
            if required:
                raise PolicySearchError("Neither d2 basis can accept all mandatory histories on this branch")
            return self.extraction_node(name, routes)

        for number, (outcome, histories) in enumerate(sorted(groups.items(), key=lambda item: -len(item[1])), 1):
            if not any(r + s <= 1 for r, s, _ in histories.logical):
                children[outcome] = 0
                continue
            label = 4 if outcome[0] == 0 else 2
            second = choices[label]
            key = histories, second.name
            if key not in completed:
                following = self.engine.transition(histories, second)
                routes = {}
                for observed, state in following.items():
                    if not any(r + s <= 1 for r, s, _ in state.logical):
                        routes[observed] = 0
                        continue
                    node = terminal(state)
                    if second.gadget.distance == 4 and (observed[0] != 0 or node is None):
                        node = followup(state)
                    elif node is None:
                        if any(self.contract.must_accept(h) for h in state.logical):
                            raise PolicySearchError(f"Mandatory history has no safe paired frame: {outcome}, {observed}")
                        node = 0
                    routes[observed] = node
                completed[key] = self.extraction_node(second.name, routes)
            children[outcome] = completed[key]
            if number % 5000 == 0:
                self.progress(f"Paired primary: {number:,}/{len(groups):,} first outcomes")
        root = self.extraction_node(first_action, children)
        self.actions = actions
        report = AdaptiveVerificationReport(self.contract, self.engine.decoder.fingerprint)
        policy = AdaptivePolicy(actions, self.contract, self.engine.decoder, tuple(self.nodes), root,
                                report, False, self.search_states, self.peak_qubits)
        compact_policy(policy)
        if verify:
            verify_adaptive_policy(policy)
        self.progress(f"Paired policy verified: {len(policy.nodes):,} nodes; {policy.report.worst_case_cnots} CX/CZ")
        return policy

    def build(self, *, first_action: str, max_cnot_cost: int | None = None,
              minimize: bool = True, verify: bool = True, paired: bool = False) -> AdaptivePolicy:
        if paired:
            return self.build_paired(first_action=first_action, verify=verify)
        if first_action not in self.actions:
            raise ValueError("Initial extraction is missing or exceeds the peak-qubit limit")
        first = self.actions[first_action]
        if first.gadget.distance < max(1, 2 * self.engine.remaining_max(self.engine.initial())):
            raise ValueError("Initial extraction does not meet the remaining-fault distance ladder")
        upper = max_cnot_cost if max_cnot_cost is not None else first.cnot_cost + 6 * max(
            a.cnot_cost for a in self.actions.values())
        if type(upper) is not int or upper < first.cnot_cost:
            raise ValueError("CNOT budget must include the initial extraction")
        obstruction = terminal_fault_obstructions(
            self.actions, self.engine.decoder, self.contract.logical_order)
        if obstruction:
            raise FTContractInfeasibleError(obstruction)
        start = time.monotonic()
        self.progress("Enumerating initial extraction outcomes")
        groups = self.transition(self.engine.initial(), first_action)

        def solve_root(budget):
            children, worst = {}, 0
            for outcome, histories in sorted(groups.items(), key=lambda item: -len(item[1])):
                result = self.solve(histories, budget - first.cnot_cost)
                if result is None:
                    return None
                cost, node = result
                children[outcome] = node
                worst = max(worst, cost)
            return first.cnot_cost + worst, self.extraction_node(first_action, children)

        self.progress(f"Searching {len(groups)} initial outcomes within {upper} CNOTs")
        self.progress("Trying the alternating distance ladder as an initial policy")
        dual = "X" if first.gadget.basis == "Z" else "Z"
        for histories in sorted(groups.values(), key=lambda h: -len(h)):
            self.seed(histories, upper - first.cnot_cost, 6, dual)
        best = solve_root(upper)
        if best is None:
            raise PolicySearchError(f"No policy found within {upper} CNOTs; "
                                    "increase the budget or extend the circuit library")
        lower = first.cnot_cost - 1
        limited = False
        if minimize:
            upper = best[0]
            self.progress("Bounding the cost of the mandatory fault-free path")
            lower = self.zero_record_lower_bound(first_action, upper) - 1
            while lower + 1 < upper:
                budget = (lower + upper) // 2
                self.progress(f"Testing CNOT bound {budget} ({self.search_states} search states)")
                try:
                    candidate = solve_root(budget)
                except PolicySearchError:
                    limited = True
                    self.progress("Search limit reached; retaining the certified feasible incumbent")
                    break
                if candidate is None:
                    lower = budget
                else:
                    best, upper = candidate, candidate[0]
        report = AdaptiveVerificationReport(self.contract, self.engine.decoder.fingerprint)
        policy = AdaptivePolicy(dict(self.actions), self.contract, self.engine.decoder,
                                tuple(self.nodes), best[1], report, minimize and not limited,
                                self.search_states, self.peak_qubits)
        policy.lower_bound_cnots = lower + 1 if minimize else 0
        policy.search_limit_reached = limited
        compact_policy(policy)
        if verify:
            verify_adaptive_policy(policy)
        policy.report.lower_bound_cnots = lower + 1 if minimize else 0
        self.progress(f"Policy complete in {time.monotonic() - start:.1f}s; "
                      f"{policy.report.worst_case_cnots or best[0]} CNOTs")
        return policy


def verify_adaptive_policy(policy: AdaptivePolicy) -> AdaptiveVerificationReport:
    """Recompute every reachable transition and check supplied leaves directly.

    This does not call the synthesizer's safe-frame or rejection decisions and
    does not trust any stored compatible sets, costs, or termination claims.
    """
    space = PauliSpace(policy.decoder.space.code)
    decoder = FixedCSSDecoder(space, policy.contract)
    report = AdaptiveVerificationReport(policy.contract, decoder.fingerprint)
    policy.report = report
    if decoder.fingerprint != policy.decoder.fingerprint:
        raise AdaptiveVerificationError("Ideal decoder changed", ())
    actions = {}
    for name, original in policy.actions.items():
        g = original.gadget
        action = ExtractionAction(name, SEGadget(
            g.distance, g.basis, g.circuit.copy(), g.readout, space, g.measurement_map.copy()))
        validate_gadget(action.gadget)
        if policy.peak_qubit_limit is not None and action.qubits > policy.peak_qubit_limit:
            raise AdaptiveVerificationError("Peak-qubit limit exceeded", ())
        actions[name] = action
    if not policy.nodes or policy.nodes[0].kind != "reject":
        raise AdaptiveVerificationError("Node zero must reject unknown outcomes", ())
    structural_done, structural_active = set(), set()

    def check_structure(index):
        if not 0 <= index < len(policy.nodes):
            raise AdaptiveVerificationError("Missing policy node", ())
        if index in structural_active:
            raise AdaptiveVerificationError("Policy does not terminate", ())
        if index in structural_done:
            return
        structural_active.add(index)
        node = policy.nodes[index]
        if node.kind == "extract":
            for child in node.children.values():
                check_structure(child)
        elif node.kind == "frame":
            if type(node.next) is not int:
                raise AdaptiveVerificationError("Missing frame continuation", ())
            check_structure(node.next)
        elif node.kind not in {"accept", "reject"}:
            raise AdaptiveVerificationError("Unknown policy node kind", ())
        structural_active.remove(index)
        structural_done.add(index)

    check_structure(policy.root)
    engine = HistoryEngine(space, policy.contract, decoder)
    initial = engine.initial()
    report.ec_a_input_syndromes = sum(r < 0 for r, _, _ in initial)
    checked, active = {}, set()

    def visit(index, histories, record):
        if index < 0 or index >= len(policy.nodes):
            raise AdaptiveVerificationError("Missing policy node", record)
        key = index, histories
        if key in active:
            raise AdaptiveVerificationError("Policy does not terminate", record)
        if key in checked:
            return checked[key]
        active.add(key)
        node = policy.nodes[index]
        if node.kind == "reject":
            for history in histories:
                if policy.contract.must_accept(history):
                    raise AdaptiveVerificationError("Mandatory-acceptance history rejected", record, history)
            report.discarded_contexts += len(histories)
            resources = 0, 0, 0
        elif node.kind == "accept":
            if node.frame < 0 or node.frame >> (2 * space.n):
                raise AdaptiveVerificationError("Invalid Pauli frame", record)
            correction = space.reduce(node.frame)
            correction_syndrome = space.pauli_syndrome(correction)
            for history in histories:
                r, s, error = history
                signature = error ^ correction_syndrome if r < 0 else space.pauli_syndrome(error ^ correction)
                sx, sz = signature & decoder.x_mask, signature >> decoder.x_width
                if ((r < 0 or r + s <= policy.contract.accepted_filter_order)
                        and (decoder.weights["X"][sx] > s or decoder.weights["Z"][sz] > s)):
                    raise AdaptiveVerificationError("Accepted output exceeds CSS fault-count filter", record, history)
                if (r >= 0 and r + s <= policy.contract.logical_order
                        and space.reduce(error ^ correction ^ decoder.correction(signature))):
                    raise AdaptiveVerificationError("Accepted output changes ideal-decoded logical state", record, history)
            report.accepted_contexts += len(histories)
            resources = 0, 0, 0
        elif node.kind == "extract":
            if node.action not in actions:
                raise AdaptiveVerificationError("Missing extraction action", record)
            action = actions[node.action]
            if action.gadget.distance < max(1, 2 * engine.remaining_max(histories)):
                raise AdaptiveVerificationError("Extraction violates the remaining-fault distance ladder", record)
            groups = engine.transition(histories, action)
            results = []
            for outcome, following in groups.items():
                report.checked_transitions += len(following)
                child = node.children.get(outcome, 0)
                results.append(visit(child, following, record + ((node.action, *outcome),)))
            resources = (action.cnot_cost + max((r[0] for r in results), default=0),
                         1 + max((r[1] for r in results), default=0),
                         max(action.qubits, max((r[2] for r in results), default=0)))
        elif node.kind == "frame":
            if node.frame < 0 or node.frame >> (2 * space.n):
                raise AdaptiveVerificationError("Invalid Pauli-frame update", record)
            resources = visit(node.next, engine.translate(histories, node.frame), record)
        else:
            raise AdaptiveVerificationError("Unknown policy node kind", record)
        active.remove(key)
        checked[key] = resources
        return resources

    report.worst_case_cnots, report.max_extractions, report.peak_qubits = visit(
        policy.root, initial, ())
    report.lower_bound_cnots = policy.lower_bound_cnots
    if report.lower_bound_cnots > report.worst_case_cnots:
        raise AdaptiveVerificationError("Cost lower bound exceeds the executed cost", ())
    if policy.optimal and report.lower_bound_cnots != report.worst_case_cnots:
        policy.optimal = False
    report.verified = True
    return report

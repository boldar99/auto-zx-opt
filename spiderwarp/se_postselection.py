"""Compile observable order-one compatibility into an adaptive SE graph.

The derived program follows the original actions and Pauli frames until its
observations have no explanation with incoming CSS order plus internal faulty
locations <= 1. Such an edge rejects immediately. No runtime fault enumerator,
noise-event oracle, or ideal final readout is needed by the saved program.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from functools import lru_cache

from spiderwarp._se_adaptive import (
    AdaptivePolicy, AdaptiveVerificationReport, CompatibleSet, FTContract,
    FixedCSSDecoder, HistoryEngine, PolicyNode, compact_policy,
)
from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol


class OrderOneEvidence:
    """Exact observation compatibility, merging equal data-syndrome histories.

    State coordinates are relative to the controller's current Pauli frame.
    Gadget flags do not depend on incoming data Paulis, and the ideal gadget
    preserves the data Pauli. Both assumptions are checked before compilation.
    """

    def __init__(self, protocol):
        policy = protocol.policy
        if policy.contract.correctable != 1 or policy.contract.logical_order != 1:
            raise ValueError("Order-one postselection requires the one-error logical contract")
        self.space, self.decoder = policy.decoder.space, policy.decoder
        self.actions = policy.actions
        self.width, self.mask = self.decoder.x_width, self.decoder.x_mask
        self.initial = frozenset(
            (r, 0, sx | (sz << self.width))
            for sx in self.decoder.classes["X"] for sz in self.decoder.classes["Z"]
            if (r := max(self.decoder.weights["X"][sx], self.decoder.weights["Z"][sz])) <= 1)
        self.responses = {}
        self._transitions = {}
        for name, action in self.actions.items():
            g = action.gadget
            for q in range(2 * protocol.code.n):
                flags, syndrome, output = g.unpack(g.input_effect(1 << q))
                error = 1 << q
                signature = self.space.pauli_syndrome(error)
                expected = signature & self.mask if g.basis == "Z" else signature >> self.width
                if flags or syndrome != expected or output != self.space.reduce(error):
                    raise ValueError(f"{name} does not have the required ideal Pauli response")
            self.responses[name] = frozenset(
                (f, sy, self.space.pauli_syndrome(data))
                for effect in g.single_fault_effects for f, sy, data in (g.unpack(effect),))

    def translate(self, state, frame):
        signature = self.space.pauli_syndrome(frame)
        return (frozenset((r, s, data ^ signature) for r, s, data in state)
                if signature else state)

    def transition(self, state, name):
        key = state, name
        if key in self._transitions:
            return self._transitions[key]
        groups = defaultdict(set)
        is_z = self.actions[name].gadget.basis == "Z"
        for r, s, data in state:
            readout = data & self.mask if is_z else data >> self.width
            groups[0, readout].add((r, s, data))
            if r + s == 0:
                for flags, syndrome, delta in self.responses[name]:
                    groups[flags, readout ^ syndrome].add((0, 1, data ^ delta))
        result = {outcome: frozenset(following) for outcome, following in groups.items()}
        self._transitions[key] = result
        return result


def verify_order_one_guard(protocol):
    """Independently check every stored non-reject edge using full Pauli histories.

    This checker uses HistoryEngine rather than the syndrome-only compiler. The
    standard policy verifier separately checks low-order mandatory acceptance,
    accepted-output logical preservation/CSS filters, resource bounds and DAG.
    """
    policy = protocol.policy
    contract = FTContract(1, 1, fault_order=1, accept_clean_through=1)
    engine = HistoryEngine(policy.decoder.space, contract, policy.decoder)
    initial = engine.initial()
    initial = CompatibleSet(initial.logical, (0, 0), engine.sectors)
    counts = dict(states=0, nonreject_edges=0, accepts=0)

    @lru_cache(maxsize=None)
    def visit(index, histories):
        node = policy.nodes[index]
        counts["states"] += 1
        if node.kind == "reject":
            return
        if not histories.logical:
            raise ValueError("Postselected graph continues without an order-zero/one explanation")
        if node.kind == "accept":
            counts["accepts"] += 1
        elif node.kind == "frame":
            visit(node.next, engine.translate(histories, node.frame))
        elif node.kind == "extract":
            groups = engine.transition(histories, policy.actions[node.action])
            for outcome, child in node.children.items():
                if policy.nodes[child].kind == "reject":
                    continue
                if outcome not in groups or not groups[outcome].logical:
                    raise ValueError(f"Postselected edge has no low-order explanation: {node.action}, {outcome}")
                counts["nonreject_edges"] += 1
                visit(child, groups[outcome])
        else:
            raise ValueError("Unknown postselected node kind")

    visit(policy.root, initial)
    return counts


def postselect_order_one(parent, *, verify=True, progress=None):
    """Prune a saved controller without resynthesis or new quantum operations.

    Shared graph nodes are paired with their compatible syndrome state before
    pruning; inspecting terminal node IDs alone would lose history information.
    Frame nodes translate that state, including an external incoming frame via
    the ordinary se_cycle_with_frame interface at execution time.
    """
    evidence = OrderOneEvidence(parent)
    old = parent.policy
    contract = replace(old.contract, accept_clean_through=1)
    decoder = FixedCSSDecoder(evidence.space, contract)
    nodes = [PolicyNode("reject")]
    interned = {("reject",): 0}

    def intern(key, node):
        if key not in interned:
            interned[key] = len(nodes)
            nodes.append(node)
        return interned[key]

    @lru_cache(maxsize=None)
    def visit(index, state):
        node = old.nodes[index]
        if not state or node.kind == "reject":
            return 0
        if node.kind == "accept":
            return intern(("accept", node.frame), PolicyNode("accept", frame=node.frame))
        if node.kind == "frame":
            child = visit(node.next, evidence.translate(state, node.frame))
            if child == 0:
                return 0
            return intern(("frame", node.frame, child), PolicyNode("frame", frame=node.frame, next=child))
        if node.kind != "extract":
            raise ValueError("Unknown parent node kind")
        children = {}
        for outcome, following in evidence.transition(state, node.action).items():
            child = visit(node.children.get(outcome, 0), following)
            if child:
                children[outcome] = child
        key = "extract", node.action, tuple(sorted(children.items()))
        return intern(key, PolicyNode("extract", action=node.action, children=children))

    root = visit(old.root, evidence.initial)
    report = AdaptiveVerificationReport(contract, decoder.fingerprint)
    policy = AdaptivePolicy(old.actions.copy(), contract, decoder, tuple(nodes), root, report,
                            peak_qubit_limit=old.peak_qubit_limit)
    compact_policy(policy)
    derived = AdaptiveSEProtocol(parent.code, policy)
    if progress:
        progress(f"Compiled order-one guard: {len(old.nodes):,} -> {len(policy.nodes):,} nodes")
    if verify:
        derived.verify()
        guard = verify_order_one_guard(derived)
        if progress:
            progress(f"Independent FT and guard checks passed: {guard}")
    return derived

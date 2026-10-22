"""Synthesize and optionally check a flag-and-recovery SE cycle.

The enumeration uses distinct *exact Pauli responses*, rather than sampling.
Responses at one depolarizing location are mutually exclusive. Histories are
merged only when their measured prefix, input/internal fault counts and residual
Pauli class agree. This retains the shared fault budget across both primaries.
"""

from __future__ import annotations

import itertools
import heapq
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import stim

from spiderwarp._se_circuit import PauliSpace, SEGadget, CircuitFault, bits
from spiderwarp.lut_decoder import LutDecoder


# (number of incoming Pauli errors, number of circuit faults, residual class).
# Incoming=-1 denotes an arbitrary-input syndrome sector for EC A; it does
# not consume the internal-fault budget and does not track a logical class.
Context = tuple[int, int, int]
# First Z primary has prefix -1. The X primary retains the accepted Z syndrome.
RecoveryKey = tuple[str, int, int]


@dataclass(frozen=True)
class ProtocolWitness:
    stage: str
    reason: str
    incoming_faults: int | None
    internal_faults: int
    residual_pauli: tuple[tuple[str, int], ...]
    flags: int = 0
    syndrome: int = 0
    previous_syndrome: int = -1
    circuit_faults: tuple[CircuitFault, ...] = ()
    counterexample_circuit: stim.Circuit | None = field(default=None, compare=False, repr=False)
    incoming_pauli: tuple[tuple[str, int], ...] = ()


class ProtocolVerificationError(ValueError):
    """The supplied gadgets cannot realize the requested adaptive FT contract."""

    def __init__(self, witness: ProtocolWitness):
        self.witness = witness
        super().__init__(
            f"{witness.stage}: {witness.reason}; incoming="
            f"{witness.incoming_faults if witness.incoming_faults is not None else 'arbitrary'}, "
            f"internal={witness.internal_faults}, flags={witness.flags}, "
            f"syndrome={witness.syndrome}, previous_syndrome={witness.previous_syndrome}, "
            f"residual={witness.residual_pauli or 'I'}")


@dataclass(frozen=True)
class RecoveryPlan:
    primary_basis: str
    primary_flags: int
    previous_syndrome: int
    distance: int
    corrections: dict[int, int]
    discarded_syndromes: tuple[int, ...]
    possible_fault_orders: tuple[int, ...]
    # Optional extraction of the discarded primary component. Correction keys
    # pack dual_syndrome | (cleanup_syndrome << number_of_checks).
    cleanup_distance: int | None = None


@dataclass
class VerificationReport:
    correctable_faults: int
    detectable_faults: int
    arbitrary_logical_state: bool = True
    residual_metric: str = "CSS weight max(w_X, w_Z); Gottesman EC A and EC B with separate X/Z filters"
    acceptance_contract: str = (
        "Postselected Gottesman conditions: EC A requires every accepted output "
        "with r <= t internal faults, for arbitrary incoming errors, to pass "
        "the CSS r-filter from some codeword (at most r X and r Z errors). EC B requires "
        "acceptance and preservation of the ideal-decoded logical state when "
        "s incoming CSS errors plus r internal faults satisfy s+r <= t. Additional "
        "bounded histories through the detectable order use the internal-fault "
        "r-filter too, without a correction-radius cap."
    )
    noise_model: str = (
        "Circuit-level Pauli faults: all 15 nonidentity two-qubit Paulis, "
        "X/Y/Z after single-qubit gates and correction gates, preparation "
        "and measurement flips; no implicit idle locations."
    )
    checked_transitions: int = 0
    accepted_contexts: int = 0
    discarded_contexts: int = 0
    ec_a_input_syndromes: int = 0
    verified: bool = False


class PolicyBuilder:
    def __init__(self, space: PauliSpace, gadgets: dict[tuple[str, int], SEGadget],
                 *, recovery_objective: str = 'qubits'):
        if recovery_objective not in {'qubits', 'gates'}:
            raise ValueError("recovery_objective must be 'qubits' or 'gates'")
        self.recovery_objective = recovery_objective
        self.space = space
        self.gadgets = gadgets
        self.correctable = (space.code.d - 1) // 2
        self.detectable = space.code.d // 2
        if self.detectable > 2:
            raise NotImplementedError("Exact adaptive synthesis currently supports up to two faults (code distance <= 5)")
        self.report = VerificationReport(self.correctable, self.detectable)
        self._correction_cache = {}
        self.standard: dict[str, dict[int, int]] = {}
        self.recoveries: dict[RecoveryKey, RecoveryPlan] = {}
        self._standard_decoders()
        self.representatives = self.space.low_weight(self.correctable + 1)
        self._arbitrary_inputs = self.space.all_syndrome_representatives()
        self.report.ec_a_input_syndromes = len(self._arbitrary_inputs)

    def _standard_decoders(self):
        code = self.space.code
        for basis, checks in (("X", code.H_x), ("Z", code.H_z)):
            decoder = LutDecoder(checks, max_decodable_weight=self.correctable)
            result = {}
            # Query only syndromes reached by correctable physical errors;
            # dependent checks need not span the whole bit-string space.
            syndromes = {tuple(int(x) for x in (checks @ np.asarray(
                [int((p >> q) & 1) for q in range(code.n)], dtype=np.uint8)) % 2)
                for w in range(self.correctable + 1)
                for qs in itertools.combinations(range(code.n), w)
                for p in [sum(1 << q for q in qs)]}
            for syndrome in sorted(syndromes):
                correction = decoder.get_mwr(np.asarray(syndrome, dtype=np.uint8))
                mask = sum(1 << int(q) for q in np.flatnonzero(correction))
                if basis == "X":
                    mask <<= code.n
                result[sum(bit << i for i, bit in enumerate(syndrome))] = mask
            self.standard[basis] = result

    def _limit(self, context: Context) -> int:
        return context[1]

    def _requires_acceptance(self, context: Context) -> bool:
        incoming, internal, _ = context
        return incoming >= 0 and incoming + internal <= self.correctable

    def _remaining(self, context: Context) -> int:
        incoming, internal, _ = context
        return (self.correctable - internal if incoming < 0
                else self.detectable - incoming - internal)

    def _normalize(self, context: Context) -> Context:
        incoming, internal, error = context
        if incoming < 0:
            error = self._arbitrary_inputs[self.space.pauli_syndrome(error)]
        return incoming, internal, error

    def _within_bound(self, context: Context) -> bool:
        _, internal, error = context
        if not self.space.within_css_filter(error, internal):
            return False
        return (not self._requires_acceptance(context)
                or self.space.within_css_bound(error, internal))

    def _fail(self, stage, reason, context, flags=0, syndrome=0, previous=-1):
        incoming, internal, error = context
        circuit_faults = ()
        counterexample = None
        incoming_pauli = ()
        # Reconstruct a fully physical witness when the failing all-zero prefix
        # is reachable by faults solely in the final primary. Other failures
        # still expose the exact residual class and fault-count constraint.
        if stage == "unflagged cycle" and incoming >= 0 and previous >= 0:
            final = self.gadgets["X", self.space.code.d]
            first = self.gadgets["Z", self.space.code.d]
            correction = self.standard["X"].get(syndrome, 0)
            before_correction = error ^ self.space.reduce(correction)
            target = (syndrome << final.num_flags) | (before_correction << final.num_detectors)
            first_correction = self.standard["Z"].get(previous)
            for initial in self.space.css_errors_of_weight(incoming):
                first_flags, first_syndrome, first_output = first.unpack(first.input_effect(initial))
                if first_flags or first_syndrome != previous or first_correction is None:
                    continue
                first_output ^= self.space.reduce(first_correction)
                effect = target ^ final.input_effect(first_output)
                found = final.fault_witness(effect, internal)
                if found is None:
                    continue
                circuit_faults = found
                incoming_pauli = self.space.product(initial)
                from spiderwarp.fault_tolerance_verification import _make_encoded_bell_input
                bell, _, _ = _make_encoded_bell_input(
                    self.space.code.H_x, self.space.code.H_z,
                    self.space.code.L_x, self.space.code.L_z,
                    max(g.circuit.num_qubits for g in self.gadgets.values()))
                counterexample = bell.copy()
                for gate, q in incoming_pauli:
                    counterexample.append(gate, [q])
                counterexample += first.circuit
                for gate, q in self.space.product(first_correction):
                    counterexample.append(gate, [q])
                counterexample += final.with_faults(found)
                for gate, q in self.space.product(correction):
                    counterexample.append(gate, [q])
                break
        raise ProtocolVerificationError(ProtocolWitness(
            stage, reason, None if incoming < 0 else incoming, internal,
            self.space.product(self.representatives.get(error, error)),
            flags, syndrome, previous, circuit_faults, counterexample, incoming_pauli))

    def _terminal(self, stage: str, context: Context, *, discarded=False,
                  flags=0, syndrome=0, previous=-1):
        incoming, internal, error = context
        if discarded:
            if self._requires_acceptance(context):
                self._fail(stage, "a correctable-order history would be discarded", context,
                           flags, syndrome, previous)
            self.report.discarded_contexts += 1
        else:
            if not self.space.within_css_filter(error, internal):
                condition = "EC A" if internal <= self.correctable else "extended r-filter"
                self._fail(stage, f"{condition}: accepted residual exceeds the FT bound", context,
                           flags, syndrome, previous)
            if self._requires_acceptance(context) and not self.space.within_css_bound(error, internal):
                self._fail(stage, "EC B: ideal-decoded logical state is not preserved", context,
                           flags, syndrome, previous)
            self.report.accepted_contexts += 1

    def _correction_faults(self, correction: int, maximum: int):
        key = correction, maximum
        if key in self._correction_cache:
            return self._correction_cache[key]
        # Emit the same X-then-Z physical operations as the executor/codegen.
        gates = [q % self.space.n for q in bits(correction)]
        layers = [{0}]
        for order in range(1, maximum + 1):
            effects = set()
            for sites in itertools.combinations(range(len(gates)), order):
                for kinds in itertools.product((1, 2, 3), repeat=order):
                    error = 0
                    for site, kind in zip(sites, kinds):
                        q = gates[site]
                        if kind & 1:
                            error ^= 1 << q
                        if kind & 2:
                            error ^= 1 << (q + self.space.n)
                    effects.add(self.space.reduce(error))
            layers.append(effects)
        result = tuple(tuple(sorted(layer)) for layer in layers)
        self._correction_cache[key] = result
        return result

    def _correct(self, context: Context, correction: int):
        incoming, internal, error = context
        error ^= self.space.reduce(correction)
        remaining = self._remaining(context)
        for order, faults in enumerate(self._correction_faults(correction, remaining)):
            for fault in faults:
                yield self._normalize((incoming, internal + order, error ^ fault))

    def _run(self, gadget: SEGadget, contexts):
        groups = defaultdict(set)
        for incoming, internal, error in sorted(contexts):
            base = gadget.input_effect(error)
            remaining = self._remaining((incoming, internal, error))
            for order in range(remaining + 1):
                for fault in gadget.fault_effects(order):
                    flags, syndrome, residual = gadget.unpack(base ^ fault)
                    groups[flags, syndrome].add(self._normalize((incoming, internal + order, residual)))
                    self.report.checked_transitions += 1
        return groups

    def _primary_histories(self, *, verify):
        initial = {(order, 0, self.space.reduce(error))
                   for order in range(self.correctable + 1)
                   for error in self.space.css_errors_of_weight(order)}
        initial.update((-1, 0, error) for error in self._arbitrary_inputs.values())
        first = self._run(self.gadgets["Z", self.space.code.d], initial)
        recovery_inputs = defaultdict(set)
        second_inputs = defaultdict(set)
        for (flags, syndrome), contexts in sorted(first.items()):
            if flags:
                recovery_inputs["Z", flags, -1].update(contexts)
                continue
            correction = self.standard["Z"].get(syndrome)
            for context in sorted(contexts):
                if correction is None:
                    if verify:
                        self._terminal("Z primary decoding", context, discarded=True, syndrome=syndrome)
                else:
                    second_inputs[syndrome].update(self._correct(context, correction))
        for previous, contexts in sorted(second_inputs.items()):
            second = self._run(self.gadgets["X", self.space.code.d], contexts)
            for (flags, syndrome), results in sorted(second.items()):
                if flags:
                    recovery_inputs["X", flags, previous].update(results)
                    continue
                correction = self.standard["X"].get(syndrome)
                if not verify:
                    continue
                for context in sorted(results, key=lambda c: (c[0] < 0, c)):
                    if correction is None:
                        self._terminal("X primary decoding", context, discarded=True,
                                       syndrome=syndrome, previous=previous)
                    else:
                        for final in self._correct(context, correction):
                            self._terminal("unflagged cycle", final,
                                           syndrome=syndrome, previous=previous)
        return recovery_inputs

    def build(self, *, verify=False):
        """Synthesize decisions; terminal verification is opt-in."""
        for key, contexts in sorted(self._primary_histories(verify=verify).items()):
            plan = self._recovery(key, contexts)
            self.recoveries[key] = plan
            if verify:
                self._verify_recovery(key, contexts, plan)
        self.report.verified = verify
        return self

    def verify(self, standard, recoveries):
        """Check the supplied decisions without replacing or reoptimizing them."""
        self.standard = standard
        self.report = VerificationReport(
            self.correctable, self.detectable, ec_a_input_syndromes=len(self._arbitrary_inputs))
        for key, contexts in sorted(self._primary_histories(verify=True).items()):
            self._verify_recovery(key, contexts, recoveries.get(key))
        self.report.verified = True
        return self.report

    def _recovery_outcomes(self, gadget, cleanup, contexts):
        grouped = self._run(gadget, contexts)
        if cleanup is None:
            return grouped
        combined = defaultdict(set)
        width = len(gadget.readout.syndromes)
        for (flags, syndrome), results in grouped.items():
            if flags:
                combined[1, syndrome].update(results)
                continue
            for (cleanup_flags, cleanup_syndrome), final in self._run(cleanup, results).items():
                combined[int(bool(cleanup_flags)), syndrome | (cleanup_syndrome << width)].update(final)
        return combined

    def _verify_recovery(self, key, contexts, plan):
        basis, flags, previous = key
        if plan is None:
            for context in sorted(contexts):
                self._terminal(f"{basis} missing recovery", context, discarded=True,
                               flags=flags, previous=previous)
            return
        dual = "Z" if basis == "X" else "X"
        gadget = self.gadgets.get((dual, plan.distance))
        cleanup = (self.gadgets.get((basis, plan.cleanup_distance))
                   if plan.cleanup_distance is not None else None)
        if gadget is None or (plan.cleanup_distance is not None and cleanup is None):
            self._fail(f"{basis} recovery", "selected gadget is missing", min(contexts),
                       flags=flags, previous=previous)
        grouped = self._recovery_outcomes(gadget, cleanup, contexts)
        for (recovery_flags, syndrome), results in sorted(grouped.items()):
            correction = plan.corrections.get(syndrome)
            for context in sorted(results):
                if recovery_flags or correction is None:
                    self._terminal(f"{basis} recovery", context, discarded=True,
                                   flags=flags, syndrome=syndrome, previous=previous)
                else:
                    for final in self._correct(context, correction):
                        self._terminal(f"{basis} recovery", final, flags=flags,
                                       syndrome=syndrome, previous=previous)

    def _candidate_corrections(self, contexts):
        low = sorted(c for c in contexts if self._requires_acceptance(c))
        high = sorted(c for c in contexts if not self._requires_acceptance(c))
        components = []
        for i, basis in enumerate(("X", "Z")):
            choices = None
            for context in low:
                error = self.space.components(context[2])[i]
                residuals = self.space.component_errors(basis, self._limit(context))
                choices = ({error ^ p for p in residuals} if choices is None
                           else {p for p in choices if p ^ error in residuals})
                if not choices:
                    return
            signatures = None
            for context in high:
                signature = self.space.component_syndrome(context[2], basis)
                allowed = {signature ^ p for p in self.space.syndrome_errors(basis, self._limit(context))}
                signatures = allowed if signatures is None else signatures & allowed
                if not signatures:
                    return
            if choices is not None:
                if signatures is not None:
                    choices = {p for p in choices if self.space.component_syndrome(p, basis) in signatures}
                reps = self.space.component_errors(basis, self.correctable + 1)
                physical = {reps.get(p, p) for p in choices}
            else:
                physical = {self.space.syndrome_representative(basis, s, self.correctable + 1)
                            for s in signatures or ()}
            components.append(sorted(physical, key=lambda p: (p.bit_count(), p)))
        if not all(components):
            return
        # Enumerate the independent CSS factors lazily in gate-count order.
        xs, zs = components
        heap = [(xs[0].bit_count() + zs[0].bit_count(), 0, 0)]
        seen = {(0, 0)}
        while heap:
            _, i, j = heapq.heappop(heap)
            yield xs[i] ^ zs[j]
            for a, b in ((i + 1, j), (i, j + 1)):
                if a < len(xs) and b < len(zs) and (a, b) not in seen:
                    seen.add((a, b))
                    heapq.heappush(heap, (xs[a].bit_count() + zs[b].bit_count(), a, b))

    def _correction_safe(self, contexts, correction):
        for context in contexts:
            for final in self._correct(context, correction):
                if not self._within_bound(final):
                    return False
        return True

    def _recovery(self, key: RecoveryKey, contexts):
        basis, flags, previous = key
        dual = "Z" if basis == "X" else "X"
        def cost(*gadgets):
            width = max(g.circuit.num_qubits for g in gadgets)
            gates = sum(len(g.operations) for g in gadgets)
            resources = (gates, width) if self.recovery_objective == 'gates' else (width, gates)
            return (*resources, tuple(g.distance for g in gadgets))

        candidates = sorted((g for (b, _), g in self.gadgets.items() if b == dual),
                            key=cost)
        cleanup_candidates = [g for (b, _), g in self.gadgets.items() if b == basis]
        pairs = sorted(((g, c) for g in candidates for c in cleanup_candidates),
                       key=lambda pair: cost(*pair))
        schedules = [(g, None) for g in candidates] + pairs
        first_failure = None
        for gadget, cleanup in schedules:
            grouped = self._recovery_outcomes(gadget, cleanup, contexts)
            by_syndrome = defaultdict(set)
            bad_recovery = False
            for (recovery_flags, syndrome), results in grouped.items():
                if recovery_flags:
                    low = next((c for c in sorted(results) if self._requires_acceptance(c)), None)
                    if low is not None:
                        first_failure = first_failure or (low, syndrome, "recovery flags discard a correctable history")
                        bad_recovery = True
                        break
                else:
                    by_syndrome[syndrome].update(results)
            if bad_recovery:
                continue
            corrections = {}
            discarded = []
            for syndrome, results in sorted(by_syndrome.items()):
                choices = self._candidate_corrections(results)
                correction = next((p for p in choices if self._correction_safe(results, p)), None)
                if correction is not None:
                    corrections[syndrome] = correction
                elif not any(self._requires_acceptance(c) for c in results):
                    discarded.append(syndrome)
                else:
                    low = min(c for c in results if self._requires_acceptance(c))
                    first_failure = first_failure or (low, syndrome, "recovery measurements cannot determine a safe correction")
                    bad_recovery = True
                    break
            if bad_recovery:
                continue
            return RecoveryPlan(basis, flags, previous, gadget.distance, corrections,
                                tuple(discarded), tuple(sorted({sum(c[:2]) for c in contexts if c[0] >= 0})),
                                cleanup.distance if cleanup is not None else None)
        context, syndrome, reason = first_failure or (min(contexts), 0, "no recovery gadget is available")
        self._fail(f"{basis} recovery synthesis", reason, context, flags, syndrome, previous)

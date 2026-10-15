"""Clifford gadgets and exact Pauli responses used by adaptive SE synthesis.

Integers encode Paulis as ``x | (z << n)`` (phases are immaterial).  A gadget
response packs its flag parities, syndrome parities, and output Pauli class.
Classes are taken modulo *stabilizers only*, so every logical Pauli is retained.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import stim

from spiderwarp.csscode import CSSCode


ANNOTATIONS = {"DETECTOR", "QUBIT_COORDS", "TICK"}
GATES = {"R", "RX", "M", "MX", "H", "CX", "CZ", "XCX", "X", "Y", "Z"}
TWO_QUBIT = {"CX", "CZ", "XCX"}


def bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def row_mask(row) -> int:
    return sum(1 << int(q) for q in np.flatnonzero(row))


class PauliSpace:
    def __init__(self, code: CSSCode):
        self.n = code.n
        self.code = code
        self.pivots: dict[int, int] = {}
        for row in [*(row_mask(r) for r in code.H_x),
                    *(row_mask(r) << code.n for r in code.H_z)]:
            row = self.reduce(row)
            if row:
                self.pivots[row.bit_length() - 1] = row
        self.pivots = dict(sorted(self.pivots.items(), reverse=True))
        self._low_weight: dict[int, dict[int, int]] = {}
        self._component_weight: dict[tuple[str, int], dict[int, int]] = {}
        self._syndrome_weights = {}
        self._pauli_syndrome_weights = {}
        self._checks = {
            "X": tuple(row_mask(r) for r in code.H_z),
            "Z": tuple(row_mask(r) for r in code.H_x),
        }

    def reduce(self, pauli: int) -> int:
        for pivot, row in sorted(self.pivots.items(), reverse=True):
            if (pauli >> pivot) & 1:
                pauli ^= row
        return pauli

    def weight(self, pauli: int) -> int:
        return ((pauli | (pauli >> self.n)) & ((1 << self.n) - 1)).bit_count()

    def product(self, pauli: int) -> tuple[tuple[str, int], ...]:
        x = pauli & ((1 << self.n) - 1)
        z = pauli >> self.n
        return tuple(("Y" if (x >> q) & (z >> q) & 1 else
                      "X" if (x >> q) & 1 else "Z", q) for q in bits(x | z))

    def errors_of_weight(self, weight: int):
        if weight == 0:
            yield 0
        for qubits in itertools.combinations(range(self.n), weight):
            for kinds in itertools.product((1, 2, 3), repeat=weight):
                yield sum(((1 << q) if kind & 1 else 0)
                          | ((1 << (q + self.n)) if kind & 2 else 0)
                          for q, kind in zip(qubits, kinds))

    def low_weight(self, maximum: int) -> dict[int, int]:
        if maximum not in self._low_weight:
            result = {}
            for weight in range(maximum + 1):
                for pauli in self.errors_of_weight(weight):
                    result.setdefault(self.reduce(pauli), pauli)
            self._low_weight[maximum] = result
        return self._low_weight[maximum]

    def components(self, pauli: int) -> tuple[int, int]:
        """Return canonical CSS components in the common 2n-bit encoding."""
        mask = (1 << self.n) - 1
        return pauli & mask, pauli & (mask << self.n)

    def component_errors(self, basis: str, maximum: int) -> dict[int, int]:
        """CSS classes with a representative of component weight <= maximum.

        Keep the two factors separate: explicitly constructing their Cartesian
        product would require O(n**(2*maximum)) memory for a CSS filter.
        """
        key = basis, maximum
        if key not in self._component_weight:
            shift = 0 if basis == "X" else self.n
            result = {}
            for weight in range(maximum + 1):
                for qubits in itertools.combinations(range(self.n), weight):
                    error = sum(1 << (q + shift) for q in qubits)
                    result.setdefault(self.reduce(error), error)
            self._component_weight[key] = result
        return self._component_weight[key]

    def within_css_bound(self, pauli: int, maximum: int) -> bool:
        x, z = self.components(pauli)
        return (x in self.component_errors("X", maximum)
                and z in self.component_errors("Z", maximum))

    def css_errors_of_weight(self, weight: int):
        """Physical representatives with minimal CSS component weight = weight."""
        xs = self.component_errors("X", weight)
        zs = self.component_errors("Z", weight)
        lower_x = self.component_errors("X", weight - 1) if weight else {}
        lower_z = self.component_errors("Z", weight - 1) if weight else {}
        for x, xp in xs.items():
            for z, zp in zs.items():
                if weight and x in lower_x and z in lower_z:
                    continue
                yield xp ^ zp

    def component_syndrome(self, pauli: int, basis: str) -> int:
        """Syndrome of one component, forgetting its logical Pauli class."""
        mask = pauli if basis == "X" else pauli >> self.n
        return sum(((mask & row).bit_count() % 2) << i
                   for i, row in enumerate(self._checks[basis]))

    def syndrome_errors(self, basis: str, maximum: int) -> dict[int, int]:
        key = basis, maximum
        if key not in self._syndrome_weights:
            self._syndrome_weights[key] = {}
            for representative in self.component_errors(basis, maximum).values():
                self._syndrome_weights[key].setdefault(
                    self.component_syndrome(representative, basis), representative)
        return self._syndrome_weights[key]

    def within_css_filter(self, pauli: int, maximum: int) -> bool:
        """Distance from *some* codeword, as required by EC property A.

        Unlike within_css_bound, an r-filter does not constrain the encoded
        logical state. Logical preservation is checked separately at orders
        within the code's correction radius (EC property B).
        """
        return all(self.component_syndrome(pauli, basis) in self.syndrome_errors(basis, maximum)
                   for basis in ("X", "Z"))

    def syndrome_representative(self, basis: str, syndrome: int, maximum: int) -> int:
        light = self.syndrome_errors(basis, maximum).get(syndrome)
        if light is not None:
            return light
        # Larger corrections can be required for propagated hook errors.
        # Solve H e = syndrome without enumerating all 2**n physical Paulis.
        pivots = {}
        shift = 0 if basis == "X" else self.n
        for q in range(self.n):
            error = 1 << (q + shift)
            column = self.component_syndrome(error, basis)
            for pivot, (row, rep) in sorted(pivots.items(), reverse=True):
                if (column >> pivot) & 1:
                    column ^= row
                    error ^= rep
            if column:
                pivots[column.bit_length() - 1] = column, error
        error = 0
        for pivot, (row, rep) in sorted(pivots.items(), reverse=True):
            if (syndrome >> pivot) & 1:
                syndrome ^= row
                error ^= rep
        if syndrome:
            raise ValueError("Inconsistent CSS syndrome")
        return error

    def pauli_syndrome(self, pauli: int) -> int:
        return (self.component_syndrome(pauli, "X")
                | (self.component_syndrome(pauli, "Z") << len(self._checks["X"])))

    def pauli_syndrome_errors(self, maximum: int) -> dict[int, int]:
        """Joint-weight filter: Y on one qubit has weight one, X_i Z_j two."""
        if maximum not in self._pauli_syndrome_weights:
            result = {}
            for error in self.low_weight(maximum).values():
                result.setdefault(self.pauli_syndrome(error), error)
            self._pauli_syndrome_weights[maximum] = result
        return self._pauli_syndrome_weights[maximum]

    def within_pauli_filter(self, pauli: int, maximum: int) -> bool:
        """Gottesman's standard r-filter: distance from some codeword."""
        return self.pauli_syndrome(pauli) in self.pauli_syndrome_errors(maximum)

    def pauli_syndrome_representative(self, syndrome: int, maximum: int) -> int:
        light = self.pauli_syndrome_errors(maximum).get(syndrome)
        if light is not None:
            return light
        width = len(self._checks["X"])
        return (self.syndrome_representative("X", syndrome & ((1 << width) - 1), maximum)
                ^ self.syndrome_representative("Z", syndrome >> width, maximum))

    def all_syndrome_representatives(self, maximum_count=65536) -> dict[int, int]:
        """Enumerate arbitrary-input syndrome sectors, without logical Paulis.

        EC A depends only on the syndrome. Linear Pauli responses and the ideal
        instrument check ensure that multiplying an input by a logical Pauli
        changes neither the observed branch nor the output syndrome.
        """
        result = {0: 0}
        for q in range(2 * self.n):
            error = 1 << q
            signature = self.pauli_syndrome(error)
            if signature in result:
                continue
            if len(result) * 2 > maximum_count:
                raise NotImplementedError(
                    f"Exact EC A verification supports at most {maximum_count} input syndrome sectors")
            result.update({s ^ signature: self.reduce(p ^ error) for s, p in tuple(result.items())})
        for signature, error in self.pauli_syndrome_errors(2).items():
            result[signature] = self.reduce(error)
        return result


@dataclass(frozen=True)
class Readout:
    flags: tuple[tuple[int, ...], ...]
    syndromes: tuple[tuple[int, ...], ...]

    def remap(self, new_to_old: dict[int, int]) -> Readout:
        old_to_new = {old: new for new, old in new_to_old.items()}
        if len(old_to_new) != len(new_to_old):
            raise ValueError("Optimization duplicated a measurement ID")
        return Readout(*(tuple(tuple(old_to_new[i] for i in parity) for parity in group)
                         for group in (self.flags, self.syndromes)))

    def annotate(self, circuit: stim.Circuit) -> stim.Circuit:
        result = stim.Circuit()
        for op in circuit.flattened():
            if op.name not in ANNOTATIONS:
                result.append(op)
        for kind, group in enumerate((self.flags, self.syndromes)):
            for index, parity in enumerate(group):
                result.append("DETECTOR", [stim.target_rec(i - result.num_measurements)
                                           for i in parity], [kind, index])
        return result

    def decode(self, measurements) -> tuple[int, int]:
        values = []
        for group in (self.flags, self.syndromes):
            value = 0
            for bit, parity in enumerate(group):
                result = 0
                for i in parity:
                    result ^= int(measurements[i])
                value |= result << bit
            values.append(value)
        return values[0], values[1]


def extract_readout(circuit: stim.Circuit, code: CSSCode) -> Readout:
    groups: list[dict[int, tuple[int, ...]]] = [{}, {}]
    count = 0
    for op in circuit.flattened():
        if op.name in {"M", "MX"}:
            count += len(op.targets_copy())
        elif op.name == "DETECTOR":
            coords = op.gate_args_copy()
            if len(coords) != 2 or coords[0] not in (0, 1) or coords[1] != int(coords[1]):
                raise ValueError("SE detectors require coordinates (0, flag_id) or (1, check_id)")
            kind, index = map(int, coords)
            if index < 0 or index in groups[kind]:
                raise ValueError("Duplicate or negative SE detector ID")
            targets = op.targets_copy()
            if any(not t.is_measurement_record_target or not 0 <= count + t.value < count
                   for t in targets):
                raise ValueError("SE detector references an unavailable measurement")
            groups[kind][index] = tuple(count + t.value for t in targets)
    if not any(groups):
        # Unoptimized Steane SE: prep flags first, then n ancilla readouts in
        # code-qubit order. Optimized/reordered circuits must carry detectors.
        flags = circuit.num_measurements - code.n
        if flags < 0:
            raise ValueError("Unannotated SE requires n final ancilla measurements")
        return Readout(tuple((i,) for i in range(flags)),
                       tuple(tuple(flags + int(q) for q in np.flatnonzero(row))
                             for row in code.H_x))
    for group in groups:
        if set(group) != set(range(len(group))):
            raise ValueError("SE detector IDs must be contiguous from zero")
    if len(groups[1]) != len(code.H_x):
        raise ValueError("X-SE must annotate every row of H_x in row order")
    return Readout(*(tuple(group[i] for i in range(len(group))) for group in groups))


def atomic_operations(circuit: stim.Circuit, n: int):
    live = set(range(n))
    measured = set()
    for op in circuit.flattened():
        if op.name in ANNOTATIONS:
            continue
        if op.name not in GATES or op.gate_args_copy():
            raise ValueError(f"Unsupported SE operation: {op}")
        targets = op.targets_copy()
        if any(not t.is_qubit_target or t.is_inverted_result_target for t in targets):
            raise ValueError(f"SE requires ordinary, uninverted qubit targets: {op}")
        width = 2 if op.name in TWO_QUBIT else 1
        if len(targets) % width:
            raise ValueError(f"Invalid target count: {op}")
        for start in range(0, len(targets), width):
            qs = tuple(t.value for t in targets[start:start + width])
            if op.name in {"R", "RX"}:
                if qs[0] < n:
                    raise ValueError("SE must not reset a data qubit")
                if qs[0] in live and qs[0] not in measured:
                    raise ValueError("Ancilla must be measured before starting another reset lifetime")
                live.add(qs[0])
                measured.discard(qs[0])
            else:
                if any(q not in live or q in measured for q in qs):
                    raise ValueError("Ancillas must be reset before use and after measurement")
                if op.name in {"M", "MX"}:
                    if qs[0] < n:
                        raise ValueError("SE must not measure a data qubit")
                    measured.add(qs[0])
            yield op.name, qs
    if live - measured != set(range(n)):
        raise ValueError("Every SE ancilla lifetime must end in measurement")


def dual_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """Conjugate the complete quantum instrument by transversal Hadamards."""
    names = {"R": "RX", "RX": "R", "M": "MX", "MX": "M",
             "X": "Z", "Z": "X", "CZ": "XCX", "XCX": "CZ"}
    result = stim.Circuit()
    for op in circuit.flattened():
        ts = op.targets_copy()
        if op.name == "CX":
            ts = [t for a, b in zip(ts[::2], ts[1::2]) for t in (b, a)]
        result.append(names.get(op.name, op.name), ts, op.gate_args_copy())
    return result


def lower_to_guppy_gates(circuit: stim.Circuit) -> stim.Circuit:
    """Expose the preparation/measurement basis rotations emitted in Guppy.

    Their fault locations must be included in synthesis, not introduced after
    verification. A reset after destructive measurement becomes a fresh qubit
    allocation in the generated program.
    """
    result = stim.Circuit()
    for op in circuit.flattened():
        qs = op.targets_copy()
        if op.name == "RX":
            result.append("R", qs)
            result.append("H", qs)
        elif op.name == "MX":
            result.append("H", qs)
            result.append("M", qs)
        elif op.name == "XCX":
            for a, b in zip(qs[::2], qs[1::2]):
                result.append("H", [a, b])
                result.append("CZ", [a, b])
                result.append("H", [a, b])
        else:
            result.append(op)
    return result


@dataclass(frozen=True)
class CircuitFault:
    operation_offset: int
    pauli_product: tuple[tuple[str, int], ...]
    before_measurement: bool = False


@dataclass
class SEGadget:
    distance: int
    basis: str
    circuit: stim.Circuit
    readout: Readout
    space: PauliSpace = field(repr=False)
    measurement_map: dict[int, int] = field(default_factory=dict)

    @cached_property
    def operations(self):
        return tuple(atomic_operations(self.circuit, self.space.n))

    @property
    def num_flags(self):
        return len(self.readout.flags)

    @property
    def num_detectors(self):
        return self.num_flags + len(self.readout.syndromes)

    def unpack(self, effect: int) -> tuple[int, int, int]:
        return (effect & ((1 << self.num_flags) - 1),
                (effect >> self.num_flags) & ((1 << len(self.readout.syndromes)) - 1),
                effect >> self.num_detectors)

    @cached_property
    def responses(self):
        """Backward Clifford propagation; each site holds exclusive alternatives.

        Model: X/Y/Z after each 1Q Clifford, all 15 nonidentity Paulis
        after each 2Q Clifford, preparation flips after reset, measurement
        record flips. TICK alone does not introduce an idle-noise location.
        """
        n = self.space.n
        nq = self.circuit.num_qubits
        shift = self.num_detectors
        xs = [self.space.reduce(1 << q) << shift if q < n else 0 for q in range(nq)]
        zs = [self.space.reduce(1 << (q + n)) << shift if q < n else 0 for q in range(nq)]
        columns = [0] * self.circuit.num_measurements
        for j, parity in enumerate(self.readout.flags + self.readout.syndromes):
            for i in parity:
                columns[i] ^= 1 << j
        record = len(columns)
        sites = []
        witnesses: dict[int, list[CircuitFault]] = {}
        for offset in reversed(range(len(self.operations))):
            name, qs = self.operations[offset]
            a = qs[0]
            if name in {"M", "MX"}:
                record -= 1
                sites.append((offset, (columns[record],)))
                witnesses.setdefault(columns[record], []).append(CircuitFault(
                    offset, (("X" if name == "M" else "Z", a),), True))
                if name == "M":
                    xs[a] ^= columns[record]
                else:
                    zs[a] ^= columns[record]
            elif name in {"R", "RX"}:
                effect = xs[a] if name == "R" else zs[a]
                sites.append((offset, (effect,)))
                witnesses.setdefault(effect, []).append(CircuitFault(
                    offset, (("X" if name == "R" else "Z", a),)))
                xs[a] = zs[a] = 0
            else:
                choices = [(0, xs[q], zs[q], xs[q] ^ zs[q]) for q in qs]
                # Include an invisible nonidentity fault when distinct physical
                # Paulis have the same response. Zero is not an identity choice.
                actual = set()
                for kinds in itertools.product(range(4), repeat=len(qs)):
                    if any(kinds):
                        effect = _xor(choices[i][kind] for i, kind in enumerate(kinds))
                        actual.add(effect)
                        product = tuple(("IXZY"[kind], q) for q, kind in zip(qs, kinds) if kind)
                        witnesses.setdefault(effect, []).append(CircuitFault(offset, product))
                sites.append((offset, tuple(sorted(actual))))
                if name == "H":
                    xs[a], zs[a] = zs[a], xs[a]
                elif name == "CX":
                    b = qs[1]
                    xs[a] ^= xs[b]
                    zs[b] ^= zs[a]
                elif name == "CZ":
                    b = qs[1]
                    xs[a] ^= zs[b]
                    xs[b] ^= zs[a]
                elif name == "XCX":
                    b = qs[1]
                    zs[a] ^= xs[b]
                    zs[b] ^= xs[a]
        self._fault_witnesses = witnesses
        return tuple(xs[:n] + zs[:n]), tuple(reversed(sites))

    @cached_property
    def input_tables(self):
        columns, _ = self.responses
        tables = []
        for start in range(0, len(columns), 8):
            table = [0] * 256
            for value in range(1, 256):
                low = value & -value
                i = start + low.bit_length() - 1
                table[value] = table[value ^ low] ^ (columns[i] if i < len(columns) else 0)
            tables.append(table)
        return tuple(tables)

    def input_effect(self, error: int) -> int:
        result = 0
        for table in self.input_tables:
            result ^= table[error & 255]
            error >>= 8
        return result

    @cached_property
    def fault_locations(self):
        locations: dict[int, set[int]] = {}
        for site, alternatives in self.responses[1]:
            for effect in alternatives:
                locations.setdefault(effect, set()).add(site)
        return locations

    @cached_property
    def single_fault_effects(self):
        return tuple(sorted(self.fault_locations))

    @cached_property
    def two_fault_effects(self):
        """Exclude pairs of exclusive alternatives at the same physical site."""
        locations = self.fault_locations
        singles = self.single_fault_effects
        pairs = set()
        for i, first in enumerate(singles):
            first_locations = locations[first]
            for second in singles[i:]:
                second_locations = locations[second]
                if len(first_locations) > 1 or len(second_locations) > 1 or first_locations != second_locations:
                    pairs.add(first ^ second)
        return tuple(sorted(pairs))

    def fault_effects(self, order):
        return (0,) if order == 0 else self.single_fault_effects if order == 1 else self.two_fault_effects

    def fault_witness(self, effect: int, order: int) -> tuple[CircuitFault, ...] | None:
        _ = self.responses
        if order == 0:
            return () if effect == 0 else None
        if order == 1:
            entries = self._fault_witnesses.get(effect)
            return (entries[0],) if entries else None
        for first in self.single_fault_effects:
            for a in self._fault_witnesses[first]:
                for b in self._fault_witnesses.get(first ^ effect, ()):
                    if a.operation_offset != b.operation_offset:
                        return a, b
        return None

    def with_faults(self, faults: tuple[CircuitFault, ...]) -> stim.Circuit:
        result = stim.Circuit()
        for offset, (name, qs) in enumerate(self.operations):
            at_site = [f for f in faults if f.operation_offset == offset]
            for fault in at_site:
                if fault.before_measurement:
                    for gate, q in fault.pauli_product:
                        result.append(gate, [q])
            result.append(name, qs)
            for fault in at_site:
                if not fault.before_measurement:
                    for gate, q in fault.pauli_product:
                        result.append(gate, [q])
        return self.readout.annotate(result)


def _xor(values):
    result = 0
    for value in values:
        result ^= value
    return result


def sample_annotation_parities(circuit: stim.Circuit, shots=1, seed=17):
    """Evaluate annotations on raw records, including deterministic sign bits.

    Stim's detector sampler subtracts its own reference sample; a deterministic
    injected Pauli (or a constant raised flag) would therefore be invisible to
    that sampler. Protocol flags and syndromes use raw measurement parities.
    """
    records = circuit.compile_sampler(seed=seed).sample(shots)
    count = 0
    parities = []
    for op in circuit.flattened():
        if op.name in {"M", "MX"}:
            count += len(op.targets_copy())
        elif op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            indices = [count + t.value for t in op.targets_copy()]
            parities.append(records[:, indices].sum(axis=1) % 2)
    return np.asarray(parities, dtype=np.uint8).T


def validate_gadget(gadget: SEGadget) -> None:
    """Check the ideal instrument on an encoded Bell input, not a logical state."""
    from spiderwarp.fault_tolerance_verification import (
        _make_encoded_bell_input, _append_bell_sensitive_measurements,
    )
    code = gadget.space.code
    _ = gadget.operations
    bell, refs, pairing = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z, gadget.circuit.num_qubits)
    for basis, checks, logicals in (("X", code.H_x, code.L_x), ("Z", code.H_z, code.L_z)):
        evaluation = _append_bell_sensitive_measurements(
            bell + gadget.circuit, checks, logicals, basis, refs, pairing)
        evaluation.detector_error_model()
        if sample_annotation_parities(evaluation).any():
            raise ValueError(f"d{gadget.distance} {gadget.basis}-SE changes the ideal encoded Bell state")
    checks = code.H_x if gadget.basis == "X" else code.H_z
    for q in range(2 * code.n):
        error = 1 << q
        flags, syndrome, output = gadget.unpack(gadget.input_effect(error))
        expected = sum(int(row[q % code.n]) << i for i, row in enumerate(checks)) if (
            (gadget.basis == "X") == (q >= code.n)) else 0
        if flags or syndrome != expected or output != gadget.space.reduce(error):
            raise ValueError(f"d{gadget.distance} {gadget.basis}-SE has an incorrect syndrome/data response at Pauli bit {q}")


def optimize_gadget(circuit: stim.Circuit, distance: int, space: PauliSpace,
                    *, verify: bool = False) -> SEGadget:
    from spiderwarp.path_cover import CoveredZXGraph
    from spiderwarp.path_cover_metrics import LexicographicCost, metric_num_paths, metric_depth
    from spiderwarp.resource_scheduling import plan_resource_aware_reuse, ReuseTarget

    readout = extract_readout(circuit, space.code)
    original = SEGadget(distance, "X", readout.annotate(circuit), readout, space)
    if verify:
        validate_gadget(original)
    if any(name in {"X", "Y", "Z"} for name, _ in original.operations):
        raise ValueError("ZX optimization of SE inputs does not support explicit Pauli gates")
    # Expand reused ancilla lifetimes before importing into ZX. PyZX's
    # ancilla labels must identify lifetimes, not recycled physical wires.
    expanded = stim.Circuit()
    physical_to_lifetime = {q: q for q in range(space.n)}
    next_lifetime = space.n
    for name, qs in original.operations:
        if name in {"R", "RX"}:
            physical_to_lifetime[qs[0]] = next_lifetime
            next_lifetime += 1
        expanded.append(name, [physical_to_lifetime[q] for q in qs])
    graph = CoveredZXGraph.from_stim(expanded, num_data_qubits=space.n)
    graph.basic_FE_rewrites()
    ids = set(range(expanded.num_measurements))
    base_cost = LexicographicCost(metric_num_paths, metric_depth)

    def cost(candidate, paths):
        terminals = {candidate.measurement_id(path[-1]) for path in paths.values() if path}
        return base_cost(candidate, paths) if ids <= terminals else float("inf")

    graph = graph.greedy_best_first_boundary_bends(cost, max_evaluations=1000)
    extracted, extraction_map = graph.extract_circuit_with_measurement_map()
    if set(extraction_map.values()) != ids or len(extraction_map) != len(ids):
        raise ValueError("ZX optimization did not preserve every SE measurement")
    reused = plan_resource_aware_reuse(extracted, space.n, heuristic="decross_exact",
                                     max_time_seconds=15.0, target=ReuseTarget.QUBITS)
    mapping = {new: extraction_map[old] for new, old in reused.meas_map.items()}
    result_readout = readout.remap(mapping)
    result = SEGadget(distance, "X", lower_to_guppy_gates(result_readout.annotate(reused.circuit)),
                      result_readout, space, mapping)
    if verify:
        validate_gadget(result)
    return result

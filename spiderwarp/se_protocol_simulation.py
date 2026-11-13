"""Monte Carlo block LER for the adaptive SE policy exported to Guppy/HUGR.

Execution uses CompiledSEProtocol.run and Stim, preserving the verified gate
schedule. This is a reference simulation, not a simulation of backend-specific
HUGR lowering. Ideal logical 0/+ preparation and final decoding surround one
noisy SE cycle. Each choice tests the corresponding logical Pauli component.
"""

from __future__ import annotations

import math
from collections import Counter
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.lut_decoder import LutDecoder
from spiderwarp.noise_model import CustomErrorModel, apply_noise, default_noise_model

if TYPE_CHECKING:
    from spiderwarp.optimized_steane_se import CompiledSEProtocol


Stage = tuple[str, str, int]  # role, basis, circuit distance


@dataclass(frozen=True)
class ProtocolDiagnostics:
    """Aggregate executed paths and rejection causes; no shot records retained.

    Stage denominators count executions, whereas rejections count shots. The
    stage attached to a rejection is the last extraction that actually ran.
    Primary flags can lead to accepted recovery and are not themselves rejects.
    accepted_cnots and accepted_stage_counts exclude every part of a trajectory
    that rejects, even if its earlier cycles completed successfully.
    """

    stage_counts: dict[Stage, int] = field(default_factory=dict)
    flagged_stage_counts: dict[Stage, int] = field(default_factory=dict)
    rejections: dict[tuple[str, str, int, str], int] = field(default_factory=dict)
    paths: dict[tuple[Stage, ...], int] = field(default_factory=dict)
    rejection_cycles: dict[int, int] = field(default_factory=dict)
    cycles_started: int = 0
    cycles_completed: int = 0
    total_cnots: int = 0
    accepted_cnots: int = 0
    accepted_stage_counts: dict[Stage, int] = field(default_factory=dict)

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts = Counter()
        for (*_, reason), count in self.rejections.items():
            counts[reason] += count
        return dict(counts)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if not trials:
        return 0.0, 1.0
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class LogicalErrorRateEstimate:
    """A block fails if any logical 0/+ output has the wrong Z/X eigenvalue.

    Discards are reported separately and excluded from the LER denominator.
    ler is NaN if no shots accept; its interval is then the uninformative [0, 1].
    """

    shots: int
    accepted: int
    logical_errors: int
    noise_model: CustomErrorModel
    seed: int
    prepared_state: Literal["0", "+"] = "0"
    diagnostics: ProtocolDiagnostics | None = None
    cycles: int = 1

    @property
    def discarded(self) -> int:
        return self.shots - self.accepted

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.shots

    @property
    def discard_rate(self) -> float:
        return self.discarded / self.shots

    @property
    def discard_interval_95(self) -> tuple[float, float]:
        return _wilson_interval(self.discarded, self.shots)

    @property
    def ler(self) -> float:
        return self.logical_errors / self.accepted if self.accepted else math.nan

    @property
    def ler_interval_95(self) -> tuple[float, float]:
        """Two-sided 95% Wilson interval, including zero observed failures."""
        return _wilson_interval(self.logical_errors, self.accepted)


def _noisy_gadget(ideal: stim.Circuit, noise: CustomErrorModel, *, persistent_data=0) -> stim.Circuit:
    """Use the shared model on the lowered quantum circuit (readout is external)."""
    circuit = stim.Circuit()
    for operation in ideal.flattened():
        if operation.name not in {"DETECTOR", "QUBIT_COORDS", "TICK"}:
            circuit.append(operation)
    if persistent_data:
        return apply_noise(circuit, noise, persistent_qubits=range(persistent_data))
    return apply_noise(circuit, noise)


class _NoisySimulator:
    """Inject noise while letting the existing policy choose every branch."""

    def __init__(self, simulator, circuits, noise, corrections):
        self.simulator, self.circuits, self.noise = simulator, circuits, noise
        self.corrections = corrections

    def current_measurement_record(self):
        return self.simulator.current_measurement_record()

    def do_circuit(self, circuit):
        self.simulator.do_circuit(self.circuits[id(circuit)])

    def do_correction(self, circuit):
        key = str(circuit)
        if key not in self.corrections:
            self.corrections[key] = apply_noise(circuit, self.noise)
        self.simulator.do_circuit(self.corrections[key])


def _observable(size, row, basis):
    observable = stim.PauliString(size)
    for qubit in np.flatnonzero(row):
        observable[int(qubit)] = basis
    return observable


class _StateExperiment:
    def __init__(self, code: CSSCode, size: int, prepared_state: str):
        self.code = code
        self.prepared_state = prepared_state
        if prepared_state not in {"0", "+"}:
            raise ValueError("prepared_state must be '0' or '+'")
        basis = "Z" if prepared_state == "0" else "X"
        checks, logicals = ((code.H_z, code.L_z) if basis == "Z"
                           else (code.H_x, code.L_x))
        self.checks = [_observable(size, row, basis) for row in checks]
        self.logicals = [_observable(size, row, basis) for row in logicals]
        stabilizers = ([_observable(size, row, "X") for row in code.H_x]
                       + [_observable(size, row, "Z") for row in code.H_z]
                       + self.logicals)
        for qubit in range(code.n, size):
            ancilla = stim.PauliString(size)
            ancilla[qubit] = "Z"
            stabilizers.append(ancilla)
        tableau = stim.Tableau.from_stabilizers(
            stabilizers, allow_redundant=True, allow_underconstrained=False)
        self.initial = stim.TableauSimulator()
        self.initial.set_inverse_tableau(tableau.inverse())
        self.logical_rows = logicals
        # Unbounded minimum-weight decoding covers every reachable syndrome,
        # including those outside the SE gadget's guaranteed correction radius.
        self.decoder = LutDecoder(checks)

    @staticmethod
    def _bits(simulator, observables):
        expectations = [simulator.peek_observable_expectation(p) for p in observables]
        if any(e == 0 for e in expectations):
            raise ValueError("SE output lost a deterministic code/logical observable")
        return np.array([e == -1 for e in expectations], dtype=np.uint8)

    def logical_failure(self, simulator, *, frame_x=0, frame_z=0) -> bool:
        frame_mask = frame_x if self.prepared_state == "0" else frame_z
        frame = np.array([(frame_mask >> q) & 1 for q in range(self.code.n)], dtype=np.uint8)
        checks = self.code.H_z if self.prepared_state == "0" else self.code.H_x
        correction = self.decoder.get_mwr(self._bits(simulator, self.checks) ^ (checks @ frame % 2))
        logical_bits = self._bits(simulator, self.logicals) ^ (self.logical_rows @ (correction ^ frame) % 2)
        return bool(np.any(logical_bits))


def scaled_noise_model(p: float) -> CustomErrorModel:
    """Set p_2q=p, scaling all other default hardware rates proportionally."""
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be finite and between 0 and 1")
    model = default_noise_model()
    scale = p / model.p_2q
    return CustomErrorModel(**{k: v * scale for k, v in vars(model).items()})


class ProtocolSampler:
    """Reusable shot-by-shot adaptive Stim reference; no compilation per batch.

    The new experiment workflow includes data idles before the first gate of
    each extraction. Set include_initial_idles=False to reproduce the historical
    one-cycle model. Classical branch latency and inter-cycle delay are zero.
    """

    def __init__(self, protocol, *, noise_model=None, prepared_state="0", cycles=1,
                 include_initial_idles=True, collect_paths=True):
        if type(cycles) is not int or cycles < 0:
            raise ValueError("cycles must be a nonnegative integer")
        if prepared_state not in {"0", "+"}:
            raise ValueError("prepared_state must be '0' or '+'")
        if cycles > 1 and not hasattr(protocol, "policy"):
            raise ValueError("Repeated cycles require an adaptive Pauli-frame protocol")
        noise_model = default_noise_model() if noise_model is None else copy(noise_model)
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in vars(noise_model).values()):
            raise ValueError("Noise probabilities must be finite and between 0 and 1")
        self.protocol, self.noise = protocol, noise_model
        self.cycles, self.prepared_state, self.collect_paths = cycles, prepared_state, collect_paths
        self.circuits, self.costs = {}, {}
        for name, g in protocol.gadgets.items():
            self.circuits[id(g.circuit)] = _noisy_gadget(
                g.circuit, noise_model,
                persistent_data=protocol.code.n if include_initial_idles else 0)
            cost = sum(len(op.targets_copy()) // 2 for op in g.circuit.flattened()
                       if op.name in {"CX", "CZ", "XCX"})
            self.costs[name] = cost
            self.costs[g.basis, g.distance] = cost
        self.experiment = _StateExperiment(protocol.code, protocol.required_qubits, prepared_state)
        self.corrections = {}

    def sample(self, shots=1000, seed=0):
        if type(shots) is not int or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        accepted = failures = started = completed = cnots = accepted_cnots = 0
        accepted_stages = Counter()
        stages, flagged, rejections, paths, rejection_cycles = (Counter() for _ in range(5))
        for _ in range(shots):
            simulator = self.experiment.initial.copy(seed=int(rng.integers(0, 2**63)))
            noisy = _NoisySimulator(simulator, self.circuits, self.noise, self.corrections)
            frame_x = frame_z = 0
            shot_path = []
            shot_cnots = 0
            shot_stages = Counter()
            survives = True
            for cycle in range(1, self.cycles + 1):
                started += 1
                # Preserve the legacy one-cycle API; later calls must carry the
                # cumulative frame returned by the previous call, not XOR it twice.
                result = (self.protocol.run(noisy) if cycle == 1 else
                          self.protocol.run(noisy, frame_x=frame_x, frame_z=frame_z))
                path = tuple((role, basis, distance)
                             for role, basis, distance, _, _ in result.measurements)
                stages.update(path)
                flagged.update((role, basis, distance)
                               for role, basis, distance, flags, _ in result.measurements if flags)
                cost = sum(self.costs.get(role, self.costs.get((basis, distance), 0))
                           for role, basis, distance in path)
                cnots += cost
                shot_cnots += cost
                shot_stages.update(path)
                if self.collect_paths:
                    shot_path.extend(path)
                if not result.accepted:
                    survives = False
                    rejection_cycles[cycle] += 1
                    stage = path[-1] if path else ("controller", "", 0)
                    rejections[(*stage, result.discard_reason)] += 1
                    break
                completed += 1
                frame_x, frame_z = result.frame_x, result.frame_z
            if self.collect_paths:
                paths[tuple(shot_path)] += 1
            if survives:
                accepted += 1
                accepted_cnots += shot_cnots
                accepted_stages.update(shot_stages)
                failures += self.experiment.logical_failure(simulator, frame_x=frame_x, frame_z=frame_z)
        return LogicalErrorRateEstimate(
            shots, accepted, failures, copy(self.noise), seed, self.prepared_state,
            ProtocolDiagnostics(dict(stages), dict(flagged), dict(rejections), dict(paths),
                                dict(rejection_cycles), started, completed, cnots,
                                accepted_cnots, dict(accepted_stages)), self.cycles)


def estimate_protocol_ler(
    protocol: CompiledSEProtocol,
    *,
    noise_model: CustomErrorModel | None = None,
    prepared_state: Literal["0", "+"] = "0",
    shots: int = 1000,
    seed: int = 0,
    cycles: int = 1,
    include_initial_idles: bool = False,
) -> LogicalErrorRateEstimate:
    """Run noisy adaptive SE cycles per fresh logical 0/+ input.

    All k logical qubits start in the selected state. For 0 inputs the block
    LER detects logical X/Y errors through logical Z readout; for + inputs it
    detects logical Z/Y errors through logical X readout. These are distinct
    basis-specific rates, not a combined arbitrary-state logical error rate.

    With no noise_model argument, use noise_model.py's default hardware model.
    Its apply_noise function handles each executed extraction/correction block,
    including its scheduled memory noise. Classical branch latency is not
    modeled. Input preparation and the final LutDecoder are noiseless.

    This executes the same branches/corrections as the Guppy/HUGR export through
    its Stim reference interpreter; it does not execute the HUGR binary itself.
    Seeds reproduce results for a fixed Stim version and machine architecture.

    Input preparation occurs once and ideal decoding occurs only at the end.
    Rejections terminate the trajectory without retrying. For backward
    compatibility initial data idles default off in this function; the new CLI
    and ProtocolSampler default them on and identify this model separately.
    """
    return ProtocolSampler(protocol, noise_model=noise_model, prepared_state=prepared_state,
                           cycles=cycles, include_initial_idles=include_initial_idles).sample(shots, seed)

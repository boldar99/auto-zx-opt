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
    """

    stage_counts: dict[Stage, int] = field(default_factory=dict)
    flagged_stage_counts: dict[Stage, int] = field(default_factory=dict)
    rejections: dict[tuple[str, str, int, str], int] = field(default_factory=dict)
    paths: dict[tuple[Stage, ...], int] = field(default_factory=dict)

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


def _noisy_gadget(ideal: stim.Circuit, noise: CustomErrorModel) -> stim.Circuit:
    """Use the shared model on the lowered quantum circuit (readout is external)."""
    circuit = stim.Circuit()
    for operation in ideal.flattened():
        if operation.name not in {"DETECTOR", "QUBIT_COORDS", "TICK"}:
            circuit.append(operation)
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

    def logical_failure(self, simulator) -> bool:
        correction = self.decoder.get_mwr(self._bits(simulator, self.checks))
        logical_bits = self._bits(simulator, self.logicals) ^ (self.logical_rows @ correction % 2)
        return bool(np.any(logical_bits))


def estimate_protocol_ler(
    protocol: CompiledSEProtocol,
    *,
    noise_model: CustomErrorModel | None = None,
    prepared_state: Literal["0", "+"] = "0",
    shots: int = 1000,
    seed: int = 0,
) -> LogicalErrorRateEstimate:
    """Run one noisy adaptive SE cycle per fresh logical 0/+ input.

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
    """
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if prepared_state not in {"0", "+"}:
        raise ValueError("prepared_state must be '0' or '+'")
    noise_model = default_noise_model() if noise_model is None else noise_model
    probabilities = (noise_model.p_1q, noise_model.p_2q, noise_model.p_init,
                     noise_model.p_meas, noise_model.p_linear_mem,
                     noise_model.p_coherent_mem, noise_model.p_leakage)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Noise probabilities must be finite and between 0 and 1")
    circuits = {id(g.circuit): _noisy_gadget(g.circuit, noise_model)
                for g in protocol.gadgets.values()}
    experiment = _StateExperiment(protocol.code, protocol.required_qubits, prepared_state)
    rng = np.random.default_rng(seed)
    accepted = failures = 0
    stages, flagged, rejections, paths = Counter(), Counter(), Counter(), Counter()
    corrections = {}
    for _ in range(shots):
        simulator = experiment.initial.copy(seed=int(rng.integers(0, 2**63)))
        result = protocol.run(_NoisySimulator(simulator, circuits, noise_model, corrections))
        path = tuple((role, basis, distance) for role, basis, distance, _, _ in result.measurements)
        stages.update(path)
        flagged.update((role, basis, distance) for role, basis, distance, flags, _
                       in result.measurements if flags)
        paths[path] += 1
        if result.accepted:
            accepted += 1
            failures += experiment.logical_failure(simulator)
        else:
            rejections[(*path[-1], result.discard_reason)] += 1
    return LogicalErrorRateEstimate(
        shots, accepted, failures, copy(noise_model), seed, prepared_state,
        ProtocolDiagnostics(dict(stages), dict(flagged), dict(rejections), dict(paths)))

"""Build, verify, execute and export an adaptive optimized Steane SE cycle.

Inputs are canonical X-stabilizer SE circuits keyed by their circuit distance.
Annotated inputs use DETECTOR(0, flag_id) and DETECTOR(1, H_x_row). Without
annotations the convention is prep flags followed by n code-ordered readouts.
The current exact mixed-Pauli enumerator supports code distances 2 through 5.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Iterable

import numpy as np
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp._se_circuit import (
    PauliSpace, SEGadget, dual_circuit, optimize_gadget, validate_gadget, lower_to_guppy_gates,
)
from spiderwarp._se_policy import (
    PolicyBuilder, ProtocolVerificationError, ProtocolWitness, RecoveryPlan,
    VerificationReport,
)

if TYPE_CHECKING:
    from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol
    from guppylang.defs import GuppyFunctionDefinition
    from hugr.package import Package


@dataclass(frozen=True)
class ProtocolResult:
    accepted: bool
    measurements: tuple[tuple[str, str, int, int, int], ...]
    corrections: tuple[tuple[tuple[str, int], ...], ...]
    discard_reason: str | None = None
    frame_x: int = 0
    frame_z: int = 0


@dataclass
class CompiledSEProtocol:
    code: CSSCode
    gadgets: dict[tuple[str, int], SEGadget]
    standard_corrections: dict[str, dict[int, int]]
    recoveries: dict[tuple[str, int, int], RecoveryPlan]
    verification: VerificationReport
    source: str
    source_path: Path
    guppy_function: GuppyFunctionDefinition
    package: Package
    package_path: Path | None = None
    _module: object = field(default=None, repr=False)
    recovery_objective: str = 'qubits'

    @property
    def required_qubits(self) -> int:
        return max(g.circuit.num_qubits for g in self.gadgets.values())

    def save(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Save source and a composable HUGR function (no decoder sidecar)."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "steane_se_protocol.py"
        package = directory / "steane_se_protocol.hugr"
        source.write_text(self.source)
        package.write_bytes(self.package.to_bytes())
        self.source_path, self.package_path = source, package
        return source, package

    def verify(self) -> VerificationReport:
        """Optionally verify the current gadgets and policy without rebuilding."""
        return verify_se_protocol(self)

    def with_recovery_objective(
        self, objective: str, *, verify: bool = False, output_dir: str | Path | None = None,
    ) -> CompiledSEProtocol:
        """Rebuild the policy/export on these exact circuits, without ZX changes.

        'qubits' minimizes recovery width; 'gates' prioritizes operation count.
        Both use the same fault contract, but can trade discard rate against
        logical error rate outside that contract's bounded fault orders.
        """
        space = next(iter(self.gadgets.values())).space
        builder = PolicyBuilder(space, self.gadgets, recovery_objective=objective)
        if verify:
            for gadget in self.gadgets.values():
                validate_gadget(gadget)
        return _compile_policy(self.code, self.gadgets, builder.build(verify=verify), output_dir)

    def run(self, simulator: stim.TableauSimulator) -> ProtocolResult:
        """Execute the same policy directly on a live Stim data block.

        A rejected block is left in the simulator for inspection; callers must
        honor ``accepted``. Accepted data qubits are never measured or reset.
        """
        measurements = []
        corrections = []
        space = next(iter(self.gadgets.values())).space

        def run_gadget(basis, distance, role):
            gadget = self.gadgets[basis, distance]
            start = len(simulator.current_measurement_record())
            simulator.do_circuit(gadget.circuit)
            flags, syndrome = gadget.readout.decode(simulator.current_measurement_record()[start:])
            measurements.append((role, basis, distance, flags, syndrome))
            return flags, syndrome

        def correct(pauli):
            product = space.product(pauli)
            correction = stim.Circuit()
            for q in range(self.code.n):
                if (pauli >> q) & 1:
                    correction.append("X", [q])
            for q in range(self.code.n):
                if (pauli >> (q + self.code.n)) & 1:
                    correction.append("Z", [q])
            handler = getattr(simulator, "do_correction", None)
            if handler is not None:
                handler(correction)
            else:
                for op in correction:
                    for target in op.targets_copy():
                        getattr(simulator, op.name.lower())(target.value)
            corrections.append(product)

        def result(accepted, reason=None):
            return ProtocolResult(accepted, tuple(measurements), tuple(corrections), reason)

        previous = -1
        for basis in ("Z", "X"):
            flags, syndrome = run_gadget(basis, self.code.d, "primary")
            if flags:
                plan = self.recoveries.get((basis, flags, previous))
                if plan is None:
                    return result(False, "unmodeled primary flags")
                dual = "Z" if basis == "X" else "X"
                recovery_flags, recovery_syndrome = run_gadget(dual, plan.distance, "recovery")
                if recovery_flags:
                    return result(False, "raised recovery flags")
                if plan.cleanup_distance is not None:
                    cleanup_flags, cleanup_syndrome = run_gadget(
                        basis, plan.cleanup_distance, "recovery_cleanup")
                    if cleanup_flags:
                        return result(False, "raised recovery cleanup flags")
                    width = len(self.gadgets[dual, plan.distance].readout.syndromes)
                    recovery_syndrome |= cleanup_syndrome << width
                correction = plan.corrections.get(recovery_syndrome)
                if correction is None:
                    return result(False, "ambiguous recovery syndrome")
                correct(correction)
                return result(True)
            correction = self.standard_corrections[basis].get(syndrome)
            if correction is None:
                return result(False, "uncorrectable primary syndrome")
            correct(correction)
            previous = syndrome
        return result(True)


def optimized_steane_SE_orchestrator(
    code: CSSCode,
    circuits: Mapping[int, stim.Circuit] | Iterable[tuple[int, stim.Circuit]],
    *,
    output_dir: str | Path | None = None,
    verify: bool = False,
    recovery_objective: str | None = None,
    policy_mode: str = "adaptive",
    optimize: bool = False,
    contract=None,
    peak_qubits: int | None = None,
    max_cnot_cost: int | None = None,
    minimize: bool = True,
    max_search_states: int = 100000,
    export: bool = True,
    progress=None,
) -> CompiledSEProtocol | AdaptiveSEProtocol:
    """Build a certified minimax adaptive cycle on finalized SE circuits.

    The adaptive mode retains every flag/syndrome, returns a terminal Pauli
    frame, and minimizes worst-case CNOT count under peak_qubits. It always
    independently verifies the policy. Set optimize=True to run ZX optimization
    before synthesis; set export=False to use Stim without installing Guppy.

    policy_mode="legacy" selects the old fixed, physically corrected cycle
    below. Its verification remains optional and controlled by verify.

    Every adaptive accepted history obeys the internal-fault-count CSS filter.
    EC A covers arbitrary incoming syndrome sectors through t internal faults;
    EC B preserves the ideal-decoded logical state through incoming plus
    internal order t. Clean inputs must also accept through floor(d/2), without
    a logical guarantee beyond t. The exact model includes joint physical
    Pauli alternatives, including all 15 alternatives per two-qubit gate.

    The legacy controller discards flagged syndromes, applies physical
    corrections, and chooses local recovery circuits by recovery_objective
    ('qubits' or 'gates'). Only that mode permits unverified export.

    Guppy is optional: install spiderwarp[guppy] for export. Without an
    output_dir, generated source is kept in a temporary directory.
    """
    if policy_mode not in {"adaptive", "legacy"}:
        raise ValueError("policy_mode must be 'adaptive' or 'legacy'")
    if policy_mode == "adaptive":
        if recovery_objective not in {None, "cnots"}:
            raise ValueError("Adaptive synthesis minimizes CNOTs; use peak_qubits for its width limit")
        from spiderwarp.adaptive_steane_se import build_adaptive_se_protocol
        items = list(circuits.items() if isinstance(circuits, Mapping) else circuits)
        if not items or len({d for d, _ in items}) != len(items):
            raise ValueError("Supply unique circuit distances")
        if any(type(d) is not int or d < 1 or not isinstance(c, stim.Circuit) for d, c in items):
            raise ValueError("circuits must contain positive integer distances and Stim circuits")
        if code.d not in dict(items):
            raise ValueError(f"Missing primary circuit at code distance {code.d}")
        return build_adaptive_se_protocol(
            code, dict(items), contract=contract, optimize=optimize, peak_qubits=peak_qubits,
            max_cnot_cost=max_cnot_cost, minimize=minimize, max_search_states=max_search_states,
            export=export, output_dir=output_dir, progress=progress)
    recovery_objective = recovery_objective or "qubits"
    if importlib.util.find_spec("guppylang") is None:
        raise ImportError("Guppy export requires guppylang; install spiderwarp[guppy]")
    if recovery_objective not in {'qubits', 'gates'}:
        raise ValueError("recovery_objective must be 'qubits' or 'gates'")
    items = list(circuits.items() if isinstance(circuits, Mapping) else circuits)
    if not items or len({d for d, _ in items}) != len(items):
        raise ValueError("Supply unique circuit distances")
    if any(type(d) is not int or d < 1 or not isinstance(c, stim.Circuit) for d, c in items):
        raise ValueError("circuits must contain positive integer distances and Stim circuits")
    if code.d not in dict(items):
        raise ValueError(f"Missing primary circuit at code distance {code.d}")
    if not 2 <= code.d <= 5:
        raise NotImplementedError("Exact adaptive synthesis currently supports code distances 2 through 5")
    if not np.array_equal(code.H_x, code.H_z):
        raise ValueError("Canonical X-only inputs currently require identical H_x and H_z (self-dual CSS checks)")
    space = PauliSpace(code)
    gadgets = {}
    for distance, circuit in sorted(items):
        x = optimize_gadget(circuit.copy(), distance, space, verify=verify)
        z = SEGadget(distance, "Z", lower_to_guppy_gates(dual_circuit(x.circuit)),
                     x.readout, space, x.measurement_map.copy())
        if verify:
            validate_gadget(z)
        gadgets["X", distance], gadgets["Z", distance] = x, z
    policy = PolicyBuilder(space, gadgets, recovery_objective=recovery_objective).build(verify=verify)
    return _compile_policy(code, gadgets, policy, output_dir)


def _compile_policy(code, gadgets, policy, output_dir):
    from spiderwarp._se_guppy import compile_protocol
    source, path, module, function, package = compile_protocol(code, gadgets, policy)
    result = CompiledSEProtocol(code, gadgets, policy.standard, policy.recoveries,
                               policy.report, source, path, function, package, _module=module,
                               recovery_objective=policy.recovery_objective)
    if output_dir is not None:
        result.save(output_dir)
    return result


def verify_se_protocol(protocol: CompiledSEProtocol) -> VerificationReport:
    """Verify current gadgets and decisions, without resynthesizing the policy.

    Raises ProtocolVerificationError (or ValueError for a malformed gadget).
    A failed check clears any previous successful verification report. The
    report covers the circuit-level Pauli model described in the report; it
    does not certify a stochastic hardware/idle noise model used for sampling.
    """
    from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol
    if isinstance(protocol, AdaptiveSEProtocol):
        return protocol.verify()
    protocol.verification = VerificationReport(
        (protocol.code.d - 1) // 2, protocol.code.d // 2)
    space = PauliSpace(protocol.code)
    gadgets = {}
    for key, original in protocol.gadgets.items():
        gadget = SEGadget(original.distance, original.basis, original.circuit,
                         original.readout, space, original.measurement_map.copy())
        validate_gadget(gadget)
        gadgets[key] = gadget
    checker = PolicyBuilder(space, gadgets)
    try:
        return checker.verify(protocol.standard_corrections, protocol.recoveries)
    finally:
        protocol.verification = checker.report


# Conventional lowercase spelling is also available.
optimized_steane_se_orchestrator = optimized_steane_SE_orchestrator

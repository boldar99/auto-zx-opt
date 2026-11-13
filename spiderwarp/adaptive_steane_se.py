"""Build and execute certified adaptive CSS Steane cycles.

Circuit optimization, offline policy synthesis, and optional Guppy compilation
are separate operations. Execution returns a classical Pauli frame and never
applies noisy physical correction gates to the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import stim

from spiderwarp._se_adaptive import (
    AdaptivePolicy, AdaptivePolicyBuilder, AdaptiveVerificationError, FTContract,
    FTContractInfeasibleError, PolicySearchError, verify_adaptive_policy,
)
from spiderwarp._se_circuit import (
    PauliSpace, SEGadget, dual_circuit, extract_readout, lower_to_guppy_gates,
    optimize_gadget,
)


@dataclass
class AdaptiveSEProtocol:
    code: object
    policy: AdaptivePolicy
    source: str = ""
    source_path: Path | None = None
    package_path: Path | None = None
    package: object = None
    guppy_function: object = None
    _module: object = None

    @property
    def verification(self):
        return self.policy.report

    @property
    def gadgets(self):
        return {name: action.gadget for name, action in self.policy.actions.items()}

    @property
    def required_qubits(self):
        return self.policy.report.peak_qubits

    def verify(self):
        return verify_adaptive_policy(self.policy)

    def save_bundle(self, output_dir):
        """Save the executable protocol without Guppy compilation."""
        from spiderwarp.se_protocol_io import save_protocol
        return save_protocol(self, output_dir)

    @classmethod
    def load(cls, path, *, verify=False):
        from spiderwarp.se_protocol_io import load_protocol
        return load_protocol(path, verify=verify)

    def export_guppy(self, output_dir):
        """Write standalone source without importing Guppy or compiling it."""
        from spiderwarp._se_adaptive_guppy import generate_adaptive_source
        from spiderwarp.se_protocol_io import atomic_write
        self.source = generate_adaptive_source(self)
        self.source_path = Path(output_dir).expanduser().resolve() / "steane_se_protocol.py"
        atomic_write(self.source_path, self.source)
        return self.source_path

    def run(self, simulator, *, frame_x=0, frame_z=0):
        """Run on an existing Stim block and return acceptance plus a frame.

        The data remain in the simulator's *physical* frame. Composers must
        retain frame_x/frame_z; supplying an existing frame makes the returned
        frame include it. The Monte Carlo helper incorporates the frame in
        ideal final decoding without adding physical correction gates.
        """
        from spiderwarp.optimized_steane_se import ProtocolResult

        measurements = []
        if any(type(value) is not int or value < 0 or value >> self.code.n
               for value in (frame_x, frame_z)):
            raise ValueError("Input Pauli-frame masks must fit the data block")
        frame = frame_x | (frame_z << self.code.n)
        space = self.policy.decoder.space
        index = self.policy.root
        while True:
            node = self.policy.nodes[index]
            if node.kind == "reject":
                return ProtocolResult(False, tuple(measurements), (), "certified rejection")
            if node.kind == "accept":
                mask = (1 << self.code.n) - 1
                correction = node.frame ^ frame
                return ProtocolResult(True, tuple(measurements), (),
                                      frame_x=correction & mask, frame_z=correction >> self.code.n)
            if node.kind == "frame":
                frame ^= node.frame
                index = node.next
                continue
            if node.kind != "extract" or node.action not in self.policy.actions:
                raise ValueError("Malformed adaptive policy")
            g = self.policy.actions[node.action].gadget
            start = len(simulator.current_measurement_record())
            simulator.do_circuit(g.circuit)
            flags, syndrome = g.readout.decode(simulator.current_measurement_record()[start:])
            measurements.append((node.action, g.basis, g.distance, flags, syndrome))
            correction_syndrome = space.component_syndrome(frame, "X" if g.basis == "Z" else "Z")
            index = node.children.get((flags, syndrome ^ correction_syndrome), 0)

    def export(self, output_dir=None):
        """Verify and compile a standalone Guppy program with a frame result."""
        self.verify()
        from spiderwarp._se_adaptive_guppy import compile_adaptive_protocol
        self.source, self.source_path, self._module, self.guppy_function, self.package = (
            compile_adaptive_protocol(self))
        if output_dir is not None:
            self.save(output_dir)
        return self

    def save(self, output_dir):
        if self.package is None:
            self.export()
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.source_path = directory / "steane_se_protocol.py"
        self.package_path = directory / "steane_se_protocol.hugr"
        self.source_path.write_text(self.source)
        self.package_path.write_bytes(self.package.to_bytes())
        circuits = []
        for index, (name, action) in enumerate(sorted(self.policy.actions.items())):
            path = directory / f"action_{index}.stim"
            path.write_text(str(action.gadget.circuit))
            circuits.append(dict(name=name, basis=action.gadget.basis,
                                 distance=action.gadget.distance, qubits=action.qubits,
                                 cnot_cost=action.cnot_cost, path=path.name,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        certificate = dict(
            code=dict(n=self.code.n, k=self.code.k, d=self.code.d),
            verification=asdict(self.verification), minimax_optimal=self.policy.optimal,
            search_states=self.policy.search_states,
            search_limit_reached=self.policy.search_limit_reached,
            peak_qubit_limit=self.policy.peak_qubit_limit, circuits=circuits,
            source_sha256=hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            hugr_sha256=hashlib.sha256(self.package_path.read_bytes()).hexdigest(),
        )
        (directory / "verification.json").write_text(json.dumps(certificate, indent=2) + "\n")
        self.save_bundle(directory)
        return self.source_path, self.package_path


def build_adaptive_se_protocol(
    code, circuits, *, contract: FTContract | None = None, optimize: bool = False,
    first_action: str | None = None, peak_qubits: int | None = None,
    max_cnot_cost: int | None = None, minimize: bool = True,
    max_search_states: int = 100000, export: bool = False, output_dir=None,
    progress=None, paired: bool = False,
) -> AdaptiveSEProtocol:
    """Synthesize and independently verify a fixed-library minimax policy.

    Integer keys label canonical X-SE circuits; Z-SE is derived for identical
    H_x/H_z checks. Alternatively pass named SEGadgets, including multiple
    variants at the same distance. Input circuits are finalized by default.
    Guppy is needed only when export=True. With export=False, output_dir saves
    a reloadable simulation bundle without compilation.

    The default contract requires acceptance/logical correctness through t,
    clean-input acceptance through floor(d/2), and accepted CSS s-filters.
    FTContract.correct_or_reject additionally requests accepted logical
    correctness through floor(d/2); silent terminal faults can produce a concrete
    FTContractInfeasibleError for that stronger even-distance target.
    """
    items = list(circuits.items())
    if not items or len({key for key, _ in items}) != len(items):
        raise ValueError("Supply unique extraction circuit keys")
    space = PauliSpace(code)
    actions = {}
    if all(type(key) is int and key > 0 and isinstance(c, stim.Circuit) for key, c in items):
        if not np.array_equal(code.H_x, code.H_z):
            raise ValueError("Canonical X-only inputs require identical H_x/H_z; supply named gadgets")
        for distance, circuit in sorted(items):
            readout = extract_readout(circuit, code)
            if optimize:
                x = optimize_gadget(circuit.copy(), distance, space, verify=True)
                z_circuit = lower_to_guppy_gates(dual_circuit(x.circuit))
            else:
                x = SEGadget(distance, "X", lower_to_guppy_gates(readout.annotate(circuit)), readout, space)
                z_circuit = lower_to_guppy_gates(readout.annotate(dual_circuit(circuit)))
            z = SEGadget(distance, "Z", z_circuit, x.readout, space)
            actions[f"X_d{distance}"] = x
            actions[f"Z_d{distance}"] = z
        first_action = first_action or f"Z_d{code.d}"
    elif all(isinstance(key, str) and isinstance(g, SEGadget) for key, g in items):
        if optimize:
            raise ValueError("Optimize named variants before supplying them to policy synthesis")
        for name, g in items:
            if (g.space.code.n != code.n or not np.array_equal(g.space.code.H_x, code.H_x)
                    or not np.array_equal(g.space.code.H_z, code.H_z)):
                raise ValueError("Named extraction uses a different code")
            actions[name] = SEGadget(g.distance, g.basis, lower_to_guppy_gates(g.circuit), g.readout, space,
                                     g.measurement_map.copy())
        if first_action is None:
            candidates = [name for name, g in actions.items() if g.basis == "Z" and g.distance == code.d]
            if len(candidates) != 1:
                raise ValueError("Specify first_action for a library with multiple initial variants")
            first_action = candidates[0]
    else:
        raise ValueError("Use integer-labelled Stim circuits or named SEGadgets")
    if paired and contract is None:
        # Order two is analyzed for detection. Correction/residual guarantees
        # cover total order one; hidden order-two cases are not overconstrained.
        from dataclasses import replace
        contract = replace(FTContract.for_code(code), correctable=1,
                           accept_clean_through=1, accepted_filter_order=1)
    builder = AdaptivePolicyBuilder(space, actions, contract=contract, peak_qubits=peak_qubits,
                                    max_search_states=max_search_states, progress=progress)
    policy = builder.build(first_action=first_action, max_cnot_cost=max_cnot_cost,
                           minimize=minimize, verify=True, paired=paired)
    result = AdaptiveSEProtocol(code, policy)
    if export:
        result.export(output_dir)
    elif output_dir is not None:
        result.save_bundle(output_dir)
    return result

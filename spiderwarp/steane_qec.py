"""Conventional Steane QEC with verified, repeat-until-success ancillas.

Retries discard only the ancilla, before any coupling to the persistent data.
Each CSS syndrome is read exactly once. Protection
labels apply to ancilla preparation, never to the distance of the data code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

from spiderwarp._se_circuit import PauliSpace, Readout, dual_circuit, lower_to_guppy_gates, row_mask
from spiderwarp.csscode import CSSCode
from spiderwarp.lut_decoder import LutDecoder
from spiderwarp.optimized_steane_se import ProtocolResult

MODES = ("fixed",)
LABELS = {"fixed": "Steane d3 RUS"}


def unique_css_syndromes(code, maximum=2):
    """Per-component correction domains, identifying errors modulo stabilizers.

    A syndrome is accepted only if every error within the stated component
    weight bound has the same stabilizer class. Empty domains reject as well.
    The bound concerns data errors, not the number of noisy circuit locations.
    """
    space = PauliSpace(code)
    domains = {}
    for basis, component, checks in (("Z", "X", code.H_z), ("X", "Z", code.H_x)):
        classes = [set() for _ in range(1 << len(checks))]
        for error in space.component_errors(component, maximum).values():
            classes[space.component_syndrome(error, component)].add(space.reduce(error))
        domains[basis] = tuple(len(group) == 1 for group in classes)
    return domains


@dataclass
class SteaneStage:
    circuit: stim.Circuit
    basis: str
    distance: int
    readout: Readout


class SteaneQECProtocol:
    """A frame-returning controller compatible with ProtocolSampler.

    Only ancilla preparation retries are unbounded. Each basis has one data
    read, followed by frame-adjusted unique-CSS-through-two postselection.
    The data code remains [[32,20,4]]. Every ancilla attempt uses d3.
    This is a simulation baseline, not an independently certified adaptive policy.
    """

    def __init__(self, code, preparations, mode="fixed", *, postselection="unique-css-2"):
        if mode not in MODES:
            raise ValueError(f"Unknown Steane RUS mode: {mode}")
        if set(preparations) != {1, 3} or code.d != 4:
            raise ValueError("Steane baselines require d1/d3 preparations and a d4 data code")
        if not np.array_equal(code.H_x, code.H_z):
            raise ValueError("These dual preparations require identical CSS checks")
        if postselection != "unique-css-2":
            raise ValueError("The retained Steane baselines require unique-css-2 data postselection")
        self.code, self.mode = code, mode
        self.start_distance = 3
        self.postselection = postselection
        self.unique_syndromes = unique_css_syndromes(code) if postselection != "none" else {}
        self.preparations = {d: c.copy() for d, c in preparations.items()}
        self.source = ""
        self.policy = None  # Repeated, cumulative-frame interface for ProtocolSampler.
        self.gadgets = {}
        n = code.n
        for basis in ("Z", "X"):
            for d, prep in sorted(self.preparations.items()):
                # Encoded |+> targets for Z readout; encoded |0> controls for X.
                ideal = dual_circuit(prep) if basis == "Z" else prep
                shifted = stim.Circuit()
                for op in lower_to_guppy_gates(ideal).flattened():
                    if op.name in {"DETECTOR", "TICK", "QUBIT_COORDS"}:
                        continue
                    shifted.append(op.name, [t.value + n for t in op.targets_copy()], op.gate_args_copy())
                self.gadgets[f"prep_{basis}_d{d}"] = SteaneStage(
                    shifted, basis, d, Readout(tuple((i,) for i in range(shifted.num_measurements)), ()))
            read = stim.Circuit()
            pairs = [(q, n + q) if basis == "Z" else (n + q, q) for q in range(n)]
            read.append("CX", [q for pair in pairs for q in pair])
            if basis == "X":
                read.append("H", range(n, 2*n))
            read.append("M", range(n, 2*n))
            checks = code.H_z if basis == "Z" else code.H_x
            groups = tuple(tuple(int(q) for q in np.flatnonzero(row)) for row in checks)
            self.gadgets[f"read_{basis}"] = SteaneStage(read, basis, 0, Readout((), groups))
            self.gadgets[f"discard_{basis}"] = SteaneStage(
                stim.Circuit(f"M {' '.join(str(n+q) for q in range(n))}"), basis, 0, Readout((), ()))
        self.decoders = {}
        for basis, checks in (("Z", code.H_z), ("X", code.H_x)):
            decoder = LutDecoder(checks)
            self.decoders[basis] = tuple(row_mask(decoder.get_mwr(
                np.array([(s >> i) & 1 for i in range(len(checks))], dtype=np.uint8)))
                for s in range(1 << len(checks)))

    @property
    def required_qubits(self):
        return max(g.circuit.num_qubits for g in self.gadgets.values())

    def run(self, simulator, *, frame_x=0, frame_z=0):
        if any(type(v) is not int or v < 0 or v >> self.code.n for v in (frame_x, frame_z)):
            raise ValueError("Input Pauli-frame masks must fit the data block")
        measurements = []

        def execute(name):
            g = self.gadgets[name]
            start = len(simulator.current_measurement_record())
            simulator.do_circuit(g.circuit)
            flags, syndrome = g.readout.decode(simulator.current_measurement_record()[start:])
            measurements.append((name, g.basis, g.distance, flags, syndrome))
            return flags, syndrome

        for basis in ("Z", "X"):
            level = self.start_distance
            while True:
                flags, _ = execute(f"prep_{basis}_d{level}")
                if flags == 0:
                    break
                execute(f"discard_{basis}")
            _, syndrome = execute(f"read_{basis}")
            checks = self.code.H_z if basis == "Z" else self.code.H_x
            frame = frame_x if basis == "Z" else frame_z
            adjusted = syndrome ^ sum(((row_mask(row) & frame).bit_count() % 2) << i
                                      for i, row in enumerate(checks))
            if not self.unique_syndromes[basis][adjusted]:
                return ProtocolResult(False, tuple(measurements), (),
                    discard_reason="ambiguous CSS syndrome through data weight two",
                    frame_x=frame_x, frame_z=frame_z)
            correction = self.decoders[basis][adjusted]
            if basis == "Z":
                frame_x ^= correction
            else:
                frame_z ^= correction
        return ProtocolResult(True, tuple(measurements), (), frame_x=frame_x, frame_z=frame_z)

    def save_bundle(self, directory):
        from spiderwarp.se_protocol_io import atomic_write, digest, versions
        import json
        core = dict(schema_version=1, kind="steane-rus", mode=self.mode,
                    code=self.code.to_dict(), preparations={str(d): str(c) for d, c in sorted(self.preparations.items())},
                    decoders={b: list(v) for b, v in self.decoders.items()},
                    preparation_start=self.start_distance, syndrome_confirmation="single-per-basis", retry_limit=None,
                    data_waits_during_retries=True, data_postselection=self.postselection,
                    unique_syndromes={b: list(v) for b, v in self.unique_syndromes.items()})
        path = Path(directory).resolve() / "protocol.json"
        atomic_write(path, json.dumps(dict(core, protocol_hash=digest(core), versions=versions()), indent=2)+"\n")
        return path

    def export_guppy(self, directory):
        from spiderwarp._steane_qec_guppy import generate_source
        from spiderwarp.se_protocol_io import atomic_write
        self.source = generate_source(self)
        self.source_path = Path(directory).resolve() / "steane_se_protocol.py"
        atomic_write(self.source_path, self.source)
        return self.source_path


def build_steane_qec(mode="fixed", *, output_dir=None, postselection="unique-css-2"):
    from spiderwarp.utils import get_project_root
    code = CSSCode.load_code("misc", "32_20_4")
    assets = get_project_root() / "assets/circuits/FTStatePrep/misc"
    protocol = SteaneQECProtocol(code, {d: stim.Circuit.from_file(assets / f"32_20_4_zero_d{d}.stim")
                                       for d in (1, 3)}, mode, postselection=postselection)
    if output_dir is not None:
        protocol.save_bundle(output_dir)
    return protocol


def load_steane_qec(manifest):
    if manifest.get("syndrome_confirmation") != "single-per-basis" or manifest.get("preparation_start") != 3:
        raise ValueError("Removed repeated-read Steane protocol; rebuild with single-per-basis d3 preparation")
    code = CSSCode.from_dict(manifest["code"])
    protocol = SteaneQECProtocol(code, {int(d): stim.Circuit(c) for d, c in manifest["preparations"].items()},
                                 manifest["mode"], postselection=manifest["data_postselection"])
    if {b: list(v) for b, v in protocol.decoders.items()} != manifest["decoders"]:
        raise ValueError("Steane decoder identity changed")
    if protocol.postselection != "none" and (
            {b: list(v) for b, v in protocol.unique_syndromes.items()} != manifest.get("unique_syndromes")):
        raise ValueError("Steane unique-correction domain changed")
    if (manifest["syndrome_confirmation"] != "single-per-basis"
            or manifest["retry_limit"] is not None or not manifest["data_waits_during_retries"]):
        raise ValueError("Unsupported Steane execution semantics")
    return protocol

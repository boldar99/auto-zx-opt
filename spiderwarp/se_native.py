"""Executable Guppy/HUGR experiment wrappers and optional Selene execution.

Selene's built-in gate noise also affects experiment preparation and final
readout and has no scheduled memory term. Native results explicitly identify
this different model; the Stim-reference sweep keeps ideal boundaries and
scheduled memory. No backend substitution is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from spiderwarp.se_protocol_io import atomic_write, digest, load_protocol, read_manifest, versions
from spiderwarp.se_protocol_simulation import _StateExperiment, _wilson_interval, scaled_noise_model


def experiment_source(protocol, *, prepared_state="0", cycles=1, incoming_x=0, incoming_z=0,
                      protocol_source=None):
    from spiderwarp._se_adaptive_guppy import generate_adaptive_source
    if prepared_state not in {"0", "+"} or type(cycles) is not int or cycles < 0:
        raise ValueError("Invalid experiment basis/cycle count")
    n = protocol.code.n
    if any(type(v) is not int or not 0 <= v < 1 << n for v in (incoming_x, incoming_z)):
        raise ValueError("Incoming masks must fit the data block")
    experiment = _StateExperiment(protocol.code, n, prepared_state)
    prep = experiment.initial.current_inverse_tableau().inverse().to_circuit()
    statements = []
    for op in prep.flattened():
        targets = [t.value for t in op.targets_copy()]
        if op.name == "CX":
            targets = list(zip(targets[::2], targets[1::2]))
        if op.name not in {"H", "S", "CX", "X", "Y", "Z", "CZ"}:
            raise ValueError(f"Unsupported ideal preparation gate: {op.name}")
        statements.append(f"    apply({op.name.lower()}, data, {tuple(targets)!r})")
    for gate, mask in (("x", incoming_x), ("z", incoming_z)):
        targets = tuple(q for q in range(n) if mask >> q & 1)
        if targets:
            statements.append(f"    apply({gate}, data, {targets!r})")
    if not statements:
        statements.append("    pass")
    tail = '''
from guppylang.std.platform import result
from guppylang.std.quantum import s

@guppy.comptime
def prepare_experiment(data: array[qubit, comptime(N_DATA)]) -> None:
PREPARATION

@guppy
def main() -> None:
    data = array(qubit() for _ in range(comptime(N_DATA)))
    prepare_experiment(data)
    accepted = True
    frame_x = 0
    frame_z = 0
    completed = 0
    for _ in range(comptime(EXPERIMENT_CYCLES)):
        if accepted:
            outcome = se_cycle_with_frame(data, frame_x, frame_z)
            accepted = outcome.accepted
            frame_x = outcome.frame_x
            frame_z = outcome.frame_z
            if accepted:
                completed += 1
    result("accepted", accepted)
    result("frame_x", frame_x)
    result("frame_z", frame_z)
    result("completed", completed)
    if comptime(EXPERIMENT_PLUS):
        for i in range(comptime(N_DATA)):
            h(data[i])
    bits = array(measure_bit(q) for q in data)
    result("measurements", bits)
'''.replace("PREPARATION", "\n".join(statements))
    if protocol_source is None:
        if hasattr(protocol, "mode"):
            from spiderwarp._steane_qec_guppy import generate_source
            protocol_source = generate_source(protocol)
        else:
            protocol_source = generate_adaptive_source(protocol)
    return (protocol_source
            + f"\nEXPERIMENT_CYCLES = {cycles}\nEXPERIMENT_PLUS = {prepared_state == '+'}\n"
            + tail)


def compile_experiment(protocol_path, output_dir, *, prepared_state="0", cycles=1,
                       incoming_x=0, incoming_z=0):
    """Compile and reload a real executable entry point; cache by exact source."""
    from hugr.package import Package
    from spiderwarp._se_adaptive_guppy import compile_adaptive_source
    _, manifest = read_manifest(protocol_path)
    protocol = load_protocol(protocol_path)
    directory = Path(output_dir).resolve()
    source = experiment_source(protocol, prepared_state=prepared_state, cycles=cycles,
                               incoming_x=incoming_x, incoming_z=incoming_z)
    identity = dict(protocol_hash=manifest["protocol_hash"], prepared_state=prepared_state,
                    cycles=cycles, incoming_x=incoming_x, incoming_z=incoming_z,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(), versions=versions())
    key = digest(identity)
    metadata = directory / "experiment.json"
    binary = directory / "experiment.hugr"
    if metadata.exists() and binary.exists():
        previous = json.loads(metadata.read_text())
        if (previous["compile_hash"] == key
                and previous["hugr_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()):
            Package.from_bytes(binary.read_bytes())
            return binary
    atomic_write(directory / "experiment.py", source)
    *_, package = compile_adaptive_source(source, entrypoint="main")
    data = package.to_bytes()
    Package.from_bytes(data)
    atomic_write(binary, data)
    atomic_write(metadata, json.dumps(dict(identity, compile_hash=key,
                 hugr_sha256=hashlib.sha256(data).hexdigest()), indent=2) + "\n")
    return binary


def run_native(protocol_path, output_dir, *, prepared_state="0", cycles=1, shots=100,
               p=8e-4, seed=0, jobs=1, incoming_x=0, incoming_z=0):
    """Execute serialized HUGR using Selene/Stim, without interpreter fallback."""
    from selene_sim import build, Stim, DepolarizingErrorModel
    if type(shots) is not int or shots <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("shots must be positive and seed must be nonnegative integers")
    if type(jobs) is not int or jobs < 1:
        raise ValueError("Native jobs must be a positive integer")
    model = scaled_noise_model(p)
    binary = compile_experiment(protocol_path, output_dir, prepared_state=prepared_state,
                                cycles=cycles, incoming_x=incoming_x, incoming_z=incoming_z)
    protocol = load_protocol(protocol_path)
    experiment = _StateExperiment(protocol.code, protocol.code.n, prepared_state)
    checks = protocol.code.H_z if prepared_state == "0" else protocol.code.H_x
    logicals = protocol.code.L_z if prepared_state == "0" else protocol.code.L_x
    runner = build(binary, build_dir=Path(output_dir).resolve() / "selene")
    noise = {k: getattr(model, k) for k in ("p_1q", "p_2q", "p_init", "p_meas")}
    accepted = failures = observed = 0
    for shot in runner.run_shots(simulator=Stim(), n_qubits=protocol.required_qubits,
                                 n_shots=shots, n_processes=jobs, random_seed=seed,
                                 error_model=DepolarizingErrorModel(**noise)):
        data = dict(shot)
        observed += 1
        if not data["accepted"]:
            continue
        accepted += 1
        if data["completed"] != cycles:
            raise RuntimeError("Native execution returned an incomplete accepted trajectory")
        bits = np.asarray(data["measurements"], dtype=np.uint8)
        if bits.shape != (protocol.code.n,):
            raise RuntimeError("Native execution returned malformed final measurements")
        frame = int(data["frame_x"] if prepared_state == "0" else data["frame_z"])
        bits ^= np.array([frame >> q & 1 for q in range(protocol.code.n)], dtype=np.uint8)
        correction = experiment.decoder.get_mwr(checks @ bits % 2)
        failures += bool(np.any(logicals @ (bits ^ correction) % 2))
    if observed != shots:
        raise RuntimeError(f"Selene returned {observed} shots, expected {shots}")
    result = dict(backend="selene-stim", noise_model="gate-noise-including-preparation-and-readout-no-memory-v1",
                  noise=noise, p=p, seed=seed, cycles=cycles, prepared_state=prepared_state,
                  shots=shots, accepted=accepted, logical_errors=failures,
                  ler=failures / accepted if accepted else None,
                  ler_interval_95=_wilson_interval(failures, accepted),
                  survival=accepted / shots, versions=versions(),
                  experiment=json.loads((Path(output_dir) / "experiment.json").read_text()))
    atomic_write(Path(output_dir) / "native_results.json", json.dumps(result, indent=2) + "\n")
    return result

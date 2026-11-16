"""Sol/Narsil Monte Carlo with PECOS Helios-alpha noise and ideal boundaries.

The separate Sol worker compiles and executes the complete adaptive program.
This module only exports source, decodes ideal final measurements, and commits
resumable counts. The old unit-layer noise model is never applied here.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass, field
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
import stim

from spiderwarp.se_native import experiment_source
from spiderwarp.se_protocol_io import atomic_write, canonical_json, digest, load_protocol, read_manifest, versions
from spiderwarp.se_protocol_simulation import _StateExperiment, _wilson_interval

MODEL_VERSION = "sol-helios-alpha-stim-ideal-boundaries-v2"
SEED_MODE = "default"
BEGIN_NOISE = 1152921504606846977
END_NOISE = 1152921504606846978
COUNT_FIELDS = ("shots", "accepted", "logical_errors", "cycles_started",
          "cycles_completed", "total_cnots", "accepted_cnots", "total_native_rzz",
          "accepted_native_rzz", "total_duration_ns", "accepted_duration_ns", "peak_live_qubits")
FIELDS = ("start", "stream_seed", "seed_mode", *COUNT_FIELDS, "seconds", "diagnostics")


@dataclass(frozen=True)
class SolConfig:
    noise_scale: float = 1.0
    repump: bool = False
    model_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.noise_scale) or self.noise_scale < 0:
            raise ValueError("Sol noise_scale must be finite and nonnegative")
        if type(self.repump) is not bool or not isinstance(self.model_overrides, dict):
            raise ValueError("Invalid Sol repump/model_overrides configuration")
        object.__setattr__(self, "noise_scale", float(self.noise_scale))

    def to_dict(self):
        return dict(noise_scale=self.noise_scale, repump=self.repump, model_overrides=self.model_overrides)


def sol_python(path=None):
    candidate = path or os.environ.get("SPIDERWARP_SOL_PYTHON")
    if candidate is None:
        candidate = Path(__file__).resolve().parents[1] / "local-sol-example/.venv/bin/python"
    candidate = Path(candidate).expanduser().absolute()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError("Set --sol-python or SPIDERWARP_SOL_PYTHON to the locked Sol environment's Python")
    return str(candidate)


def worker(python, request, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="request_", suffix=".json", dir=directory)
    os.close(fd)
    path = Path(filename)
    try:
        atomic_write(path, json.dumps(request, allow_nan=False))
        result = subprocess.run([python, str(Path(__file__).with_name("_sol_worker.py")), str(path)],
                                text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("Sol worker failed:\n" + result.stdout + result.stderr)
    finally:
        path.unlink(missing_ok=True)


def engine_identity(python, config, directory):
    path = Path(directory) / "description.json"
    worker(python, dict(action="describe", config=config.to_dict(), output=str(path)), directory)
    runtime = json.loads(path.read_text())
    here = Path(__file__).parent
    sources = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in (
        "se_sol.py", "_sol_worker.py", "_sol_boundary.c", "se_native.py", "_se_adaptive_guppy.py",
        "_se_adaptive_guppy_template.py.in", "_se_guppy_template.py.in", "_steane_qec_guppy.py",
        "_se_decisions.py", "lut_decoder.py", "se_protocol_io.py", "se_protocol_simulation.py",
        "_se_circuit.py", "_se_adaptive.py", "steane_qec.py", "adaptive_steane_se.py", "csscode.py")}
    return dict(model=MODEL_VERSION, sources=sources, runtime=runtime,
                reference_versions={k: v for k, v in versions().items() if k in {"numpy", "stim"}},
                policy_calibration="Frozen controller; probability policies retain their original training model",
                certificate_scope="Original abstract circuit fault model; no Sol hardware certificate")


def input_sync_checks(code, prepared_state="0", mode="cover"):
    """Deterministic generator cover, sufficient to force every data dependency.

    Measuring a known stabilizer does not change the prepared quantum state.
    Full mode retains the original preparation/layout for controls. Changing
    the cover can change the ion layout, so mode is part of the task identity.
    """
    if mode not in {"cover", "full"}:
        raise ValueError("Sol input_sync must be cover or full")
    generators = [("X", tuple(int(q) for q in np.flatnonzero(row))) for row in code.H_x]
    generators += [("Z", tuple(int(q) for q in np.flatnonzero(row))) for row in code.H_z]
    generators += [("Z" if prepared_state == "0" else "X", tuple(int(q) for q in np.flatnonzero(row)))
                   for row in (code.L_z if prepared_state == "0" else code.L_x)]
    generators = [(basis, support) for basis, support in generators if support]
    uncovered = set(range(code.n))
    if set().union(*(set(support) for _, support in generators)) != uncovered:
        raise ValueError("Sol input stabilizers must cover every data qubit")
    if mode == "full":
        return generators
    selected = []
    while uncovered:
        # Prefer maximum newly covered data; then fewer gates, then original
        # generator order. In [[32,20,4]], the first full-support X check wins.
        index = max(range(len(generators)), key=lambda i:
                    (len(uncovered.intersection(generators[i][1])), -len(generators[i][1]), -i))
        item = generators.pop(index)
        selected.append(item)
        uncovered.difference_update(item[1])
    return selected


def css_input_preparation(code, prepared_state="0"):
    """Prepare the uniform superposition of the CSS state's X-generator span.

    Logical 0 uses H_x; logical + additionally uses L_x. Gaussian elimination
    gives independent pivot controls and disjoint nonpivot targets, so the
    circuit needs one H per independent row and one CX per nonpivot 1.
    """
    if prepared_state not in {"0", "+"}:
        raise ValueError("prepared_state must be '0' or '+'")
    matrix = (code.H_x if prepared_state == "0" else np.vstack((code.H_x, code.L_x))).copy()
    rank, pivots = 0, []
    for column in range(code.n):
        candidates = np.flatnonzero(matrix[rank:, column])
        if not len(candidates):
            continue
        first = rank + int(candidates[0])
        matrix[[rank, first]] = matrix[[first, rank]]
        for row in range(len(matrix)):
            if row != rank and matrix[row, column]:
                matrix[row] ^= matrix[rank]
        pivots.append(column)
        rank += 1
        if rank == len(matrix):
            break
    circuit = stim.Circuit()
    if pivots:
        circuit.append("H", pivots)
    for row, pivot in enumerate(pivots):
        for target in np.flatnonzero(matrix[row]):
            if target != pivot:
                circuit.append("CX", [pivot, int(target)])
    return circuit


def sol_experiment_source(protocol, *, prepared_state="0", cycles=1, incoming_x=0, incoming_z=0,
                          protocol_source=None, input_sync="cover"):
    if protocol.required_qubits < protocol.code.n + 1:
        raise ValueError("Sol ideal-input synchronization needs one temporary ancilla")
    source = experiment_source(protocol, prepared_state=prepared_state, cycles=cycles,
                               incoming_x=incoming_x, incoming_z=incoming_z, protocol_source=protocol_source)
    source = source.replace("from guppylang.std.platform import result",
                            "from guppylang.std.platform import result\nfrom qtm_platform.ops import sleep")
    # Make result available to the instrumented gadget functions as well.
    source = "from guppylang.std.platform import result\n" + source
    if input_sync == "cover":
        # Keep full mode's complete original prefix for compatibility audits.
        statements = []
        for op in css_input_preparation(protocol.code, prepared_state):
            targets = [target.value for target in op.targets_copy()]
            if op.name == "CX":
                targets = list(zip(targets[::2], targets[1::2]))
            statements.append(f"    apply({op.name.lower()}, data, {tuple(targets)!r})")
        for gate, mask in (("x", incoming_x), ("z", incoming_z)):
            targets = tuple(q for q in range(protocol.code.n) if mask >> q & 1)
            if targets:
                statements.append(f"    apply({gate}, data, {targets!r})")
        begin = source.index("@guppy.comptime\ndef prepare_experiment(")
        end = source.index("@guppy\ndef main()", begin)
        preparation = ("@guppy.comptime\ndef prepare_experiment(data: array[qubit, comptime(N_DATA)]) -> None:\n"
                       + "\n".join(statements or ["    pass"]) + "\n\n")
        source = source[:begin] + preparation + source[end:]
    source = source.replace("    prepare_experiment(data)\n",
                            f"    prepare_experiment(data)\n    synchronize_input(data)\n    sleep(data, {BEGIN_NOISE})\n")
    source = source.replace('    result("accepted", accepted)',
                            f'    sleep(data, {END_NOISE})\n    result("accepted", accepted)')
    source = source.replace("        if accepted:\n            outcome =",
                            '        if accepted:\n            result("_sw_cycle", 1)\n            outcome =')
    # Narsil 1.0 defers unitary batches across local barriers and its supplied
    # platform utility cannot drain Sol. Read a generator cover of the known input
    # stabilizer state nondestructively instead. Their outcomes are deterministic
    # even with the requested incoming Pauli. Each data qubit participates, so
    # reading these futures forces every preceding preparation gate to finish
    # while the error adapter is still ideal. This adds one temporary ancilla.
    sync = ['@guppy', 'def check_input(observed: bool, expected: bool) -> None:',
            '    if observed != expected:', '        exit("Ideal Sol input synchronization failed")',
            '', '@guppy.comptime',
            'def synchronize_input(data: array[qubit, comptime(N_DATA)]) -> None:']
    for component, support in input_sync_checks(protocol.code, prepared_state, input_sync):
        incoming = incoming_z if component == "X" else incoming_x
        expected = bool(sum(incoming >> q & 1 for q in support) % 2)
        sync += ['    probe = qubit()', '    h(probe)']
        for q in support:
            sync += [f'    {"cx" if component == "X" else "cz"}(probe, data[{q}])']
        sync += ['    h(probe)', '    observed = measure_bit(probe)', f'    check_input(observed, {expected})']
    source += "\n" + "\n".join(sync) + "\n"
    stages = []

    def stage(name, basis, distance, gadget):
        i = len(stages)
        cost = sum(len(op.targets_copy()) // 2 for op in gadget.circuit.flattened()
                   if op.name in {"CX", "CZ", "XCX"})
        stages.append(dict(name=name, basis=basis, distance=distance, cnots=cost))
        return i

    if hasattr(protocol, "mode"):
        for basis in ("Z", "X"):
            for d in sorted(protocol.preparations):
                name = f"prep_{basis}_d{d}"
                gadget = protocol.gadgets[name]
                i = stage(name, basis, d, gadget)
                old = f"    return pack_parities(measurements, {gadget.readout.flags!r})"
                # Each function has its own return, even if the flag groups match.
                function_start = source.index(f"def prep_{basis.lower()}_d{d}(")
                position = source.index(old, function_start)
                new = (f"    flags = pack_parities(measurements, {gadget.readout.flags!r})\n"
                       f'    result("_sw_stage_{i}", flags)\n    return flags')
                source = source[:position] + source[position:].replace(old, new, 1)
            name = f"read_{basis}"
            i = stage(name, basis, 0, protocol.gadgets[name])
            old = f"    return syndrome_{basis.lower()}(measurements)"
            source = source.replace(old, f'    result("_sw_stage_{i}", 0)\n' + old, 1)
    else:
        used = {node.action for node in protocol.policy.nodes if node.kind == "extract"}
        actions = sorted((name, action) for name, action in protocol.policy.actions.items()
                         if name in used or not used)
        for i, (name, action) in enumerate(actions):
            stage(name, action.gadget.basis, action.gadget.distance, action.gadget)
            start = source.index(f"def run_action_{i}(")
            position = source.index("    return ExtractionResult(flags, syndrome)", start)
            source = (source[:position] + f'    result("_sw_stage_{i}", flags)\n'
                      + f'    result("_sw_syndrome_{i}", syndrome)\n' + source[position:])
    return source, stages


def _metrics(record):
    # MetricStore nests component prefixes: "runtime:spiderwarp:metric".
    found = {}
    for category, values in record.get("metrics", {}).items():
        if isinstance(values, dict):
            for name, value in values.items():
                full = category + ":" + name
                if "spiderwarp:" in full:
                    found[full.split("spiderwarp:", 1)[1]] = value
    required = {"phase_switches", "peak_live_qubits", "noisy_rzz", "protocol_duration_ns"}
    if not required <= found.keys() or found["phase_switches"] != 2:
        raise RuntimeError("Missing Sol boundary metrics or incomplete phase switch")
    return found


def summarize_shots(path, protocol, stages, basis, cycles, start, shots, peak_limit, *, experiment=None,
                    stream_seed=None):
    experiment = experiment or _StateExperiment(protocol.code, protocol.code.n, basis)
    checks = protocol.code.H_z if basis == "0" else protocol.code.H_x
    logicals = protocol.code.L_z if basis == "0" else protocol.code.L_x
    counts = {name: 0 for name in COUNT_FIELDS}
    stage_counts, accepted_stages, flagged, rejection_cycles = (Counter() for _ in range(4))
    with Path(path).open() as stream:
        for line in stream:
            data = json.loads(line)
            if stream_seed is not None and (data.get("shot_id") != start + counts["shots"]
                    or data.get("stream_seed") != stream_seed or data.get("seed_mode") != SEED_MODE):
                raise RuntimeError("Sol returned an incorrect seed stream or global shot index")
            accepted = data["accepted"]
            completed, started = data["completed"], data["cycles_started"]
            if accepted not in (0, 1) or not 0 <= completed <= started <= cycles:
                raise RuntimeError("Malformed Sol acceptance/cycle results")
            if (accepted and completed != cycles) or (not accepted and completed + 1 != started):
                raise RuntimeError("Sol returned an inconsistent trajectory")
            metrics = _metrics(data)
            if metrics["peak_live_qubits"] > peak_limit:
                raise RuntimeError("Sol exceeded the protocol's live-qubit limit")
            shot_stages = Counter()
            cost = 0
            for i, flags in data["stages"]:
                if type(i) is not int or not 0 <= i < len(stages) or type(flags) is not int or flags < 0:
                    raise RuntimeError("Malformed Sol stage record")
                key = stages[i]["name"]
                shot_stages[key] += 1
                flagged[key] += bool(flags)
                cost += stages[i]["cnots"]
            stage_counts.update(shot_stages)
            counts["shots"] += 1
            counts["cycles_started"] += started
            counts["cycles_completed"] += completed
            counts["total_cnots"] += cost
            counts["total_native_rzz"] += metrics["noisy_rzz"]
            counts["total_duration_ns"] += metrics["protocol_duration_ns"]
            counts["peak_live_qubits"] = max(counts["peak_live_qubits"], metrics["peak_live_qubits"])
            if not accepted:
                rejection_cycles[started] += 1
                continue
            counts["accepted"] += 1
            counts["accepted_cnots"] += cost
            counts["accepted_native_rzz"] += metrics["noisy_rzz"]
            counts["accepted_duration_ns"] += metrics["protocol_duration_ns"]
            accepted_stages.update(shot_stages)
            bits = np.asarray(data["measurements"])
            if bits.shape != (protocol.code.n,) or not np.isin(bits, [0, 1]).all():
                raise RuntimeError("Malformed Sol final measurements")
            frame = data["frame_x"] if basis == "0" else data["frame_z"]
            if type(frame) is not int or not 0 <= frame < 1 << protocol.code.n:
                raise RuntimeError("Malformed Sol Pauli frame")
            bits = bits.astype(np.uint8) ^ np.array([frame >> q & 1 for q in range(protocol.code.n)], dtype=np.uint8)
            correction = experiment.decoder.get_mwr(checks @ bits % 2)
            counts["logical_errors"] += bool(np.any(logicals @ (bits ^ correction) % 2))
    if counts["shots"] != shots:
        raise RuntimeError("Sol returned an incorrect number of shots")
    return dict(start=start, seconds=0.0, **counts, diagnostics=canonical_json(dict(
        stage_counts=dict(stage_counts), accepted_stage_counts=dict(accepted_stages),
        flagged_stage_counts=dict(flagged), rejection_cycles=dict(rejection_cycles))))


def sampling_seed(task):
    """Stable stream across batch/process changes, separate for each experiment."""
    return int(digest(dict(model=MODEL_VERSION, seed=task["seed"], protocol_hash=task["protocol_hash"],
        noise=task["noise"], basis=task["prepared_state"], cycles=task["cycles"],
        input_sync=task["input_sync"], incoming_x=task["incoming_x"], incoming_z=task["incoming_z"]))[:15], 16)


def _terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _task_batches(python, bundle, directory, task, stages, protocol, start, shots, batch_size,
                  retain_shots, cancelled, processes, process_lock):
    """One isolated Python/native run, yielding committed-size shot ranges.

    Every native shot is still reset/reseeded by Selene. Only process startup,
    HUGR loading and decoder construction are amortized across batches.
    """
    directory = Path(directory)
    begin = time.monotonic()
    if read_manifest(bundle)[1]["protocol_hash"] != task["protocol_hash"]:
        raise RuntimeError("Sol protocol changed after the task was planned")
    seed = sampling_seed(task)
    request = dict(action="run", source=str(directory / "experiment.py"),
        directory=str(directory), config=task["noise"], peak_qubits=task["peak_qubits"],
        shots=shots, seed=seed, start=start, seed_mode=SEED_MODE,
        stream_batches=True, batch_size=batch_size)
    experiment = _StateExperiment(protocol.code, protocol.code.n, task["prepared_state"])
    fd, filename = tempfile.mkstemp(prefix="request_", suffix=".json", dir=directory)
    os.close(fd)
    path = Path(filename)
    process = None
    try:
        atomic_write(path, json.dumps(request, allow_nan=False))
        with tempfile.TemporaryFile(mode="w+t") as errors:
            if cancelled.is_set():
                return
            process = subprocess.Popen([python, "-u", str(Path(__file__).with_name("_sol_worker.py")), str(path)],
                stdout=subprocess.PIPE, stderr=errors, text=True, start_new_session=True)
            with process_lock:
                processes.add(process)
            expected = start
            for line in process.stdout:
                if cancelled.is_set():
                    return
                message = json.loads(line)
                count = min(batch_size, start + shots - expected)
                raw = directory / f"shots_{expected}.jsonl"
                if (count <= 0 or message.get("start") != expected or message.get("shots") != count
                        or Path(message.get("output", "")) != raw
                        or message.get("stream_seed") != seed or message.get("seed_mode") != SEED_MODE):
                    raise RuntimeError("Invalid or overlapping Sol batch notification")
                if read_manifest(bundle)[1]["protocol_hash"] != task["protocol_hash"]:
                    raise RuntimeError("Sol protocol changed during sampling")
                row = summarize_shots(raw, protocol, stages, task["prepared_state"], task["cycles"],
                    expected, count, task["peak_qubits"], experiment=experiment, stream_seed=seed)
                row.update(seconds=time.monotonic() - begin, stream_seed=seed, seed_mode=SEED_MODE)
                if not retain_shots:
                    raw.unlink()
                yield row
                expected += count
                begin = time.monotonic()
            process.wait()
            if process.returncode:
                errors.seek(0)
                raise RuntimeError("Sol worker failed:\n" + errors.read())
            if expected != start + shots:
                raise RuntimeError("Sol worker returned an incomplete shot range")
    finally:
        if process is not None:
            _terminate(process)
            process.stdout.close()
            with process_lock:
                processes.discard(process)
        path.unlink(missing_ok=True)


def _csv(rows, fields):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _batches(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    position = 0
    for row in rows:
        for name in ("start", "stream_seed", *COUNT_FIELDS):
            row[name] = int(row[name])
        row["seconds"] = float(row["seconds"])
        if (row["start"] != position or row["shots"] <= 0
                or not 0 <= row["logical_errors"] <= row["accepted"] <= row["shots"]
                or any(row[n] < 0 for n in ("start", "stream_seed", *COUNT_FIELDS))
                or row["seed_mode"] != SEED_MODE
                or not math.isfinite(row["seconds"]) or row["seconds"] < 0
                or row["accepted_cnots"] > row["total_cnots"]
                or row["accepted_native_rzz"] > row["total_native_rzz"]
                or row["accepted_duration_ns"] > row["total_duration_ns"]):
            raise ValueError(f"Invalid/duplicate Sol batch ranges in {path}")
        position += row["shots"]
    return rows


def summarize(directory):
    directory = Path(directory)
    summaries = []
    for metadata in sorted(directory.glob("tasks/*.json")):
        record = json.loads(metadata.read_text())
        task = record["task"]
        if digest(task) != metadata.stem:
            raise ValueError("Sol task identity mismatch")
        rows = _batches(directory / "batches" / (metadata.stem + ".csv"))
        if not rows:
            continue
        if any(row["stream_seed"] != sampling_seed(task) for row in rows):
            raise ValueError("Sol batch seed stream differs from its task")
        totals = {name: sum(row[name] for row in rows) for name in (*COUNT_FIELDS, "seconds")
                  if name != "peak_live_qubits"}
        accepted, shots, failures = (totals[n] for n in ("accepted", "shots", "logical_errors"))
        low, high = _wilson_interval(failures, accepted)
        sl, sh = _wilson_interval(accepted, shots)
        summaries.append(dict(task_hash=metadata.stem, protocol_hash=task["protocol_hash"],
            engine_hash=digest(task["engine"]), backend=MODEL_VERSION, noise_scale=task["noise"]["noise_scale"],
            noise=canonical_json(task["noise"]), cycles=task["cycles"], prepared_state=task["prepared_state"],
            seed=task["seed"], stream_seed=sampling_seed(task), seed_mode=SEED_MODE,
            input_sync=task["input_sync"], **totals, target_shots=record["target_shots"],
            ler=failures/accepted if accepted else math.nan, ler_low=low, ler_high=high,
            survival=accepted/shots, survival_low=sl, survival_high=sh,
            mean_cnots=totals["accepted_cnots"]/accepted if accepted else math.nan,
            mean_native_rzz=totals["accepted_native_rzz"]/accepted if accepted else math.nan,
            mean_duration_seconds=totals["accepted_duration_ns"]/accepted/1e9 if accepted else math.nan,
            peak_live_qubits=max(row["peak_live_qubits"] for row in rows)))
    if summaries:
        atomic_write(directory / "summary.csv", _csv(summaries, list(summaries[0])))
    return summaries


def run_sol_sweep(protocol_path, configurations, results_dir="benchmarks/sol_v2", *, shots=1000,
                  batch_size=250, jobs=1, seed=0, python=None, repump=False,
                  model_overrides=None, retain_shots=False, incoming_x=0, incoming_z=0,
                  input_sync="cover", progress=print):
    """Sample fixed attempted-shot budgets; resume only exact matching tasks.

    Sol probability-policy bundles are evaluated as frozen controllers. Their
    historical 10%/25% training thresholds are not recalibrated by this runner.
    """
    for name, value in (("shots", shots), ("batch_size", batch_size), ("jobs", jobs)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if shots >= 1 << 64:
        raise ValueError("Sol shot indices must fit an unsigned 64-bit integer")
    python = sol_python(python)
    bundle, manifest = read_manifest(protocol_path)
    protocol = load_protocol(bundle)
    directory = Path(results_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    configurations = list(configurations)
    if not configurations:
        raise ValueError("Supply at least one Sol configuration")
    import fcntl
    with (directory / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another Sol sweep is writing {directory}") from None
        # Do not mix old-reference CSVs or another result schema into this tree.
        for old in directory.glob("tasks/*.json"):
            if json.loads(old.read_text()).get("task", {}).get("backend") != MODEL_VERSION:
                raise ValueError("Sol requires its own results directory")
        tasks, records, stage_tables, engines = {}, {}, {}, {}
        # Large V2 tables are emitted once, shared by every basis/cycle wrapper.
        if hasattr(protocol, "mode"):
            from spiderwarp._steane_qec_guppy import generate_source
            protocol_source = generate_source(protocol)
        else:
            from spiderwarp._se_adaptive_guppy import generate_adaptive_source
            protocol_source = generate_adaptive_source(protocol)
        for config in configurations:
            if set(config) - {"noise_scale", "cycles", "prepared_state"}:
                raise ValueError("Sol configurations use noise_scale, cycles, prepared_state; legacy p/noise are not supported")
            basis, cycles = config["prepared_state"], config["cycles"]
            if basis not in {"0", "+"} or type(cycles) is not int or cycles < 0:
                raise ValueError("Invalid Sol basis/cycle count")
            noise = SolConfig(config.get("noise_scale", 1.0), repump, dict(model_overrides or {}))
            noise_key = canonical_json(noise.to_dict())
            if noise_key not in engines:
                with tempfile.TemporaryDirectory(prefix="sol_identity_") as temporary:
                    engines[noise_key] = engine_identity(python, noise, temporary)
            engine = engines[noise_key]
            source, stages = sol_experiment_source(protocol, prepared_state=basis, cycles=cycles,
                incoming_x=incoming_x, incoming_z=incoming_z, protocol_source=protocol_source, input_sync=input_sync)
            task = dict(protocol_hash=manifest["protocol_hash"], backend=MODEL_VERSION, engine=engine,
                        noise=noise.to_dict(), prepared_state=basis, cycles=cycles, seed=seed,
                        peak_qubits=protocol.required_qubits, incoming_x=incoming_x, incoming_z=incoming_z,
                        source_sha256=hashlib.sha256(source.encode()).hexdigest(), input_sync=input_sync,
                        input_preparation="css-systematic-v1" if input_sync == "cover" else "legacy-tableau")
            key = digest(task)
            tasks[key] = task
            work = directory / "work" / key
            atomic_write(work / "experiment.py", source)
            stage_tables[key] = stages
            records[key] = _batches(directory / "batches" / (key + ".csv"))
            atomic_write(directory / "tasks" / (key + ".json"), json.dumps(dict(
                task_hash=key, task=task, target_shots=max(shots, sum(r["shots"] for r in records[key])),
                stages=stages), indent=2) + "\n")
        # Each task owns one persistent Python/native run. Threads only dispatch
        # to those isolated runtimes and decode batches; no nested process pool.
        def commit(key, row):
            records[key].append(row)
            atomic_write(directory / "batches" / (key + ".csv"), _csv(records[key], FIELDS))
            if progress:
                progress(f"Sol scale={tasks[key]['noise']['noise_scale']:g} N={tasks[key]['cycles']} "
                         f"state={tasks[key]['prepared_state']}: {row['start']+row['shots']}/{shots}")

        cancelled, process_lock, processes = threading.Event(), threading.Lock(), set()

        def sample_task(key):
            done = sum(row["shots"] for row in records[key])
            if done < shots:
                for row in _task_batches(python, str(bundle), directory / "work" / key,
                        tasks[key], stage_tables[key], protocol, done, shots - done, batch_size,
                        retain_shots, cancelled, processes, process_lock):
                    commit(key, row)

        pool = ThreadPoolExecutor(max_workers=jobs)
        try:
            for future in as_completed([pool.submit(sample_task, key) for key in tasks]):
                future.result()
        except BaseException:
            cancelled.set()
            with process_lock:
                for process in tuple(processes):
                    _terminate(process)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return summarize(directory)

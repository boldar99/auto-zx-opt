"""Resumable, bounded-memory Monte Carlo batches for saved SE protocols."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from functools import lru_cache
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from spiderwarp.noise_model import CustomErrorModel, default_noise_model
from spiderwarp.se_protocol_io import atomic_write, canonical_json, digest, load_protocol, read_manifest, versions
from spiderwarp.se_protocol_simulation import ProtocolSampler, _wilson_interval, scaled_noise_model

MODEL_VERSION = "stim-adaptive-initial-data-idles-v1"
BATCH_FIELDS = ("start", "shots", "seed", "accepted", "logical_errors", "seconds",
                "cycles_started", "cycles_completed", "total_cnots", "diagnostics")


def engine_identity():
    """Conservatively separate results when simulator/decoder semantics change."""
    here = Path(__file__).parent
    sources = {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in
               ("se_sweep.py", "se_protocol_simulation.py", "noise_model.py", "lut_decoder.py",
                "adaptive_steane_se.py", "se_protocol_io.py", "_se_circuit.py", "_se_adaptive.py",
                "steane_qec.py")}
    v = versions()
    return dict(model=MODEL_VERSION, sources=sources,
                versions={k: v[k] for k in ("stim", "numpy")},
                python=list(sys.version_info[:3]), architecture=platform.machine())


def small_sweep():
    """Scaled hardware p sweep and exact default-hardware repetition sweep."""
    physical = [dict(p=float(p), cycles=1, prepared_state=basis, sweep="physical_error")
                for p in np.geomspace(1e-4, 1e-2, 11) for basis in ("0", "+")]
    repeated = [dict(noise=vars(default_noise_model()), cycles=n, prepared_state=basis, sweep="cycles")
                for n in (1, 3, 5, 7, 10) for basis in ("0", "+")]
    return physical + repeated


def num_samples(p, cycles=1, *, shots=100_000, max_shots=None, reference_p=None):
    """Fixed production budget for approximate relative-error consistency.

    In the low-noise regime p_L is proportional to cycles*p². Allocating its
    inverse approximately equalizes expected failures and relative Wilson CI
    widths (also after dividing p_L by p² or cycles). `shots` is a minimum;
    the default cap is ten times that minimum. The cap bounds runtime where the
    asymptotic approximation is inaccurate. Budgets never depend on sampled
    failures, so ordinary fixed-sample Wilson intervals remain appropriate.
    """
    if type(shots) is not int or shots <= 0 or type(cycles) is not int or cycles < 0:
        raise ValueError("shots must be positive and cycles nonnegative integers")
    max_shots = 10 * shots if max_shots is None else max_shots
    if type(max_shots) is not int or max_shots < shots:
        raise ValueError("max_shots must be an integer at least shots")
    reference_p = default_noise_model().p_2q if reference_p is None else reference_p
    if not math.isfinite(p) or not 0 <= p <= 1 or not math.isfinite(reference_p) or not 0 < reference_p <= 1:
        raise ValueError("p must be in [0,1] and reference_p in (0,1]")
    if p == 0 or cycles == 0:
        return shots
    ratio = reference_p / p
    # Check the cap before squaring, including extremely small positive p.
    if ratio >= math.sqrt((max_shots / shots) * cycles):
        return max_shots
    return min(max_shots, max(shots, math.ceil(shots * ratio * ratio / cycles)))


def _csv_text(rows, fields):
    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fields)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _read_batches(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    position = 0
    for row in rows:
        for name in ("start", "shots", "seed", "accepted", "logical_errors", "cycles_started",
                     "cycles_completed", "total_cnots"):
            row[name] = int(row[name])
        row["seconds"] = float(row["seconds"])
        if (row["start"] != position or row["shots"] <= 0
                or not 0 <= row["logical_errors"] <= row["accepted"] <= row["shots"]):
            raise ValueError(f"Invalid/duplicate batch ranges in {path}")
        position += row["shots"]
    return rows


@lru_cache(maxsize=2)
def _protocol(path, protocol_hash):
    _, m = read_manifest(path)
    if m["protocol_hash"] != protocol_hash:
        raise ValueError("Protocol changed after the sweep was planned")
    return load_protocol(path)


@lru_cache(maxsize=2)
def _sampler(path, task_json):
    task = json.loads(task_json)
    return ProtocolSampler(_protocol(path, task["protocol_hash"]),
                           noise_model=CustomErrorModel(**task["noise"]),
                           prepared_state=task["prepared_state"], cycles=task["cycles"],
                           collect_paths=False)


def _sample_batch(path, task, start, shots):
    # Batch seeds depend on the disjoint starting shot, not scheduling or process
    # number. Changing --jobs preserves results; changing batch size creates a
    # new random partition but cannot replay a completed batch on resume.
    seed = int(digest(dict(task_hash=digest(task), start=start))[:15], 16)
    begin = time.monotonic()
    result = _sampler(path, canonical_json(task)).sample(shots, seed)
    d = result.diagnostics
    diagnostics = dict(
        stage_counts=[[list(k), v] for k, v in sorted(d.stage_counts.items())],
        flagged_stage_counts=[[list(k), v] for k, v in sorted(d.flagged_stage_counts.items())],
        rejections=[[list(k), v] for k, v in sorted(d.rejections.items())],
        rejection_cycles=sorted(d.rejection_cycles.items()))
    return dict(start=start, shots=shots, seed=seed, accepted=result.accepted,
                logical_errors=result.logical_errors, seconds=time.monotonic() - begin,
                cycles_started=d.cycles_started, cycles_completed=d.cycles_completed,
                total_cnots=d.total_cnots, diagnostics=canonical_json(diagnostics))


def summarize(directory):
    """Aggregate raw counts, never averages of batch rates or CI endpoints."""
    directory = Path(directory)
    summaries = []
    diagnostics = {}
    for path in sorted(directory.glob("tasks/*.json")):
        m = json.loads(path.read_text())
        task = m["task"]
        if digest(task) != path.stem or m["task_hash"] != path.stem:
            raise ValueError(f"Task manifest hash mismatch: {path}")
        rows = _read_batches(directory / "batches" / f"{path.stem}.csv")
        if not rows:
            continue
        totals = {k: sum(r[k] for r in rows) for k in
                  ("shots", "accepted", "logical_errors", "seconds", "cycles_started",
                   "cycles_completed", "total_cnots")}
        shots, accepted, errors = (totals[k] for k in ("shots", "accepted", "logical_errors"))
        low, high = _wilson_interval(errors, accepted)
        sl, sh = _wilson_interval(accepted, shots)
        summaries.append(dict(task_hash=path.stem, protocol_hash=task["protocol_hash"],
                              engine_hash=digest(task["engine"]), backend=task["backend"],
                              p=task["noise"]["p_2q"], cycles=task["cycles"],
                              prepared_state=task["prepared_state"], **totals,
                              discarded=shots - accepted,
                              ler=errors / accepted if accepted else math.nan,
                              ler_low=low, ler_high=high, survival=accepted / shots,
                              survival_low=sl, survival_high=sh,
                              mean_cnots=totals["total_cnots"] / shots,
                              bad_outcome_rate=(shots - accepted + errors) / shots,
                              noise=canonical_json(task["noise"]), seed=task["seed"]))
        summaries[-1]["sweeps"] = canonical_json(m.get("sweeps", []))
        summaries[-1]["target_shots"] = m.get("target_shots", shots)
        counters = {k: Counter() for k in ("stage_counts", "flagged_stage_counts", "rejections", "rejection_cycles")}
        for row in rows:
            for key, pairs in json.loads(row["diagnostics"]).items():
                counters[key].update({tuple(k) if isinstance(k, list) else k: v for k, v in pairs})
        diagnostics[path.stem] = {k: [[key, count] for key, count in sorted(c.items())]
                                  for k, c in counters.items()}
    if summaries:
        atomic_write(directory / "summary.csv", _csv_text(summaries, list(summaries[0])))
        atomic_write(directory / "diagnostics.json", json.dumps(diagnostics, indent=2) + "\n")
    return summaries


def run_sweep(protocol_path, configurations, results_dir="benchmarks/se", *, shots=100_000,
              batch_size=10_000, jobs=1, seed=0, allocation="fixed", max_shots=None, progress=print):
    """Reach a fixed attempted-shot target per task; committed batches resume.

    Only one batch per task is in flight. A single parent writes atomic CSV
    replacements. A directory lock prevents independent concurrent writers;
    uncommitted work after an interruption is safely repeated with its seed.
    """
    for name, value in (("shots", shots), ("batch_size", batch_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0 or type(jobs) is not int or jobs == 0:
        raise ValueError("seed must be nonnegative and jobs must be nonzero integers")
    if allocation not in {"fixed", "inverse-square"}:
        raise ValueError("allocation must be fixed or inverse-square")
    if max_shots is not None and (type(max_shots) is not int or max_shots < shots):
        raise ValueError("max_shots must be an integer at least shots")
    path, manifest = read_manifest(protocol_path)
    configurations = list(configurations)
    if not configurations:
        raise ValueError("Supply at least one simulation configuration")
    engine = engine_identity()
    tasks, targets, sweeps = {}, {}, {}
    for config in configurations:
        basis, cycles = config["prepared_state"], config["cycles"]
        if basis not in {"0", "+"} or type(cycles) is not int or cycles < 0:
            raise ValueError("Invalid basis or cycle count")
        noise = (CustomErrorModel(**config["noise"]) if "noise" in config
                 else scaled_noise_model(config["p"]))
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in vars(noise).values()):
            raise ValueError("Noise probabilities must be finite and between 0 and 1")
        task = dict(protocol_hash=manifest["protocol_hash"], backend="stim-reference",
                    engine=engine, noise=vars(noise), prepared_state=basis, cycles=cycles, seed=seed)
        key = digest(task)
        tasks[key] = task
        target = (num_samples(noise.p_2q, cycles, shots=shots, max_shots=max_shots)
                  if allocation == "inverse-square" else shots)
        targets[key] = max(targets.get(key, 0), target)
        sweeps.setdefault(key, set()).update([config["sweep"]] if config.get("sweep") else [])
    directory = Path(results_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # OS advisory locks are released even on SIGKILL; no stale lock cleanup.
    import fcntl
    with (directory / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another sweep is writing {directory}") from None
        pending = []
        records = {}
        for key, task in tasks.items():
            metadata_path = directory / "tasks" / f"{key}.json"
            previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            roles = sorted(sweeps[key] | set(previous.get("sweeps", [])))
            atomic_write(directory / "tasks" / f"{key}.json",
                         json.dumps(dict(task_hash=key, task=task, sweeps=roles,
                                         target_shots=targets[key], allocation=allocation,
                                         shot_floor=shots, max_shots=max_shots), indent=2) + "\n")
            records[key] = _read_batches(directory / "batches" / f"{key}.csv")
            done = sum(r["shots"] for r in records[key])
            if done < targets[key]:
                pending.append((key, done))
        workers = jobs if jobs > 0 else max(1, (os.cpu_count() or 1) + 1 + jobs)
        if progress:
            progress(f"{len(tasks)} tasks; {len(pending)} need shots; {workers} workers; "
                     f"{sum(targets.values()):,} total target shots ({allocation})")

        def commit(key, row):
            records[key].append(row)
            atomic_write(directory / "batches" / f"{key}.csv", _csv_text(records[key], BATCH_FIELDS))
            done = row["start"] + row["shots"]
            if progress:
                task = tasks[key]
                progress(f"p={task['noise']['p_2q']:.4g} N={task['cycles']} "
                         f"state={task['prepared_state']}: {done}/{targets[key]}")
            return done

        if workers == 1:
            for key, start in pending:
                while start < targets[key]:
                    row = _sample_batch(str(path), tasks[key], start, min(batch_size, targets[key] - start))
                    start = commit(key, row)
        else:
            # One task at a time per worker, with bounded outstanding futures.
            # Explicit spawn avoids inheriting simulator/compiler thread state.
            import multiprocessing
            from collections import deque
            queue = deque(pending)
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {}
                while queue or futures:
                    while queue and len(futures) < workers:
                        key, start = queue.popleft()
                        future = pool.submit(_sample_batch, str(path), tasks[key], start,
                                             min(batch_size, targets[key] - start))
                        futures[future] = key
                    ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in ready:
                        key = futures.pop(future)
                        start = commit(key, future.result())
                        if start < targets[key]:
                            queue.append((key, start))
        return summarize(directory)

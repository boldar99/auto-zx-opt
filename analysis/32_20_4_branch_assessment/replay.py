"""Replay saved one-cycle production seeds; condition outcomes on path flags.

This measures pruning of the existing controller. It neither forces a distance
switch nor builds a new protocol. Replayed shots must reproduce every saved
batch's acceptance/error counts exactly and must not be added to production.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import multiprocessing

import numpy as np

from spiderwarp.se_protocol_io import atomic_write, load_protocol, versions
from spiderwarp.se_protocol_simulation import ProtocolSampler, _NoisySimulator, _wilson_interval
from spiderwarp.noise_model import CustomErrorModel, default_noise_model

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRODUCTION = ROOT / "benchmarks/se_100k"
PROTOCOL = ROOT / "build/se_32_20_4"


def replay(task_path):
    metadata = json.loads(Path(task_path).read_text())
    task = metadata["task"]
    basis = task["prepared_state"]
    path = PRODUCTION / "batches" / (Path(task_path).stem + ".csv")
    batches = list(csv.DictReader(path.open()))
    protocol = load_protocol(PROTOCOL)
    sampler = ProtocolSampler(protocol, noise_model=CustomErrorModel(**task["noise"]),
                              prepared_state=basis, cycles=1, collect_paths=False)
    groups, paths = {}, {}
    checked = []
    for batch in batches:
        accepted = errors = 0
        rng = np.random.default_rng(int(batch["seed"]))
        for _ in range(int(batch["shots"])):
            simulator = sampler.experiment.initial.copy(seed=int(rng.integers(0, 2**63)))
            result = protocol.run(_NoisySimulator(simulator, sampler.circuits,
                                                 sampler.noise, sampler.corrections))
            failed = bool(result.accepted and sampler.experiment.logical_failure(
                simulator, frame_x=result.frame_x, frame_z=result.frame_z))
            accepted += result.accepted
            errors += failed
            distances = {d for _, _, d, _, _ in result.measurements}
            d4_flagged = any(d == 4 and f for _, _, d, f, _ in result.measurements)
            any_flags = any(f for _, _, _, f, _ in result.measurements)
            flagged_extractions = sum(bool(f) for _, _, _, f, _ in result.measurements)
            definitions = {
                "baseline": True,
                "all_d4_flags_zero": not d4_flagged,
                "all_flags_zero": not any_flags,
                "only_d4_and_all_flags_zero": distances == {4} and not any_flags,
                "keep_current_paths_without_d2": 2 not in distances,
                "keep_current_paths_without_d1": 1 not in distances,
                "at_most_one_flagged_extraction": flagged_extractions <= 1,
                "observed_d4_flag": d4_flagged,
                "observed_no_d4_flag": not d4_flagged,
                "observed_d1_used": 1 in distances,
                "observed_d2_used_without_d1": 2 in distances and 1 not in distances,
                "observed_only_d4": distances == {4},
            }
            for name, selected in definitions.items():
                if selected:
                    counts = groups.setdefault(name, Counter())
                    counts["selected_shots"] += 1
                    counts["accepted"] += result.accepted
                    counts["logical_errors"] += failed
            signature = ";".join(f"{b}{d}{'!' if f else ''}" for _, b, d, f, _ in result.measurements)
            counts = paths.setdefault(signature, Counter())
            counts["shots"] += 1
            counts["accepted"] += result.accepted
            counts["logical_errors"] += failed
        assert (accepted, errors) == (int(batch["accepted"]), int(batch["logical_errors"])), (
            basis, batch["start"], accepted, errors, batch["accepted"], batch["logical_errors"])
        checked.append(dict(start=int(batch["start"]), shots=int(batch["shots"]), seed=int(batch["seed"]),
                            accepted=accepted, logical_errors=errors))
        print(f"basis={basis}: replayed and matched {sum(b['shots'] for b in checked):,} shots", flush=True)
    total = sum(b["shots"] for b in checked)
    baseline_errors = groups["baseline"]["logical_errors"]
    rows = []
    for name, counts in sorted(groups.items()):
        a, e = counts["accepted"], counts["logical_errors"]
        lo, hi = _wilson_interval(e, a)
        sl, sh = _wilson_interval(a, total)
        rows.append(dict(prepared_state=basis, condition=name, shots=total, **counts,
                         ler=e/a if a else None, ler_low=lo, ler_high=hi, survival=a/total,
                         survival_low=sl, survival_high=sh, baseline_failure_fraction=e/baseline_errors))
    assert sum(groups[n]["logical_errors"] for n in ("observed_d4_flag", "observed_no_d4_flag")) == baseline_errors
    assert sum(groups[n]["logical_errors"] for n in ("observed_d1_used", "observed_d2_used_without_d1", "observed_only_d4")) == baseline_errors
    return dict(prepared_state=basis, task_hash=Path(task_path).stem,
                input_batches_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                batches_matched=checked, groups=rows, paths=paths)


if __name__ == "__main__":
    tasks = []
    for path in sorted((PRODUCTION / "tasks").glob("*.json")):
        task = json.loads(path.read_text())["task"]
        if task["cycles"] == 1 and task["noise"] == vars(default_noise_model()):
            tasks.append(str(path))
    assert len(tasks) == 2
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(replay, tasks))
    record = dict(methodology="Exact replay of saved default-noise N=1 production batches; 100k per basis",
                  scope="Posthoc filters and path attribution on the existing adaptive policy; no resynthesis or forced switching",
                  caveats=["Replayed shots are the original observations, not independent extra samples",
                           "No-d1/no-d2 filters prune existing paths; they do not implement d4-to-d2/d1 controllers",
                           "Flagged-extraction count is an observable filter, not a count of physical faults",
                           "Survival includes original policy rejection; no early-abort cost is estimated"],
                  versions=versions(), results=results)
    atomic_write(OUT / "assessment.json", json.dumps(record, indent=2)+"\n")
    rows = [row for r in results for row in r["groups"]]
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(OUT / "comparison.csv", buffer.getvalue())
    for row in rows:
        if row["condition"] in {"baseline", "all_d4_flags_zero", "all_flags_zero", "only_d4_and_all_flags_zero"}:
            print(row, flush=True)

"""Reproduce the three Steane RUS sweeps and overlay the saved adaptive data.

python -m spiderwarp.se_comparison --jobs 8
All budgets and settings match benchmarks/se_100k; completed batches resume.
"""
from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
import math
from pathlib import Path
import time

from spiderwarp.se_protocol_io import atomic_write, read_manifest
from spiderwarp.se_sweep import num_samples, run_sweep, small_sweep
from spiderwarp.steane_qec import LABELS, MODES, build_steane_qec


def save_resources(protocol, directory="benchmarks/steane_comparison"):
    from spiderwarp.noise_model import add_idling_gates, stim_circuit_to_abstract_circuit, partition_circ
    stages = {}
    for name, gadget in protocol.gadgets.items():
        abstract = stim_circuit_to_abstract_circuit(gadget.circuit)
        scheduled = add_idling_gates(abstract, persistent_qubits=range(protocol.code.n))
        stages[name] = dict(qubits=gadget.circuit.num_qubits,
            cnots=sum(len(op.targets_copy())//2 for op in gadget.circuit if op.name in {"CX", "CZ"}),
            depth=len(partition_circ(abstract)), measurements=gadget.circuit.num_measurements,
            live_idle_units=sum(op.arg for op in scheduled if op.name == "IDLE"),
            data_idle_units=sum(op.arg for op in scheduled if op.name == "IDLE" and op.targets[0] < protocol.code.n))
    atomic_write(Path(directory) / "resources.json", json.dumps(dict(
        peak_qubits=protocol.required_qubits, ideal_cycle_cnots=632, stages=stages), indent=2)+"\n")


def save_comparison(directories, labels, directory="benchmarks/steane_comparison"):
    """Save a combined count table and audit like-for-like experiment settings."""
    from spiderwarp.se_protocol_simulation import _wilson_interval
    all_rows, settings, totals = [], [], {}
    for i, (source, label) in enumerate(zip(directories, labels)):
        with (source / "summary.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        points = {}
        for row in rows:
            shots, accepted, errors = (int(row[k]) for k in ("shots", "accepted", "logical_errors"))
            if shots != int(row["target_shots"]):
                raise ValueError(f"Unfinished production budget in {source}")
            if not 0 <= errors <= accepted <= shots or (i and accepted != shots):
                raise ValueError(f"Invalid acceptance/error counts in {source}")
            for observed, expected in zip(("ler_low", "ler_high"), _wilson_interval(errors, accepted)):
                if not math.isclose(float(row[observed]), expected, abs_tol=1e-15):
                    raise ValueError(f"Invalid Wilson interval in {source}")
            key = (float(row["p"]), int(row["cycles"]), row["prepared_state"],
                   row["noise"], row["sweeps"], int(row["seed"]))
            if key in points:
                raise ValueError(f"Duplicate production point in {source}")
            points[key] = shots
            all_rows.append(dict(method=label, peak_qubits=60 if i == 0 else 66,
                                 results=str(source), **row))
        settings.append(points)
        totals[label] = sum(int(r["shots"]) for r in rows)
    if any(s != settings[0] for s in settings[1:]):
        raise ValueError("Comparison configurations, noise, seeds or budgets differ")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
    directory = Path(directory)
    atomic_write(directory / "comparison.csv", buffer.getvalue())
    atomic_write(directory / "validation.json", json.dumps(dict(
        complete=True, methods=labels, configurations_per_method=len(settings[0]),
        attempted_shots=totals, equal_noise_seeds_budgets_and_configurations=True,
        intervals="95% Wilson checked against raw counts",
        interpretation="Conditional 20-logical-qubit block LER; endpoint LER/N; survival separate",
        timing="Serial noisy ancilla attempts, including persistent data idles; ideal experiment boundaries",
        resource_difference="Current adaptive: 60 qubits; all conventional Steane variants: 66 qubits",
        syndrome_confirmation="Steane: two consecutive readings per CSS basis; adaptive: its existing policy",
        certificates="Adaptive certificate retained unchanged; no adaptive-policy certificate claimed for Steane RUS"
    ), indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--max-shots", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/se_100k"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    configs = small_sweep()
    directories = []
    for mode in args.modes:
        name = "steane_rus_" + mode.replace("-", "_")
        bundle, results = Path("build") / name, Path("benchmarks") / (name + "_100k")
        directories.append(results)
        if args.plot_only:
            continue
        if not (bundle / "protocol.json").exists():
            from spiderwarp.se_experiment import emit_protocol
            protocol = build_steane_qec(mode, output_dir=bundle)
            emit_protocol(protocol, bundle.resolve(), "hugr")
        targets = [num_samples(c.get("p", c.get("noise", {}).get("p_2q")), c["cycles"],
                               shots=args.shots, max_shots=args.max_shots) for c in configs]
        record = dict(status="running", mode=mode, label=LABELS[mode],
                      command=f"python -m spiderwarp.se_experiment sweep {bundle} --config {results}/config.json "
                              f"--shots {args.shots} --max-shots {args.max_shots} --batch-size {args.batch_size} "
                              f"--allocation inverse-square --jobs {args.jobs} --seed {args.seed} --results {results}",
                      protocol_hash=read_manifest(bundle)[1]["protocol_hash"], seed=args.seed,
                      configurations=len(configs), total_target_shots=sum(targets),
                      shot_floor=args.shots, max_shots=args.max_shots, batch_size=args.batch_size,
                      jobs=args.jobs, allocation="inverse-square", target_shots=targets,
                      preparation="ideal input once; all Steane ancilla attempts noisy",
                      final_decoding="ideal after last cycle", initial_data_idles=True,
                      retry_limit=None, syndrome_confirmation="two consecutive readings per basis",
                      timing="serial on-demand ancillas; zero classical branch latency/inter-cycle delay",
                      data_waits_during_retries=True, peak_qubits=66,
                      statistic="20-logical-qubit block LER in each basis; Wilson 95% intervals")
        atomic_write(results / "config.json", json.dumps(configs, indent=2) + "\n")
        atomic_write(results / "run.json", json.dumps(record, indent=2) + "\n")
        begin = time.monotonic()
        rows = run_sweep(bundle, configs, results, shots=args.shots, max_shots=args.max_shots,
                         batch_size=args.batch_size, jobs=args.jobs, seed=args.seed,
                         allocation="inverse-square")
        if len(rows) != len(configs) or any(r["shots"] != r["target_shots"] for r in rows):
            raise RuntimeError("Incomplete Steane sweep")
        if any(r["accepted"] != r["shots"] for r in rows):
            raise RuntimeError("Unbounded Steane RUS must not reject the data")
        record.update(status="complete", actual_attempted_shots=sum(r["shots"] for r in rows),
                      wall_seconds=time.monotonic() - begin)
        atomic_write(results / "run.json", json.dumps(record, indent=2) + "\n")
    from spiderwarp.se_plotting import plot_results
    labels = ["Current adaptive (60 qubits)"] + [LABELS[m] + " (66 qubits)" for m in args.modes]
    save_comparison([args.baseline] + directories, labels)
    from spiderwarp.se_protocol_io import load_protocol
    save_resources(load_protocol(Path("build") / ("steane_rus_" + args.modes[0].replace("-", "_"))))
    for path in plot_results(args.baseline, "benchmarks/steane_comparison/figures",
                             compare=directories, labels=labels):
        print(path, flush=True)


if __name__ == "__main__":
    main()

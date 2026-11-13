"""Reproduce the retained Steane RUS sweeps and overlay saved adaptive data.

python -m spiderwarp.se_comparison --jobs 8
Every plotted method must match the attempted-shot budget at each configuration.
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
from spiderwarp.se_comparison_names import METHODS

COMPARISON_MODES = ("d4", "ladder-ancilla")


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


def save_comparison(directories, labels, directory="benchmarks/steane_comparison", *,
                    qubit_counts=None, always_accept=None, require_equal_budgets=True):
    """Save a combined count table and audit like-for-like experiment settings."""
    from spiderwarp.se_protocol_simulation import _wilson_interval
    if not directories or len(directories) != len(labels):
        raise ValueError("Supply one label for each nonempty result-directory list")
    qubit_counts = [60] + [66]*(len(directories)-1) if qubit_counts is None else qubit_counts
    always_accept = [False] + [True]*(len(directories)-1) if always_accept is None else always_accept
    if len(qubit_counts) != len(directories) or len(always_accept) != len(directories):
        raise ValueError("Supply qubit counts and acceptance requirements for every method")
    all_rows, settings, totals = [], [], {}
    for i, (source, label) in enumerate(zip(directories, labels)):
        with (source / "summary.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No completed results in {source}")
        points = {}
        for row in rows:
            shots, accepted, errors = (int(row[k]) for k in ("shots", "accepted", "logical_errors"))
            if shots != int(row["target_shots"]):
                raise ValueError(f"Unfinished production budget in {source}")
            if not 0 <= errors <= accepted <= shots or (always_accept[i] and accepted != shots):
                raise ValueError(f"Invalid acceptance/error counts in {source}")
            for observed, expected in zip(("ler_low", "ler_high"), _wilson_interval(errors, accepted)):
                if not math.isclose(float(row[observed]), expected, abs_tol=1e-15):
                    raise ValueError(f"Invalid Wilson interval in {source}")
            key = (float(row["p"]), int(row["cycles"]), row["prepared_state"],
                   row["noise"], row["sweeps"], int(row["seed"]))
            if key in points:
                raise ValueError(f"Duplicate production point in {source}")
            points[key] = shots
            all_rows.append(dict(method=label, peak_qubits=qubit_counts[i],
                                 results=str(source), **row))
        settings.append(points)
        totals[label] = sum(int(r["shots"]) for r in rows)
    equal_budgets = all(s == settings[0] for s in settings[1:])
    if any(s.keys() != settings[0].keys() for s in settings[1:]) or (require_equal_budgets and not equal_budgets):
        raise ValueError("Comparison configurations, noise, seeds or budgets differ")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
    directory = Path(directory)
    atomic_write(directory / "comparison.csv", buffer.getvalue())
    atomic_write(directory / "validation.json", json.dumps(dict(
        complete=True, methods=labels, configurations_per_method=len(settings[0]),
        attempted_shots=totals, equal_noise_seeds_budgets_and_configurations=equal_budgets,
        equal_noise_seeds_and_configurations=True, equal_shot_budgets=equal_budgets,
        equal_attempted_shots_per_configuration=equal_budgets,
        sampling_note=("Matched fixed budgets" if equal_budgets else
                       "Unequal attempted-shot budgets; intervals use actual accepted counts"),
        intervals="95% Wilson checked against raw counts",
        interpretation="Conditional 20-logical-qubit block LER; endpoint LER/N; survival separate",
        timing="Serial noisy ancilla attempts, including persistent data idles; ideal experiment boundaries",
        peak_qubits=dict(zip(labels, qubit_counts)),
        full_acceptance_required=[label for label, required in zip(labels, always_accept) if required],
        syndrome_confirmation="Steane: two consecutive readings per CSS basis; adaptive: its existing policy",
        certificates="See each saved adaptive contract/certificate; no adaptive-policy certificate claimed for Steane RUS"
    ), indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--max-shots", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/se_100k"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(COMPARISON_MODES))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--postselected-results", type=Path,
                        help="Include this completed adaptive-PS run; default auto-detects benchmarks/se_postselected_100k")
    parser.add_argument("--steane-postselected-results", "--quick-postselected-results", type=Path, nargs="+",
                        dest="steane_postselected_results",
                        help="Completed Steane syndrome-PS runs with the same full budgets; the old flag is an alias")
    parser.add_argument("--adaptive-css-results", type=Path, nargs="+",
                        help="Completed adaptive CSS experiments; default auto-detects the selected full-budget methods")
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
    steane_names = {'d4': 'steane_d4', 'ladder-ancilla': 'steane_descending'}
    labels = [METHODS['adaptive_baseline'].label] + [
        METHODS[steane_names[m]].label if m in steane_names else LABELS[m] + ' (66 qubits)'
        for m in args.modes]
    qubits, full_acceptance = [60]+[66]*len(directories), [False]+[True]*len(directories)
    candidate = args.postselected_results or Path("benchmarks/se_postselected_100k")
    run_path = candidate / "run.json"
    complete = ((candidate / "summary.csv").exists() and
                (not run_path.exists() or json.loads(run_path.read_text()).get("status") == "complete"))
    if args.postselected_results and not complete:
        raise ValueError("The requested adaptive-PS result directory is not complete")
    if complete:
        directories.append(candidate)
        labels.append(METHODS['adaptive_history'].label)
        qubits.append(60)
        full_acceptance.append(False)
    steane_candidates = args.steane_postselected_results or [
        Path("benchmarks") / ("steane_css_ps_" + mode.replace("-", "_") + "_100k")
        for mode in args.modes]
    steane_ps_included = False
    for candidate in steane_candidates:
        run_path = candidate / "run.json"
        ready = run_path.exists() and json.loads(run_path.read_text()).get("status") == "complete"
        if not ready:
            archived = candidate.with_name(candidate.name.replace('_100k', '_10k'))
            if args.steane_postselected_results or run_path.exists() or archived.exists():
                raise ValueError(f"Steane syndrome-PS run is incomplete: {candidate}. Finish the full-budget sampling before plotting.")
            continue
        record = json.loads(run_path.read_text())
        key = 'steane_d4_ps' if record['mode'] == 'd4' else 'steane_descending_ps'
        directories.append(candidate)
        labels.append(METHODS[key].label)
        qubits.append(66)
        full_acceptance.append(False)
        steane_ps_included = True
    adaptive_candidates = args.adaptive_css_results or [Path('benchmarks/adaptive_css_100k'),
        Path('benchmarks/adaptive_css_recover_safe_100k')]
    for candidate in adaptive_candidates:
        run_path = candidate / 'run.json'
        ready = run_path.exists() and json.loads(run_path.read_text()).get('status') == 'complete'
        if not ready:
            if args.adaptive_css_results:
                raise ValueError(f'Requested adaptive CSS run is incomplete: {candidate}')
            continue
        record = json.loads(run_path.read_text())
        label = (METHODS['adaptive_syndrome'].label if record['variant'] == 'css' else
                 METHODS['adaptive_recovery'].label if record['variant'] == 'css-recover-safe' else
                 'Adaptive experimental recovery (relaxed EC; 60 qubits)')
        directories.append(candidate)
        labels.append(label)
        qubits.append(60)
        full_acceptance.append(False)
    save_comparison([args.baseline] + directories, labels,
                    qubit_counts=qubits, always_accept=full_acceptance,
                    require_equal_budgets=True)
    from spiderwarp.se_protocol_io import load_protocol
    save_resources(load_protocol(Path("build") / ("steane_rus_" + args.modes[0].replace("-", "_"))))
    for path in plot_results(args.baseline, "benchmarks/steane_comparison/figures",
                             compare=directories, labels=labels, physical_survival=complete or steane_ps_included,
                             sample_note='Equal attempted shots at every matching configuration; 95% Wilson intervals'):
        print(path, flush=True)


if __name__ == "__main__":
    main()

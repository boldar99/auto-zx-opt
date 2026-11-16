"""Matched fixed-budget comparison using the Sol runtime and Helios alpha model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spiderwarp.se_comparison_names import METHODS
from spiderwarp.se_protocol_io import atomic_write
from spiderwarp.se_sol import MODEL_VERSION, _csv, run_sol_sweep

BUNDLES = {
    "adaptive_history": "se_postselected_32_20_4",
    "adaptive_scheduled": "adaptive_scheduled_32_20_4",
    "adaptive_order_two": "adaptive_order_two_32_20_4",
    "adaptive_probability_10": "adaptive_probability_10_32_20_4",
    "adaptive_probability_25": "adaptive_probability_25_32_20_4",
    "adaptive_no_flags": "adaptive_css_no_flags_32_20_4",
    "steane_fixed_ps": "steane_single_d3_fixed_ps",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(BUNDLES), nargs="+", help="Run just these methods")
    parser.add_argument("--bundle-root", type=Path, default=Path("build"))
    parser.add_argument("--results", type=Path, default=Path("benchmarks/sol_v2_comparison"))
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--cycles", type=int, nargs="+", default=[1, 3, 5, 7, 10])
    parser.add_argument("--basis", choices=["0", "+", "both"], default="both")
    parser.add_argument("--sol-python", type=Path)
    parser.add_argument("--repump", action="store_true")
    parser.add_argument("--input-sync", choices=["cover", "full"], default="cover")
    parser.add_argument("--retain-shots", action="store_true")
    args = parser.parse_args(argv)
    bases = ("0", "+") if args.basis == "both" else (args.basis,)
    configurations = [dict(noise_scale=scale, cycles=cycles, prepared_state=basis)
                      for scale in args.noise_scales for cycles in args.cycles for basis in bases]
    combined, settings = [], []
    for key in (dict.fromkeys(args.only) if args.only else BUNDLES):
        method = METHODS[key]
        rows = run_sol_sweep(args.bundle_root / BUNDLES[key], configurations,
                            args.results / key, shots=args.shots, batch_size=args.batch_size,
                            jobs=args.jobs, seed=args.seed, python=args.sol_python,
                            repump=args.repump, retain_shots=args.retain_shots, input_sync=args.input_sync)
        expected = {(c["noise_scale"], c["cycles"], c["prepared_state"]) for c in configurations}
        # A results tree may contain previous grids. Compare only this request.
        rows = [r for r in rows if (r["noise_scale"], r["cycles"], r["prepared_state"]) in expected
                and r["seed"] == args.seed and json.loads(r["noise"])["repump"] == args.repump
                and r["input_sync"] == args.input_sync]
        points = {}
        for row in rows:
            if row["shots"] < args.shots or row["backend"] != MODEL_VERSION:
                raise RuntimeError("Incomplete Sol comparison")
            point = (row["noise_scale"], row["cycles"], row["prepared_state"],
                     row["noise"], row["seed"], row["engine_hash"], row["input_sync"])
            if point in points:
                raise RuntimeError("Multiple Sol engine identities for one comparison point; use a fresh results tree")
            points[point] = row["shots"]
            combined.append(dict(method=key, display_name=method.name, **row))
        if len(points) != len(expected):
            raise RuntimeError("Missing Sol comparison points")
        settings.append(points)
    if any(points != settings[0] for points in settings[1:]):
        raise RuntimeError("Sol comparison noise, engine, seeds or attempted budgets differ")
    atomic_write(args.results / "comparison.csv", _csv(combined, list(combined[0])))
    atomic_write(args.results / "validation.json", json.dumps(dict(
        backend=MODEL_VERSION, complete=True, methods=sorted({r["method"] for r in combined}),
        configurations_per_method=len(settings[0]), equal_noise_engine_seeds_and_attempted_budgets=True,
        data_preparation="Ideal once; nondestructive stabilizer synchronization before noise starts",
        input_sync=args.input_sync, seed_mode="default",
        final_readout="Ideal once; noiseless classical LutDecoder after the last accepted cycle",
        statistics="Conditional block LER and gate/duration costs on fully accepted trajectories only",
        approximations="Incoherent memory and leakage-to-depolarization; Helios alpha used for Sol",
        probability_policies="Frozen under their original training model, not Sol-calibrated",
        certificate_scope="Existing abstract-circuit certificates; no Sol hardware certificate"), indent=2) + "\n")
    print(f"Sol comparison: {(args.results / 'comparison.csv').resolve()}")


if __name__ == "__main__":
    main()

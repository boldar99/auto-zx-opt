"""Static figures matching the 713 notebook's two SE experiment views."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from spiderwarp.se_protocol_simulation import _wilson_interval
from spiderwarp.noise_model import default_noise_model


def plot_results(results, output_dir=None, *, cycle_p=None, compare=None, labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cycle_p = default_noise_model().p_2q if cycle_p is None else cycle_p

    results = Path(results)
    source = results / "summary.csv" if results.is_dir() else results
    directory = Path(output_dir) if output_dir else source.parent / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    sources = [source] + [Path(p) / "summary.csv" if Path(p).is_dir() else Path(p) for p in (compare or [])]
    if labels is not None and len(labels) != len(sources):
        raise ValueError("Supply one label for each result directory")
    rows, names = [], {}
    for i, csv_path in enumerate(sources):
        with csv_path.open(newline="") as f:
            selected = list(csv.DictReader(f))
        for row in selected:
            if labels is not None:
                names[row["protocol_hash"]] = labels[i]
            elif compare:
                names[row["protocol_hash"]] = csv_path.parent.name
        rows.extend(selected)
    if not rows:
        raise ValueError("No simulation results to plot")
    groups = defaultdict(lambda: defaultdict(lambda: dict(counts=[0, 0, 0, 0], sweeps=set())))
    for row in rows:
        p, cycles = float(row["p"]), int(row["cycles"])
        if p <= 0 or cycles <= 0:
            continue  # Controls are retained in CSV, excluded from log plots.
        noise = json.loads(row["noise"])
        profile = tuple((k, round(v / p, 12)) for k, v in sorted(noise.items()))
        key = (row["protocol_hash"], row["engine_hash"], profile, row["prepared_state"])
        point = groups[key][p, cycles]
        point["sweeps"].update(json.loads(row.get("sweeps") or "[]"))
        totals = point["counts"]
        for i, name in enumerate(("shots", "accepted", "logical_errors", "total_cnots")):
            totals[i] += int(row[name])
    if not groups:
        raise ValueError("No positive-p, positive-cycle results to plot")

    def series(ax, x, totals, scale, label, *, survival=False, resources=False):
        if resources:
            ax.plot(x, [c[3] / c[0] / divisor for c, divisor in zip(totals, scale)],
                    "o-", label=label)
            return
        y, lo, hi, zero = [], [], [], []
        for (shots, accepted, errors, _), divisor in zip(totals, scale):
            k, n = (accepted, shots) if survival else (errors, accepted)
            low, high = _wilson_interval(k, n)
            y.append(k / n / divisor if n else np.nan)
            lo.append(low / divisor if n else np.nan)
            hi.append(high / divisor if n else np.nan)
            zero.append(k == 0 and n > 0 and not survival)
        x, y, lo, hi, zero = map(np.asarray, (x, y, lo, hi, zero))
        valid = np.isfinite(y) & ~zero
        line = ax.errorbar(x[valid], y[valid],
                           yerr=[np.maximum(0, y[valid] - lo[valid]), hi[valid] - y[valid]],
                           fmt="o-", capsize=3, label=label)
        color = line.lines[0].get_color()
        ax.fill_between(x[valid], lo[valid], hi[valid], color=color, alpha=.12)
        if zero.any():
            # At zero failures the point estimate cannot appear on a log axis.
            ax.scatter(x[zero], hi[zero], marker="v", facecolors="none", edgecolors=color,
                       label=f"{label}: {'zero accepted' if survival else 'zero failures'} (95% CI upper endpoint)")

    paths = []
    profile_ids = {key: i + 1 for i, key in enumerate(sorted({k[:3] for k in groups}))}
    from spiderwarp.se_protocol_io import atomic_write
    atomic_write(directory / "models.json", json.dumps([
        dict(model=i, protocol_hash=key[0], label=names.get(key[0]), engine_hash=key[1], noise_ratios=dict(key[2]))
        for key, i in profile_ids.items()], indent=2) + "\n")
    modes = ("physical_error", "cycles", "survival") + (("cnot_cost",) if compare else ())
    for mode in modes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), squeeze=False)
        plotted = False
        tick_values = set()
        for key, points in groups.items():
            protocol, engine, profile, basis = key
            ax = axes[0, 0 if basis == "0" else 1]
            selected = []
            for (p, n), point in points.items():
                roles = point["sweeps"]
                belongs = ((not roles or "physical_error" in roles) and n == 1
                           if mode == "physical_error" else
                           (not roles or "cycles" in roles) and np.isclose(p, cycle_p, rtol=1e-9, atol=0))
                if belongs:
                    selected.append((p, n, point["counts"]))
            selected.sort()
            if not selected:
                continue
            plotted = True
            xs = [p if mode == "physical_error" else n for p, n, _ in selected]
            if mode != "physical_error":
                tick_values.update(xs)
            scales = [p * p if mode == "physical_error" else n if mode in {"cycles", "cnot_cost"} else 1
                      for p, n, _ in selected]
            series(ax, xs, [c for _, _, c in selected], scales,
                   names.get(protocol, f"{protocol[:8]} / model {profile_ids[key[:3]]}"),
                   survival=mode == "survival", resources=mode == "cnot_cost")
        if not plotted:
            plt.close(fig)
            continue
        for i, ax in enumerate(axes[0]):
            ax.set_title("Logical 0 (Z readout)" if i == 0 else "Logical + (X readout)")
            ax.grid(True, which="both", ls="--", alpha=.4)
            if mode == "physical_error":
                ax.set_xscale("log")
                ax.set_yscale("log")
                positive = [float(r["p"]) for r in rows if float(r["p"]) > 0 and int(r["cycles"]) == 1]
                if positive:
                    x = np.geomspace(min(positive), max(positive), 100)
                    ax.plot(x, 1 / x, ":", color="gray", label="p_L = p guide")
                ax.set_xlabel("Two-qubit error probability p")
                ax.set_ylabel("Conditional block failure probability / p²")
            else:
                ax.set_xlabel("Number of QEC cycles N")
                ax.set_xticks(sorted(tick_values))
                ax.set_ylabel("Conditional block failure probability / N" if mode == "cycles"
                              else "Mean CNOTs per attempted trajectory / N" if mode == "cnot_cost"
                              else "Probability all cycles accept")
                if mode == "cycles":
                    ax.set_yscale("log")
                elif mode == "survival":
                    ax.set_yscale("linear")
                    # Include the measured confidence intervals and a little
                    # padding, rather than forcing the entire probability range.
                    ax.margins(y=.08)
                    bottom, top = ax.get_ylim()
                    ax.set_ylim(max(0, bottom), min(1, top))
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize="small")
        fig.suptitle("Basis-specific block performance"
                     + (f"; p={cycle_p:g}" if mode != "physical_error" else ""))
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            path = directory / f"{mode}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            paths.append(path.resolve())
        plt.close(fig)
    return paths

"""Readable mask tables and holdout postselection tradeoff figures."""
from collections import Counter, defaultdict
import csv
import gzip
from io import StringIO
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from patterns import OUT
from spiderwarp.se_protocol_io import atomic_write
from spiderwarp.se_protocol_simulation import _wilson_interval


def write_csv(path, rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue())


def main():
    record = json.loads((OUT / "patterns_assessment.json").read_text())
    names = sorted({v["action"] for masks in record["learned_bad_masks"].values() for v in masks}
                   | {"X_d1", "X_d2", "X_d4", "Z_d1", "Z_d2", "Z_d4"})
    patterns = list(csv.DictReader((OUT / "pattern_risk.csv").open()))
    grouped = defaultdict(dict)
    for row in patterns:
        if row["split"] == "validation" and row["kind"] == "action_mask":
            grouped[row["action"], row["mask_hex"]][row["prepared_state"]] = row
    summary = []
    for (action, mask), bases in sorted(grouped.items()):
        first = next(iter(bases.values()))
        values = dict(action=action, mask_hex=mask, bits_lsb_first=first["bits_lsb_first"],
                      set_bits=first["set_bits"], min_internal_order_for_mask=first["min_internal_order_for_mask"])
        for basis, suffix in (("0", "zero"), ("+", "plus")):
            row = bases.get(basis, {})
            for key in ("encountered", "accepted", "logical_errors", "p_error_given_accepted", "low", "high"):
                values[key+"_"+suffix] = row.get(key, "")
        summary.append(values)
    write_csv(OUT / "flag_pattern_summary.csv", summary)

    flag_rows = []
    for split in ("discovery", "validation"):
        for basis, suffix in (("0", "zero"), ("+", "plus")):
            counts = defaultdict(Counter)
            with gzip.open(OUT / f"{split}_{suffix}.jsonl.gz", "rt") as stream:
                for line in stream:
                    _, _, accepted, failed, observations, _, compatible = json.loads(line)
                    has_d4_flag = any(names[aid].endswith("d4") and f for aid, f, _ in observations)
                    for label in ("d4_flag" if has_d4_flag else "no_d4_flag",
                                  "no_order01_explanation" if not compatible else "has_order01_explanation"):
                        counts[label]["encountered"] += 1
                        counts[label]["accepted"] += accepted
                        counts[label]["logical_errors"] += failed
            for label, values in counts.items():
                a, e = values["accepted"], values["logical_errors"]
                lo, hi = _wilson_interval(e, a)
                flag_rows.append(dict(split=split, prepared_state=basis, condition=label, **values,
                                      p_error_given_accepted=e/a, low=lo, high=hi))
    write_csv(OUT / "conditional_risk.csv", flag_rows)

    policies = list(csv.DictReader((OUT / "postselection_policies.csv").open()))
    labels = {
        "baseline": "Current acceptance",
        "all_d4_flags_zero": "All d4 flags zero",
        "reject_flag_evidence_ge2": "PS on 2+ faults from flags",
        "reject_no_order01_explanation": "PS when no order-0/1 history remains",
        "learned_bad_masks_0.05": "Raw-mask risk PS (5% threshold)",
    }
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, basis in enumerate(("0", "+")):
        ax = axes[i]
        for j, (rule, label) in enumerate(labels.items()):
            row = next(r for r in policies if r["split"] == "validation"
                       and r["prepared_state"] == basis and r["rule"] == rule)
            x, y = 100*float(row["survival"]), 100*float(row["p_error_given_accepted"])
            ax.errorbar(x, y,
                xerr=[[x-100*float(row["survival_low"])] , [100*float(row["survival_high"])-x]],
                yerr=[[y-100*float(row["low"])] , [100*float(row["high"])-y]],
                fmt=("o", "s", "^", "D", "v")[j], capsize=3, markersize=6, label=label)
        ax.set_title("Logical 0 (Z readout)" if basis == "0" else "Logical + (X readout)")
        ax.set_xlabel("Probability cycle accepts (%)")
        ax.set_ylabel("Conditional block LER (%)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=.3, linestyle="--")
        ax.legend(fontsize=8)
    fig.suptitle("One-cycle postselection tradeoff: independent 100k-shot validation per basis")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"postselection_tradeoff.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    common = [r for r in summary if r["action"].endswith("d4") and r["mask_hex"] != "0x000"
              and r["accepted_zero"] and r["accepted_plus"]
              and min(int(r["accepted_zero"]), int(r["accepted_plus"])) >= 100]
    common.sort(key=lambda r: max(float(r["p_error_given_accepted_zero"]),
                                 float(r["p_error_given_accepted_plus"])), reverse=True)
    common = common[:10]
    fig, ax = plt.subplots(figsize=(9, 6))
    values = np.array([[100*float(r["p_error_given_accepted_"+s]) for s in ("zero", "plus")] for r in common])
    plot = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=max(20, values.max()), aspect="auto")
    ax.set_xticks([0, 1], ["Logical 0", "Logical +"])
    ax.set_yticks(range(len(common)), [r["action"] + " / " + r["mask_hex"] for r in common])
    for i, row in enumerate(common):
        for j, suffix in enumerate(("zero", "plus")):
            ax.text(j, i, f"{values[i,j]:.2f}%\n{row['logical_errors_'+suffix]} / {row['accepted_'+suffix]}",
                    ha="center", va="center", color="white" if values[i,j] > 12 else "black", fontsize=10)
    ax.set_title("Highest-risk common d4 masks: independent validation\nAt least 100 accepted cases per basis; error counts shown")
    fig.colorbar(plot, ax=ax, label="P(block error | accepted and mask appears) (%)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"common_d4_mask_risk.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Wrote wide mask table, conditional-risk table, and both figures.")


if __name__ == "__main__":
    main()

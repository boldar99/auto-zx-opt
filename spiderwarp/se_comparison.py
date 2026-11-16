"""Run the bounded, no-d1 adaptive PS and fixed Steane comparison.

python -m spiderwarp.se_comparison --jobs 8
Every plotted method must match the attempted-shot budget at each configuration.
"""
from __future__ import annotations

import csv
from io import StringIO
import json
import math
from pathlib import Path

from spiderwarp.se_protocol_io import atomic_write, read_manifest



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
        peak_qubits=protocol.required_qubits, ideal_cycle_cnots=sum(
            stages[name]["cnots"] for name in ("prep_Z_d3", "read_Z", "prep_X_d3", "read_X")), stages=stages), indent=2)+"\n")


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
        syndrome_confirmation="Steane: one reading per CSS basis; adaptive: its existing policy",
        certificates="See each saved adaptive contract/certificate; no adaptive-policy certificate claimed for Steane RUS"
    ), indent=2) + "\n")


def main(argv=None):
    """Run Sol by default; retain explicit access to the historical reference."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=['sol', 'stim-reference'], default='sol')
    args, remaining = parser.parse_known_args(argv)
    if args.backend == 'sol':
        from spiderwarp.se_sol_comparison import main as sol_main
        return sol_main(remaining)
    from importlib import import_module
    return import_module('analysis.32_20_4_bounded_comparison.study').main(remaining)


if __name__ == "__main__":
    main()

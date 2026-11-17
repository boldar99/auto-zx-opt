"""Prepare an encoded state, run QEC cycles, and save memory AR/LER results.

The registered initial preparations target all twenty logical Z outcomes zero.
Each QEC method is a complete Steane cycle; its suffix selects the preparation
of its ancillas. RUS methods include up to four retries after their first
attempt. Initialization, QEC, and terminal readout all use the Sol noise model.

Acceptance requires successful initialization, every requested QEC cycle, and
accepted decoded terminal Z readout. LER is the fraction of accepted blocks
with any wrong logical bit. AR uses all attempted trajectories. Rejected shots
stop immediately and still emit both summary bits, so they remain in the AR
denominator. Zero cycles runs an initialization/readout baseline.

From the repository root::

    uv run --project local_sol_example --no-sync python -m sol_experiment.experiment_memory --initial-prep d4_RUS --qec-method steane_d3_RUS --num-cycles 10 --shots 10000

The JSON report is saved under out/memory/ unless --output specifies a file.
The factory can also accept custom Guppy methods. Z readout is used throughout;
expected_logicals optionally specifies a nonzero computational-basis target as
a twenty-bit mask, with bit i the expected outcome of logical Z operator i.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
from typing import Literal

from guppylang import guppy, OptimizationLevel
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, output
from guppylang.std.quantum import discard_array, qubit, h

from local_sol_example.sol.sim import run_hugr

from sol_experiment.logical_measurement import logical_z_meas
from sol_experiment.logical_state_preparation import PREPARATION_METHODS
from sol_experiment.experiment_SPAM import (
    SPAMStatistics, print_summary, summarize_result,
)
from sol_experiment.QEC_optimized_steane import make_optimized_steane_qec_cycle
from sol_experiment.QEC_steane import make_steane_qec_cycle
from sol_experiment.optimized_steane_dispatchers import zero_flag_dispatcher

QEC_METHODS = {
    f"steane_{name}": make_steane_qec_cycle(prepare_state)
    for name, prepare_state in PREPARATION_METHODS.items()
} | {
    "optimized_steane": make_optimized_steane_qec_cycle(zero_flag_dispatcher)
}


def make_memory_experiment(
    initial_state_prep: GuppyFunctionDefinition,
    qec_method: GuppyFunctionDefinition,
    num_cycles: int,
    basis: Literal["Z", "X"]
) -> GuppyFunctionDefinition:
    """Return a zero-argument Guppy memory program emitting two bits per shot.

    initial_state_prep borrows a fresh array[qubit, 32] and returns an integer
    verification mask, with zero meaning success. qec_method borrows the same
    data array, applies its corrections, and returns a bool acceptance bit.
    Methods and settings are bound before compilation; the QEC loop and all
    outcome-dependent decisions execute in Guppy at runtime.

    On rejection the remaining cycles and final readout are skipped, and the
    data is discarded. On success logical_z_meas consumes the block. The two
    outputs are always accepted and logical_error, compatible with the SPAM
    result summarizer. A registered QEC method needs at most 66 live qubits.
    """
    if isinstance(num_cycles, bool) or not isinstance(num_cycles, int):
        raise TypeError("num_cycles must be an integer")
    if num_cycles < 0:
        raise ValueError("num_cycles must be nonnegative")

    @guppy
    def experiment() -> None:
        data = array(qubit() for _ in range(32))
        flags = initial_state_prep(data)
        if comptime(basis == "X"):
            for i in range(32): h(data[i])
        if flags != 0:
            discard_array(data)
            output("accepted", False)
            output("logical_error", False)
            return

        for _ in range(comptime(num_cycles)):
            if not qec_method(data):
                discard_array(data)
                output("accepted", False)
                output("logical_error", False)
                return

        if comptime(basis == "X"):
            for i in range(32): h(data[i])
        readout = logical_z_meas(data)
        logical_error = False
        for i in range(20):
            expected = (0 & (1 << i)) != 0
            logical_error |= readout.logicals[i] != expected
        output("accepted", readout.accepted)
        output("logical_error", readout.accepted and logical_error)

    return experiment


def save_results(
    path: str | Path,
    stats: SPAMStatistics,
    configuration: dict[str, object],
) -> Path:
    """Save settings, counts and fractional rates; undefined rates become null."""
    path = Path(path).expanduser().resolve()
    record = {
        "experiment": "memory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "shots": stats.shots,
        "accepted": stats.accepted,
        "logical_errors": stats.logical_errors,
        "AR": stats.ar,
        "LER": stats.ler,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> Path:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--initial-prep", choices=PREPARATION_METHODS, default="d3_RUS",
        help="Initial data-block preparation",
    )
    parser.add_argument(
        "--qec-method", choices=QEC_METHODS, default="optimized_steane",
        help="Complete QEC cycle; suffix names its ancilla preparation",
    )
    parser.add_argument("--num-cycles", type=int, default=1, help="QEC cycles per trajectory")
    parser.add_argument("--basis", type=str, default="Z", choices=["Z", "X"], help="QEC cycles per trajectory")
    parser.add_argument("--shots", type=int, default=1000, help="Number of trajectories")
    parser.add_argument("--processes", type=int, default=8, help="Parallel simulation processes")
    parser.add_argument("--seed", type=int, help="Simulation seed; generated and saved if omitted")
    parser.add_argument("--output", type=Path, help="JSON report path; otherwise out/memory/<timestamp>.json")
    args = parser.parse_args(argv)
    if args.num_cycles < 0:
        parser.error("--num-cycles must be nonnegative")
    if args.shots < 1 or args.processes < 1:
        parser.error("--shots and --processes must be positive")
    if args.seed is not None and not 0 <= args.seed < (1 << 64):
        parser.error("--seed must be an unsigned 64-bit integer")

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    program = make_memory_experiment(
        PREPARATION_METHODS[args.initial_prep], QEC_METHODS[args.qec_method],
        args.num_cycles, args.basis,
    )
    hugr = program.with_opt_level(OptimizationLevel.Default).compile()
    result = run_hugr(
        hugr, n_shots=args.shots, n_qubits=66, stabilizer=True,
        n_processes=args.processes, seed=seed, leakage_repumping=True,
    )
    stats = summarize_result(result)
    if stats.shots != args.shots:
        raise RuntimeError(f"Expected {args.shots} shots, received {stats.shots}")

    configuration = {
        "initial_state_prep": args.initial_prep,
        "qec_method": args.qec_method,
        "num_cycles": args.num_cycles,
        "seed": seed,
        "processes": args.processes,
        "n_qubits": 66,
        "platform": "sol",
        "simulator": "stim",
        "noise_model": "qsystem_model_alpha",
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f" + args.basis)
    path = save_results(
        args.output or Path("out/memory") / f"{timestamp}.json", stats, configuration,
    )
    print_summary(stats)
    print(f"Saved results: {path}")
    return path


if __name__ == "__main__":
    main()

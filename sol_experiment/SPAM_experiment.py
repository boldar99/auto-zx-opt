"""Estimate accepted-block SPAM error and acceptance rates for d3 preparation.

Each shot makes one preparation attempt, without RUS retries. Acceptance
requires both zero preparation flags and accepted decoded Z readout. A logical
failure means any of the 20 corrected logical bits differs from the expected
all-zero state. LER is conditional on acceptance; AR uses all attempted shots.

Run from the repository root, for example::

    uv run --project local_sol_example --no-sync python -m sol_experiment.SPAM_experiment --shots 10000
"""

from dataclasses import dataclass

from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit

from sol_experiment.logical_measurement import logical_z_meas
from sol_experiment.state_preparation import state_prep_32_20_4_zero_d3


@guppy
def estimate_SPAM_LER() -> None:
    """One d3 preparation/readout attempt, emitting only two summary bits."""
    data = array(qubit() for _ in range(32))
    flags = state_prep_32_20_4_zero_d3(data)
    logical_measurement = logical_z_meas(data)
    accepted = flags == 0 and logical_measurement.accepted
    logical_error = False
    for bit in logical_measurement.logicals:
        logical_error |= bit
    output("accepted", accepted)
    output("logical_error", accepted and logical_error)


@dataclass(frozen=True)
class SPAMStatistics:
    shots: int
    accepted: int
    logical_errors: int

    @property
    def ar(self) -> float | None:
        return self.accepted / self.shots if self.shots else None

    @property
    def ler(self) -> float | None:
        return self.logical_errors / self.accepted if self.accepted else None


def summarize_result(result) -> SPAMStatistics:
    """Count accepted logical failures while iterating over a QsysResult."""
    shots = accepted = logical_errors = 0
    for shot in result:
        shots += 1
        values = shot.as_dict()
        if values["accepted"]:
            accepted += 1
            logical_errors += bool(values["logical_error"])
    return SPAMStatistics(shots, accepted, logical_errors)


def print_summary(stats: SPAMStatistics) -> None:
    ar = f"{stats.ar:.6%}" if stats.ar is not None else "undefined"
    print(f"AR:  {ar} ({stats.accepted:,}/{stats.shots:,} shots accepted)")
    if stats.ler is None:
        print("LER: undefined (no accepted shots)")
    else:
        print(
            f"LER: {stats.ler:.6%} "
            f"({stats.logical_errors:,}/{stats.accepted:,} accepted blocks have a logical error)"
        )


def main() -> None:
    import argparse

    from guppylang import OptimizationLevel
    from local_sol_example.sol.sim import run_hugr, shot_viewer

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=int, default=10000, help="Number of preparation attempts")
    parser.add_argument("--seed", type=int, help="Simulation seed")
    parser.add_argument("--processes", type=int, default=8, help="Parallel simulation processes")
    parser.add_argument("--view-shot", action="store_true", help="Open the shot viewer after sampling")
    args = parser.parse_args()
    if args.shots < 1 or args.processes < 1:
        parser.error("--shots and --processes must be positive")

    hugr = estimate_SPAM_LER.with_opt_level(OptimizationLevel.Minimal).compile()
    result = run_hugr(
        hugr, n_shots=args.shots, n_qubits=34, stabilizer=True,
        n_processes=args.processes, seed=args.seed,
    )
    print_summary(summarize_result(result))

    if args.view_shot:
        shot_viewer(hugr)


if __name__ == "__main__":
    main()

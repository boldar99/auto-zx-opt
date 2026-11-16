"""Run executable Sol controls against all seven saved comparison bundles."""
import argparse
import json
from pathlib import Path

from spiderwarp.se_protocol_io import atomic_write
from spiderwarp.se_sol import run_sol_sweep
from spiderwarp.se_sol_comparison import BUNDLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/sol_v2_controls"))
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cycles", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--only", choices=tuple(BUNDLES))
    parser.add_argument("--input-sync", choices=["cover", "full"], default="cover")
    args = parser.parse_args()
    configurations = [dict(noise_scale=0, cycles=cycles, prepared_state=basis)
                      for cycles in args.cycles for basis in ("0", "+")]
    report = {}
    for key in ([args.only] if args.only else BUNDLES):
        print("Sol controls:", key, flush=True)
        rows = run_sol_sweep(Path("build") / BUNDLES[key], configurations, args.out / key,
                            shots=args.shots, batch_size=args.shots, jobs=args.jobs,
                            seed=20261113, incoming_x=1, incoming_z=1 << 17, retain_shots=True,
                            input_sync=args.input_sync)
        assert len(rows) == len(configurations)
        assert all(r["accepted"] == args.shots and r["logical_errors"] == 0 for r in rows)
        # These Clifford exports lower each CX/CZ to one native RPP. Equality
        # proves no ideal preparation or readout entangler crosses the boundary.
        assert all(r["total_cnots"] == r["total_native_rzz"] for r in rows)
        report[key] = rows
        atomic_write(args.out / "validation.json", json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

"""Build, export, reload, simulate and plot adaptive Steane SE experiments.

Run ``python -m spiderwarp.se_experiment --help`` from an installed checkout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from spiderwarp.se_protocol_io import atomic_write, load_protocol, read_manifest, versions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Synthesize, verify and save a protocol bundle")
    build.add_argument("--code", default="misc/32_20_4", help="Code family/name")
    build.add_argument("--circuits", type=Path, help="Directory with <code>_d<label>.stim assets")
    build.add_argument("--distances", nargs="+", type=int, default=[1, 2, 4])
    build.add_argument("--out", type=Path, default=Path("build/se_32_20_4"))
    build.add_argument("--peak-qubits", type=int, default=60)
    build.add_argument("--max-cnot-cost", type=int)
    build.add_argument("--max-search-states", type=int, default=100000)
    build.add_argument("--optimize", action="store_true", help="Run ZX optimization before synthesis")
    build.add_argument("--feasible-only", action="store_true", help="Skip minimax search")
    build.add_argument("--emit", choices=["none", "guppy", "hugr"], default="guppy")
    compile_ = sub.add_parser("compile", help="Generate source/HUGR from an existing bundle")
    compile_.add_argument("protocol", type=Path)
    compile_.add_argument("--emit", choices=["guppy", "hugr"], default="hugr")
    compile_.add_argument("--verify", action="store_true", help="Re-run the independent fault checker")
    run = sub.add_parser("run", help="Sample one configuration, with resumable batches")
    sweep = sub.add_parser("sweep", help="Sample a grid, with resumable batches")
    for cmd in (run, sweep):
        cmd.add_argument("protocol", type=Path)
        cmd.add_argument("--shots", type=int, default=100000, help="Minimum attempted shots per configuration")
        cmd.add_argument("--batch-size", type=int, default=10000)
        cmd.add_argument("--allocation", choices=["fixed", "inverse-square"],
                         default="inverse-square" if cmd is sweep else "fixed",
                         help="Fixed budgets, or bounded inverse cycles*p² allocation")
        cmd.add_argument("--max-shots", type=int, help="Per-point cap for inverse-square allocation; default 10*shots")
        cmd.add_argument("--jobs", type=int, default=1, help="Processes; -2 leaves one CPU free")
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--results", type=Path, default=Path("benchmarks/se"))
        cmd.add_argument("--basis", choices=["0", "+", "both"], default="both")
    noise = run.add_mutually_exclusive_group()
    noise.add_argument("--p", type=float, help="p_2q; all default hardware rates scale together")
    noise.add_argument("--noise-model", type=Path, help="JSON object of explicit CustomErrorModel rates")
    run.add_argument("--cycles", type=int, default=1)
    sweep.add_argument("--preset", choices=["small"], default="small")
    sweep.add_argument("--config", type=Path, help="JSON list of {p, cycles, prepared_state} or {noise, ...}")
    sweep.add_argument("--ps", type=float, nargs="+")
    sweep.add_argument("--cycles", type=int, nargs="+", help="With --ps, sample the Cartesian product")
    plot = sub.add_parser("plot", help="Generate PDF/PNG figures from aggregate CSV counts")
    plot.add_argument("results", type=Path)
    plot.add_argument("--out", type=Path)
    plot.add_argument("--cycle-p", type=float, help="Repetition-sweep p_2q; default is default_noise_model().p_2q")
    inspect = sub.add_parser("inspect", help="Validate and describe a saved protocol")
    inspect.add_argument("protocol", type=Path)
    inspect.add_argument("--verify", action="store_true")
    inspect.add_argument("--matrices", action="store_true", help="Include the complete code matrices")
    native = sub.add_parser("native", help="Compile an executable HUGR wrapper and run Selene/Stim")
    native.add_argument("protocol", type=Path)
    native.add_argument("--out", type=Path, default=Path("build/se_native"))
    native.add_argument("--basis", choices=["0", "+"], default="0")
    native.add_argument("--cycles", type=int, default=1)
    native.add_argument("--shots", type=int, default=100)
    native.add_argument("--p", type=float, default=8e-4,
                        help="Native gate noise also affects preparation/readout; no memory noise")
    native.add_argument("--seed", type=int, default=0)
    native.add_argument("--jobs", type=int, default=1)
    return p


def emit_protocol(protocol, directory, emit):
    """Write source first so an interrupted compile leaves a usable bundle."""
    if emit == "none":
        return
    path = protocol.export_guppy(directory)
    print(f"Standalone Guppy: {path}", flush=True)
    import hashlib
    record_path = Path(directory) / "export.json"
    target = Path(directory) / "steane_se_protocol.hugr"
    identity = dict(protocol_hash=read_manifest(directory)[1]["protocol_hash"],
                    source_sha256=hashlib.sha256(protocol.source.encode()).hexdigest(),
                    versions=versions())
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    reusable = (all(previous.get(k) == v for k, v in identity.items())
                and target.exists()
                and previous.get("hugr_sha256") == hashlib.sha256(target.read_bytes()).hexdigest())
    record = dict(previous, hugr_compiled=True) if reusable else dict(identity, hugr_compiled=False)
    # A new source-only build must not advertise an old binary as current.
    atomic_write(record_path, json.dumps(record, indent=2) + "\n")
    if emit == "guppy":
        return
    from spiderwarp._se_adaptive_guppy import compile_adaptive_source
    from hugr.package import Package
    if reusable:
        Package.from_bytes(target.read_bytes())
        print(f"Existing HUGR validated and reloaded: {target}")
        return
    begin = time.monotonic()
    source, source_path, module, function, package = compile_adaptive_source(protocol.source)
    data = package.to_bytes()
    # Validate the serialized package before publishing it.
    Package.from_bytes(data)
    atomic_write(target, data)
    atomic_write(record_path, json.dumps(dict(identity,
        hugr_sha256=hashlib.sha256(data).hexdigest(), hugr_compiled=True,
        seconds=time.monotonic() - begin, hugr_reloaded=True), indent=2) + "\n")
    print(f"HUGR compiled and reloaded: {target} ({len(data):,} bytes; {time.monotonic()-begin:.1f}s)")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "build":
        import stim
        from spiderwarp.adaptive_steane_se import build_adaptive_se_protocol
        from spiderwarp.csscode import CSSCode
        from spiderwarp.utils import get_project_root
        family, name = args.code.rsplit("/", 1)
        code = CSSCode.load_code(family, name)
        assets = args.circuits or get_project_root() / "assets/circuits/SECircuits"
        circuits = {d: stim.Circuit.from_file(assets / f"{name}_d{d}.stim") for d in args.distances}
        if len(circuits) != len(args.distances):
            p.error("Supply unique distances")
        protocol = build_adaptive_se_protocol(
            code, circuits, peak_qubits=args.peak_qubits, optimize=args.optimize,
            minimize=not args.feasible_only, max_cnot_cost=args.max_cnot_cost,
            max_search_states=args.max_search_states, export=False,
            output_dir=args.out, progress=lambda s: print(s, flush=True))
        print(f"Protocol bundle: {args.out.resolve()}", flush=True)
        print(json.dumps(vars(protocol.verification), default=vars, indent=2), flush=True)
        emit_protocol(protocol, args.out.resolve(), args.emit)
    elif args.command == "compile":
        path, _ = read_manifest(args.protocol)
        emit_protocol(load_protocol(path, verify=args.verify), path.parent, args.emit)
    elif args.command == "inspect":
        path, manifest = read_manifest(args.protocol)
        protocol = load_protocol(path, verify=args.verify)
        export_path = path.parent / "export.json"
        export = json.loads(export_path.read_text()) if export_path.exists() else None
        code = protocol.code
        code_info = (code.to_dict() if args.matrices else
                     dict(n=code.n, k=code.k, d=code.d, name=code.name,
                          x_checks=len(code.H_x), z_checks=len(code.H_z)))
        print(json.dumps(dict(protocol_hash=manifest["protocol_hash"], code=code_info,
                              verification=manifest["verification"], versions=manifest["versions"],
                              export=export,
                              export_matches_protocol=bool(export and export.get("protocol_hash") == manifest["protocol_hash"]),
                              nodes=len(protocol.policy.nodes), independently_rechecked=args.verify), indent=2))
    elif args.command in {"run", "sweep"}:
        from spiderwarp.se_sweep import run_sweep, small_sweep
        bases = ("0", "+") if args.basis == "both" else (args.basis,)
        if args.command == "run":
            from spiderwarp.noise_model import default_noise_model
            noise = ({"noise": json.loads(args.noise_model.read_text())} if args.noise_model
                     else {"p": args.p} if args.p is not None
                     else {"noise": vars(default_noise_model())})
            configurations = [dict(noise, cycles=args.cycles, prepared_state=b) for b in bases]
        elif args.config:
            if args.ps or args.cycles:
                p.error("Use --config or --ps/--cycles")
            configurations = json.loads(args.config.read_text())
            configurations = [c for c in configurations if c["prepared_state"] in bases]
        elif args.ps or args.cycles:
            if not (args.ps and args.cycles):
                p.error("Supply both --ps and --cycles")
            configurations = [dict(p=v, cycles=n, prepared_state=b)
                              for v in args.ps for n in args.cycles for b in bases]
        else:
            configurations = [c for c in small_sweep() if c["prepared_state"] in bases]
        summary = run_sweep(args.protocol, configurations, args.results, shots=args.shots,
                            batch_size=args.batch_size, jobs=args.jobs, seed=args.seed,
                            allocation=args.allocation, max_shots=args.max_shots)
        print(f"Results: {(args.results / 'summary.csv').resolve()}")
        for row in summary:
            print(f"p={row['p']:.4g} N={row['cycles']} state={row['prepared_state']} "
                  f"LER={row['ler']:.5g} [{row['ler_low']:.5g}, {row['ler_high']:.5g}], "
                  f"survival={row['survival']:.5g}; accepted={row['accepted']}/{row['shots']}")
    elif args.command == "plot":
        from spiderwarp.se_plotting import plot_results
        for path in plot_results(args.results, args.out, cycle_p=args.cycle_p):
            print(path)
    elif args.command == "native":
        from spiderwarp.se_native import run_native
        print("Selene gate-noise model includes preparation/readout and excludes memory.", flush=True)
        print(json.dumps(run_native(args.protocol, args.out, prepared_state=args.basis,
                                    cycles=args.cycles, shots=args.shots, p=args.p,
                                    seed=args.seed, jobs=args.jobs), indent=2))


if __name__ == "__main__":
    main()

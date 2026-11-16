"""Standalone worker for the separately locked Sol Python environment.

Run by filename, not ``python -m spiderwarp...``: importing SpiderWarp here
would require installing the reference simulator's incompatible dependencies.
The JSON request/result protocol is private and versioned by source hashes.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import warnings

PACKAGES = ("guppylang", "guppylang-internals", "hugr", "selene-sim", "selene-core",
            "selene-narsil", "selene-eldarion", "eldarion", "pecos-selene",
            "qtm-platform", "selene-qtm-platform", "selene-formats")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data.encode() if isinstance(data, str) else data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def error_model(config):
    from pecos_selene import qsystem_model_alpha
    overrides = dict(config.get("model_overrides", {}))
    reserved = {"scale", "coherent_dephasing", "leak2depolar", "noiseless_gates",
                "noiseless_trace_dir", "random_seed"}
    if reserved.intersection(overrides):
        raise ValueError("Reserved Sol model overrides: " + ", ".join(sorted(reserved.intersection(overrides))))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pecos_selene.error_models.named")
        # These approximations are part of the model identity, not implicit Stim
        # conversions. They also permit clock-preserving ideal boundary replay.
        return qsystem_model_alpha(scale=config["noise_scale"], coherent_dephasing=False,
                                   leak2depolar=True, **overrides)


def describe(config):
    from selene_narsil import NarsilRuntimePlugin
    model = error_model(config)
    runtime = NarsilRuntimePlugin(leakage_repump_enabled=config["repump"])
    params = asdict(model)
    params.pop("random_seed", None)
    params.pop("noiseless_trace_dir", None)
    return dict(versions={name: version(name) for name in PACKAGES},
                python=list(sys.version_info[:3]), architecture=platform.machine(),
                parameters=params, runtime_args=runtime.get_init_args(),
                libraries={"pecos": sha(model.library_file), "narsil": sha(runtime.library_file)},
                platform="sol", simulator="stim", ion_pool=192,
                optimization="Minimal", boundaries="ideal-prepare-cover-sync-and-readout-v2",
                input_preparation="systematic CSS encoder and generator cover; legacy full mode available",
                seeding=dict(mode="default", indexing="fixed task stream plus global shot_offset"),
                execution="one Python worker and native run per unfinished task; atomic batch notifications",
                compiler=compiler_identity())


def compiler_identity():
    cc = os.environ.get("CC", "cc")
    # CC is one executable, never shell-expanded.
    return subprocess.run([cc, "--version"], check=True, text=True, capture_output=True).stdout


def boundary_library(directory):
    from selene_core.headers import get_include_directory
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("The Sol boundary adapter currently supports macOS and Linux")
    source = Path(__file__).with_name("_sol_boundary.c")
    include = get_include_directory()
    identity = sha(source) + compiler_identity() + "".join(
        sha(include / "selene" / name) for name in ("runtime.h", "error_model.h", "core_types.h"))
    key = hashlib.sha256(identity.encode()).hexdigest()
    target = directory / ("boundary_" + key + (".dylib" if platform.system() == "Darwin" else ".so"))
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        flags = ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-ldl"]
        result = subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-Wall", "-Wextra",
                                 "-fPIC", *flags, "-I", str(include), str(source), "-o", str(temporary)],
                                capture_output=True, text=True)
        if result.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Sol boundary adapter compilation failed:\n" + result.stderr)
        os.replace(temporary, target)
    return target


def components(config, library, peak):
    from selene_core import ErrorModel, Runtime
    from selene_narsil import NarsilRuntimePlugin

    class BoundaryRuntime(Runtime):
        @property
        def library_file(self):
            return library

        def get_init_args(self):
            runtime = NarsilRuntimePlugin(leakage_repump_enabled=config["repump"])
            return [str(runtime.library_file), str(peak), *runtime.get_init_args()]

    class BoundaryErrorModel(ErrorModel):
        @property
        def library_file(self):
            return library

        def get_init_args(self):
            model = error_model(config)
            return [str(model.library_file), *model.get_init_args()]

    return BoundaryRuntime(), BoundaryErrorModel()


def compile_source(source_path, directory, repump):
    from guppylang import OptimizationLevel
    from hugr.package import Package
    source_path = Path(source_path)
    metadata = directory / "compile.json"
    binary = directory / "experiment.hugr"
    identity = dict(source=sha(source_path), versions={n: version(n) for n in PACKAGES},
                    optimization="Minimal", repump=repump)
    if metadata.exists() and binary.exists():
        previous = json.loads(metadata.read_text())
        if previous.get("identity") == identity and previous.get("hugr_sha256") == sha(binary):
            return Package.from_bytes(binary.read_bytes())
    name = "_spiderwarp_sol_" + sha(source_path)
    spec = importlib.util.spec_from_file_location(name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        package = module.main.with_opt_level(OptimizationLevel.Minimal).compile()
        # Repump is configured on Narsil itself. The internal HUGR config helper
        # in qtm-platform 0.8.1 emits newer QIS symbols absent from the supplied
        # local example's Sol interface, even when all options are defaults.
        data = package.to_bytes()
        Package.from_bytes(data)
        atomic(binary, data)
        atomic(metadata, json.dumps(dict(identity=identity, hugr_sha256=sha(binary)), indent=2))
        return package
    finally:
        sys.modules.pop(name, None)


def run(request):
    from selene_sim import Stim, build
    from selene_sim.instance import SeleneInstance
    from selene_sim.event_hooks import MetricStore
    from selene_eldarion import register_eldarion, QtmPlatformPlugin
    config = request["config"]
    seed_mode = request.get("seed_mode", "default")
    if seed_mode != "default":
        raise ValueError("Sol sampling pins seed_mode=default")
    streaming = request.get("stream_batches", False)
    batch_size = request.get("batch_size", request["shots"]) if streaming else request["shots"]
    if type(batch_size) is not int or batch_size <= 0 or request["shots"] <= 0 or request["start"] < 0:
        raise ValueError("Invalid Sol shot range or batch size")
    directory = Path(request["directory"])
    library = boundary_library(directory)
    runtime, model = components(config, library, request["peak_qubits"])
    package = compile_source(request["source"], directory, config["repump"])
    register_eldarion()
    cached = directory / "native_instance.json"
    identity = dict(hugr=sha(directory / "experiment.hugr"), utility=sha(QtmPlatformPlugin().library_file))
    previous = json.loads(cached.read_text()) if cached.exists() else {}
    executable = Path(previous.get("instance", {}).get("executable", "missing"))
    if previous.get("identity") == identity and executable.is_file() and previous.get("sha256") == sha(executable):
        options = previous["instance"]
        runner = SeleneInstance(**{key: [Path(p) for p in value] if key == "library_search_dirs" else Path(value)
                                   for key, value in options.items()})
    else:
        runner = build(package, eldarion=True, utilities=[QtmPlatformPlugin()], platform="sol",
                       build_dir=directory / "selene")
        options = {key: [str(p) for p in value] if key == "library_search_dirs" else str(value)
                   for key, value in asdict(runner).items()}
        atomic(cached, json.dumps(dict(identity=identity, instance=options, sha256=sha(runner.executable)), indent=2))
    metrics = MetricStore()
    count = 0
    batch_count = 0
    stream = None
    temporary = None

    def publish(output, temporary, start, shots):
        os.replace(temporary, output)
        if streaming:
            print(json.dumps(dict(start=start, shots=shots, output=str(output),
                                  stream_seed=request["seed"], seed_mode=seed_mode)), flush=True)

    try:
        for shot in runner.run_shots(simulator=Stim(), runtime=runtime, error_model=model,
                n_qubits=192, n_shots=request["shots"], n_processes=1,
                random_seed=request["seed"], shot_offset=request["start"],
                seed_mode=seed_mode, event_hook=metrics):
            if stream is None:
                batch_start = request["start"] + count
                output = (directory / f"shots_{batch_start}.jsonl" if streaming else Path(request["output"]))
                temporary = output.with_suffix(".tmp")
                stream = temporary.open("w")
            data, stages, syndromes = {}, [], []
            started = 0
            for tag, value in shot:
                if tag == "_sw_cycle":
                    started += 1
                elif tag.startswith("_sw_stage_"):
                    stages.append([int(tag[len("_sw_stage_"):]), value])
                elif tag.startswith("_sw_syndrome_"):
                    syndromes.append([int(tag[len("_sw_syndrome_"):]), value])
                elif tag in data:
                    raise RuntimeError("Duplicate Sol output: " + tag)
                else:
                    data[tag] = value
            shot_id = metrics.shots[-1]["emulator"]["shot_number"]
            if shot_id != request["start"] + count:
                raise RuntimeError("Sol returned an unexpected global shot index")
            data.update(stages=stages, syndromes=syndromes, cycles_started=started,
                        shot_id=shot_id, stream_seed=request["seed"], seed_mode=seed_mode)
            data["runtime_metrics"] = metrics.shots[-1].get("runtime", {})
            # Selene namespaces metrics by component, with the adapter's
            # spiderwarp category nested below the runtime namespace.
            data["metrics"] = metrics.shots[-1]
            stream.write(json.dumps(data, separators=(",", ":")) + "\n")
            metrics.shots.clear()
            count += 1
            batch_count += 1
            if batch_count == batch_size and count < request["shots"]:
                stream.close()
                publish(output, temporary, batch_start, batch_count)
                stream = None
                temporary = None
                batch_count = 0
        if count != request["shots"]:
            raise RuntimeError(f"Sol returned {count} shots, expected {request['shots']}")
        # Wait for native process completion before publishing the final batch.
        stream.close()
        publish(output, temporary, batch_start, batch_count)
        temporary = None
    finally:
        if stream is not None:
            stream.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    if request["action"] == "describe":
        atomic(request["output"], json.dumps(describe(request["config"]), indent=2))
    elif request["action"] == "run":
        run(request)
    else:
        raise ValueError("Unknown Sol worker action")


if __name__ == "__main__":
    main()

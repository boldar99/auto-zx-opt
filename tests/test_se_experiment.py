"""Persistence, physical cycle composition, reproducibility and resume checks."""
import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import stim

from spiderwarp.adaptive_steane_se import build_adaptive_se_protocol
from spiderwarp._se_adaptive import PolicyNode
from spiderwarp.csscode import CSSCode
from spiderwarp.noise_model import CustomErrorModel, apply_noise
from spiderwarp.optimized_steane_se import ProtocolResult
from spiderwarp.se_protocol_io import load_protocol, read_manifest
from spiderwarp.se_protocol_simulation import ProtocolSampler, scaled_noise_model
from spiderwarp.se_sweep import run_sweep, small_sweep, num_samples
from test_adaptive_se_policy import bell_actions


@pytest.fixture
def saved_protocol(tmp_path):
    code, _, actions = bell_actions()
    protocol = build_adaptive_se_protocol(code, actions, first_action="Z", export=False,
                                          output_dir=tmp_path / "protocol")
    return protocol, tmp_path / "protocol"


def test_bundle_round_trip_without_synthesis_or_guppy(saved_protocol, monkeypatch):
    original, path = saved_protocol
    from spiderwarp._se_adaptive import AdaptivePolicyBuilder
    monkeypatch.setattr(AdaptivePolicyBuilder, "build", lambda *a, **k: pytest.fail("Resynthesized"))
    loaded = load_protocol(path)
    assert not (path / "steane_se_protocol.hugr").exists()
    assert original.policy.nodes == loaded.policy.nodes
    assert original.verification == loaded.verification
    first_hash = read_manifest(path)[1]["protocol_hash"]
    loaded.save_bundle(path)
    assert read_manifest(path)[1]["protocol_hash"] == first_hash
    for x in range(4):
        for z in range(4):
            sim = stim.TableauSimulator(seed=9)
            sim.do_circuit(stim.Circuit("H 0\nCX 0 1"))
            for q in range(2):
                if x >> q & 1:
                    sim.x(q)
                if z >> q & 1:
                    sim.z(q)
            a, b = sim.copy(seed=7), sim.copy(seed=7)
            assert original.run(a, frame_x=x, frame_z=z) == loaded.run(b, frame_x=x, frame_z=z)
            assert a.canonical_stabilizers() == b.canonical_stabilizers()
    # Source-only emission must not import Guppy or invoke the compiler.
    import spiderwarp._se_adaptive_guppy as emitter
    monkeypatch.setattr(emitter, "compile_adaptive_source", lambda *a, **k: pytest.fail("Compiled"))
    assert loaded.export_guppy(path).exists()


def test_bundle_detects_changed_files_and_nontermination(saved_protocol, tmp_path):
    protocol, path = saved_protocol
    m = read_manifest(path)[1]
    circuit = path / m["actions"][0]["circuit"]
    original = circuit.read_bytes()
    circuit.write_bytes(original + b"\nX 0\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_protocol(path)
    circuit.write_bytes(original)
    root = protocol.policy.root
    nodes = list(protocol.policy.nodes)
    nodes[root] = PolicyNode("frame", frame=1, next=root)
    protocol.policy = replace(protocol.policy, nodes=tuple(nodes))
    protocol.save_bundle(tmp_path / "loop")
    with pytest.raises(ValueError, match="does not terminate"):
        load_protocol(tmp_path / "loop")


def test_source_only_cli_marks_a_stale_binary(saved_protocol, capsys):
    from spiderwarp.se_experiment import emit_protocol, main
    protocol, path = saved_protocol
    (path / "steane_se_protocol.hugr").write_bytes(b"old binary")
    (path / "export.json").write_text(json.dumps(dict(protocol_hash="different")))
    emit_protocol(protocol, path, "guppy")
    record = json.loads((path / "export.json").read_text())
    assert record["protocol_hash"] == read_manifest(path)[1]["protocol_hash"]
    assert not record["hugr_compiled"]
    assert (path / "steane_se_protocol.hugr").read_bytes() == b"old binary"
    capsys.readouterr()
    main(["inspect", str(path)])
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["code"]["n"] == 2 and "H_x" not in inspected["code"]
    assert inspected["export_matches_protocol"]


def test_initial_data_wait_is_noisy_but_unused_ancillas_are_not_prepared():
    circuit = stim.Circuit("R 1\nH 1\nCX 1 0\nM 1")
    historical = apply_noise(circuit, CustomErrorModel(p_linear_mem=.01))
    current = apply_noise(circuit, CustomErrorModel(p_linear_mem=.01), persistent_qubits=range(1))
    assert current[0].name == "Z_ERROR"
    assert [t.value for t in current[0].targets_copy()] == [0]
    assert current[0].gate_args_copy() == [.02]
    assert str(current).endswith(str(historical))


@pytest.mark.parametrize("basis", ["0", "+"])
def test_noiseless_repeated_protocol(saved_protocol, basis):
    protocol, path = saved_protocol
    sampler = ProtocolSampler(load_protocol(path), noise_model=CustomErrorModel(),
                              prepared_state=basis, cycles=5)
    result = sampler.sample(20, 5)
    assert result.accepted == 20 and result.logical_errors == 0
    assert result.diagnostics.cycles_started == result.diagnostics.cycles_completed == 100
    assert result.diagnostics.total_cnots == 400


def test_repetitions_keep_state_and_returned_frame_and_stop_at_rejection():
    code = CSSCode.load_code("MQT", "7_1_3")
    logical = sum(int(v) << q for q, v in enumerate(code.L_x[0]))
    calls = []
    g = SimpleNamespace(circuit=stim.Circuit(), basis="Z", distance=3)
    def run(sim, *, frame_x=0, frame_z=0):
        calls.append(frame_x)
        for q, bit in enumerate(code.L_x[0]):
            if bit:
                sim.simulator.x(q)
        return ProtocolResult(True, (), (), frame_x=frame_x ^ logical, frame_z=frame_z)
    protocol = SimpleNamespace(code=code, required_qubits=code.n, gadgets={"g": g}, policy=True, run=run)
    result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), cycles=3).sample(1, 9)
    assert calls == [0, logical, 0]
    assert result.logical_errors == 0
    calls.clear()
    def reject(sim, **kwargs):
        calls.append(1)
        return ProtocolResult(False, (), (), "forced")
    protocol.run = reject
    result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), cycles=5).sample(4, 9)
    assert len(calls) == 4
    assert result.accepted == 0 and result.diagnostics.rejection_cycles == {1: 4}
    assert result.diagnostics.cycles_completed == 0


def test_p_scaling_and_zero_cycle_control(saved_protocol):
    from spiderwarp.noise_model import default_noise_model
    assert vars(scaled_noise_model(8e-4)) == vars(default_noise_model())
    assert not any(vars(scaled_noise_model(0)).values())
    with pytest.raises(ValueError):
        scaled_noise_model(float("nan"))
    result = ProtocolSampler(saved_protocol[0], cycles=0).sample(5, 2)
    assert result.accepted == 5 and result.logical_errors == result.diagnostics.cycles_started == 0
    configurations = small_sweep()
    assert len(configurations) == 32
    repeated = [c for c in configurations if c["sweep"] == "cycles"]
    physical = [c for c in configurations if c["sweep"] == "physical_error"]
    assert len(repeated) == 10 and len(physical) == 22
    assert all(c["noise"] == vars(default_noise_model()) for c in repeated)
    assert all(c["cycles"] == 1 and "noise" not in c for c in physical)


def test_shot_allocation_normalizes_rare_event_statistics_with_bounds():
    assert num_samples(8e-4) == 100000
    assert num_samples(4e-4) == 400000
    assert num_samples(4e-4, cycles=2) == 200000
    assert num_samples(4e-4, cycles=10) == 100000
    assert num_samples(1e-4) == 1000000
    assert num_samples(1e-4, max_shots=10000000) == 6400000
    assert num_samples(1e-300) == 1000000
    assert num_samples(0) == num_samples(8e-4, cycles=0) == 100000
    with pytest.raises(ValueError):
        num_samples(8e-4, max_shots=99999)
    with pytest.raises(ValueError):
        num_samples(float("nan"))


def test_per_task_targets_and_resume(saved_protocol, tmp_path):
    configs = [dict(p=p, cycles=1, prepared_state="0", sweep="physical_error")
               for p in (8e-4, 4e-4)]
    path = tmp_path / "allocated"
    results = run_sweep(saved_protocol[1], configs, path, shots=5, max_shots=30,
                        allocation="inverse-square", batch_size=3, progress=None)
    assert {r["p"]: r["shots"] for r in results} == {8e-4: 5, 4e-4: 20}
    assert all(r["shots"] == r["target_shots"] for r in results)
    snapshots = {p: p.read_bytes() for p in path.glob("batches/*.csv")}
    run_sweep(saved_protocol[1], configs, path, shots=5, max_shots=30,
              allocation="inverse-square", batch_size=3, progress=None)
    assert all(p.read_bytes() == b for p, b in snapshots.items())


def test_cli_uses_current_default_model_when_p_is_omitted(saved_protocol, monkeypatch):
    from spiderwarp.se_experiment import main
    import spiderwarp.noise_model as noise_module
    import spiderwarp.se_sweep as sweep_module
    model = CustomErrorModel(p_2q=.002, p_init=.003, p_linear_mem=.004)
    monkeypatch.setattr(noise_module, "default_noise_model", lambda: model)
    captured = []
    def run(path, configurations, *args, **kwargs):
        captured.extend(configurations)
        return []
    monkeypatch.setattr(sweep_module, "run_sweep", run)
    main(["run", str(saved_protocol[1]), "--cycles", "3"])
    assert len(captured) == 2
    assert all(c["noise"] == vars(model) and "p" not in c for c in captured)


def test_resume_parallelism_counts_and_plotting(saved_protocol, tmp_path, monkeypatch):
    _, protocol = saved_protocol
    configs = [dict(p=p, cycles=n, prepared_state=b)
               for p, n in ((.001, 1), (.001, 3), (.01, 1)) for b in ("0", "+")]
    results = tmp_path / "results"
    first = run_sweep(protocol, configs, results, shots=10, batch_size=5, seed=8, progress=None)
    assert len(first) == 6 and all(r["shots"] == 10 for r in first)
    snapshots = {p: p.read_bytes() for p in results.glob("batches/*.csv")}
    run_sweep(protocol, configs, results, shots=10, batch_size=5, seed=8, progress=None)
    assert all(p.read_bytes() == data for p, data in snapshots.items())
    extended = run_sweep(protocol, configs, results, shots=20, batch_size=5, jobs=2, seed=8, progress=None)
    independent = run_sweep(protocol, configs, tmp_path / "independent", shots=20,
                            batch_size=5, jobs=1, seed=8, progress=None)
    for a, b in zip(extended, independent):
        assert all(a[k] == b[k] for k in a if k != "seconds")
    from spiderwarp.se_plotting import plot_results
    from matplotlib.figure import Figure
    save = Figure.savefig
    def check_scales(fig, path, **kwargs):
        survival = Path(path).stem == "survival"
        assert all(ax.get_yscale() == ("linear" if survival else "log") for ax in fig.axes)
        if survival:
            assert all(ax.get_ylim()[0] > 0 and ax.get_ylim()[1] <= 1 for ax in fig.axes)
        return save(fig, path, **kwargs)
    monkeypatch.setattr(Figure, "savefig", check_scales)
    paths = plot_results(results, cycle_p=.001)
    assert len(paths) == 6
    assert all(p.stat().st_size > 1000 for p in paths)
    # Duplicate ranges cannot silently inflate the sample count.
    batch = next(results.glob("batches/*.csv"))
    with batch.open(newline="") as f:
        lines = list(csv.reader(f))
    with batch.open("a", newline="") as f:
        csv.writer(f).writerow(lines[1])
    with pytest.raises(ValueError, match="batch ranges"):
        run_sweep(protocol, configs, results, shots=30, progress=None)


def test_native_hugr_compiles_reloads_and_executes(saved_protocol, tmp_path):
    pytest.importorskip("guppylang")
    pytest.importorskip("selene_sim")
    from spiderwarp.se_native import run_native
    result = run_native(saved_protocol[1], tmp_path / "native", shots=4, cycles=3,
                        p=0, seed=7, incoming_x=1, incoming_z=2)
    assert result["accepted"] == 4 and result["logical_errors"] == 0
    assert result["experiment"]["cycles"] == 3
    assert result["backend"] == "selene-stim"

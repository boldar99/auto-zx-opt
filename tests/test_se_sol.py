"""Sol boundary isolation, native execution, accepted costs and exact resumes."""
import json
from pathlib import Path
import sys

import pytest

from spiderwarp.se_sol import (
    SolConfig, _batches, css_input_preparation, input_sync_checks, run_sol_sweep, sampling_seed,
    sol_experiment_source, sol_python, summarize_shots,
)
from test_se_experiment import saved_protocol


@pytest.mark.parametrize("scale", [-1, float("nan"), float("inf")])
def test_invalid_noise_scale(scale):
    with pytest.raises(ValueError, match="noise_scale"):
        SolConfig(scale)


def test_source_instruments_actions_and_forces_ideal_preparation(saved_protocol):
    source, stages = sol_experiment_source(saved_protocol[0], cycles=3, incoming_x=1, incoming_z=2)
    assert len(stages) == 2 and sum(s["cnots"] for s in stages) == 4
    assert 'synchronize_input(data)' in source and 'check_input(observed, True)' in source
    assert source.index('synchronize_input(data)') < source.index('sleep(data, 1152921504606846977)')
    assert 'result("_sw_cycle", 1)' in source
    assert 'result("_sw_stage_0", flags)' in source


def test_input_sync_cover_and_full_modes():
    from spiderwarp.csscode import CSSCode
    code = CSSCode.load_code("misc", "32_20_4")
    for basis in ("0", "+"):
        cover = input_sync_checks(code, basis)
        full = input_sync_checks(code, basis, "full")
        assert cover == [("X", tuple(range(32)))]
        assert len(full) == 32 and sum(len(support) for _, support in full) == 316
    with pytest.raises(ValueError, match="input_sync"):
        input_sync_checks(code, mode="unsupported")


@pytest.mark.parametrize("code_name", ["misc/32_20_4", "MQT/7_1_3"])
@pytest.mark.parametrize("basis", ["0", "+"])
def test_css_encoder_matches_independent_tableau_state(code_name, basis):
    import stim
    from spiderwarp.csscode import CSSCode
    from spiderwarp.se_protocol_simulation import _StateExperiment
    code = CSSCode.load_code(*code_name.split("/"))
    circuit = css_input_preparation(code, basis)
    sim = stim.TableauSimulator()
    sim.set_num_qubits(code.n)
    sim.do_circuit(circuit)
    reference = _StateExperiment(code, code.n, basis)
    assert sim.canonical_stabilizers() == reference.initial.canonical_stabilizers()
    if code.n == 32:
        assert sum(len(op.targets_copy()) // 2 for op in circuit if op.name == "CX") == 90


def test_accepted_costs_exclude_earlier_cycles_of_rejected_shots(saved_protocol, tmp_path):
    protocol = saved_protocol[0]
    _, stages = sol_experiment_source(protocol)
    metric = {"runtime": {"spiderwarp:phase_switches": 2, "spiderwarp:peak_live_qubits": 3,
                         "spiderwarp:noisy_rzz": 8, "spiderwarp:protocol_duration_ns": 10}}
    base = dict(frame_x=0, frame_z=0, measurements=[0, 0], metrics=metric)
    records = [dict(base, accepted=1, completed=2, cycles_started=2, stages=[[0, 0], [1, 0]] * 2),
               dict(base, accepted=0, completed=1, cycles_started=2, stages=[[0, 0], [1, 0]] * 2)]
    raw = tmp_path / "shots.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in records))
    result = summarize_shots(raw, protocol, stages, "0", 2, 0, 2, 3)
    assert result["accepted"] == 1 and result["logical_errors"] == 0
    assert result["total_cnots"] == 16 and result["accepted_cnots"] == 8
    assert result["accepted_native_rzz"] == 8 and result["accepted_duration_ns"] == 10
    assert json.loads(result["diagnostics"])["rejection_cycles"] == {"2": 1}
    records[0]["metrics"] = {}
    raw.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(RuntimeError, match="boundary metrics"):
        summarize_shots(raw, protocol, stages, "0", 2, 0, 1, 3)


def test_cli_sol_passes_native_scale_and_isolated_python(saved_protocol, monkeypatch, tmp_path):
    from spiderwarp.se_experiment import main
    import spiderwarp.se_sol as module
    seen = []
    monkeypatch.setattr(module, "run_sol_sweep", lambda *a, **k: seen.append((a, k)) or [])
    main(["sol", str(saved_protocol[1]), "--noise-scales", "0", "1", "--cycles", "1", "3",
          "--shots", "2", "--repump", "--sol-python", "/some/python", "--results", str(tmp_path)])
    assert len(seen[0][0][1]) == 8 and seen[0][1]["repump"]
    assert str(seen[0][1]["python"]) == "/some/python"
    assert {c["noise_scale"] for c in seen[0][0][1]} == {0, 1}


def test_comparison_defaults_to_sol_and_preserves_explicit_reference(monkeypatch):
    from types import SimpleNamespace
    import importlib
    from spiderwarp.se_comparison import main
    import spiderwarp.se_sol_comparison as module
    calls = []
    monkeypatch.setattr(module, "main", lambda args: calls.append(("sol", args)))
    main(["--shots", "2"])
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(
        main=lambda args: calls.append((name, args))))
    main(["--backend", "stim-reference", "--plot-only"])
    assert calls == [("sol", ["--shots", "2"]),
                     ("analysis.32_20_4_bounded_comparison.study", ["--plot-only"])]


def test_comparison_selects_two_methods_in_one_combined_table(monkeypatch, tmp_path):
    import csv
    import spiderwarp.se_sol_comparison as module
    from spiderwarp.se_sol import MODEL_VERSION
    calls = []

    def sample(bundle, configurations, directory, **options):
        calls.append(bundle.name)
        return [dict(backend=MODEL_VERSION, shots=options["shots"], seed=options["seed"],
                     noise='{"repump":false,"noise_scale":1.0}', engine_hash="same-engine",
                     input_sync="cover", **c) for c in configurations]

    monkeypatch.setattr(module, "run_sol_sweep", sample)
    module.main(["--only", "adaptive_history", "steane_fixed_ps", "--shots", "10",
                 "--cycles", "1", "--results", str(tmp_path)])
    assert calls == ["se_postselected_32_20_4", "steane_single_d3_fixed_ps"]
    with (tmp_path / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and {r["method"] for r in rows} == {"adaptive_history", "steane_fixed_ps"}


@pytest.fixture(scope="module")
def sol_interpreter():
    try:
        return sol_python()
    except ValueError:
        pytest.skip("Separately installed Sol environment unavailable")


def test_real_sol_noiseless_frames_and_batch_independent_resume(saved_protocol, sol_interpreter, tmp_path):
    config = [dict(noise_scale=0, prepared_state=b, cycles=3) for b in ("0", "+")]
    results = tmp_path / "results"
    rows = run_sol_sweep(saved_protocol[1], config, results, shots=2, batch_size=1,
                         incoming_x=1, incoming_z=2, python=sol_interpreter, retain_shots=True, progress=None)
    assert len(rows) == 2
    assert all(r["accepted"] == 2 and r["logical_errors"] == 0 for r in rows)
    assert all(r["total_cnots"] == r["total_native_rzz"] == 24 for r in rows)
    assert all(r["peak_live_qubits"] == 3 for r in rows)
    for task in results.glob("tasks/*.json"):
        runs = list((results / "work" / task.stem / "selene" / "runs").iterdir())
        assert len(runs) == 1  # Both atomic batches came from one native process.
        assert "count: 2" in (runs[0] / "configuration.yaml").read_text()
        assert "seed_mode: default" in (runs[0] / "configuration.yaml").read_text()
    before = {p: p.read_bytes() for p in results.glob("batches/*.csv")}
    run_sol_sweep(saved_protocol[1], config, results, shots=2, batch_size=2,
                  incoming_x=1, incoming_z=2, python=sol_interpreter, progress=None)
    assert all(p.read_bytes() == text for p, text in before.items())
    other = tmp_path / "independent"
    run_sol_sweep(saved_protocol[1], config, other, shots=2, batch_size=2,
                  incoming_x=1, incoming_z=2, python=sol_interpreter, retain_shots=True, progress=None)
    for task in results.glob("tasks/*.json"):
        key = task.stem
        first = [json.loads(line) for p in sorted((results / "work" / key).glob("shots_*.jsonl"))
                 for line in p.read_text().splitlines()]
        second = [json.loads(line) for line in (other / "work" / key / "shots_0.jsonl").read_text().splitlines()]
        assert first == second
    path = next(results.glob("batches/*.csv"))
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n")
    with pytest.raises(ValueError, match="batch ranges"):
        _batches(path)


def test_real_sol_ideal_boundaries_remain_clean_at_extreme_spam(saved_protocol, sol_interpreter, tmp_path):
    rows = run_sol_sweep(saved_protocol[1], [dict(noise_scale=1, cycles=0, prepared_state=b)
                                            for b in ("0", "+")], tmp_path / "ideal",
                         shots=3, batch_size=3, python=sol_interpreter, progress=None,
                         model_overrides=dict(p_meas_0=.49, p_meas_1=.49, p_prep=.49))
    assert all(r["accepted"] == 3 and r["logical_errors"] == 0 for r in rows)
    assert all(r["total_native_rzz"] == r["total_duration_ns"] == r["cycles_started"] == 0 for r in rows)


def test_sol_rejects_legacy_noise_configuration(saved_protocol, sol_interpreter, tmp_path):
    with pytest.raises(ValueError, match="legacy p/noise"):
        run_sol_sweep(saved_protocol[1], [dict(p=.001, cycles=1, prepared_state="0")],
                      tmp_path, python=sol_interpreter)


def _raw_records(root, task_key):
    paths = sorted((root / "work" / task_key).glob("shots_*.jsonl"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return [json.loads(line) for p in paths for line in p.read_text().splitlines()]


def test_noisy_stream_is_exact_across_batches_parallelism_and_resume(saved_protocol, sol_interpreter, tmp_path):
    configs = [dict(noise_scale=1, prepared_state=b, cycles=2) for b in ("0", "+")]
    options = dict(python=sol_interpreter, seed=73, retain_shots=True, progress=None,
                   model_overrides=dict(p_meas_0=.15, p_meas_1=.15, p_prep=.02))
    a, b = tmp_path / "a", tmp_path / "b"
    run_sol_sweep(saved_protocol[1], configs, a, shots=4, batch_size=1, jobs=1, **options)
    run_sol_sweep(saved_protocol[1], configs, b, shots=6, batch_size=3, jobs=2, **options)
    run_sol_sweep(saved_protocol[1], configs, a, shots=6, batch_size=2, jobs=2, **options)
    seeds = set()
    for metadata in a.glob("tasks/*.json"):
        task = json.loads(metadata.read_text())["task"]
        first, second = _raw_records(a, metadata.stem), _raw_records(b, metadata.stem)
        assert first == second
        assert [r["shot_id"] for r in first] == list(range(6))
        assert all(r["stream_seed"] == sampling_seed(task) and r["seed_mode"] == "default" for r in first)
        seeds.add(sampling_seed(task))
        runs = list((a / "work" / metadata.stem / "selene" / "runs").iterdir())
        assert len(runs) == 2  # Initial range [0,4), then resumed range [4,6).
    assert len(seeds) == 2  # Independent basis streams.


def test_worker_failure_preserves_committed_ranges_and_resumes(saved_protocol, monkeypatch, tmp_path):
    """Commit one batch, fail the worker, then resume without replaying it."""
    import subprocess
    import spiderwarp.se_sol as module
    fake = tmp_path / "fake_worker.py"
    fake.write_text('''import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());start=r['start'];remaining=r['shots'];size=r['batch_size']
while remaining:
    n=min(size,remaining);out=Path(r['directory'])/f'shots_{start}.jsonl'
    metrics={'runtime':{'spiderwarp:phase_switches':2,'spiderwarp:peak_live_qubits':3,
        'spiderwarp:noisy_rzz':4,'spiderwarp:protocol_duration_ns':10}}
    records=[dict(accepted=1,completed=1,cycles_started=1,stages=[[0,0],[1,0]],
        frame_x=0,frame_z=0,measurements=[0,0],metrics=metrics,shot_id=i,
        stream_seed=r['seed'],seed_mode='default') for i in range(start,start+n)]
    out.write_text(''.join(json.dumps(x)+'\\n' for x in records))
    print(json.dumps(dict(start=start,shots=n,output=str(out),stream_seed=r['seed'],seed_mode='default')),flush=True)
    if r['start']==0:sys.exit('deliberate worker failure after first batch')
    start+=n;remaining-=n
''')
    original = subprocess.Popen

    def popen(args, **kwargs):
        if len(args) > 2 and args[1] == "-u":
            args = [args[0], "-u", str(fake), args[-1]]
        return original(args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "engine_identity", lambda *a: {"test": "stable"})
    configs = [dict(noise_scale=0, cycles=1, prepared_state="0")]
    results = tmp_path / "results"
    with pytest.raises(RuntimeError, match="deliberate worker failure"):
        run_sol_sweep(saved_protocol[1], configs, results, shots=5, batch_size=2,
                      python=sys.executable, progress=None, retain_shots=True)
    batch = next(results.glob("batches/*.csv"))
    committed = _batches(batch)
    assert [(r["start"], r["shots"]) for r in committed] == [(0, 2)]
    rows = run_sol_sweep(saved_protocol[1], configs, results, shots=5, batch_size=2,
                         python=sys.executable, progress=None, retain_shots=True)
    assert rows[0]["shots"] == rows[0]["accepted"] == 5
    assert [(r["start"], r["shots"]) for r in _batches(batch)] == [(0, 2), (2, 2), (4, 1)]
    assert _batches(batch)[0] == committed[0]

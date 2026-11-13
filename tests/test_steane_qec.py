"""Physical Steane execution, retry boundaries, persistence and HUGR checks."""
from collections import Counter
import ast
import csv
from dataclasses import dataclass
import json

import numpy as np
import pytest
import stim

from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler, _StateExperiment
from spiderwarp.steane_qec import MODES, build_steane_qec


@pytest.fixture(scope="module")
def protocols():
    return {mode: build_steane_qec(mode) for mode in MODES}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("basis", ("0", "+"))
def test_physical_noiseless_repeated_cycles(protocols, mode, basis):
    protocol = protocols[mode]
    result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), prepared_state=basis,
                             cycles=3).sample(20, 7)
    assert result.accepted == 20 and result.logical_errors == 0
    assert result.diagnostics.cycles_completed == 60
    assert result.diagnostics.total_cnots == 20 * 3 * 288
    assert protocol.required_qubits == 66


class ScriptedReadout:
    def __init__(self, protocol):
        self.protocol = protocol
        self.record = []
        self.names = {id(g.circuit): name for name, g in protocol.gadgets.items()}
        self.counts = Counter()
        self.trace = []

    def current_measurement_record(self):
        return self.record

    def do_circuit(self, circuit):
        name = self.names[id(circuit)]
        self.counts[name] += 1
        self.trace.append(name)
        # Reject the first two preparations of either distance/basis; d1 has
        # no flags. The first Z read is nonzero and must be used without confirmation.
        values = [False] * circuit.num_measurements
        prep_count = sum(v for k, v in self.counts.items() if k.startswith("prep_"))
        if name.startswith("prep_") and prep_count <= 2 and values:
            values[0] = True
        if name == "read_Z" and self.counts[name] == 1:
            values[0] = True
        self.record.extend(values)


@pytest.mark.parametrize("mode", MODES)
def test_fixed_retry_and_single_read(protocols, mode):
    protocol = protocols[mode]
    sim = ScriptedReadout(protocol)
    outcome = protocol.run(sim)
    prep = [n for n in sim.trace if n.startswith("prep_")]
    assert prep == ["prep_Z_d3"] * 3 + ["prep_X_d3"]
    assert sim.trace[:2] == [prep[0], "discard_Z"]
    assert sim.counts["read_Z"] == sim.counts["read_X"] == 1
    assert outcome.accepted and outcome.frame_x == 1 and outcome.frame_z == 0
    second = ScriptedReadout(protocol)
    protocol.run(second)
    assert second.trace[0] == "prep_Z_d3"  # Every new QEC cycle resets.


@pytest.mark.parametrize("mode", MODES)
def test_bundle_roundtrip_and_integrity(protocols, mode, tmp_path):
    original = protocols[mode]
    path = original.save_bundle(tmp_path)
    loaded = load_protocol(path)
    assert loaded.mode == mode
    assert loaded.decoders == original.decoders
    source = loaded.export_guppy(tmp_path).read_text()
    assert "while True:" in source and "syndrome == previous" not in source
    manifest = json.loads(path.read_text())
    manifest["mode"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_protocol(path)


def test_every_single_css_input_and_existing_frame(protocols):
    protocol = protocols["fixed"]
    # Split X/Z support is CSS weight one, including incoming tracked frames.
    for basis in ("0", "+"):
        experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
        for q in range(protocol.code.n):
            sim = experiment.initial.copy(seed=q)
            sim.x(q)
            sim.z((q + 17) % protocol.code.n)
            result = protocol.run(sim)
            assert not experiment.logical_failure(sim, frame_x=result.frame_x, frame_z=result.frame_z)
            # The same physical errors remain, so a repeated call must retain
            # the cumulative frame instead of applying the correction twice.
            again = protocol.run(sim, frame_x=result.frame_x, frame_z=result.frame_z)
            assert (again.frame_x, again.frame_z) == (result.frame_x, result.frame_z)


@pytest.mark.parametrize("mode", MODES)
def test_arbitrary_logical_state_is_preserved_with_split_css_input(protocols, mode):
    from spiderwarp.fault_tolerance_verification import _make_encoded_bell_input
    protocol = protocols[mode]
    code = protocol.code
    bell, refs, pairing = _make_encoded_bell_input(code.H_x, code.H_z, code.L_x, code.L_z,
                                                 protocol.required_qubits)
    simulator = stim.TableauSimulator(seed=7)
    simulator.do_circuit(bell)
    simulator.x(1)
    simulator.z(17)
    outcome = protocol.run(simulator)
    for q in range(code.n):
        if outcome.frame_x >> q & 1:
            simulator.x(q)
        if outcome.frame_z >> q & 1:
            simulator.z(q)
    for basis, checks, logicals in (("X", code.H_x, code.L_x), ("Z", code.H_z, code.L_z)):
        for i, row in enumerate(list(checks) + list(logicals)):
            observable = stim.PauliString(bell.num_qubits)
            for q in np.flatnonzero(row):
                observable[int(q)] = basis
            if i >= len(checks):
                logical_index = i-len(checks)
                for j, q in enumerate(refs):
                    if (j == logical_index if basis == "X" else pairing[j, logical_index]):
                        observable[q] = basis
            assert simulator.peek_observable_expectation(observable) == 1


def test_data_waits_through_every_failed_preparation(protocols):
    sampler = ProtocolSampler(protocols["fixed"], noise_model=CustomErrorModel(p_linear_mem=.001))
    g = sampler.protocol.gadgets["prep_Z_d3"]
    noisy = sampler.circuits[id(g.circuit)]
    idle = noisy[0]
    assert idle.name == "Z_ERROR"
    assert {t.value for t in idle.targets_copy()} == set(range(32))
    assert idle.gate_args_copy()[0] > .001


@pytest.mark.parametrize("mode", MODES)
def test_flagged_corrupted_ancilla_never_reaches_data(protocols, mode):
    protocol = protocols[mode]
    experiment = _StateExperiment(protocol.code, protocol.required_qubits, "+")

    class Injected:
        def __init__(self):
            self.sim = experiment.initial.copy(seed=7)
            self.injected = False

        def current_measurement_record(self):
            return self.sim.current_measurement_record()

        def do_circuit(self, circuit):
            if not self.injected:
                modified = stim.Circuit()
                flipped = False
                for op in circuit:
                    if op.name == "M" and not flipped:
                        modified.append("X", [op.targets_copy()[0]])
                        flipped = True
                    modified.append(op)
                modified.append("X", list(range(32, 40)))
                modified.append("Z", list(range(40, 48)))
                self.sim.do_circuit(modified)
                self.injected = True
            else:
                self.sim.do_circuit(circuit)

    simulator = Injected()
    outcome = protocol.run(simulator)
    assert outcome.measurements[0][3] != 0
    assert outcome.measurements[1][0] == "discard_Z"
    assert not experiment.logical_failure(simulator.sim, frame_x=outcome.frame_x, frame_z=outcome.frame_z)


@pytest.mark.parametrize("mode", MODES)
def test_exported_controller_matches_retry_paths_and_all_decoder_inputs(protocols, mode):
    from spiderwarp._steane_qec_guppy import generate_source
    protocol = protocols[mode]
    tree = ast.parse(generate_source(protocol))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            definitions.append(node)
        elif isinstance(node, ast.ClassDef):
            node.decorator_list = [ast.Name(id="dataclass", ctx=ast.Load())]
            definitions.append(node)
        elif isinstance(node, ast.Assign):
            definitions.append(node)
    exported = dict(comptime=lambda v: v, dataclass=dataclass, array=list, qubit=lambda: 0,
                    measure_bit=lambda q: False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 "<exported Steane>", "exec"), exported)
    sim = ScriptedReadout(protocol)

    def preparation(name):
        sim.do_circuit(protocol.gadgets[name].circuit)
        return protocol.gadgets[name].readout.decode(sim.record[-protocol.gadgets[name].circuit.num_measurements:])[0]

    def read(basis):
        sim.do_circuit(protocol.gadgets[f"read_{basis}"].circuit)
        return protocol.gadgets[f"read_{basis}"].readout.decode(sim.record[-32:])[1]

    for basis in ("Z", "X"):
        for d in (1, 3):
            name = f"prep_{basis}_d{d}"
            exported[f"prep_{basis.lower()}_d{d}"] = lambda ancilla, name=name: preparation(name)
        exported[f"read_{basis.lower()}"] = lambda data, ancilla, basis=basis: read(basis)
        for s, correction in enumerate(protocol.decoders[basis]):
            assert exported[f"DECODER_{basis}"][s] == correction
        checks = protocol.code.H_z if basis == "Z" else protocol.code.H_x
        for mask in [0, (1 << 32)-1] + [1 << q for q in range(32)]:
            expected = sum(int(v) << i for i, v in enumerate(checks @ np.array(
                [mask >> q & 1 for q in range(32)], dtype=np.uint8) % 2))
            assert exported[f"frame_syndrome_{basis.lower()}"](mask) == expected
    outcome = exported["se_cycle_with_frame"]([0] * 32, 0, 0)
    reference = ScriptedReadout(protocol)
    expected = protocol.run(reference)
    assert (outcome.accepted, outcome.frame_x, outcome.frame_z) == (expected.accepted, expected.frame_x, expected.frame_z)
    assert sim.trace == [n for n in reference.trace if not n.startswith("discard_")]


def test_comparison_plots_and_count_audit(tmp_path, monkeypatch):
    from spiderwarp.se_comparison import save_comparison
    from spiderwarp.se_plotting import plot_results
    from spiderwarp.se_protocol_simulation import scaled_noise_model, _wilson_interval
    from matplotlib.figure import Figure
    directories = [tmp_path / str(i) for i in range(4)]
    labels = ["Adaptive", "Steane d4", "Steane per ancilla", "Steane per cycle"]
    for i, directory in enumerate(directories):
        directory.mkdir()
        rows = []
        for basis in ("0", "+"):
            for p, n in ((1e-4, 1), (.01, 1), (.0008, 1), (.0008, 3)):
                accepted, errors = (990 if i == 0 else 1000), 10 + i
                low, high = _wilson_interval(errors, accepted)
                rows.append(dict(protocol_hash=f"protocol-{i}", engine_hash="engine", prepared_state=basis,
                                 p=p, cycles=n, shots=1000, target_shots=1000, accepted=accepted,
                                 logical_errors=errors, total_cnots=1000*(300 + i*100)*n,
                                 accepted_cnots=accepted*(300 + i*100)*n,
                                 noise=json.dumps(vars(scaled_noise_model(p))), seed=7,
                                 ler_low=low, ler_high=high,
                                 sweeps=json.dumps(["physical_error" if p != .0008 else "cycles"])))
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit = tmp_path / "audit"
    save_comparison(directories, labels, audit)
    assert json.loads((audit / "validation.json").read_text())["complete"]
    original = Figure.savefig

    def check_figure(figure, path, **kwargs):
        mode = path.stem
        for ax in figure.axes:
            assert all(label in ax.get_legend_handles_labels()[1] for label in labels)
            assert ax.get_yscale() == ("log" if mode in {"physical_error", "cycles"} else "linear")
            if mode == "survival":
                assert .9 < ax.get_ylim()[0] < ax.get_ylim()[1] <= 1
        return original(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_figure)
    paths = plot_results(directories[0], audit / "figures", compare=directories[1:], labels=labels)
    assert len(paths) == 8 and all(p.exists() for p in paths)


@pytest.mark.parametrize("mode", MODES)
def test_native_hugr_executes_both_bases(protocols, mode, tmp_path):
    pytest.importorskip("guppylang")
    pytest.importorskip("selene_sim")
    from spiderwarp.se_native import run_native
    protocol = protocols[mode]
    path = protocol.save_bundle(tmp_path / "protocol")
    for basis in ("0", "+"):
        result = run_native(path, tmp_path / basis, prepared_state=basis, cycles=3,
                            incoming_x=1, incoming_z=1 << 17, p=0, shots=4, seed=7)
        assert result["accepted"] == 4 and result["logical_errors"] == 0

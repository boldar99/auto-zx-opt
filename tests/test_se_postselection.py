"""Compiled history guards, physical frames, independent checks and exports."""
from dataclasses import replace
import csv
import json
from pathlib import Path

import pytest
import stim

from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol, build_adaptive_se_protocol
from spiderwarp._se_adaptive import PolicyNode
from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_postselection import postselect_order_one, verify_order_one_guard
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler
from test_adaptive_se_policy import bell_actions


@pytest.fixture
def small_protocol():
    code, _, actions = bell_actions()
    return build_adaptive_se_protocol(code, actions, first_action="Z", export=False)


def test_contract_pruning_roundtrip_and_physical_frames(small_protocol, tmp_path):
    original = small_protocol
    old_nodes, old_contract, old_report = original.policy.nodes, original.policy.contract, original.verification
    derived = postselect_order_one(original)
    assert derived.verification.verified
    assert derived.policy.contract.accept_clean_through == 1
    assert derived.policy.contract.logical_order == 1
    assert original.policy.nodes == old_nodes and original.policy.contract == old_contract
    assert original.verification is old_report
    assert verify_order_one_guard(derived)["nonreject_edges"] > 0
    loaded = load_protocol(derived.save_bundle(tmp_path))
    assert loaded.policy.nodes == derived.policy.nodes
    for x in range(4):
        for z in range(4):
            a = stim.TableauSimulator(seed=17)
            a.do_circuit(stim.Circuit("H 0\nCX 0 1"))
            for q in range(2):
                if x >> q & 1:
                    a.x(q)
                if z >> q & 1:
                    a.z(q)
            b = a.copy(seed=17)
            expected = original.run(a, frame_x=x, frame_z=z)
            actual = loaded.run(b, frame_x=x, frame_z=z)
            assert actual == expected
            assert a.canonical_stabilizers() == b.canonical_stabilizers()
    result = ProtocolSampler(loaded, noise_model=CustomErrorModel(), cycles=5).sample(20, 17)
    assert result.accepted == 20 and result.logical_errors == 0


def test_internal_frame_updates_keep_the_guard_in_correct_coordinates(small_protocol):
    p = small_protocol
    frame = 1 | (2 << p.code.n)
    nodes = []
    for node in p.policy.nodes:
        if node.kind == "extract":
            g = p.policy.actions[node.action].gadget
            syndrome = p.policy.decoder.space.component_syndrome(frame, "X" if g.basis == "Z" else "Z")
            node = replace(node, children={(f, s ^ syndrome): child for (f, s), child in node.children.items()})
        elif node.kind == "accept":
            node = replace(node, frame=node.frame ^ frame)
        nodes.append(node)
    nodes.append(PolicyNode("frame", frame=frame, next=p.policy.root))
    # An initial frame offset, permuted observation keys, and the matching
    # terminal offset describe exactly the original physical instrument.
    shifted = AdaptiveSEProtocol(p.code, replace(p.policy, nodes=tuple(nodes), root=len(nodes)-1))
    shifted.verify()
    derived = postselect_order_one(shifted)
    for x in range(4):
        for z in range(4):
            a = stim.TableauSimulator(seed=17)
            a.do_circuit(stim.Circuit("H 0\nCX 0 1"))
            for q in range(2):
                if x >> q & 1:
                    a.x(q)
                if z >> q & 1:
                    a.z(q)
            b = a.copy(seed=17)
            assert p.run(a) == derived.run(b)
            assert a.canonical_stabilizers() == b.canonical_stabilizers()


def test_independent_guard_rejects_an_unsupported_continuation(small_protocol):
    derived = postselect_order_one(small_protocol)
    nodes = list(derived.policy.nodes)
    root = derived.policy.root
    action = derived.policy.actions[nodes[root].action].gadget
    outcome = (0, 1 << len(action.readout.syndromes))
    accept = next(i for i, node in enumerate(nodes) if node.kind == "accept")
    nodes[root] = replace(nodes[root], children=dict(nodes[root].children) | {outcome: accept})
    derived.policy = replace(derived.policy, nodes=tuple(nodes))
    with pytest.raises(ValueError, match="no low-order explanation"):
        verify_order_one_guard(derived)


@pytest.mark.parametrize("basis", ("0", "+"))
def test_native_compiles_reloads_and_carries_frames(small_protocol, tmp_path, basis):
    pytest.importorskip("guppylang")
    pytest.importorskip("selene_sim")
    from spiderwarp.se_native import run_native
    derived = postselect_order_one(small_protocol)
    path = derived.save_bundle(tmp_path / "protocol")
    result = run_native(path, tmp_path / "native", prepared_state=basis, cycles=3,
                        incoming_x=1, incoming_z=2, p=0, shots=4, seed=17)
    assert result["accepted"] == 4 and result["logical_errors"] == 0


def test_full_size_saved_guard():
    path = Path(__file__).resolve().parents[1] / "build/se_postselected_32_20_4"
    if not (path / "protocol.json").exists():
        pytest.skip("Full-size bundle is created by the integration build")
    protocol = load_protocol(path)
    assert protocol.verification.verified and protocol.required_qubits == 60
    assert protocol.policy.contract.accept_clean_through == 1
    assert protocol.policy.contract.input_scope == 'total-order'
    assert {protocol.policy.actions[node.action].gadget.distance
            for node in protocol.policy.nodes if node.kind == "extract"} == {2, 4}
    for basis in ("0", "+"):
        result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), prepared_state=basis,
                                 cycles=5).sample(20, 17)
        assert result.accepted == 20 and result.logical_errors == 0


def test_comparison_keeps_postselection_yield_and_resource_count(tmp_path):
    from spiderwarp.se_comparison import save_comparison
    from spiderwarp.se_protocol_simulation import _wilson_interval
    directories = [tmp_path / "original", tmp_path / "postselected"]
    for directory, accepted, errors in zip(directories, (90, 70), (5, 1)):
        directory.mkdir()
        lo, hi = _wilson_interval(errors, accepted)
        rows = [dict(protocol_hash=directory.name, engine_hash="engine", prepared_state=basis, p=.0008, cycles=1,
                     shots=100, target_shots=100, accepted=accepted, logical_errors=errors,
                     total_cnots=100*400, accepted_cnots=accepted*400, ler_low=lo, ler_high=hi, noise='{"p_2q":0.0008}',
                     sweeps='["cycles","physical_error"]', seed=17)
                for basis in ("0", "+")]
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    out = tmp_path / "comparison"
    save_comparison(directories, ["original", "PS"], out,
                    qubit_counts=[60, 60], always_accept=[False, False])
    rows = list(csv.DictReader((out / "comparison.csv").open()))
    assert [int(r["accepted"]) for r in rows] == [90, 90, 70, 70]
    assert all(int(r["peak_qubits"]) == 60 for r in rows)
    assert json.loads((out / "validation.json").read_text())["full_acceptance_required"] == []
    with pytest.raises(ValueError, match="Invalid acceptance"):
        save_comparison(directories, ["original", "PS"], out)
    from spiderwarp.se_plotting import plot_results
    paths = plot_results(directories[0], out / "figures", compare=directories[1:],
                         labels=["original", "PS"], physical_survival=True)
    assert len(paths) == 10
    assert (out / "figures/physical_survival.png").exists()


@pytest.mark.parametrize("shots", (1000000, 160761))
def test_physical_survival_handles_full_acceptance_at_large_shot_counts(tmp_path, shots):
    from spiderwarp.se_plotting import plot_results
    rows = [dict(protocol_hash="protocol", engine_hash="engine", prepared_state=b, p=.0001,
                 cycles=1, shots=shots, accepted=shots, logical_errors=0,
                 total_cnots=400*shots, noise='{"p_2q":0.0001}', sweeps='["physical_error"]')
            for b in ("0", "+")]
    with (tmp_path / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = plot_results(tmp_path, physical_survival=True)
    assert len(paths) == 4
    assert (tmp_path / "figures/physical_survival.png").exists()

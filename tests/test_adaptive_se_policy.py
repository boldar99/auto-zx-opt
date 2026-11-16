"""Fault contracts, physical response replays, and independent policy checks."""

from dataclasses import replace

import numpy as np
import pytest
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp._se_circuit import (
    PauliSpace, Readout, SEGadget, dual_circuit, extract_readout,
    lower_to_guppy_gates, sample_annotation_parities, validate_gadget,
)
from spiderwarp._se_adaptive import (
    AdaptivePolicyBuilder, AdaptiveVerificationError, FTContract,
    FTContractInfeasibleError, FixedCSSDecoder, HistoryEngine, PolicyNode,
    CompatibleSet, ExtractionAction, terminal_fault_obstructions, verify_adaptive_policy,
)
from spiderwarp.fault_tolerance_verification import _make_encoded_bell_input
from spiderwarp.utils import get_project_root
from spiderwarp.adaptive_steane_se import build_adaptive_se_protocol


def bell_actions():
    h = np.array([[1, 1]], dtype=np.uint8)
    empty = np.empty((0, 2), dtype=np.uint8)
    code = CSSCode(2, 0, 3, h, h, empty, empty, True)
    space = PauliSpace(code)
    c = stim.Circuit("RX 2\nCX 2 0 2 1\nMX 2\nDETECTOR(1, 0) rec[-1]")
    readout = extract_readout(c, code)
    actions = {basis: SEGadget(3, basis, lower_to_guppy_gates(raw), readout, space)
               for basis, raw in (("X", c), ("Z", dual_circuit(c)))}
    return code, space, actions


def real_actions():
    code = CSSCode.load_code("misc", "32_20_4")
    space = PauliSpace(code)
    actions = {}
    for d in (1, 2, 4):
        c = stim.Circuit.from_file(get_project_root() / f"assets/circuits/SECircuits/32_20_4_d{d}.stim")
        readout = extract_readout(c, code)
        for basis, raw in (("X", c), ("Z", dual_circuit(c))):
            actions[f"{basis}{d}"] = SEGadget(
                d, basis, lower_to_guppy_gates(raw), readout, space)
    return code, space, actions


@pytest.fixture(scope="session")
def verified_32_adaptive_protocol():
    code, _, actions = real_actions()
    return build_adaptive_se_protocol(code, actions, first_action="Z4", minimize=False,
                                      max_cnot_cost=1092)


def test_small_policy_is_minimax_and_checks_all_physical_fault_alternatives():
    _, space, actions = bell_actions()
    policy = AdaptivePolicyBuilder(space, actions).build(first_action="Z")
    assert policy.optimal
    assert policy.report.verified
    assert policy.report.worst_case_cnots == 4
    assert policy.report.max_extractions == 2
    assert policy.report.peak_qubits == 3
    assert policy.report.discarded_contexts == 0
    # A gate fault carrying X on one target and Z on the other is one event.
    contract = policy.contract
    assert contract.remaining((0, 0, 0)) == 1
    assert contract.remaining((0, 1, 0)) == 0


def test_verifier_rejects_changed_acceptance_frame_and_nontermination():
    _, space, actions = bell_actions()
    policy = AdaptivePolicyBuilder(space, actions).build(first_action="Z")
    accept = next(i for i, n in enumerate(policy.nodes) if n.kind == "accept")
    nodes = list(policy.nodes)
    nodes[accept] = PolicyNode("reject")
    with pytest.raises(AdaptiveVerificationError, match="Mandatory-acceptance"):
        verify_adaptive_policy(replace(policy, nodes=tuple(nodes)))
    nodes[accept] = replace(policy.nodes[accept], frame=policy.nodes[accept].frame ^ 1)
    with pytest.raises(AdaptiveVerificationError, match="CSS fault-count filter"):
        verify_adaptive_policy(replace(policy, nodes=tuple(nodes)))
    nodes = list(policy.nodes)
    root = nodes[policy.root]
    nodes[policy.root] = replace(root, children={o: policy.root for o in root.children})
    with pytest.raises(AdaptiveVerificationError, match="does not terminate"):
        verify_adaptive_policy(replace(policy, nodes=tuple(nodes)))


def test_peak_qubit_limit_and_first_action_are_enforced():
    _, space, actions = bell_actions()
    builder = AdaptivePolicyBuilder(space, actions, peak_qubits=2)
    with pytest.raises(ValueError, match="peak-qubit limit"):
        builder.build(first_action="Z")
    builder = AdaptivePolicyBuilder(space, actions)
    with pytest.raises(ValueError, match="CNOT budget"):
        builder.build(first_action="Z", max_cnot_cost=1)


def test_confirmed_contract_and_d1_zero_budget_execution(monkeypatch):
    _, space, actions = bell_actions()
    contract = FTContract(1, 1, fault_order=2, accept_clean_through=2)
    assert contract.must_accept((0, 2, 0))
    assert not contract.must_accept((1, 1, 0))
    assert not contract.must_accept((2, 0, 0))
    engine = HistoryEngine(space, contract)
    assert engine.remaining_max(engine.initial()) == 2
    spent = CompatibleSet(frozenset({(0, 2, 0)}), (0, 0, 0), engine.sectors)
    assert engine.remaining_max(spent) == 0
    g = replace(actions["X"], distance=1)
    original = g.fault_effects

    def fault_free_only(order):
        assert order == 0, "A d1 action with exhausted budget must not enumerate faults"
        return original(order)

    monkeypatch.setattr(g, "fault_effects", fault_free_only)
    result = engine.transition(spent, ExtractionAction("ideal_d1", g))
    assert set(result) == {(0, 0)}
    builder = AdaptivePolicyBuilder(space, {"Z": replace(actions["Z"], distance=1)}, contract=contract)
    with pytest.raises(ValueError, match="distance ladder"):
        builder.build(first_action="Z")


def test_joint_syndrome_bitsets_retain_conditioning_correlations():
    _, space, actions = bell_actions()
    engine = HistoryEngine(space, FTContract(0, 0))
    # Only (sx,sz)=(0,0),(1,1), rather than all four independent combinations.
    mask = (1 << engine.indices[0]) | (1 << engine.indices[3])
    histories = CompatibleSet(frozenset(), (mask,), engine.sectors)
    a = ExtractionAction("X", actions["X"])
    groups = engine.transition(histories, a)
    assert set(groups[(0, 0)]) == {(-1, 0, 0)}
    assert set(groups[(0, 1)]) == {(-1, 0, 3)}
    for outcome, child in groups.items():
        assert engine.transition(histories, a, outcome) == {outcome: child}


def test_filtered_real_transition_matches_full_enumeration():
    _, space, actions = real_actions()
    engine = HistoryEngine(space, FTContract.for_code(space.code))
    action = ExtractionAction("Z4", actions["Z4"])
    initial = engine.initial()
    full = engine.transition(initial, action)
    for outcome in ((0, 0), (0, 32), (1, 0)):
        assert engine.transition(initial, action, outcome) == {outcome: full[outcome]}


def test_distance_four_two_fault_logical_guarantee_has_physical_no_go_certificate():
    code, space, actions = real_actions()
    builder = AdaptivePolicyBuilder(space, actions, contract=FTContract.correct_or_reject(code))
    with pytest.raises(FTContractInfeasibleError) as raised:
        builder.build(first_action="Z4")
    assert {w.action for w in raised.value.obstructions} == set(actions)
    for witness in raised.value.obstructions:
        g = actions[witness.action]
        assert len(witness.faults) <= 2
        assert witness.decoded_logical != 0
        assert space.pauli_syndrome(witness.decoded_logical) == 0
        bell, _, _ = _make_encoded_bell_input(
            code.H_x, code.H_z, code.L_x, code.L_z, g.circuit.num_qubits)
        # Replaying *actual physical faults* must give the mandatory clean
        # classical record, despite the nontrivial final decoded logical Pauli.
        assert not sample_annotation_parities(bell + g.with_faults(witness.faults), shots=8).any()
        assert builder.engine.decoder.logical(space.reduce(witness.output_error)) == witness.decoded_logical


def test_css_boundary_filter_does_not_split_the_physical_fault_budget():
    _, space, actions = bell_actions()
    engine = HistoryEngine(space, FTContract(1, 1))
    assert engine.safe_frame({(-1, 1, space.pauli_syndrome(1 | (1 << (space.n + 1))))}) == 0
    for action in actions.values():
        validate_gadget(action)


def test_reference_executor_returns_frame_without_applying_correction_gates():
    code, _, actions = bell_actions()
    protocol = build_adaptive_se_protocol(code, actions, first_action="Z")
    bell, _, _ = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z, protocol.required_qubits)
    for x in range(4):
        for z in range(4):
            simulator = stim.TableauSimulator()
            simulator.do_circuit(bell)
            for q in range(2):
                if (x >> q) & 1:
                    simulator.x(q)
                if (z >> q) & 1:
                    simulator.z(q)
            result = protocol.run(simulator)
            assert result.accepted
            assert result.corrections == ()
            # The physical data retain the input syndrome before frame use.
            for basis, expected in (("Z", x.bit_count() % 2), ("X", z.bit_count() % 2)):
                observable = stim.PauliString(protocol.required_qubits)
                observable[0] = observable[1] = basis
                assert simulator.peek_observable_expectation(observable) == (-1) ** expected
            for q in range(2):
                if (result.frame_x >> q) & 1:
                    simulator.x(q)
                if (result.frame_z >> q) & 1:
                    simulator.z(q)
            for basis in ("X", "Z"):
                observable = stim.PauliString(protocol.required_qubits)
                observable[0] = observable[1] = basis
                assert simulator.peek_observable_expectation(observable) == 1


def test_adaptive_guppy_export_compiles_reloads_and_agrees_on_control_flow(tmp_path):
    pytest.importorskip("guppylang")
    from hugr.package import Package
    from test_optimized_steane_se import generated_python

    code, _, actions = bell_actions()
    protocol = build_adaptive_se_protocol(code, actions, first_action="Z", output_dir=tmp_path, export=True)
    Package.from_bytes(protocol.package_path.read_bytes())
    import hashlib
    import json
    certificate = json.loads((tmp_path / "verification.json").read_text())
    assert certificate["verification"]["verified"]
    assert certificate["hugr_sha256"] == hashlib.sha256(protocol.package_path.read_bytes()).hexdigest()
    for action in certificate["circuits"]:
        assert action["sha256"] == hashlib.sha256((tmp_path / action["path"]).read_bytes()).hexdigest()
    exported = generated_python(protocol.source, protocol.source_path)
    assert exported["VERIFIED_AT_BUILD"]
    assert exported["WORST_CASE_CNOTS"] == 4
    assert "apply_correction" not in protocol.source
    for x in range(4):
        for z in range(4):
            simulator = stim.TableauSimulator()
            simulator.do_circuit(stim.Circuit("H 0\nCX 0 1"))
            for q in range(2):
                if (x >> q) & 1:
                    simulator.x(q)
                if (z >> q) & 1:
                    simulator.z(q)
            expected = protocol.run(simulator)
            records = iter(expected.measurements)

            def execute(data, index):
                name, _, _, flags, syndrome = next(records)
                assert exported["ACTION_NAMES"][index] == name
                return exported["ExtractionResult"](flags, syndrome)

            exported["execute_action"] = execute
            actual = exported["se_cycle"](list(range(code.n)))
            assert (actual.accepted, actual.frame_x, actual.frame_z) == (
                expected.accepted, expected.frame_x, expected.frame_z)
            assert next(records, None) is None


def _apply_frame(simulator, n, frame_x, frame_z):
    for q in range(n):
        if (frame_x >> q) & 1:
            simulator.x(q)
        if (frame_z >> q) & 1:
            simulator.z(q)


def _data_syndrome(simulator, code, size):
    result = 0
    index = 0
    for basis, rows in (("Z", code.H_z), ("X", code.H_x)):
        for row in rows:
            observable = stim.PauliString(size)
            for q in np.flatnonzero(row):
                observable[int(q)] = basis
            value = simulator.peek_observable_expectation(observable)
            assert value != 0
            result |= int(value < 0) << index
            index += 1
    return result


def _bell_logical_expectations(simulator, code, size, refs, pairing):
    values = []
    for basis, rows in (("X", code.L_x), ("Z", code.L_z)):
        for i, row in enumerate(rows):
            observable = stim.PauliString(size)
            for q in np.flatnonzero(row):
                observable[int(q)] = basis
            for j, q in enumerate(refs):
                if (j == i if basis == "X" else pairing[j, i]):
                    observable[q] = basis
            values.append(simulator.peek_observable_expectation(observable))
    return values


def test_real_one_fault_old_rejection_is_recovered_and_preserves_logical_state(verified_32_adaptive_protocol):
    protocol = verified_32_adaptive_protocol
    code, policy = protocol.code, protocol.policy
    primary = policy.actions["Z4"].gadget
    space = policy.decoder.space
    effect = next(e for e in primary.single_fault_effects
                  if primary.unpack(e) == (0, 32, space.reduce(1 << (code.n + 16))))
    faults = primary.fault_witness(effect, 1)
    bell, refs, pairing = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z, protocol.required_qubits)
    simulator = stim.TableauSimulator()
    simulator.do_circuit(bell)

    class InjectFirst:
        first = True

        def current_measurement_record(self):
            return simulator.current_measurement_record()

        def do_circuit(self, circuit):
            simulator.do_circuit(primary.with_faults(faults) if self.first else circuit)
            self.first = False

    result = protocol.run(InjectFirst())
    assert result.measurements[0][3:] == (0, 32)
    assert result.accepted
    _apply_frame(simulator, code.n, result.frame_x, result.frame_z)
    signature = _data_syndrome(simulator, code, bell.num_qubits)
    assert policy.decoder.weights["X"][signature & policy.decoder.x_mask] <= 1
    assert policy.decoder.weights["Z"][signature >> policy.decoder.x_width] <= 1
    correction = policy.decoder.correction(signature)
    _apply_frame(simulator, code.n, correction & ((1 << code.n) - 1), correction >> code.n)
    assert set(_bell_logical_expectations(simulator, code, bell.num_qubits, refs, pairing)) == {1}


def test_real_clean_two_fault_output_accepts_without_claiming_logical_correctness(verified_32_adaptive_protocol):
    protocol = verified_32_adaptive_protocol
    code, policy = protocol.code, protocol.policy
    bell, refs, pairing = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z, protocol.required_qubits)
    clean = stim.TableauSimulator()
    clean.do_circuit(bell)
    baseline = protocol.run(clean)
    last_name = baseline.measurements[-1][0]
    action = policy.actions[last_name]
    witness = terminal_fault_obstructions({last_name: action}, policy.decoder, 2)[0]
    simulator = stim.TableauSimulator()
    simulator.do_circuit(bell)

    class InjectLast:
        stage = 0

        def current_measurement_record(self):
            return simulator.current_measurement_record()

        def do_circuit(self, circuit):
            self.stage += 1
            simulator.do_circuit(action.gadget.with_faults(witness.faults)
                                 if self.stage == len(baseline.measurements) else circuit)

    result = protocol.run(InjectLast())
    assert result.measurements == baseline.measurements
    assert result.accepted
    _apply_frame(simulator, code.n, result.frame_x, result.frame_z)
    signature = _data_syndrome(simulator, code, bell.num_qubits)
    assert policy.decoder.weights["X"][signature & policy.decoder.x_mask] <= 2
    assert policy.decoder.weights["Z"][signature >> policy.decoder.x_width] <= 2
    correction = policy.decoder.correction(signature)
    _apply_frame(simulator, code.n, correction & ((1 << code.n) - 1), correction >> code.n)
    assert -1 in _bell_logical_expectations(simulator, code, bell.num_qubits, refs, pairing)


def test_real_export_preserves_every_stored_transition(verified_32_adaptive_protocol):
    from spiderwarp._se_adaptive_guppy import generate_adaptive_source
    from test_optimized_steane_se import generated_python

    protocol = verified_32_adaptive_protocol
    exported = generated_python(generate_adaptive_source(protocol))
    policy = protocol.policy
    from importlib import import_module
    audit = import_module('analysis.32_20_4_bounded_comparison.probability_controls').audit_export
    checked = audit(protocol, exported)
    assert checked['transitions_checked'] == sum(len(n.children) for n in policy.nodes if n.kind == 'extract')
    assert checked['all_absent_outcomes_checked']


def test_frame_coordinate_change_is_invertible_and_normalization_is_idempotent():
    _, space, actions = bell_actions()
    engine = HistoryEngine(space, FTContract.for_code(space.code))
    histories = engine.initial()
    for action in actions.values():
        for child in engine.transition(histories, ExtractionAction(action.basis, action)).values():
            for frame in range(16):
                shifted = engine.translate(child, frame)
                assert engine.translate(shifted, frame) == child
                normalized, offset = engine.normalize(shifted)
                assert engine.translate(normalized, offset) == shifted
                assert engine.normalize(normalized) == (normalized, 0)


def test_existing_frame_composes_in_reference_and_exported_execution():
    from spiderwarp._se_adaptive_guppy import generate_adaptive_source
    from test_optimized_steane_se import generated_python
    code, _, actions = bell_actions()
    protocol = build_adaptive_se_protocol(code, actions, first_action="Z")
    exported = generated_python(generate_adaptive_source(protocol))
    for x in range(4):
        for z in range(4):
            simulator = stim.TableauSimulator()
            simulator.do_circuit(stim.Circuit("H 0\nCX 0 1"))
            _apply_frame(simulator, 2, x, z)
            expected = protocol.run(simulator, frame_x=x, frame_z=z)
            records = iter(expected.measurements)

            def execute(data, index):
                name, _, _, flags, syndrome = next(records)
                assert exported["ACTION_NAMES"][index] == name
                return exported["ExtractionResult"](flags, syndrome)

            exported["execute_action"] = execute
            actual = exported["se_cycle_with_frame"](list(range(code.n)), x, z)
            assert (actual.accepted, actual.frame_x, actual.frame_z) == (
                expected.accepted, expected.frame_x, expected.frame_z)
            assert expected.accepted
            _apply_frame(simulator, code.n, actual.frame_x, actual.frame_z)
            assert _data_syndrome(simulator, code, protocol.required_qubits) == 0

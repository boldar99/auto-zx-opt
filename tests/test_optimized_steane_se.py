from pathlib import Path
from dataclasses import dataclass, replace
import ast
import subprocess
import sys

import numpy as np
import pytest
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp._se_circuit import (
    PauliSpace, SEGadget, Readout, CircuitFault, dual_circuit,
    extract_readout, validate_gadget, lower_to_guppy_gates,
    sample_annotation_parities,
)
from spiderwarp._se_policy import (
    PolicyBuilder, ProtocolVerificationError, RecoveryPlan, VerificationReport,
)
from spiderwarp.optimized_steane_se import (
    CompiledSEProtocol, optimized_steane_SE_orchestrator, verify_se_protocol,
)
from spiderwarp.fault_tolerance_verification import _make_encoded_bell_input
from spiderwarp.utils import get_project_root


def bell_code():
    # A fully stabilized Bell block, with distance 3 as a conservative bound
    # (k=0 has no logical operator of finite distance).
    h = np.array([[1, 1]], dtype=np.uint8)
    logicals = np.empty((0, 2), dtype=np.uint8)
    return CSSCode(2, 0, 3, h, h, logicals, logicals, True)


def bell_gadget():
    return stim.Circuit(
        "R 3\nM 3\nRX 2\nCX 2 0 2 1\nMX 2\n"
        "DETECTOR(0, 0) rec[-2]\nDETECTOR(1, 0) rec[-1]"
    )


def raw_gadgets(code, circuits):
    space = PauliSpace(code)
    gadgets = {}
    for distance, circuit in circuits.items():
        readout = extract_readout(circuit, code)
        for basis, c in [("X", circuit), ("Z", dual_circuit(circuit))]:
            c = lower_to_guppy_gates(readout.annotate(c))
            g = SEGadget(distance, basis, c, readout, space)
            validate_gadget(g)
            gadgets[basis, distance] = g
    return space, gadgets


def test_pauli_quotient_keeps_arbitrary_logical_information():
    code = CSSCode.load_code("misc", "32_20_4")
    space = PauliSpace(code)
    for row in code.L_x:
        assert space.reduce(sum(1 << int(q) for q in np.flatnonzero(row))) != 0
    assert space.reduce(3) not in space.low_weight(1)
    assert space.reduce(1 | (1 << code.n)) in space.low_weight(1)


def test_record_parities_survive_permutation():
    readout = Readout(((0,), (2,)), ((1, 3),))
    remapped = readout.remap({0: 3, 1: 0, 2: 2, 3: 1})
    assert readout.decode([1, 0, 1, 1]) == remapped.decode([1, 1, 1, 0])
    c = stim.Circuit("R 0 1 2 3\nM 0 1 2 3")
    assert str(remapped.annotate(c)).endswith("DETECTOR(1, 0) rec[-1] rec[-4]")
    with pytest.raises(ValueError, match="duplicated"):
        readout.remap({0: 0, 1: 0})


def test_cannot_use_data_measurement_or_uninitialized_ancilla():
    code = bell_code()
    space = PauliSpace(code)
    for text, message in [("M 0", "measure a data"), ("CX 0 2\nM 2", "reset before use"),
                          ("R 2\nH 2\nR 2\nM 2", "another reset lifetime")]:
        gadget = SEGadget(3, "X", stim.Circuit(text), Readout((), ((0,),)), space)
        with pytest.raises(ValueError, match=message):
            _ = gadget.operations


def test_raw_flag_signs_are_checked_without_subtracting_stim_reference():
    circuit = bell_gadget()
    # The spectator flag is deterministically raised. Its detector *changes*
    # relative to Stim's reference are zero, but its protocol value is one.
    circuit.insert(1, stim.CircuitInstruction("X", [3]))
    code = bell_code()
    space = PauliSpace(code)
    gadget = SEGadget(3, "X", circuit, extract_readout(circuit, code), space)
    with pytest.raises(ValueError, match="ideal encoded Bell"):
        validate_gadget(gadget)


def test_joint_responses_match_stim_for_input_paulis_and_physical_faults():
    code = bell_code()
    space, gadgets = raw_gadgets(code, {3: bell_gadget()})
    g = gadgets["X", 3]
    bell, _, _ = _make_encoded_bell_input(code.H_x, code.H_z, code.L_x, code.L_z, g.circuit.num_qubits)
    observables = [(stim.PauliString("XX"), 0b1100),
                   (stim.PauliString("ZZ"), 0b0011)]
    for error in (0, 1, 2, 4, 8, 5, 9):
        for effect in g.single_fault_effects:
            fault = g.fault_witness(effect, 1)
            sim = stim.TableauSimulator()
            sim.do_circuit(bell)
            for gate, q in space.product(error):
                getattr(sim, gate.lower())(q)
            sim.do_circuit(g.with_faults(fault))
            flags, syndrome, output = g.unpack(g.input_effect(error) ^ effect)
            assert g.readout.decode(sim.current_measurement_record()) == (flags, syndrome)
            for observable, anticommuting_mask in observables:
                expected = -1 if (output & anticommuting_mask).bit_count() % 2 else 1
                assert sim.peek_observable_expectation(observable) == expected


def test_fault_alternatives_at_one_location_cannot_form_a_two_fault_history():
    code = bell_code()
    g = SEGadget(3, "X", stim.Circuit(), Readout((), ()), PauliSpace(code))
    g.responses = ((0, 0, 0, 0), ((7, (1, 2, 3)),))
    assert g.two_fault_effects == ()
    g = SEGadget(3, "X", stim.Circuit(), Readout((), ()), PauliSpace(code))
    g.responses = ((0, 0, 0, 0), ((7, (1,)), (8, (2,))))
    assert g.two_fault_effects == (3,)


@pytest.fixture(scope="module")
def compiled_bell_protocol(tmp_path_factory):
    pytest.importorskip("guppylang")
    return optimized_steane_SE_orchestrator(
        bell_code(), [(1, stim.Circuit("RX 2\nCX 2 0 2 1\nMX 2\nDETECTOR(1, 0) rec[-1]")),
                      (3, bell_gadget())],
        output_dir=tmp_path_factory.mktemp("compiled_se"),
        verify=True,
    )


def test_builder_corrects_all_incoming_paulis_and_preserves_data(compiled_bell_protocol):
    p = compiled_bell_protocol
    assert p.verification.verified
    assert p.verification.correctable_faults == p.verification.detectable_faults == 1
    assert {plan.distance for plan in p.recoveries.values()} == {1}
    for error in ("X", "Y", "Z"):
        for q in range(p.code.n):
            sim = stim.TableauSimulator()
            sim.h(0)
            sim.cx(0, 1)
            getattr(sim, error.lower())(q)
            result = p.run(sim)
            assert result.accepted
            assert [m[:2] for m in result.measurements] == [("primary", "Z"), ("primary", "X")]
            assert sim.peek_observable_expectation(stim.PauliString("XX")) == 1
            assert sim.peek_observable_expectation(stim.PauliString("ZZ")) == 1
    # All measured/reset qubits remain outside the data prefix after reuse.
    for gadget in p.gadgets.values():
        assert all(q >= 2 for name, qs in gadget.operations if name in {"R", "M"} for q in qs)


@pytest.mark.parametrize("prepared_state", ["0", "+"])
def test_ler_runs_adaptive_policy_with_noise_and_reproducible_seeds(
    compiled_bell_protocol, monkeypatch, prepared_state,
):
    from spiderwarp.noise_model import CustomErrorModel
    from spiderwarp.se_protocol_simulation import estimate_protocol_ler

    protocol = compiled_bell_protocol
    traces = []
    original_run = protocol.run

    def record_run(simulator):
        result = original_run(simulator)
        traces.append(result)
        return result

    monkeypatch.setattr(protocol, "run", record_run)
    ideal = estimate_protocol_ler(
        protocol, noise_model=CustomErrorModel(), prepared_state=prepared_state, shots=8, seed=17)
    assert ideal.accepted == 8 and ideal.logical_errors == 0
    # Every flipped spectator flag takes the recovery branch. The unflagged
    # recovery can still accept, so raised primary flags alone are not discards.
    recovered = estimate_protocol_ler(
        protocol, noise_model=CustomErrorModel(p_meas=1), prepared_state=prepared_state, shots=8)
    assert recovered.accepted == 8 and recovered.logical_errors == 0
    assert all(any(m[0] == "recovery" for m in r.measurements) for r in traces[-8:])
    noise = CustomErrorModel(p_1q=0.03, p_2q=0.03, p_init=0.03, p_meas=0.03)
    first = estimate_protocol_ler(protocol, noise_model=noise, prepared_state=prepared_state, shots=100, seed=17)
    second = estimate_protocol_ler(protocol, noise_model=noise, prepared_state=prepared_state, shots=100, seed=17)
    assert (first.accepted, first.logical_errors) == (second.accepted, second.logical_errors)
    assert vars(first.noise_model) == vars(second.noise_model)
    assert 0 < first.accepted <= first.shots
    with pytest.raises(ValueError, match="Noise probabilities"):
        estimate_protocol_ler(protocol, noise_model=CustomErrorModel(p_linear_mem=-0.01))
    with pytest.raises(ValueError, match="prepared_state"):
        estimate_protocol_ler(protocol, noise_model=CustomErrorModel(), prepared_state="bell")


def test_source_and_hugr_reload_without_spiderwarp(compiled_bell_protocol, tmp_path):
    p = compiled_bell_protocol
    source, package = p.save(tmp_path)
    assert "lut_decoder" not in source.read_text()
    assert "stim" not in source.read_text()
    result = subprocess.run(
        [sys.executable, "-I", "-c", "\n".join([
            "import importlib.util, sys",
            "from pathlib import Path",
            "from hugr.package import Package",
            f"path = Path({str(source)!r})",
            "spec = importlib.util.spec_from_file_location('saved_se', path)",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            "module.se_cycle.check()",
            "module.postselected_se_cycle.check()",
            f"loaded = Package.from_bytes(Path({str(package)!r}).read_bytes())",
            "assert loaded.modules",
            "assert module.compile_se_cycle().modules",
        ])], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_verification_is_off_by_default_and_sampling_uses_default_noise(monkeypatch, tmp_path):
    pytest.importorskip("guppylang")
    import spiderwarp._se_circuit as circuit_module
    import spiderwarp.optimized_steane_se as protocol_module
    import spiderwarp.se_protocol_simulation as simulation
    from spiderwarp.noise_model import default_noise_model

    def unexpected(*args, **kwargs):
        raise AssertionError("Verification ran without being requested")

    with monkeypatch.context() as patch:
        patch.setattr(circuit_module, "validate_gadget", unexpected)
        patch.setattr(protocol_module, "validate_gadget", unexpected)
        patch.setattr(PolicyBuilder, "_terminal", unexpected)
        protocol = optimized_steane_SE_orchestrator(
            bell_code(), {3: bell_gadget()}, output_dir=tmp_path)
    assert not protocol.verification.verified
    assert "VERIFIED_AT_BUILD = False" in protocol.source

    calls = []
    original_apply_noise = simulation.apply_noise

    def record_noise(circuit, model):
        noisy = original_apply_noise(circuit, model)
        calls.append((circuit.copy(), noisy.copy(), vars(model).copy()))
        return noisy

    monkeypatch.setattr(simulation, "apply_noise", record_noise)
    estimate = simulation.estimate_protocol_ler(protocol, shots=16, seed=8)
    expected = vars(default_noise_model())
    assert vars(estimate.noise_model) == expected
    assert all(parameters == expected for _, _, parameters in calls)
    assert any(op.name == "Z_ERROR" for _, noisy, _ in calls for op in noisy)
    assert any(not c.num_measurements for c, _, _ in calls)  # Correction blocks.
    assert not protocol.verification.verified
    assert protocol.verify().verified


def test_optional_verifier_checks_the_existing_corrections(compiled_bell_protocol):
    protocol = replace(compiled_bell_protocol, standard_corrections={
        basis: dict(table) for basis, table in compiled_bell_protocol.standard_corrections.items()
    })
    assert verify_se_protocol(protocol).verified
    # An erroneous clean-path correction must fail rather than be resynthesized.
    protocol.standard_corrections["Z"][0] = 1
    with pytest.raises(ProtocolVerificationError):
        protocol.verify()
    assert not protocol.verification.verified


def control_flow_protocol():
    # Distinct readout bits allow the fake simulator to expose accidental use
    # of a discarded primary syndrome, and to test recovery early returns.
    code = bell_code()
    space = PauliSpace(code)
    circuit = stim.Circuit("R 2 3\nM 2 3")
    ro = Readout(((0,),), ((1,),))
    gadgets = {(basis, distance): SEGadget(distance, basis, circuit, ro, space)
               for basis in ("X", "Z") for distance in (1, 3)}
    recoveries = {
        ("Z", 1, -1): RecoveryPlan("Z", 1, -1, 1, {0: 1}, (), (1,)),
        ("X", 1, 1): RecoveryPlan("X", 1, 1, 1, {0: 4}, (), (1,)),
    }
    return CompiledSEProtocol(code, gadgets, {"Z": {0: 0, 1: 1}, "X": {0: 0, 1: 4}},
                              recoveries, VerificationReport(1, 1), "", Path("unused"), None, None)


class FakeSimulator:
    def __init__(self, results):
        self.results = iter(results)
        self.record = []
        self.applied = []

    def current_measurement_record(self):
        return list(self.record)

    def do_circuit(self, _):
        flags, syndrome = next(self.results)
        self.record.extend([flags, syndrome])

    def x(self, q):
        self.applied.append(("X", q))

    def z(self, q):
        self.applied.append(("Z", q))


def test_first_primary_recovery_finishes_cycle_and_discards_primary_syndrome():
    p = control_flow_protocol()
    sim = FakeSimulator([(1, 1), (0, 0)])
    result = p.run(sim)
    assert result.accepted
    assert [m[:3] for m in result.measurements] == [("primary", "Z", 3), ("recovery", "X", 1)]
    assert sim.applied == [("X", 0)]


def test_clean_first_primary_corrects_then_runs_max_distance_secondary_and_recovers():
    p = control_flow_protocol()
    sim = FakeSimulator([(0, 1), (1, 1), (0, 0)])
    result = p.run(sim)
    assert result.accepted
    assert [m[:3] for m in result.measurements] == [
        ("primary", "Z", 3), ("primary", "X", 3), ("recovery", "Z", 1)]
    assert sim.applied == [("X", 0), ("Z", 0)]


@pytest.mark.parametrize("recovery", [(1, 0), (0, 1)])
def test_recovery_flag_or_unresolved_syndrome_discards(recovery):
    sim = FakeSimulator([(1, 0), recovery])
    result = control_flow_protocol().run(sim)
    assert not result.accepted
    assert sim.applied == []


def test_unknown_primary_flags_and_invalid_primary_syndrome_discard():
    p = control_flow_protocol()
    p.standard_corrections["Z"] = {0: 0}
    assert not p.run(FakeSimulator([(0, 1)])).accepted
    p.recoveries.clear()
    assert not p.run(FakeSimulator([(1, 0)])).accepted


def test_cycle_fault_budget_is_not_reset_at_secondary():
    code = bell_code()
    code.d = 4
    p = PolicyBuilder(PauliSpace(code), {})
    # One incoming error + one circuit fault leaves no fault budget for a
    # following correction. A fresh budget here would invent extra histories.
    assert list(p._correct((1, 1, 0), 1)) == [(1, 1, p.space.reduce(1))]


def test_two_internal_faults_can_leave_two_unflagged_errors():
    code = CSSCode.load_code("misc", "32_20_4")
    circuit = stim.Circuit.from_file(get_project_root() / "assets/circuits/SECircuits/32_20_4_d4.stim")
    space, gadgets = raw_gadgets(code, {4: circuit})
    policy = PolicyBuilder(space, gadgets)
    residual = space.reduce(3)  # X0 X1, with no subsequent syndrome extraction.
    final = gadgets["X", 4]
    faults = final.fault_witness(residual << final.num_detectors, 2)
    assert faults is not None and len(faults) == 2
    bell, _, _ = _make_encoded_bell_input(code.H_x, code.H_z, code.L_x, code.L_z,
                                         final.circuit.num_qubits)
    witness = bell + gadgets["Z", 4].circuit + final.with_faults(faults)
    assert not sample_annotation_parities(witness, shots=8).any()
    assert residual not in space.low_weight(1)
    assert policy._limit((0, 2, residual)) == 2
    policy._terminal("unflagged cycle", (0, 2, residual))
    assert policy.report.accepted_contexts == 1
    # EC A still allows only one residual error if only one fault is internal.
    assert policy._limit((1, 1, residual)) == 1
    with pytest.raises(ProtocolVerificationError, match="EC A"):
        policy._terminal("unflagged cycle", (1, 1, residual))


def test_mixed_pauli_fault_passes_css_filter_and_preserves_raw_syndromes():
    code = CSSCode.load_code("misc", "32_20_4")
    circuit = stim.Circuit.from_file(get_project_root() / "assets/circuits/SECircuits/32_20_4_d4.stim")
    space, gadgets = raw_gadgets(code, {4: circuit})
    policy = PolicyBuilder(space, gadgets)
    final = gadgets["X", 4]
    for effect in final.single_fault_effects:
        flags, syndrome, error = final.unpack(effect)
        correction = policy.standard["X"].get(syndrome)
        if flags or correction is None or not syndrome:
            continue
        residual = error ^ space.reduce(correction)
        if residual not in space.low_weight(1) and space.within_css_bound(residual, 1):
            break
    else:
        pytest.fail("Expected a mixed fault with separate X/Z residual support")
    assert not space.within_pauli_filter(residual, 1)
    policy._terminal("unflagged cycle", (0, 1, residual), syndrome=syndrome)
    bell, _, _ = _make_encoded_bell_input(code.H_x, code.H_z, code.L_x, code.L_z,
                                         final.circuit.num_qubits)
    witness = bell + gadgets["Z", 4].circuit + final.with_faults(final.fault_witness(effect, 1))
    # Raw parity evaluation must retain this nonzero syndrome. A detector
    # sampler with its automatically subtracted reference would hide it.
    parities = sample_annotation_parities(witness)
    expected = [(syndrome >> i) & 1 for i in range(code.H_x.shape[0])]
    assert parities[0, -len(expected):].tolist() == expected


def test_codespace_filter_and_logical_preservation_are_distinct_conditions():
    code = CSSCode.load_code("misc", "32_20_4")
    space = PauliSpace(code)
    logical = space.reduce(sum(1 << int(q) for q in np.flatnonzero(code.L_x[0])))
    assert space.within_pauli_filter(logical, 0)
    assert logical not in space.low_weight(1)
    policy = PolicyBuilder(space, {})
    # At correctable orders the original logical information must survive.
    with pytest.raises(ProtocolVerificationError, match="EC B"):
        policy._terminal("logical check", (0, 1, logical))
    # The higher-order filter bounds distance to some codeword.
    policy._terminal("filter check", (0, 2, logical))
    # This also governs recovery synthesis, rather than only final validation.
    assert next(policy._candidate_corrections({(0, 2, logical)})) == 0
    assert policy._correction_safe({(0, 2, logical)}, 0)


def test_ec_a_covers_arbitrary_input_sectors_with_an_independent_fault_budget():
    code = bell_code()
    space, gadgets = raw_gadgets(code, {3: bell_gadget()})
    policy = PolicyBuilder(space, gadgets)
    # Exhaustively compare sector representatives against all physical Paulis.
    expected = {space.pauli_syndrome(p) for w in range(code.n + 1)
                for p in space.errors_of_weight(w)}
    assert set(policy._arbitrary_inputs) == expected
    assert policy.report.ec_a_input_syndromes == 4
    assert all(space.pauli_syndrome(p) == s for s, p in policy._arbitrary_inputs.items())
    assert policy._remaining((-1, 0, 0)) == policy.correctable
    assert policy._remaining((-1, 1, 0)) == 0
    groups = policy._run(gadgets["Z", 3], {(-1, 0, 0)})
    assert {c[1] for cs in groups.values() for c in cs} == {0, 1}
    assert all(not policy._requires_acceptance(c) for cs in groups.values() for c in cs)


def test_css_filter_decouples_x_and_z_components_in_every_policy_check():
    code = CSSCode.load_code("misc", "32_20_4")
    space = PauliSpace(code)
    split = space.reduce((1 << 7) | (1 << (23 + code.n)))
    assert space.within_css_bound(split, 1)
    assert not space.within_pauli_filter(split, 1)
    assert space.within_pauli_filter(split, 2)
    policy = PolicyBuilder(space, {})
    assert policy._within_bound((0, 1, split))
    assert policy._within_bound((1, 1, split))
    policy._terminal("accepted split CSS error", (1, 1, split))
    assert next(policy._candidate_corrections({(0, 1, split)})) == 0
    assert policy._correction_safe({(0, 1, split)}, 0)
    assert split in {space.reduce(p) for p in space.css_errors_of_weight(1)}
    y = space.reduce((1 << 7) | (1 << (7 + code.n)))
    assert space.within_pauli_filter(y, 1)


def test_shared_decoders_preserve_every_reachable_correction_and_rejection():
    from spiderwarp._se_guppy import share_recovery_decoders
    plans = {
        ("Z", 1, -1): RecoveryPlan("Z", 1, -1, 1, {0: 0, 1: 5}, (2,), (1, 2)),
        ("X", 1, 0): RecoveryPlan("X", 1, 0, 1, {0: 0, 3: 7}, (), (2,)),
        ("X", 2, 0): RecoveryPlan("X", 2, 0, 1, {2: 9}, (), (2,)),
    }
    tables, routing = share_recovery_decoders(plans)
    assert len(tables) == 2
    assert routing["Z", 1, -1] == routing["X", 1, 0]
    assert routing["Z", 1, -1] != routing["X", 2, 0]
    for key, plan in plans.items():
        decoder = tables[routing[key]]
        assert all(decoder[s] == p for s, p in plan.corrections.items())
        assert all(decoder[s] is None for s in plan.discarded_syndromes)


def test_shared_integer_decisions_preserve_sparse_and_product_decoders():
    from spiderwarp._se_guppy import IntegerDecisions
    decisions = IntegerDecisions()
    sparse = {0: 7, 3: 19, 14: 19, 4095: 5}
    product = {s: (s & 63) for s in range(4096)}
    a = decisions.build(sparse, 12)
    b = decisions.build(product, 12)
    assert all(decisions.evaluate(a, s) == sparse.get(s, 0) for s in range(4096))
    assert all(decisions.evaluate(b, s) == product[s] for s in range(4096))
    before = len(decisions.nodes)
    assert decisions.build(sparse, 12) == a
    assert len(decisions.nodes) == before
    rows, roots = decisions.compact([a, b])
    assert len(rows) < len(decisions.nodes)
    for root, table in zip(roots, (sparse, product)):
        for key in range(4096):
            node = root
            while node >= 0:
                mask, pattern, matched, other = rows[node]
                node = matched if key & mask == pattern else other
            assert -(node + 1) == table.get(key, 0)


def generated_python(source):
    """Execute generated bodies with Python values to exhaustively check tables.

    Compilation/reload is checked separately. Removing only Guppy annotations
    lets these tests cover the actual exported lookup/control-flow bodies.
    """
    tree = ast.parse(source)
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            definitions.append(node)
        elif isinstance(node, ast.ClassDef):
            node.decorator_list = [ast.Name(id='dataclass', ctx=ast.Load())]
            definitions.append(node)
        elif isinstance(node, ast.Assign):
            definitions.append(node)
    namespace = {'comptime': lambda value: value, 'dataclass': dataclass}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 '<generated SE>', 'exec'), namespace)
    return namespace


def test_export_expands_the_exact_gate_schedule_and_measurement_parities(compiled_bell_protocol):
    protocol = compiled_bell_protocol
    exported = generated_python(protocol.source)
    assert 'decision_0(' not in protocol.source
    for (basis, distance), gadget in protocol.gadgets.items():
        # Include all-zero, single-bit, and all-one readouts to exercise every
        # parity column, even outcomes that this noiseless circuit cannot reach.
        count = gadget.circuit.num_measurements
        for mask in [0, (1 << count) - 1] + [1 << i for i in range(count)]:
            operations = iter(gadget.operations)
            measured = []

            def gate(name, *targets):
                actual_name, actual_targets = next(operations)
                assert (name, targets) == (actual_name, actual_targets)

            def allocate():
                name, targets = next(operations)
                assert name == 'R'
                return targets[0]

            def measure(q):
                gate('M', q)
                measured.append(bool((mask >> len(measured)) & 1))
                return measured[-1]

            exported.update(qubit=allocate, measure_bit=measure)
            for name in ('h', 'x', 'y', 'z', 'cx', 'cz'):
                exported[name] = lambda *qs, name=name: gate(name.upper(), *qs)
            result = exported[f'run_{basis.lower()}_d{distance}'](list(range(protocol.code.n)))
            assert next(operations, None) is None
            assert (result.flags, result.syndrome) == gadget.readout.decode(measured)


def test_exported_tables_preserve_all_real_decoder_cases_and_recovery_routes(verified_raw_32_policy):
    from spiderwarp._se_guppy import generate_source, share_recovery_decoders
    code, _, gadgets, policy = verified_raw_32_policy
    source = generate_source(code, gadgets, policy)
    exported = generated_python(source)
    # Keep source growth tied to compact circuit/data descriptions, rather than
    # generating a function for every branch of every decoder.
    assert len(source.splitlines()) < 6000
    tables, decoder_ids = share_recovery_decoders(policy.recoveries)
    for decoder_id, table in enumerate([policy.standard['Z'], policy.standard['X']] + tables):
        width = max(1, max((s for s, p in table.items() if p is not None), default=0).bit_length())
        for syndrome in [-1, *range(1 << width), 1 << width]:
            result = exported['decode'](decoder_id, syndrome)
            pauli = table.get(syndrome)
            assert result.accepted == (pauli is not None)
            assert result.x_mask | (result.z_mask << code.n) == (pauli or 0)
    for key, plan in policy.recoveries.items():
        basis, flags, previous = key
        route = (exported['select_z_recovery'](flags) if basis == 'Z'
                 else exported['select_x_recovery'](flags, previous))
        assert (route.distance, route.cleanup_distance, route.decoder_id) == (
            plan.distance, plan.cleanup_distance or 0, decoder_ids[key])
    assert exported['select_z_recovery'](0).decoder_id == -1
    assert exported['select_z_recovery'](1 << gadgets['Z', code.d].num_flags).decoder_id == -1
    assert exported['select_x_recovery'](1, -1).decoder_id == -1
    assert exported['select_x_recovery'](1, 1 << exported['PREVIOUS_WIDTH']).decoder_id == -1
    for previous in range(1 << exported['PREVIOUS_WIDTH']):
        for flags in range(1 << gadgets['X', code.d].num_flags):
            route = exported['select_x_recovery'](flags, previous)
            assert (route.decoder_id >= 0) == (('X', flags, previous) in policy.recoveries)
    for flags in range(1 << gadgets['Z', code.d].num_flags):
        route = exported['select_z_recovery'](flags)
        assert (route.decoder_id >= 0) == (('Z', flags, -1) in policy.recoveries)


@pytest.fixture(scope="module")
def verified_raw_32_policy():
    code = CSSCode.load_code("misc", "32_20_4")
    assets = get_project_root() / "assets/circuits/SECircuits"
    space, gadgets = raw_gadgets(code, {
        d: stim.Circuit.from_file(assets / f"32_20_4_d{d}.stim") for d in (1, 2, 4)
    })
    assert [gadgets["X", d].num_flags for d in (1, 2, 4)] == [0, 6, 11]
    return code, space, gadgets, PolicyBuilder(space, gadgets).build(verify=True)


def test_real_32_code_verifies_css_ec_a_b_with_necessary_recovery_cleanup(verified_raw_32_policy):
    code, space, gadgets, policy = verified_raw_32_policy
    assert policy.report.verified
    assert policy.report.correctable_faults == 1
    assert policy.report.detectable_faults == 2
    assert policy.report.ec_a_input_syndromes == 4096
    assert any(plan.cleanup_distance is not None for plan in policy.recoveries.values())
    assert {plan.primary_basis for plan in policy.recoveries.values()} == {"X", "Z"}


def test_recovery_cleanup_uses_both_new_syndromes_and_returns():
    p = control_flow_protocol()
    old = p.recoveries["Z", 1, -1]
    p.recoveries["Z", 1, -1] = replace(old, cleanup_distance=1, corrections={2: 1})
    # Primary syndrome is discarded. New dual and cleanup syndromes pack as 0 | 1<<1.
    sim = FakeSimulator([(1, 1), (0, 0), (0, 1)])
    result = p.run(sim)
    assert result.accepted
    assert [m[:3] for m in result.measurements] == [
        ("primary", "Z", 3), ("recovery", "X", 1), ("recovery_cleanup", "Z", 1)]
    assert sim.applied == [("X", 0)]
    rejected = p.run(FakeSimulator([(1, 0), (0, 0), (1, 0)]))
    assert not rejected.accepted
    assert rejected.discard_reason == "raised recovery cleanup flags"


def test_css_weight_one_incoming_x_and_z_on_distinct_qubits_are_both_corrected(verified_raw_32_policy):
    code, space, gadgets, policy = verified_raw_32_policy
    protocol = CompiledSEProtocol(code, gadgets, policy.standard, policy.recoveries,
                                  policy.report, "", Path("unused"), None, None)
    bell, refs, pairing = _make_encoded_bell_input(
        code.H_x, code.H_z, code.L_x, code.L_z, protocol.required_qubits)
    sim = stim.TableauSimulator()
    sim.do_circuit(bell)
    sim.x(1)
    sim.z(17)
    result = protocol.run(sim)
    assert result.accepted
    for basis, checks, logicals in [("X", code.H_x, code.L_x), ("Z", code.H_z, code.L_z)]:
        for i, row in enumerate(list(checks) + list(logicals)):
            observable = stim.PauliString(bell.num_qubits)
            for q, bit in enumerate(row):
                if bit:
                    observable[q] = basis
            if i >= len(checks):
                logical_index = i - len(checks)
                for j, q in enumerate(refs):
                    if (j == logical_index if basis == "X" else pairing[j, logical_index]):
                        observable[q] = basis
            assert sim.peek_observable_expectation(observable) == 1


def test_invalid_distance_inputs_are_rejected_before_optimization():
    pytest.importorskip("guppylang")
    code = bell_code()
    with pytest.raises(ValueError, match="Missing primary"):
        optimized_steane_SE_orchestrator(code, {1: bell_gadget()})
    with pytest.raises(ValueError, match="unique"):
        optimized_steane_SE_orchestrator(code, [(3, bell_gadget()), (3, bell_gadget())])

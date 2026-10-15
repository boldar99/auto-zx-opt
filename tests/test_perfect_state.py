import numpy as np
import pytest
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.stim_utils import perfect_state_from_code


def _asymmetric_code() -> CSSCode:
    return CSSCode(
        n=4, k=1, d=1,
        H_x=np.array([[1, 1, 0, 0]], dtype=np.uint8),
        H_z=np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8),
        L_x=np.array([[0, 0, 1, 1]], dtype=np.uint8),
        L_z=np.array([[0, 0, 1, 0]], dtype=np.uint8),
    )


def _repetition_code() -> CSSCode:
    return CSSCode(
        n=3, k=1, d=1,
        H_x=np.zeros((0, 3), dtype=np.uint8),
        H_z=np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        L_x=np.array([[1, 1, 1]], dtype=np.uint8),
        L_z=np.array([[1, 0, 0]], dtype=np.uint8),
    )


def _redundant_code() -> CSSCode:
    code = _asymmetric_code()
    code.H_x = np.vstack([code.H_x, code.H_x[0], np.zeros(4, dtype=np.uint8)])
    code.H_z = np.vstack([code.H_z, code.H_z[0] ^ code.H_z[1]])
    return code


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("code_factory", [
    lambda: CSSCode.load_code("misc", "32_20_4"),
    _asymmetric_code,
    _repetition_code,
    _redundant_code,
])
def test_perfect_state_has_positive_stabilizers_and_requested_logicals(
    basis, code_factory,
) -> None:
    code = code_factory()
    circuit = perfect_state_from_code(code, basis)
    simulator = stim.TableauSimulator(seed=17)
    simulator.do(circuit)

    logicals = code.L_x if basis == "X" else code.L_z
    for pauli, rows in [("X", code.H_x), ("Z", code.H_z), (basis, logicals)]:
        for row in rows:
            observable = stim.PauliString("".join(pauli if bit else "_" for bit in row))
            assert simulator.peek_observable_expectation(observable) == 1

    opposite_basis = "Z" if basis == "X" else "X"
    opposite_logicals = code.L_z if basis == "X" else code.L_x
    for row in opposite_logicals:
        observable = stim.PauliString("".join(opposite_basis if bit else "_" for bit in row))
        assert simulator.peek_observable_expectation(observable) == 0

    check_rows = code.H_z if basis == "X" else code.H_x
    assert circuit.num_measurements == np.count_nonzero(np.any(check_rows, axis=1))
    assert circuit.num_detectors == 0

    # The ideal prefix is resets followed immediately by projections, not a
    # synthesized encoder. Every correction is controlled by a record bit.
    assert circuit[0].name == ("RX" if basis == "X" else "R")
    if circuit.num_measurements:
        assert circuit[1].name == "MPP"
    for operation in circuit[1:]:
        if operation.name != "MPP":
            assert operation.name == ("CX" if basis == "X" else "CZ")
            assert all(target.is_measurement_record_target
                       for target in operation.targets_copy()[::2])

    # Test the corrected state across random projection branches. The raw
    # preparation results need not be zero, but all subsequent checks must be.
    readout = circuit.copy()
    for pauli, rows in [("X", code.H_x), ("Z", code.H_z), (basis, logicals)]:
        for row in rows:
            support = np.flatnonzero(row)
            if not len(support):
                continue
            targets = []
            for qubit in support:
                if targets:
                    targets.append(stim.target_combiner())
                targets.append(stim.target_x(int(qubit)) if pauli == "X"
                               else stim.target_z(int(qubit)))
            readout.append("MPP", targets)
    samples = readout.compile_sampler(seed=19).sample(64)
    assert not samples[:, circuit.num_measurements:].any()


def test_perfect_state_rejects_unknown_basis() -> None:
    with pytest.raises(ValueError, match="basis"):
        perfect_state_from_code(_asymmetric_code(), "Y")

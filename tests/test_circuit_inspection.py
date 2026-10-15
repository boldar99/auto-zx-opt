import numpy as np
import pytest
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.stim_utils import circuit_for_inspection, perfect_state_from_code


def test_inspection_removes_requested_operations_without_mutating_input() -> None:
    circuit = stim.Circuit(
        "R 0 1\nTICK\nMPP X0*X1\nM(0.1) !0\n"
        "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-2]\nX 1"
    )
    original = circuit.copy()

    view = circuit_for_inspection(circuit)

    assert view == stim.Circuit("R 0 1\nTICK\nM(0.1) !0\nX 1")
    assert view.num_detectors == view.num_observables == 0
    assert circuit == original
    view.diagram("timeline-svg")


def test_inspection_remaps_retained_feedback_and_removes_mpp_feedback() -> None:
    circuit = stim.Circuit(
        "R 0 1 2 3\nM 0\nMPP Z1\n"
        "CX rec[-1] 2 rec[-2] 3 0 1\nM 2\n"
        "CZ rec[-2] 3\nCX rec[-3] 2\nM 3"
    )

    view = circuit_for_inspection(circuit)

    assert view == stim.Circuit(
        "R 0 1 2 3\nM 0\nCX rec[-1] 3 0 1\n"
        "M 2\nCX rec[-2] 2\nM 3"
    )
    view.compile_sampler().sample(1)
    view.diagram("timeline-svg")


def test_inspection_remaps_references_across_repeat_iterations() -> None:
    circuit = stim.Circuit(
        "M 0\nREPEAT 2 {\nMPP Z1\nCX rec[-2] 2\n"
        "M 3\nDETECTOR rec[-1]\n}"
    )

    view = circuit_for_inspection(circuit)

    assert view == stim.Circuit("M 0\nCX rec[-1] 2\nM 3\nCX rec[-1] 2\nM 3")
    view.compile_sampler().sample(1)


def test_inspection_counts_each_product_in_a_batched_mpp() -> None:
    circuit = stim.Circuit("M 0\nMPP X1*X2 Z3\nM 4\nCX rec[-4] 5")

    assert circuit_for_inspection(circuit) == stim.Circuit("M 0 4\nCX rec[-2] 5")


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_inspection_removes_projection_feedback_from_perfect_state(basis) -> None:
    code = CSSCode(
        n=2, k=0, d=1,
        H_x=np.array([[1, 1]], dtype=np.uint8),
        H_z=np.array([[1, 1]], dtype=np.uint8),
        L_x=np.zeros((0, 2), dtype=np.uint8),
        L_z=np.zeros((0, 2), dtype=np.uint8),
    )
    circuit = perfect_state_from_code(code, basis)

    view = circuit_for_inspection(circuit)

    assert view == stim.Circuit("RX 0 1" if basis == "X" else "R 0 1")
    view.diagram("timeline-svg")

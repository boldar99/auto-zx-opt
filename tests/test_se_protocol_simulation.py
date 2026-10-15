import math

import numpy as np
import pytest

from spiderwarp.csscode import CSSCode
from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.se_protocol_simulation import _StateExperiment, LogicalErrorRateEstimate


@pytest.mark.parametrize("prepared_state", ["0", "+"])
def test_state_ler_removes_correctable_residuals_and_detects_logical_errors(prepared_state):
    code = CSSCode.load_code("MQT", "7_1_3")
    experiment = _StateExperiment(code, code.n + 2, prepared_state)
    assert experiment.initial.num_qubits == code.n + 2
    # Include split-support X/Z errors, whose CSS weight is one.
    for x in range(-1, code.n):
        for z in range(-1, code.n):
            simulator = experiment.initial.copy(seed=1)
            if x >= 0:
                simulator.x(x)
            if z >= 0:
                simulator.z(z)
            assert not experiment.logical_failure(simulator)
    for basis, rows in [("x", code.L_x), ("z", code.L_z)]:
        for row in rows:
            simulator = experiment.initial.copy(seed=1)
            for qubit in np.flatnonzero(row):
                getattr(simulator, basis)(int(qubit))
            detectable_basis = "x" if prepared_state == "0" else "z"
            assert experiment.logical_failure(simulator) == (basis == detectable_basis)


def test_ler_denominator_and_zero_failure_uncertainty():
    estimate = LogicalErrorRateEstimate(100, 80, 4, CustomErrorModel(), 1)
    assert estimate.discarded == 20
    assert estimate.acceptance_rate == 0.8
    assert estimate.ler == 4 / 80
    low, high = estimate.ler_interval_95
    assert low < estimate.ler < high
    zero = LogicalErrorRateEstimate(100, 100, 0, CustomErrorModel(), 1)
    assert zero.ler_interval_95[0] == pytest.approx(0)
    assert zero.ler_interval_95[1] == pytest.approx(0.0369935, abs=1e-7)
    rejected = LogicalErrorRateEstimate(100, 0, 0, CustomErrorModel(), 1)
    assert math.isnan(rejected.ler)
    assert rejected.ler_interval_95 == (0, 1)

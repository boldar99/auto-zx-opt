"""Optimized Steane extraction and correction for the [[32, 20, 4]] code.

The Python circuit description contains operations and two named detector
groups: flags and syndrome. Detectors use end-of-circuit Stim rec offsets.
Compile-time tracing emits the circuit and its parity calculations; runtime
Guppy code dispatches extractions and applies the terminal physical correction.
Classical acceptance and decoding policies live in sol_experiment.optimized_steane_dispatchers.
"""
from typing import Literal

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import h, qubit, x, z

from sol_experiment.circuit_descriptions import (
    DetectorGroups, optimized_steane_32_20_4_X_d2, optimized_steane_32_20_4_X_d4,
)
from sol_experiment.circuit_utils import emit_circuit
from sol_experiment.optimized_steane_dispatchers import (
    HISTORY_CAPACITY, X_BASIS, Z_BASIS, DispatcherAction,
)
from sol_experiment.measurement_utils import pack_detector_parities


def make_optimized_steane_circuit(
    basis: Literal["X", "Z"],
    distance: Literal[4, 2],
) -> GuppyFunctionDefinition:
    """Return raw extract(data) -> (flags, syndrome), without decoding.

    Z checks correct X errors; X checks correct Z errors. The Z variant wraps
    the X extraction in transversal H on data and restores the data basis.
    The d4/d2 circuits need at most 60/51 live qubits including the data.
    """
    if distance not in (4, 2):
        raise ValueError("Circuit distance must be 4 or 2")
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")

    qec_method = {
        2: optimized_steane_32_20_4_X_d2,
        4: optimized_steane_32_20_4_X_d4,
    }[distance]
    operations, detectors = qec_method()

    @guppy.comptime
    def extract(data: array[qubit, 32]) -> tuple[int, int]:
        if basis == "Z":
            for q in data:
                h(q)
        measurements = emit_circuit(data, operations)
        if basis == "Z":
            for q in data:
                h(q)
        readout = [measurement.read() for measurement in measurements]
        flags = pack_detector_parities(readout, detectors.flags)
        syndrome = pack_detector_parities(readout, detectors.syndrome)
        return flags, syndrome

    return extract


def make_optimized_steane_qec_cycle(
    dispatcher_function: GuppyFunctionDefinition,
) -> GuppyFunctionDefinition:
    """Bind a dispatcher and return cycle(data) -> bool.

    Call dispatcher_function(history, length) first with an empty history,
    then after appending each (basis, distance, flags, syndrome) observation.
    Basis values and decision actions are named in sol_experiment.optimized_steane_dispatchers.
    The dispatcher must preserve the history and return DispatchDecision.

    Z4/X4/Z2/X2 request another extraction; ACCEPT applies X/Z correction masks
    once and returns True; REJECT returns False. Unknown actions and requests
    beyond the three-entry history reject without applying a correction. Data are
    borrowed and retained on every path. Each invocation starts a fresh history.
    """
    qec_z4 = make_optimized_steane_circuit("Z", 4)
    qec_x4 = make_optimized_steane_circuit("X", 4)
    qec_z2 = make_optimized_steane_circuit("Z", 2)
    qec_x2 = make_optimized_steane_circuit("X", 2)

    @guppy
    def cycle(data: array[qubit, 32]) -> bool:
        history = array((0, 0, 0, 0) for _ in range(comptime(HISTORY_CAPACITY)))
        for history_length in range(comptime(HISTORY_CAPACITY + 1)):
            decision = dispatcher_function(history, history_length)
            if decision.action == comptime(DispatcherAction.REJECT.value):
                return False
            if decision.action == comptime(DispatcherAction.ACCEPT.value):
                for i in range(32):
                    if (decision.correction_x & (1 << i)) != 0:
                        x(data[i])
                    if (decision.correction_z & (1 << i)) != 0:
                        z(data[i])
                return True
            if history_length == comptime(HISTORY_CAPACITY):
                return False

            if decision.action == comptime(DispatcherAction.Z4.value):
                flags, syndrome = qec_z4(data)
                basis, distance = comptime(Z_BASIS), 4
            elif decision.action == comptime(DispatcherAction.X4.value):
                flags, syndrome = qec_x4(data)
                basis, distance = comptime(X_BASIS), 4
            elif decision.action == comptime(DispatcherAction.Z2.value):
                flags, syndrome = qec_z2(data)
                basis, distance = comptime(Z_BASIS), 2
            elif decision.action == comptime(DispatcherAction.X2.value):
                flags, syndrome = qec_x2(data)
                basis, distance = comptime(X_BASIS), 2
            else:
                return False

            history[history_length] = (basis, distance, flags, syndrome)
        return False

    return cycle

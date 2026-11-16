"""Optimized Steane extraction and correction for the [[32, 20, 4]] code.

The Python circuit description contains operations and two named detector
groups: flags and syndrome. Detectors use end-of-circuit Stim rec offsets.
Compile-time tracing emits the circuit and its parity calculations; runtime
Guppy code handles acceptance, syndrome decoding, and physical correction.
"""
from typing import Literal, NamedTuple

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import h, qubit, x, z

from sol_experiment.circuit_utils import emit_circuit
from sol_experiment.logical_measurement import decode_syndrome
from sol_experiment.measurement_utils import pack_detector_parities


class DetectorGroups(NamedTuple):
    """Classical detector metadata; list order determines each output bit."""

    flags: list[tuple[int, ...]]
    syndrome: list[tuple[int, ...]]


@guppy.struct
class OptimizedSteaneQECResult:
    """One extraction's outcome, with any accepted correction already applied.

    flags packs the flag detectors; syndrome packs the syndrome detectors in
    check row order. A flag or ambiguous syndrome rejects with correction zero.
    The borrowed data remains available on every path, including rejection.
    """

    accepted: bool
    flags: int
    syndrome: int
    correction: int


def optimized_steane_32_20_4_X_d4() -> tuple[list[tuple[str, list[int]]], DetectorGroups]:
    """Describe X-check extraction with 11 flags and six syndrome detectors."""
    ops = [
        ("RX", [32, 33, 34]),
        ("R", [35, 36]),
        ("RX", [37]),
        ("R", [38, 39, 40]),
        ("RX", [41]),
        ("R", [42, 43, 44, 45]),
        ("RX", [46]),
        ("R", [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]),
        ("RX", [59]),
        ("CX", [34, 35, 37, 36, 41, 9, 46, 45, 59, 58, 32, 34, 33, 35, 37, 39, 36, 38, 41, 53, 45, 47, 46, 48, 33, 37, 35, 39, 34, 38, 32, 36, 47, 51, 45, 49, 46, 50, 48, 52, 33, 41, 37, 43, 39, 44, 47, 54, 51, 56, 49, 55, 46, 53, 52, 57, 41, 43, 33, 46, 59, 53, 39, 52, 41, 40, 59, 39, 41, 44, 40, 42, 39, 7, 44, 57, 35, 41]),
        ("MX", [39]),
        ("R", [39]),
        ("CX", [38, 39, 38, 51, 42, 39, 39, 56, 34, 42, 59, 39, 34, 47, 42, 54, 59, 51, 39, 14, 47, 18, 42, 10, 51, 22]),
        ("MX", [39]),
        ("R", [39]),
        ("CX", [36, 39, 36, 49, 40, 39, 49, 20, 39, 55, 32, 40, 39, 12, 55, 47, 57, 47, 58, 55, 59, 57, 46, 47, 58, 33, 59, 34]),
        ("M", [47]),
        ("CX", [46, 17, 33, 1]),
        ("R", [47]),
        ("MX", [33]),
        ("CX", [50, 47]),
        ("R", [33]),
        ("CX", [37, 50, 43, 47, 45, 33, 47, 29, 40, 33, 32, 45, 58, 40, 32, 0, 58, 45, 40, 8]),
        ("MX", [32]),
        ("CX", [58, 54]),
        ("RX", [32]),
        ("CX", [58, 43, 54, 26, 58, 36, 43, 13, 58, 50]),
        ("MX", [43]),
        ("R", [43]),
        ("CX", [48, 43, 41, 43, 35, 48, 59, 41, 35, 3, 59, 48, 41, 11, 59, 58]),
        ("MX", [41, 59]),
        ("M", [58]),
        ("RX", [41]),
        ("R", [58]),
        ("CX", [41, 58]),
        ("H", [58]),
        ("CX", [33, 41, 36, 58, 43, 41, 37, 58, 36, 4, 47, 41, 51, 58]),
        ("MX", [47]),
        ("CX", [56, 41]),
        ("MX", [51]),
        ("CX", [52, 58]),
        ("H", [58]),
        ("CX", [41, 58]),
        ("MX", [41]),
        ("M", [58]),
        ("RX", [41]),
        ("R", [47]),
        ("CX", [32, 47]),
        ("H", [47]),
        ("CX", [33, 32, 33, 24, 57, 32]),
        ("MX", [33]),
        ("CX", [57, 31, 52, 32]),
        ("R", [33]),
        ("MX", [57]),
        ("CX", [52, 23, 41, 33]),
        ("MX", [52]),
        ("H", [33]),
        ("CX", [37, 41, 34, 33, 40, 41, 37, 5, 35, 33, 34, 2]),
        ("MX", [40]),
        ("CX", [42, 41]),
        ("MX", [37, 35, 34, 42]),
        ("CX", [39, 41]),
        ("RX", [34]),
        ("R", [35]),
        ("MX", [39]),
        ("CX", [34, 35]),
        ("H", [34]),
        ("CX", [45, 35, 38, 34, 53, 35, 45, 32, 38, 6, 44, 34, 54, 35, 53, 25, 45, 33]),
        ("MX", [38]),
        ("CX", [44, 47, 56, 34]),
        ("MX", [54]),
        ("CX", [48, 35, 45, 41, 49, 33, 44, 15, 43, 47, 46, 34, 56, 30, 48, 19, 45, 16]),
        ("MX", [49, 44]),
        ("CX", [55, 47, 43, 33]),
        ("MX", [46]),
        ("CX", [50, 34]),
        ("MX", [45]),
        ("CX", [55, 28, 43, 41, 56, 33, 48, 34, 50, 47]),
        ("MX", [55]),
        ("CX", [43, 27]),
        ("H", [33]),
        ("MX", [56]),
        ("H", [34]),
        ("MX", [48]),
        ("CX", [36, 47, 50, 21]),
        ("MX", [43]),
        ("CX", [41, 33, 34, 35]),
        ("MX", [36]),
        ("CX", [53, 47]),
        ("MX", [50, 41]),
        ("M", [33, 35]),
        ("MX", [34]),
        ("H", [47]),
        ("MX", [53]),
        ("CX", [32, 47]),
        ("MX", [32]),
        ("M", [47]),
    ]
    flag_detectors = [
        (-39,),
        (-33,),
        (-34,),
        (-29,),
        (-30,),
        (-4,),
        (-5,),
        (-1,),
        (-2,),
        (-6,),
        (-7,),
    ]
    syndrome_detectors = [
        (-37, -38, -22, -23, -9, -24, -19, -41, -25, -21, -35, -20, -36, -40, -16, -14, -15,
         -11, -17, -8, -31, -26, -28, -3, -18, -10, -13, -32, -12, -27),
        (-37, -38, -22, -23, -9, -24, -19, -41, -25, -21, -35, -20, -36, -40, -16),
        (-37, -38, -22, -23, -9, -24, -19, -41, -14, -15, -11, -17, -8, -31, -26),
        (-37, -38, -22, -23, -25, -21, -35, -14, -15, -11, -28, -3, -18, -10),
        (-37, -38, -9, -24, -25, -20, -36, -14, -15, -17, -8, -28, -3, -13, -32),
        (-37, -22, -9, -19, -25, -21, -20, -40, -14, -17, -31, -28, -18, -13, -12),
    ]

    return ops, DetectorGroups(flags=flag_detectors, syndrome=syndrome_detectors)


def optimized_steane_32_20_4_X_d2() -> tuple[list[tuple[str, list[int]]], DetectorGroups]:
    """Describe X-check extraction with 11 flags and six syndrome detectors."""
    ops = [
        ("RX", [32, 33]),
        ("R", [34]),
        ("RX", [35]),
        ("R", [36, 37]),
        ("RX", [38]),
        ("R", [39, 40, 41]),
        ("RX", [42]),
        ("R", [43, 44, 45, 46, 47, 48, 49]),
        ("RX", [50]),
        ("CX", [33, 1, 35, 34, 38, 37, 42, 41, 50, 49, 34, 36, 37, 39, 41, 43, 42, 44, 32, 34, 37, 40, 41, 45, 42, 46, 34, 40, 41, 47, 42, 48, 40, 12, 45, 40, 34, 45, 49, 40, 45, 20, 49, 33, 40, 28]),
        ("MX", [45]),
        ("CX", [33, 42]),
        ("MX", [40]),
        ("R", [40]),
        ("CX", [42, 17]),
        ("R", [45]),
        ("CX", [35, 40]),
        ("MX", [42]),
        ("R", [42]),
        ("CX", [38, 42, 38, 45, 33, 38, 38, 48, 33, 35, 50, 48, 38, 9, 35, 45, 48, 25]),
        ("MX", [38]),
        ("RX", [38]),
        ("CX", [38, 33, 32, 38, 33, 40, 32, 37, 38, 36, 37, 47, 32, 41, 49, 37, 32, 0, 49, 41, 37, 8]),
        ("MX", [32, 37]),
        ("R", [32, 37]),
        ("CX", [39, 32, 43, 37, 38, 39, 36, 32, 39, 10, 43, 39, 49, 39, 38, 43, 39, 26, 43, 18]),
        ("MX", [43]),
        ("R", [43]),
        ("CX", [37, 43, 32, 43, 36, 37, 43, 30, 36, 6]),
        ("MX", [36]),
        ("R", [36]),
        ("CX", [46, 36, 45, 36, 35, 46, 36, 29, 49, 45, 35, 5, 49, 34, 45, 13, 49, 46]),
        ("MX", [45]),
        ("CX", [46, 21]),
        ("R", [45]),
        ("MX", [46]),
        ("RX", [46]),
        ("CX", [46, 45]),
        ("H", [45]),
        ("CX", [47, 46, 34, 45, 47, 24, 35, 45, 34, 4]),
        ("MX", [35, 34]),
        ("RX", [34]),
        ("CX", [34, 47]),
        ("H", [34]),
        ("CX", [41, 34, 48, 34, 41, 16]),
        ("MX", [48]),
        ("CX", [39, 34]),
        ("R", [35]),
        ("MX", [39]),
        ("CX", [42, 35]),
        ("R", [39]),
        ("CX", [40, 35, 33, 42, 44, 39, 35, 15, 39, 35, 40, 39, 50, 40, 50, 32, 40, 7, 50, 37, 32, 14]),
        ("MX", [40]),
        ("CX", [50, 35, 37, 22]),
        ("MX", [32]),
        ("CX", [50, 38, 35, 47, 37, 45, 38, 2, 35, 31]),
        ("MX", [37]),
        ("CX", [39, 45]),
        ("MX", [38, 35]),
        ("R", [32]),
        ("H", [45]),
        ("CX", [39, 47, 44, 32, 41, 47, 39, 23, 42, 32, 33, 44]),
        ("MX", [41, 39]),
        ("CX", [32, 27, 50, 42, 33, 3, 32, 46, 50, 44, 42, 11]),
        ("MX", [33, 32]),
        ("CX", [36, 46, 44, 19, 50, 49]),
        ("MX", [42, 36]),
        ("CX", [43, 46, 44, 34]),
        ("MX", [50]),
        ("M", [49]),
        ("MX", [43]),
        ("CX", [46, 45]),
        ("H", [34]),
        ("MX", [44, 46]),
        ("M", [45]),
        ("CX", [34, 47]),
        ("MX", [34]),
        ("M", [47]),
    ]
    flag_detectors = [
        (-7, ),
        (-8, ),
        (-3, ),
        (-4, ),
        (-2, ),
        (-1, ),
    ]
    syndrome_detectors = [
        (-29, -16, -12, -22, -23, -26, -19, -28, -30, -10, -25, -18, -14, -31, -27, -5, -33, -24, -17, -13, -21, -20, -11, -32, -9, -6, -15),
        (-29, -16, -12, -22, -23, -26, -19, -28, -30, -10, -25, -18),
        (-29, -16, -12, -22, -23, -26, -19, -14, -31, -27, -5, -33, -24, -17, -13),
        (-29, -16, -12, -28, -30, -10, -14, -31, -27, -5, -21, -20, -11),
        (-29, -22, -23, -28, -30, -25, -14, -31, -33, -24, -21, -32, -9),
        (-29, -16, -22, -26, -28, -18, -14, -27, -33, -17, -20, -32, -6),
    ]

    return ops, DetectorGroups(flags=flag_detectors, syndrome=syndrome_detectors)


def make_optimized_steane_d4(
    basis: Literal["X", "Z"],
) -> GuppyFunctionDefinition:
    """Return qec(data: array[qubit, 32]) -> OptimizedSteaneQECResult.

    The basis names the measured checks: X checks correct Z errors, and Z
    checks correct X errors. The Z variant conjugates the entire X extraction
    with transversal H on data, restoring its basis before any rejection.
    Each extraction uses at most 60 live qubits, including the data block.

    Nonzero flags are postselected; otherwise use the shared syndrome decoder.
    Rejection happens after data interaction, so the caller must reject the
    shot. This function does not retry or reset the data.
    """
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")

    operations, detectors = optimized_steane_32_20_4_X_d4()

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

    @guppy
    def qec(data: array[qubit, 32]) -> OptimizedSteaneQECResult:
        flags, syndrome = extract(data)
        if flags != 0:
            return OptimizedSteaneQECResult(False, flags, syndrome, 0)

        decoded = decode_syndrome(syndrome)
        if not decoded.accepted:
            return OptimizedSteaneQECResult(False, flags, syndrome, 0)

        for i in range(32):
            if (decoded.correction & (1 << i)) != 0:
                if comptime(basis == "Z"):
                    x(data[i])
                else:
                    z(data[i])
        return OptimizedSteaneQECResult(True, flags, syndrome, decoded.correction)

    return qec


def make_optimized_steane_qec_cycle() -> GuppyFunctionDefinition:
    """Return cycle(data) -> bool, running Z then X checks; stop on rejection."""
    qec_z = make_optimized_steane_d4("Z")
    qec_x = make_optimized_steane_d4("X")

    @guppy
    def cycle(data: array[qubit, 32]) -> bool:
        first = qec_z(data)
        if not first.accepted:
            return False
        second = qec_x(data)
        return second.accepted

    return cycle

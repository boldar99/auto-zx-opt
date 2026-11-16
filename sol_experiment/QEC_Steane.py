"""Steane syndrome extraction and physical correction for [[32, 20, 4]].

Factories bind the preparation and basis in Python. Their returned Guppy
functions borrow array[qubit, 32], preserving the caller's data block and its
logical information. The preparation must produce encoded |0> and return an
integer verification mask, with zero meaning success. Bounded RUS wrappers
from state_preparation can be supplied directly.

The basis names the stabilizers measured: Z checks correct X errors, and X
checks correct Z errors. Each extraction uses one verified 32-qubit ancilla.
Together with data and preparation flags this needs at most 66 live qubits
for the supplied d2/d3/d4 preparations, including their RUS versions.

Each basis is extracted once, using the shared single-bit correction and
two-bit ambiguity rejection policy. These are syndrome-domain guarantees;
they do not certify correction of arbitrary faults throughout a noisy cycle.
"""

from typing import Literal

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import cx, discard_array, h, measure_array, qubit, x, z

from sol_experiment.logical_measurement import decode_readout_syndrome


@guppy.struct
class SteaneQECResult:
    """Outcome of one basis, with correction already applied to the data.

    Nonzero prep_flags means the ancilla failed before any data interaction;
    syndrome and correction are zero placeholders in that case. A rejected
    readout has prep_flags zero and the observed syndrome, with no correction
    applied. The caller must check accepted before using the logical shot.
    """

    accepted: bool
    prep_flags: int
    syndrome: int
    correction: int


def make_steane_qec_from_state_prep(
    prepare_state: GuppyFunctionDefinition,
    basis: Literal["X", "Z"],
) -> GuppyFunctionDefinition:
    """Return qec(data: array[qubit, 32]) -> SteaneQECResult.

    prepare_state borrows a fresh 32-qubit block and returns its flag mask.
    It owns its preparation ancillas. Failed preparations are discarded before
    coupling to data; retry policy is entirely supplied by prepare_state.

    Z extraction: convert the ancilla to |+>, CX(data, ancilla), measure Z,
    then apply the decoded X correction to data.
    X extraction: CX(ancilla, data), measure ancilla in X, then apply Z.
    Only the ancilla changes basis. The data is never destructively measured.
    """
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")

    @guppy
    def qec(data: array[qubit, 32]) -> SteaneQECResult:
        ancilla = array(qubit() for _ in range(32))
        flags = prepare_state(ancilla)
        if flags != 0:
            discard_array(ancilla)
            return SteaneQECResult(False, flags, 0, 0)

        if comptime(basis == "Z"):
            for i in range(32):
                h(ancilla[i])
            for i in range(32):
                cx(data[i], ancilla[i])
        else:
            for i in range(32):
                cx(ancilla[i], data[i])
            for i in range(32):
                h(ancilla[i])

        measurements = measure_array(ancilla)
        readout = array(m.read() for m in measurements)
        decoded = decode_readout_syndrome(readout)
        if not decoded.accepted:
            return SteaneQECResult(False, 0, decoded.syndrome, 0)

        for i in range(32):
            if (decoded.correction & (1 << i)) != 0:
                if comptime(basis == "Z"):
                    x(data[i])
                else:
                    z(data[i])
        return SteaneQECResult(True, 0, decoded.syndrome, decoded.correction)

    return qec


def make_steane_qec_cycle(prepare_state: GuppyFunctionDefinition) -> GuppyFunctionDefinition:
    """Return cycle(data) -> bool, extracting Z checks and then X checks.

    Stop on a failed stage and return False; the caller should reject that
    logical shot. Previously applied corrections are retained. Use the
    single-basis factories directly when per-stage diagnostics are needed.
    Ancilla blocks are released between stages, so the peak width is 66.
    """
    qec_z = make_steane_qec_from_state_prep(prepare_state, "Z")
    qec_x = make_steane_qec_from_state_prep(prepare_state, "X")

    @guppy
    def cycle(data: array[qubit, 32]) -> bool:
        first = qec_z(data)
        if not first.accepted:
            return False
        second = qec_x(data)
        return second.accepted

    return cycle

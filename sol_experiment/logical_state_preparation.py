"""Composable Guppy preparations of the [[32, 20, 4]] encoded zero state.

Requires guppylang >=1,<2.

The ``*_ops_*`` functions in circuit_descriptions are ordinary Python descriptions. The
``@guppy.comptime`` functions below translate them into gates during compilation;
there is no operation-list interpreter running on the quantum computer.

Each single-attempt preparation borrows a fresh, zero-initialised 32-qubit array.
The caller keeps that array, even when verification fails. Qubits 32 and 33 in
the descriptions are local verification ancillas, allocated and destructively
measured internally. Distances 2, 3, and 4 need 34 live qubits total, including
these flags; distance 1 needs 32. Initial data R operations are supplied by the
caller's ``qubit()`` allocations; RX means prepare |+>, implemented by adding H.

The return value is a measurement mask: bit i is the i-th verification result
in circuit order, with 0 meaning accept. Distances 1/2/3/4 produce 0/6/9/11 bits.
Distance 1 has no verification and always returns 0. These are preparation
circuit distances; the underlying code remains [[32, 20, 4]].

For a single attempt inside a QEC controller::

    ancilla = array(qubit() for _ in range(32))
    flags = state_prep_32_20_4_zero_d4(ancilla)
    # Branch on flags in a regular @guppy function, then use/dispose of ancilla.

For a bounded repeat-until-success method, wrap any preparation::

    rus_d3 = repeat_until_success(state_prep_32_20_4_zero_d3, max_retries=4)

The returned Guppy function has the same borrowed-block/flag-mask interface,
so it can also be passed to ``make_spam_experiment``. It makes one initial
attempt and up to four retries by default, resetting the block between failed
attempts. A nonzero return value means every allowed attempt failed.

The executable SPAM experiment lives in ``sol_experiment.experiment_SPAM``.
It combines preparation verification with corrected logical Z readout and
reports acceptance and logical error rates. Use the sol_experiment uv
environment, with the stabilizer simulator, to run the 34-qubit experiment.
"""

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, exit
from guppylang.std.quantum import qubit, reset

from sol_experiment.circuit_descriptions import (
    state_prep_ops_32_20_4_zero_d1, state_prep_ops_32_20_4_zero_d2,
    state_prep_ops_32_20_4_zero_d3, state_prep_ops_32_20_4_zero_d4,
)
from sol_experiment.circuit_utils import emit_circuit
from sol_experiment.measurement_utils import pack_bits


def _pack_flags(measurements):
    """Read and pack preparation verification results during Guppy tracing."""
    return pack_bits([measurement.read() for measurement in measurements])


@guppy.comptime
def state_prep_32_20_4_zero_d1(data: array[qubit, 32]) -> int:
    """Prepare a fresh zero-initialised block; no verification measurements."""
    return _pack_flags(emit_circuit(data, state_prep_ops_32_20_4_zero_d1(), fresh_data=True))


@guppy.comptime
def state_prep_32_20_4_zero_d2(data: array[qubit, 32]) -> int:
    """Prepare a fresh zero-initialised block; return the six verification bits."""
    return _pack_flags(emit_circuit(data, state_prep_ops_32_20_4_zero_d2(), fresh_data=True))


@guppy.comptime
def state_prep_32_20_4_zero_d3(data: array[qubit, 32]) -> int:
    """Prepare a fresh zero-initialised block; return the nine verification bits."""
    return _pack_flags(emit_circuit(data, state_prep_ops_32_20_4_zero_d3(), fresh_data=True))


@guppy.comptime
def state_prep_32_20_4_zero_d4(data: array[qubit, 32]) -> int:
    """Prepare a fresh zero-initialised block; return the eleven verification bits."""
    return _pack_flags(emit_circuit(data, state_prep_ops_32_20_4_zero_d4(), fresh_data=True))


@guppy
def state_prep_32_20_4_zero(data: array[qubit, 32], distance: int) -> int:
    """Choose a preparation circuit at runtime; data must be fresh |0> qubits."""
    if distance == 1:
        return state_prep_32_20_4_zero_d1(data)
    if distance == 2:
        return state_prep_32_20_4_zero_d2(data)
    if distance == 3:
        return state_prep_32_20_4_zero_d3(data)
    if distance == 4:
        return state_prep_32_20_4_zero_d4(data)
    exit("State-preparation distance must be 1, 2, 3, or 4", 1)
    return -1


def repeat_until_success(
    prepare_state: GuppyFunctionDefinition, max_retries: int = 4,
) -> GuppyFunctionDefinition:
    """Wrap a preparation with bounded, measurement-controlled Guppy retries.

    prepare_state must borrow a fresh array[qubit, 32] and return an int flag
    mask, with zero meaning success. The returned function has that same
    interface. The caller retains the final block on success or exhaustion.

    max_retries counts attempts AFTER the first: zero means one attempt, and
    the default four means at most five attempts. Stop at the first zero flag;
    otherwise return the last nonzero mask without aborting the shot. Reset
    all data qubits to |0> before each retry, since these preparations assume
    a fresh block. The wrapped preparation manages its own flag ancillas.

    The method and retry limit are bound in Python; the retry loop executes
    in Guppy at runtime, so it can branch on actual verification outcomes.
    """
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise TypeError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")

    @guppy
    def preparation(data: array[qubit, 32]) -> int:
        flags = prepare_state(data)
        retries = 0
        while flags != 0 and retries < comptime(max_retries):
            for q in range(32):
                reset(data[q])
            flags = prepare_state(data)
            retries += 1
        return flags

    return preparation


PREPARATION_METHODS = {
    "d1": state_prep_32_20_4_zero_d1,
    "d2": state_prep_32_20_4_zero_d2,
    "d3": state_prep_32_20_4_zero_d3,
    "d4": state_prep_32_20_4_zero_d4,
    "d2_RUS": repeat_until_success(state_prep_32_20_4_zero_d2),
    "d3_RUS": repeat_until_success(state_prep_32_20_4_zero_d3),
    "d4_RUS": repeat_until_success(state_prep_32_20_4_zero_d4),
}

"""Composable Guppy preparations of the [[32, 20, 4]] encoded zero state.

Requires guppylang >=1,<2.

The ``*_ops_*`` functions are ordinary Python circuit descriptions. The
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
reports acceptance and logical error rates. Use the local_sol_example Python
environment, with the stabilizer simulator, to run the 34-qubit experiment.
"""

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, exit
from guppylang.std.quantum import qubit, reset

from sol_experiment.circuit_utils import emit_circuit


def state_prep_ops_32_20_4_zero_d4() -> list[tuple[str, list[int]]]:
    return state_prep_ops_32_20_4_zero_d1() + [
        ("R", [32]),
        ("CX", [28, 32, 18, 32, 31, 32, 17, 32]),
        ("M", [32]),
        ("RX", [32]),
        ("R", [33]),
        ("CX", [32, 33, 32, 28, 33, 25, 32, 1, 33, 7, 32, 8, 33, 14, 32, 16, 33, 22, 32, 26, 33, 31, 32, 13, 33, 2, 32, 4, 33, 11, 32, 21, 33, 19, 32, 33]),
        ("MX", [32]),
        ("M", [33]),
        ("RX", [32, 33]),
        ("CZ", [33, 32]),
        ("CX", [4, 32, 5, 32, 22, 32, 23, 32]),
        ("CX", [24, 33, 27, 33, 29, 33, 30, 33]),
        ("CZ", [33, 32]),
        ("MX", [32, 33]),
        ("RX", [32, 33]),
        ("CZ", [32, 33]),
        ("CX", [16, 33, 25, 33, 26, 33, 19, 33]),
        ("CX", [6, 32, 15, 32, 30, 32, 17, 32, 21, 32, 19, 32]),
        ("CZ", [32, 33]),
        ("MX", [32, 33]),
        ("RX", [32]),
        ("RX", [33]),
        ("CZ", [33, 32]),
        ("CX", [15, 32, 27, 32, 28, 32, 21, 32, 4, 32, 25, 32]),
        ("CX", [24, 33, 31, 33, 23, 33, 16, 33]),
        ("CZ", [33, 32]),
        ("MX", [32]),
        ("MX", [33]),
        ("RX", [32]),
        ("RX", [33]),
        ("CZ", [33, 32]),
        ("CX", [2, 32, 3, 32, 16, 32, 20, 32, 27, 32, 30, 32]),
        ("CX", [5, 33, 8, 33, 10, 33, 12, 33, 16, 33, 27, 33]),
        ("CZ", [33, 32]),
        ("MX", [32]),
        ("MX", [33]),
    ]


def state_prep_ops_32_20_4_zero_d3() -> list[tuple[str, list[int]]]:
    return state_prep_ops_32_20_4_zero_d1() + [
        ("R", [32]),
        ("CX", [28, 32, 18, 32, 31, 32, 17, 32]),
        ("M", [32]),
        ("RX", [32]),
        ("R", [33]),
        ("CX", [32, 33, 32, 28, 33, 25, 32, 1, 33, 7, 32, 8, 33, 14, 32, 16, 33, 22, 32, 26, 33, 31, 32, 13, 33, 2, 32, 4, 33, 11, 32, 21, 33, 19, 32, 33]),
        ("MX", [32]),
        ("M", [33]),
        ("RX", [32, 33]),
        ("CZ", [33, 32]),
        ("CX", [4, 32, 5, 32, 22, 32, 23, 32]),
        ("CX", [24, 33, 27, 33, 29, 33, 30, 33]),
        ("CZ", [33, 32]),
        ("MX", [32, 33]),
        ("RX", [32, 33]),
        ("CZ", [32, 33]),
        ("CX", [16, 33, 25, 33, 26, 33, 19, 33]),
        ("CX", [6, 32, 15, 32, 30, 32, 17, 32, 21, 32, 19, 32]),
        ("CZ", [32, 33]),
        ("MX", [32, 33]),
        ("RX", [32]),
        ("RX", [33]),
        ("CZ", [33, 32]),
        ("CX", [15, 32, 27, 32, 28, 32, 21, 32, 4, 32, 25, 32]),
        ("CX", [24, 33, 31, 33, 23, 33, 16, 33]),
        ("CZ", [33, 32]),
        ("MX", [32]),
        ("MX", [33])
    ]


def state_prep_ops_32_20_4_zero_d2() -> list[tuple[str, list[int]]]:
    return state_prep_ops_32_20_4_zero_d1() + [
        ("RX", [32]),
        ("R", [33]),
        ("CX", [32, 33, 32, 28, 33, 25, 32, 1, 33, 7, 32, 8, 33, 14, 32, 16, 33, 22, 32, 26, 33, 31, 32, 13, 33, 2, 32, 4, 33, 11, 32, 21, 33, 19, 32, 33]),
        ("MX", [32]),
        ("M", [33]),
        ("RX", [32, 33]),
        ("CZ", [33, 32]),
        ("CX", [4, 32, 5, 32, 22, 32, 23, 32]),
        ("CX", [24, 33, 27, 33, 29, 33, 30, 33]),
        ("CZ", [33, 32]),
        ("MX", [32, 33]),
        ("RX", [32, 33]),
        ("CZ", [32, 33]),
        ("CX", [16, 32, 25, 32, 26, 32, 19, 32]),
        ("CX", [24, 33, 31, 33, 23, 33, 16, 33]),
        ("CZ", [33, 32]),
        ("MX", [32]),
        ("MX", [33]),
    ]


def state_prep_ops_32_20_4_zero_d1() -> list[tuple[str, list[int]]]:
    return [
        ("RX", [0, 1, 2]),
        ("R", [3]),
        ("RX", [4]),
        ("R", [5, 6, 7]),
        ("RX", [8]),
        ("R", [9, 10, 11, 12, 13, 14, 15]),
        ("RX", [16]),
        ("R", [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
        ("CX", [2, 3, 4, 5, 8, 9, 16, 17, 0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 16, 18, 17, 19, 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15, 16, 20, 17, 21, 18, 22, 19, 23, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 23, 31, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]),
    ]


@guppy
def _append_bit(mask: int, bit: bool, index: int) -> int:
    # Runtime branching belongs in @guppy, since bit may be a measurement.
    if bit:
        return mask | (1 << index)
    return mask


def _pack_flags(measurements):
    """Read and pack preparation verification results during Guppy tracing."""
    if len(measurements) > 63:
        raise ValueError("Preparation flag masks support at most 63 measurements")
    mask = 0
    for i, measurement in enumerate(measurements):
        mask = _append_bit(mask, measurement.read(), i)
    return mask


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

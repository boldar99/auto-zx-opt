"""Classical policies for choosing optimized Steane extractions and corrections.

History entries are (basis, distance, flags, syndrome), using Z_BASIS/X_BASIS.
Only entries before history_length are valid. The fixed capacity covers the
current paths of up to three extractions without experimental Guppy lists.
Dispatchers inspect this history and return a next action or terminal decision;
they do not operate on the quantum data or apply corrections.
"""

from enum import IntEnum

from guppylang import guppy
from guppylang.std.builtins import array, comptime

from sol_experiment.decoding import decode_syndrome


HISTORY_CAPACITY = 3
Z_BASIS, X_BASIS = 0, 1


class DispatcherAction(IntEnum):
    """Named action codes embedded as integers for Guppy 1.1.4.

    This compiler supports @guppy.enum constructors, but not matching or
    comparing their variants. Keep that workaround at the compile-time edge.
    """

    Z4 = 0
    X4 = 1
    Z2 = 2
    X2 = 3
    ACCEPT = 4
    REJECT = 5


@guppy.struct
class DispatchDecision:
    """Next extraction, or ACCEPT with X/Z masks, or REJECT.

    Correction masks are used only on ACCEPT and refer to the physical data
    frame: the runner has not applied any earlier syndrome corrections.
    """

    action: int
    correction_x: int
    correction_z: int


@guppy
def zero_flag_dispatcher(
    history: array[tuple[int, int, int, int], comptime(HISTORY_CAPACITY)],
    history_length: int,
) -> DispatchDecision:
    """Start with Z4, follow either unflagged d4 basis with its dual, then decode."""
    if history_length == 0:
        return DispatchDecision(comptime(DispatcherAction.Z4.value), 0, 0)
    if history_length < 0 or history_length > 2:
        return DispatchDecision(comptime(DispatcherAction.REJECT.value), 0, 0)

    first_basis, first_distance, first_flags, first_syndrome = history[0]
    if first_basis != comptime(Z_BASIS) and first_basis != comptime(X_BASIS):
        return DispatchDecision(comptime(DispatcherAction.REJECT.value), 0, 0)
    if first_distance != 4 or first_flags != 0:
        return DispatchDecision(comptime(DispatcherAction.REJECT.value), 0, 0)

    if history_length == 1:
        if first_basis == comptime(Z_BASIS):
            return DispatchDecision(comptime(DispatcherAction.X4.value), 0, 0)
        return DispatchDecision(comptime(DispatcherAction.Z4.value), 0, 0)

    second_basis, second_distance, second_flags, second_syndrome = history[1]
    if second_basis != comptime(Z_BASIS) and second_basis != comptime(X_BASIS):
        return DispatchDecision(comptime(DispatcherAction.REJECT.value), 0, 0)
    if second_basis == first_basis or second_distance != 4 or second_flags != 0:
        return DispatchDecision(comptime(DispatcherAction.REJECT.value), 0, 0)

    first = decode_syndrome(first_syndrome)
    second = decode_syndrome(second_syndrome)
    if not first.accepted or not second.accepted:
        return DispatchDecision(comptime(DispatcherAction.REJECT.value), 0, 0)
    if first_basis == comptime(Z_BASIS):
        return DispatchDecision(comptime(DispatcherAction.ACCEPT.value), first.correction, second.correction)
    return DispatchDecision(comptime(DispatcherAction.ACCEPT.value), second.correction, first.correction)

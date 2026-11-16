"""Destructive, corrected logical Z readout for the [[32, 20, 4]] code.

Requires guppylang >=1,<2. The checks and logical operators are the ones in
spiderwarp/assets/qeccs/misc/32_20_4.json, in the same row order as the sketch.
This self-dual code uses the same supports for its X and Z operators. Here
they label Z observables, since the physical measurements are in the Z basis.

The decoder corrects any one flipped physical readout bit and rejects every
two-bit flip. These bounds concern the total bit error relative to an encoded
readout; higher-weight errors can share an accepted syndrome. Preparation
flags must be checked separately by the caller.

All lookup tables are built in Python and embedded at compile time. The
compiled program uses only integer arithmetic and fixed-size arrays.
"""

from guppylang import guppy
from guppylang.std.builtins import array, comptime, owned
from guppylang.std.quantum import measure_array, qubit


# Bit q denotes physical qubit q. Syndrome bit i denotes check row i.
Z_CHECK_MASKS = (
    0xFFFFFFFF,
    0x0000FFFF,
    0x00FF00FF,
    0x0F0F0F0F,
    0x33333333,
    0x55555555,
)

LOGICAL_Z_SUPPORTS = (
    (0, 2, 8, 10),
    (1, 2, 8, 11),
    (0, 4, 8, 12),
    (1, 4, 8, 13),
    (2, 4, 8, 14),
    (0, 1, 2, 4, 8, 15),
    (0, 2, 16, 18),
    (1, 2, 16, 19),
    (0, 4, 16, 20),
    (1, 4, 16, 21),
    (2, 4, 16, 22),
    (0, 1, 2, 4, 16, 23),
    (0, 8, 16, 24),
    (1, 8, 16, 25),
    (2, 8, 16, 26),
    (0, 1, 2, 8, 16, 27),
    (4, 8, 16, 28),
    (0, 1, 4, 8, 16, 29),
    (0, 2, 4, 8, 16, 30),
    (1, 2, 4, 8, 16, 31),
)
LOGICAL_Z_MASKS = tuple(sum(1 << q for q in support) for support in LOGICAL_Z_SUPPORTS)


def make_lut() -> tuple[list[int], list[bool]]:
    """Index corrections/rejections by the six-bit, little-endian syndrome.

    Accept the zero syndrome and the 32 distinct single-bit syndromes. The
    remaining 31 syndromes have ambiguous two-bit explanations and reject.
    Rejected entries have a zero correction, which must not be used to accept.
    """
    corrections = [0] * 64
    post_select = [True] * 64
    post_select[0] = False
    for q in range(32):
        syndrome = sum(
            ((check >> q) & 1) << i for i, check in enumerate(Z_CHECK_MASKS)
        )
        if not post_select[syndrome]:
            raise ValueError("The code does not have distinct nonzero single-bit syndromes")
        corrections[syndrome] = 1 << q
        post_select[syndrome] = False
    return corrections, post_select


CORRECTIONS, POST_SELECT = make_lut()


@guppy.struct
class LogicalZResult:
    """Twenty logical outcomes, valid only when accepted is True.

    syndrome records the raw readout checks; correction is a physical bit mask.
    Rejected results carry zero correction and placeholder all-zero logicals.
    """

    accepted: bool
    syndrome: int
    correction: int
    logicals: array[bool, 20]


@guppy
def _masked_parity(word: int, mask: int) -> bool:
    """Parity of the selected bits in a 32-bit word."""
    parity = word & mask
    parity ^= parity >> 16
    parity ^= parity >> 8
    parity ^= parity >> 4
    parity ^= parity >> 2
    parity ^= parity >> 1
    return (parity & 1) != 0


@guppy
def decode_z_readout(readout: array[bool, 32]) -> LogicalZResult:
    """Decode classical Z readout, preserving the caller's raw bit array."""
    word = 0
    for q in range(32):
        if readout[q]:
            word |= 1 << q

    checks = comptime(list(Z_CHECK_MASKS))
    syndrome = 0
    for i in range(6):
        if _masked_parity(word, checks[i]):
            syndrome |= 1 << i

    post_select = comptime(POST_SELECT)
    if post_select[syndrome]:
        return LogicalZResult(False, syndrome, 0, array(False for _ in range(20)))

    corrections = comptime(CORRECTIONS)
    correction = corrections[syndrome]
    corrected = word ^ correction
    logical_masks = comptime(list(LOGICAL_Z_MASKS))
    logicals = array(_masked_parity(corrected, logical_masks[i]) for i in range(20))
    return LogicalZResult(True, syndrome, correction, logicals)


@guppy
def logical_z_meas(data: array[qubit, 32] @ owned) -> LogicalZResult:
    """Consume a block with transversal Z measurements and decode its readout.

    No additional qubits or entangling gates are needed. All measurements are
    requested before their results are read. This function reports acceptance;
    it does not abort or retry a destructively measured block.
    """
    measurements = measure_array(data)
    readout = array(m.read() for m in measurements)
    return decode_z_readout(readout)

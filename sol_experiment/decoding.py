"""Classical CSS syndrome and logical readout decoding for [[32, 20, 4]].

Requires guppylang >=1,<2. The checks and logical operators are the ones in
spiderwarp/assets/qeccs/misc/32_20_4.json, in the same row order as the sketch.
This self-dual code uses the same supports for its X and Z operators. Here
they label Z observables for decoding classical Z-basis readout.

The decoder corrects any one flipped physical readout bit and rejects every
two-bit flip. These bounds concern the total bit error relative to an encoded
readout; higher-weight errors can share an accepted syndrome. Preparation
flags must be checked separately by the caller.

All lookup tables are built in Python and embedded at compile time. The
compiled program uses only integer arithmetic and fixed-size arrays.
"""

from guppylang import guppy
from guppylang.std.builtins import array, comptime


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
class SyndromeResult:
    """A decoded CSS syndrome, without measuring any logical observables."""

    accepted: bool
    syndrome: int
    correction: int


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
def _pack_readout(readout: array[bool, 32]) -> int:
    word = 0
    for q in range(32):
        if readout[q]:
            word |= 1 << q
    return word


@guppy
def decode_syndrome(syndrome: int) -> SyndromeResult:
    """Decode six packed check outcomes, with bit i denoting check row i.

    Accept zero or a single-qubit syndrome; reject ambiguous syndromes and
    values outside the six-bit range. The correction is a physical Pauli mask,
    which the caller applies as X for Z checks or Z for X checks.
    """
    if syndrome < 0 or syndrome >= 64:
        return SyndromeResult(False, syndrome, 0)
    post_select = comptime(POST_SELECT)
    if post_select[syndrome]:
        return SyndromeResult(False, syndrome, 0)

    corrections = comptime(CORRECTIONS)
    return SyndromeResult(True, syndrome, corrections[syndrome])


@guppy
def _decode_word_syndrome(word: int) -> SyndromeResult:
    checks = comptime(list(Z_CHECK_MASKS))
    syndrome = 0
    for i in range(6):
        if _masked_parity(word, checks[i]):
            syndrome |= 1 << i
    return decode_syndrome(syndrome)


@guppy
def decode_readout_syndrome(readout: array[bool, 32]) -> SyndromeResult:
    """Decode just the six checks, for terminal readout or Steane extraction.

    The code's X and Z checks have identical supports, so this decoder also
    handles X-basis ancilla readout after a transversal H. The input is borrowed
    and preserved. A rejected syndrome has correction zero and must be rejected
    by the caller; it does not mean that no error was present.
    """
    return _decode_word_syndrome(_pack_readout(readout))


@guppy
def decode_z_readout(readout: array[bool, 32]) -> LogicalZResult:
    """Decode classical Z readout, preserving the caller's raw bit array."""
    word = _pack_readout(readout)
    decoded = _decode_word_syndrome(word)
    if not decoded.accepted:
        return LogicalZResult(False, decoded.syndrome, 0, array(False for _ in range(20)))

    correction = decoded.correction
    corrected = word ^ correction
    logical_masks = comptime(list(LOGICAL_Z_MASKS))
    logicals = array(_masked_parity(corrected, logical_masks[i]) for i in range(20))
    return LogicalZResult(True, decoded.syndrome, correction, logicals)

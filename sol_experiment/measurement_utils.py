"""Measurement packing and detector parities during Guppy compile-time tracing.

Call these Python helpers inside @guppy.comptime functions, after reading each
Measurement handle once. They accept ordinary Python lists of traced booleans;
the resulting integer masks and XORs are evaluated by the compiled program.
"""

from collections.abc import Sequence

from guppylang import guppy


@guppy
def _append_bit(mask: int, bit: bool, index: int) -> int:
    if bit:
        return mask | (1 << index)
    return mask


def pack_bits(bits):
    """Pack booleans in little-endian order into a nonnegative Guppy int."""
    if len(bits) > 63:
        raise ValueError("Measurement masks support at most 63 bits")
    mask = 0
    for i, bit in enumerate(bits):
        mask = _append_bit(mask, bit, i)
    return mask


def pack_detector_parities(readout, detectors: Sequence[Sequence[int]]):
    """Pack one parity per detector, in the order supplied.

    Each detector lists negative Stim rec offsets into the complete readout:
    -1 is its last measurement. Repeated references cancel under XOR. An empty
    detector has parity zero. Separate calls can reuse the same readout bits.
    """
    parities = []
    for offsets in detectors:
        parity = False
        for offset in offsets:
            if type(offset) is not int or not -len(readout) <= offset < 0:
                raise ValueError(f"Detector offset {offset!r} is outside the measurement record")
            parity ^= readout[offset]
        parities.append(parity)
    return pack_bits(parities)

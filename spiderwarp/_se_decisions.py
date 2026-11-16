"""Shared integer decision diagrams with XOR offsets on their edges.

Acceptance is a total function (missing keys mean zero). Correction functions
may instead be partial: their caller must check acceptance before using them.
An unspecified correction is not a rejection and can share either branch.
"""

from __future__ import annotations


class XorDecisions:
    """Intern functions up to a constant XOR, including below their roots.

    A reference is ``(node, offset)``. Node zero returns zero; each other node
    tests one key bit and XORs a delta on its high branch. Factoring the low
    branch's offset out of each node makes translated tables share storage.
    """

    def __init__(self):
        self.nodes = []
        self._ids = {}

    def build(self, table, width, *, partial=False):
        if width < 0 or any(k < 0 or k >> width or v < 0 for k, v in table.items()):
            raise ValueError("Decision keys must fit their width and values must be nonnegative")
        entries = tuple(sorted(table.items() if partial else
                               ((k, v) for k, v in table.items() if v)))

        def visit(values, bit):
            if not values:
                return 0, 0
            first = values[0][1]
            if (partial or len(values) == 1 << (bit + 1)) and all(v == first for _, v in values):
                return 0, first
            mask = 1 << bit
            low_values = tuple((k, v) for k, v in values if k < mask)
            high_values = tuple((k - mask, v) for k, v in values if k >= mask)
            if partial and not low_values:
                return visit(high_values, bit - 1)
            if partial and not high_values:
                return visit(low_values, bit - 1)
            low, low_offset = visit(low_values, bit - 1)
            high, high_offset = visit(high_values, bit - 1)
            if low == high and low_offset == high_offset:
                return low, low_offset
            row = bit, low, high, low_offset ^ high_offset
            if row not in self._ids:
                self.nodes.append(row)
                self._ids[row] = len(self.nodes)
            return self._ids[row], low_offset

        return visit(entries, width - 1)

    def evaluate(self, reference, key):
        node, value = reference
        while node:
            bit, low, high, delta = self.nodes[node - 1]
            if key & (1 << bit):
                value ^= delta
                node = high
            else:
                node = low
        return value

    def compact(self, roots):
        """Drop unused nodes and encode roots/rows using shared XOR values.

        Constant roots keep the existing ``-(value + 1)`` convention. Other
        roots pack a node ID and an index into the delta pool. Each decision
        row fits one signed integer when possible, with a tuple fallback.
        """
        reachable, pending = set(), [node for node, _ in roots]
        while pending:
            node = pending.pop()
            if node and node not in reachable:
                reachable.add(node)
                pending.extend(self.nodes[node - 1][1:3])
        mapping = {old: new for new, old in enumerate(sorted(reachable), 1)}
        mapping[0] = 0
        rows = [(bit, mapping[low], mapping[high], delta)
                for old in sorted(reachable)
                for bit, low, high, delta in [self.nodes[old - 1]]]
        deltas = sorted({0, *(offset for _, offset in roots), *(row[3] for row in rows)})
        delta_ids = {delta: i for i, delta in enumerate(deltas)}
        reference_bits = max(1, len(rows).bit_length())
        bit_bits = max(1, max((row[0] for row in rows), default=0).bit_length())
        delta_bits = max(1, (len(deltas) - 1).bit_length())
        if reference_bits + delta_bits > 63:
            raise ValueError("Decision root references exceed a signed machine word")
        references = [(mapping[node] | (delta_ids[offset] << reference_bits))
                      if node else -(offset + 1) for node, offset in roots]
        rows = [(bit, low, high, delta_ids[delta]) for bit, low, high, delta in rows]
        packed = bit_bits + 2 * reference_bits + delta_bits <= 63
        metadata = dict(
            N_DECISION_NODES=len(rows), DECISION_ROWS_PACKED=packed,
            DECISION_BIT_BITS=bit_bits, DECISION_BIT_MASK=(1 << bit_bits) - 1,
            DECISION_REFERENCE_BITS=reference_bits,
            DECISION_REFERENCE_MASK=(1 << reference_bits) - 1,
            DECISION_HIGH_SHIFT=bit_bits + reference_bits,
            DECISION_DELTA_SHIFT=bit_bits + 2 * reference_bits,
        )
        if packed:
            rows = [(bit | (low << bit_bits) | (high << (bit_bits + reference_bits))
                     | (delta << (bit_bits + 2 * reference_bits)))
                    for bit, low, high, delta in rows]
        return rows, references, deltas, metadata


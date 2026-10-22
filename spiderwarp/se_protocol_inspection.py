"""Notebook views and resource counts for the actual SE circuit schedules."""

from __future__ import annotations

from collections import Counter
from html import escape
from xml.etree import ElementTree

import stim

from spiderwarp._se_circuit import lower_to_guppy_gates
from spiderwarp.noise_model import (
    add_idling_gates, partition_circ, stim_circuit_to_abstract_circuit,
)
from spiderwarp.stim_utils import circuit_for_inspection


def circuit_resources(circuit: stim.Circuit, n_data: int) -> dict[str, int]:
    """Count the basis-lowered gates and the noise model's inferred idle time.

    live_idle_units excludes intervals after a destructive ancilla measurement.
    Data qubits are live throughout. As in the current noise model, intervals
    before a qubit's first operation are not counted. These are unit-duration
    gate layers, not calibrated hardware seconds or classical branch latency.
    """
    lowered = lower_to_guppy_gates(circuit)
    quantum = stim.Circuit()
    for op in lowered.flattened():
        if op.name not in {'TICK', 'DETECTOR', 'QUBIT_COORDS'}:
            quantum.append(op)
    abstract = stim_circuit_to_abstract_circuit(quantum)
    counts = Counter(g.name for g in abstract)
    live = set(range(n_data))
    idle = live_idle = 0
    for gate in add_idling_gates(abstract):
        if gate.name == 'R':
            live.update(gate.targets)
        elif gate.name == 'M':
            live.difference_update(gate.targets)
        elif gate.name == 'IDLE':
            idle += gate.arg
            if gate.targets[0] in live:
                live_idle += gate.arg
    return dict(
        qubits=circuit.num_qubits,
        preparations=counts['R'], measurements=counts['M'],
        one_qubit_gates=sum(counts[g] for g in ('H', 'X', 'Y', 'Z')),
        two_qubit_gates=sum(counts[g] for g in ('CX', 'CZ')),
        depth=len(partition_circ(abstract)),
        live_idle_units=live_idle, idle_units=idle,
    )


def protocol_circuit_roles(protocol) -> dict[tuple[str, int], tuple[str, ...]]:
    """Describe how each available gadget is selected by the current policy."""
    roles = {key: set() for key in protocol.gadgets}
    for basis in ('X', 'Z'):
        roles[basis, protocol.code.d].add('primary')
    for plan in protocol.recoveries.values():
        dual = 'X' if plan.primary_basis == 'Z' else 'Z'
        roles[dual, plan.distance].add('recovery')
        if plan.cleanup_distance is not None:
            roles[plan.primary_basis, plan.cleanup_distance].add('cleanup')
    return {key: tuple(sorted(value)) for key, value in roles.items()}


def circuit_inspection_html(circuit: stim.Circuit, title: str, n_data: int) -> str:
    """A collapsible, scrollable SVG plus the complete annotated Stim source.

    Return HTML without requiring IPython; notebooks can display it with HTML().
    SVG dimensions are kept at their native size so large circuits stay legible.
    """
    diagram = str(circuit_for_inspection(circuit).diagram('timeline-svg'))
    _, _, width, height = ElementTree.fromstring(diagram).attrib['viewBox'].split()
    diagram = diagram.replace('<svg ',
        f'<svg style="max-width:none;width:{width}px;height:{height}px" ', 1)
    return (
        f'<details><summary><strong>{escape(title)}</strong> '
        f'({circuit.num_qubits} qubits, {circuit.num_measurements} measurements)</summary>'
        f'<p>Data: q0–q{n_data - 1}; remaining wires are ancillas. '
        'The diagram shows quantum operations; the source below retains flag '
        'and syndrome measurement parities.</p>'
        f'<div style="overflow:auto;max-height:650px;border:1px solid #888;resize:vertical">{diagram}</div>'
        '<details><summary>Annotated Stim instructions</summary>'
        f'<pre style="overflow:auto;max-height:450px">{escape(str(circuit))}</pre>'
        '</details></details>'
    )

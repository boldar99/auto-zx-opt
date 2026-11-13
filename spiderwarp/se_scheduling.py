"""Shorter d2 allocations under a fixed adaptive-protocol qubit limit.

Split recycled ancillas into lifetimes, then remove expensive reuse links.
The operation order, measurements and controller are preserved. Selection uses
the production idle scheduler, including persistent data at the input boundary.
This is a bounded search over an allocation frontier, not a depth optimum.
"""
from dataclasses import replace

import stim

from spiderwarp._se_adaptive import AdaptiveVerificationReport, ExtractionAction
from spiderwarp._se_circuit import SEGadget, validate_gadget
from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol
from spiderwarp.noise_model import (
    add_idling_gates, partition_circ, stim_circuit_to_abstract_circuit,
)
from spiderwarp.resource_scheduling import (
    _build_reuse_frontier, _mapping_from_merge_links, allocate_interval_qubits,
    build_resource_event_dag, emit_scheduled_circuit,
)


def scheduled_resources(circuit, n_data):
    """Measure the same annotation-free circuit used by ProtocolSampler."""
    stripped = stim.Circuit()
    for op in circuit.flattened():
        if op.name not in {'DETECTOR', 'QUBIT_COORDS', 'TICK'}:
            stripped.append(op)
    abstract = stim_circuit_to_abstract_circuit(stripped)
    scheduled = add_idling_gates(abstract, persistent_qubits=range(n_data))
    idle = [op for op in scheduled if op.name == 'IDLE']
    return dict(qubits=circuit.num_qubits, depth=len(partition_circ(abstract)),
                cnots=sum(len(op.targets_copy()) // 2 for op in stripped
                          if op.name in {'CX', 'CZ'}),
                data_idle_units=sum(op.arg for op in idle if op.targets[0] < n_data),
                ancilla_idle_units=sum(op.arg for op in idle if op.targets[0] >= n_data))


def _score(resources):
    return (resources['data_idle_units'], resources['depth'],
            resources['ancilla_idle_units'], resources['qubits'])


def _reschedule(gadget, peak_qubits):
    n = gadget.space.n
    expanded = stim.Circuit()
    mapping, next_qubit = {q: q for q in range(n)}, n
    for name, qubits in gadget.operations:
        if name in {'R', 'RX'}:
            mapping[qubits[0]] = next_qubit
            next_qubit += 1
        expanded.append(name, [mapping[q] for q in qubits])
    problem = build_resource_event_dag(expanded, n)
    order = list(range(problem.dag.number_of_nodes()))
    minimum, _ = allocate_interval_qubits(problem.lifetimes, order, n)
    candidates, links = _build_reuse_frontier(problem, order, problem.lifetimes, n, minimum)
    best, resources = gadget, scheduled_resources(gadget.circuit, n)
    feasible = 0
    for candidate in candidates:
        if candidate.num_qubits > peak_qubits:
            continue
        feasible += 1
        allocation, _ = _mapping_from_merge_links(
            problem.lifetimes, n, links[:candidate.num_reuse_merges])
        circuit, meas_map = emit_scheduled_circuit(problem, order, allocation)
        following = scheduled_resources(circuit, n)
        if _score(following) < _score(resources):
            readout = gadget.readout.remap(meas_map)
            provenance = {new: gadget.measurement_map.get(old, old)
                          for new, old in meas_map.items()}
            best = SEGadget(gadget.distance, gadget.basis, readout.annotate(circuit),
                            readout, gadget.space, provenance)
            resources = following
    return best, feasible


def reschedule_d2(parent, *, peak_qubits=60, verify=True):
    """Keep the standard r+s<=1 controller and shorten only its d2 gadgets."""
    if type(peak_qubits) is not int or peak_qubits < parent.required_qubits:
        raise ValueError('Qubit limit must accommodate the parent protocol')
    if (parent.policy.contract.correctable != 0 or parent.policy.contract.logical_order != 1
            or set(parent.gadgets) != {'X_d2', 'Z_d2', 'X_d4', 'Z_d4'}):
        raise ValueError('Supply the standard flag-conditioned r+s<=1 protocol')
    actions, stages = {}, {}
    for name, gadget in parent.gadgets.items():
        before = scheduled_resources(gadget.circuit, parent.code.n)
        new, feasible = (_reschedule(gadget, peak_qubits) if gadget.distance == 2
                         else (gadget, 0))
        after = scheduled_resources(new.circuit, parent.code.n)
        validate_gadget(new)
        if set(new.single_fault_effects) != set(gadget.single_fault_effects):
            raise ValueError(f'{name}: changed single-fault responses')
        if after['qubits'] > peak_qubits or after['cnots'] != before['cnots']:
            raise ValueError(f'{name}: changed gate count or exceeded qubit limit')
        actions[name] = ExtractionAction(name, new)
        stages[name] = dict(before=before, after=after, feasible_allocations=feasible,
                            identical_single_fault_responses=True,
                            circuit_changed=new is not gadget)
    report = AdaptiveVerificationReport(parent.policy.contract, parent.policy.decoder.fingerprint)
    policy = replace(parent.policy, actions=actions, report=report, peak_qubit_limit=peak_qubits)
    protocol = AdaptiveSEProtocol(parent.code, policy)
    if verify:
        protocol.verify()
        from spiderwarp.se_postselection import verify_order_one_guard
        guard = verify_order_one_guard(protocol)
    else:
        guard = None
    return protocol, dict(method='standard-order-one-shorter-d2', peak_qubit_limit=peak_qubits,
        unchanged_controller=True, unchanged_frames=True, unchanged_d4=True,
        selection='Minimize scheduled data idles, depth, ancilla idles, then qubits on reuse frontier',
        optimal=False, stages=stages, order_one_guard=guard)

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import count
from pprint import pprint

import pyzx as zx

# from code_examples import *

# Global counter for unique gate IDs
_id_counter = count()


class Operation(ABC):
    uid: int = field(default_factory=lambda: next(_id_counter), init=False)

    @abstractmethod
    def add_to_pyzx_circuit(self, circ: zx.Circuit, index_mapping: dict[int, int]):
        pass

    @property
    @abstractmethod
    def qubits_involved(self) -> tuple[int, ...]:
        pass


@dataclass(frozen=True)
class SingleQubitOperation(Operation, ABC):
    i: int
    uid: int = field(default_factory=lambda: next(_id_counter), init=False, compare=False, hash=False)

    def __lt__(self, other: SingleQubitOperation):
        return self.i < other.i

    @property
    def qubits_involved(self) -> tuple[int, ...]:
        return (self.i,)


@dataclass(frozen=True)
class TwoQubitOperation(Operation, ABC):
    i: int
    j: int
    uid: int = field(default_factory=lambda: next(_id_counter), init=False, compare=False, hash=False)

    @property
    def qubits_involved(self) -> tuple[int, ...]:
        return self.i, self.j


@dataclass(frozen=True)
class StatePreparation(Operation, ABC):
    pass


@dataclass(frozen=True)
class Measurement(Operation, ABC):
    pass


class ZState(SingleQubitOperation, StatePreparation):
    def add_to_pyzx_circuit(self, circ, index_mapping):
        circ.add_gate("InitAncilla", label=index_mapping.get(self.i, self.i), basis="Z")


class XState(SingleQubitOperation, StatePreparation):
    def add_to_pyzx_circuit(self, circ, index_mapping):
        circ.add_gate("InitAncilla", label=index_mapping.get(self.i, self.i), basis="X")


class ZBasisMeasurement(SingleQubitOperation, Measurement):
    def add_to_pyzx_circuit(self, circ, index_mapping):
        circ.add_gate("PostSelect", label=index_mapping.get(self.i, self.i), basis="Z")


class XBasisMeasurement(SingleQubitOperation, Measurement):
    def add_to_pyzx_circuit(self, circ, index_mapping):
        circ.add_gate("PostSelect", label=index_mapping.get(self.i, self.i), basis="X")


class H(SingleQubitOperation):
    def add_to_pyzx_circuit(self, circ, index_mapping):
        circ.add_gate("H", index_mapping.get(self.i, self.i))


class CNOT(TwoQubitOperation):
    def add_to_pyzx_circuit(self, circ, index_mapping):
        circ.add_gate("CNOT", index_mapping.get(self.i, self.i), index_mapping.get(self.j, self.j))


class BellState(TwoQubitOperation, StatePreparation):
    def add_to_pyzx_circuit(self, circ, index_mapping):
        circ.add_gate("InitAncilla", label=index_mapping.get(self.i, self.i), basis="X")
        circ.add_gate("InitAncilla", label=index_mapping.get(self.j, self.j), basis="Z")
        circ.add_gate("CNOT", index_mapping.get(self.i, self.i), index_mapping.get(self.j, self.j))


class Circuit:
    def __init__(self, n_data: int, n_ancilla: int):
        self.n_data = n_data
        self.n_ancilla = n_ancilla
        self.ops: list[Operation] = []
        self._gate_pos: dict[int, int] = {}
        self.qubit_schedules: list[list[int]] = [list() for _ in range(n_data + n_ancilla)]

    def _remove_op(self, op: Operation):
        if op.uid not in self._gate_pos:
            return

        idx = self._gate_pos[op.uid]
        self.ops.pop(idx)
        del self._gate_pos[op.uid]

        for i in range(idx, len(self.ops)):
            other_op = self.ops[i]
            self._gate_pos[other_op.uid] = i

        for q in op.qubits_involved:
            if op.uid in self.qubit_schedules[q]:
                self.qubit_schedules[q].remove(op.uid)

    def _replace_op(self, op: Operation, new_op: Operation):
        """
        Replaces 'op' with 'new_op' at the exact same global index.
        Updates self.ops, self._gate_pos, and self.qubit_schedules.
        """
        if op.uid not in self._gate_pos:
            return
        if new_op.uid in self._gate_pos:
            raise ValueError(f"Operation with uid {new_op.uid} already exists in circuit.")


        idx = self._gate_pos[op.uid]
        self.ops[idx] = new_op

        del self._gate_pos[op.uid]
        self._gate_pos[new_op.uid] = idx

        # Update Qubit Schedules
        # A. Remove old op from its specific qubit schedules
        for q in op.qubits_involved:
            if op.uid in self.qubit_schedules[q]:
                self.qubit_schedules[q].remove(op.uid)

        # B. Insert new op into its specific qubit schedules
        # We must insert it in the correct order relative to other gates on that qubit.
        for q in new_op.qubits_involved:
            sched = self.qubit_schedules[q]
            inserted = False

            # Iterate through the schedule to find where the global index 'idx' fits
            for i, existing_uid in enumerate(sched):
                # Retrieve global position of the existing gate on this qubit
                existing_global_pos = self._gate_pos[existing_uid]

                # If we find a gate that happens *after* our new gate, insert before it
                if existing_global_pos > idx:
                    sched.insert(i, new_op.uid)
                    inserted = True
                    break

            # If no later gate was found, the new gate goes at the end of the schedule
            if not inserted:
                sched.append(new_op.uid)

    def _register_op(self, op: Operation):
        """Internal: registers a new operation at the end of the circuit."""
        pos = len(self.ops)
        self.ops.append(op)
        self._gate_pos[op.uid] = pos

        for q in op.qubits_involved:
            self.qubit_schedules[q].append(op.uid)

        return op

    def ket_zero(self, i: int):
        return self._register_op(ZState(i))

    def ket_plus(self, i: int):
        return self._register_op(XState(i))

    def measure_z(self, i: int):
        return self._register_op(ZBasisMeasurement(i))

    def measure_x(self, i: int):
        return self._register_op(XBasisMeasurement(i))

    def h(self, i: int):
        return self._register_op(H(i))

    def cnot(self, i: int, j: int):
        sched_i = self.qubit_schedules[i]
        sched_j = self.qubit_schedules[j]

        if sched_i and sched_j:
            last_uid_i = sched_i[-1]
            last_uid_j = sched_j[-1]

            op_i = self.ops[self._gate_pos[last_uid_i]]
            op_j = self.ops[self._gate_pos[last_uid_j]]

            if isinstance(op_i, XState) and isinstance(op_j, ZState):
                self._remove_op(op_i)
                self._remove_op(op_j)

                return self._register_op(BellState(i, j))

        return self._register_op(CNOT(i, j))

    def bell_state(self, i: int, j: int):
        return self._register_op(BellState(i, j))

    def get_bell_states(self) -> list[BellState]:
        return [o for o in self.ops if isinstance(o, BellState)]

    def get_op_by_uid(self, uid: int) -> Operation:
        return self.ops[self._gate_pos[uid]]

    def _update_gate_pos(self, start_idx: int, end_idx: int):
        for idx in range(start_idx, end_idx + 1):
            op = self.ops[idx]
            self._gate_pos[op.uid] = idx

    def can_swap(self, idx: int):
        if idx < 1 or idx >= len(self.ops) - 1:
            return False

        op0 = self.ops[idx - 1]
        op1 = self.ops[idx]
        op2 = self.ops[idx + 1]
        op3 = idx + 2 < len(self.ops) and self.ops[idx + 2]

        q1 = set(op1.qubits_involved)
        q2 = set(op2.qubits_involved)
        intersection = q1.intersection(q2)

        if not intersection:
            return True
        if len(intersection) > 1:
            return False
        shared_qubit = next(iter(intersection))

        op_i = self.qubit_schedules[shared_qubit].index(op1.uid)
        if op_i == 1:
            op0 = self.ops[self._gate_pos[self.qubit_schedules[shared_qubit][op_i - 1]]]
            if isinstance(op0, (ZState, XState)) and isinstance(op1, CNOT) and isinstance(op2, CNOT):
                if op1.i == op2.i == op0.i:  # Same control
                    return True
                elif op1.j == op2.j == op0.i:  # Same target
                    return True
        if op_i + 3 == len(self.qubit_schedules[shared_qubit]):
            op3 = self.ops[self._gate_pos[self.qubit_schedules[shared_qubit][op_i + 2]]]
            if isinstance(op3, (ZBasisMeasurement, XBasisMeasurement)) and isinstance(op1, CNOT) and isinstance(op2, CNOT):
                if op1.i == op2.i == op3.i:  # Same control
                    return True
                elif op1.j == op2.j == op3.i:  # Same target
                    return True

        return False

    def swap_adjacent_ops(self, idx: int) -> bool:
        """
        Rule 1 & Rule 3 (Partial):
        Swaps ops[idx] and ops[idx+1] if allowed.
        Returns True if swapped.
        """
        if self.can_swap(idx):
            op1 = self.ops[idx]
            op2 = self.ops[idx + 1]

            q1 = set(op1.qubits_involved)
            q2 = set(op2.qubits_involved)
            intersection = q1.intersection(q2)

            self.ops[idx], self.ops[idx + 1] = self.ops[idx + 1], self.ops[idx]

            self._gate_pos[op1.uid] = idx + 1
            self._gate_pos[op2.uid] = idx

            if intersection:
                for q in intersection:
                    sched = self.qubit_schedules[q]
                    s_idx = sched.index(op1.uid)
                    if s_idx + 1 < len(sched) and sched[s_idx + 1] == op2.uid:
                        sched[s_idx], sched[s_idx + 1] = sched[s_idx + 1], sched[s_idx]
            return True
        else:
            return False

    def can_bell_push(self, gate_idx: int) -> bool:
        if gate_idx < 1 or gate_idx >= len(self.ops) - 1:
            return False

        op_bell = self.ops[gate_idx - 1]
        op_next = self.ops[gate_idx]

        if not isinstance(op_bell, BellState):
            return False

        if isinstance(op_next, TwoQubitOperation):
            return op_next.i in op_bell.qubits_involved or op_next.j in op_bell.qubits_involved

        if isinstance(op_next, SingleQubitOperation):
            return op_next.i in op_bell.qubits_involved

        return False

    def apply_bell_push(self, gate_idx: int) -> bool:
        op_next = self.ops[gate_idx]
        if isinstance(op_next, SingleQubitOperation):
            return self.apply_bell_push_1(gate_idx)
        if isinstance(op_next, TwoQubitOperation):
            return self.apply_bell_push_2(gate_idx)
        return False

    def apply_bell_push_1(self, gate_idx: int) -> bool:
        if not self.can_bell_push(gate_idx):
            return False

        op_bell = self.ops[gate_idx - 1]
        op_next = self.ops[gate_idx]

        if not isinstance(op_next, (ZBasisMeasurement, XBasisMeasurement)):
            return False

        old_qubit = op_next.i
        [new_qubit] = set(op_bell.qubits_involved) - {old_qubit}

        new_op: StatePreparation
        if isinstance(op_next, ZBasisMeasurement):
            new_op = ZState(new_qubit)
        else:
            new_op = XState(new_qubit)

        self._remove_op(op_next)
        self._replace_op(op_bell, new_op)

        return True

    def apply_bell_push_2(self, gate_idx: int) -> bool:
        """
        Rule 2: Bell commutation.
        Pattern: BELL(a,b); CNOT(a,c)
        Result:  BELL(a,b); CNOT(b,c)
        """
        if not self.can_bell_push(gate_idx):
            return False

        op_bell = self.ops[gate_idx - 1]
        op_next = self.ops[gate_idx]

        if not isinstance(op_next, CNOT):
            return False

        bell_qubits = set(op_bell.qubits_involved)
        op_qubits = set(op_next.qubits_involved)

        op_stabiles_ls = list(op_qubits - bell_qubits)
        move_to_ls = list(bell_qubits - op_qubits)
        shared_qubits_ls = list(bell_qubits.intersection(op_qubits))

        assert len(op_stabiles_ls) == len(move_to_ls) == len(shared_qubits_ls) == 1

        stabile_qubit = op_stabiles_ls[0]
        move_to_qubit = move_to_ls[0]
        shared_qubit = shared_qubits_ls[0]

        if stabile_qubit == op_next.i:  # Control
            new_cnot = CNOT(stabile_qubit, move_to_qubit)
        else:  # Target
            new_cnot = CNOT(move_to_qubit, stabile_qubit)

        old_uid = op_next.uid
        new_uid = new_cnot.uid

        self.ops[gate_idx] = new_cnot

        del self._gate_pos[old_uid]
        self._gate_pos[new_uid] = gate_idx

        self.qubit_schedules[shared_qubit].remove(old_uid)
        s_idx_target = self.qubit_schedules[stabile_qubit].index(old_uid)
        self.qubit_schedules[stabile_qubit][s_idx_target] = new_uid

        sched_b = self.qubit_schedules[move_to_qubit]
        bell_pos_in_b = sched_b.index(op_bell.uid)
        sched_b.insert(bell_pos_in_b + 1, new_uid)

        return True

    def bubble_left(self, gate_idx: int) -> int:
        """Repeatedly try to swap a gate left until blocked."""
        while self.can_swap(gate_idx - 1):
            self.swap_adjacent_ops(gate_idx - 1)
            gate_idx -= 1
        return gate_idx

    def cluster_bubble_left(self, gate_idx: int, stop_at_gate: int = -1) -> int:
        """
        Pushes a gate left as much as possible. If it gets blocked by another gate,
        that blocking gate is added to the moving 'mass', and we try to push
        the whole group left. Stops at StatePreparation or circuit start.
        """
        cluster = {gate_idx}
        busy = True

        while busy:
            busy = False
            # We sort indices to try moving the left-most gates first,
            # which clears the path for the rest of the cluster.
            sorted_indices = sorted(list(cluster))

            for idx in sorted_indices:
                if idx <= 0:
                    continue

                prev_idx = idx - 1

                if prev_idx in cluster:
                    continue

                if self.swap_adjacent_ops(prev_idx):
                    cluster.remove(idx)
                    cluster.add(prev_idx)
                    busy = True
                    break

                else:
                    if prev_idx == stop_at_gate:
                        break
                    blocker = self.ops[prev_idx]
                    if not isinstance(blocker, StatePreparation):
                        cluster.add(prev_idx)
                        busy = True
                        break

        return max(cluster)

    def bubble_state_right(self, gate_idx: int) -> int:
        while isinstance(self.ops[gate_idx + 1], StatePreparation):
            self.swap_adjacent_ops(gate_idx)
            gate_idx += 1
        return gate_idx

    def to_pyzx_circuit(self):
        state_preps, gates = [], []
        indices = set()

        for op in self.ops:
            if isinstance(op, BellState):
                state_preps.append(XState(op.i))
                state_preps.append(ZState(op.j))
            elif isinstance(op, StatePreparation):
                state_preps.append(op)
            indices = indices | set(op.qubits_involved)
        state_preps.sort()

        for op in self.ops:
            if not isinstance(op, StatePreparation):
                gates.append(op)
            elif isinstance(op, BellState):
                gates.append(CNOT(op.i, op.j))

        to_index = {ix: i for i, ix in enumerate(sorted(list(indices)))}

        circ = zx.Circuit(self.n_data)
        for op in state_preps + gates:
            op.add_to_pyzx_circuit(circ, to_index)
        return circ

    @classmethod
    def from_list(cls, n_data: int, n_ancillae: int, x_states: list[int], x_measurements: list[int], cnots: list[tuple[int, int]]):
        ret = cls(n_data, n_ancillae)
        for i in range(n_data, n_data + n_ancillae):
            if i in x_states:
                ret.ket_plus(i)
            else:
                ret.ket_zero(i)

        for c, n in cnots:
            ret.cnot(c, n)

        for i in range(n_data, n_data + n_ancillae):
            if i in x_measurements:
                ret.measure_x(i)
            else:
                ret.measure_z(i)

        return ret

    def copy(self):
        ret = type(self)(self.n_data, self.n_ancilla)
        ret.ops = self.ops[:]
        ret._gate_pos = self._gate_pos.copy()
        ret.qubit_schedules = [s[:] for s in self.qubit_schedules]
        return ret

    def try_bend_at_bel_state(self, bell_idx: int, push_to_top: bool = True):
        bell_idx = self.bubble_state_right(bell_idx)
        op = self.ops[bell_idx]
        assert isinstance(op, BellState)

        if push_to_top:
            to_push_through = self.qubit_schedules[op.i][1:]
        else:
            to_push_through = self.qubit_schedules[op.j][1:]

        for gate in to_push_through:
            gate_idx = self.cluster_bubble_left(self._gate_pos[gate], bell_idx)
            if self.can_bell_push(gate_idx):
                self.apply_bell_push(gate_idx)
            else:
                return False
        return True

    def try_reduce_non_local_cnot(self, qubit_idx: int):
        qubit_ops = self.qubit_schedules[qubit_idx]
        if len(qubit_ops) != 4:
            return False

        [state, cn1, cn2, meas] = tuple(self.ops[self._gate_pos[o]] for o in qubit_ops)
        if not(
            isinstance(state, (ZState, XState)) and
            isinstance(cn1, CNOT) and  isinstance(cn2, CNOT)
            and isinstance(meas, (ZBasisMeasurement, XBasisMeasurement))
        ):
            return False

        self.cluster_bubble_left(self._gate_pos[qubit_ops[2]], self._gate_pos[qubit_ops[1]])
        if self._gate_pos[qubit_ops[2]] != self._gate_pos[qubit_ops[1]] + 1:
            return False

        if cn1.i == cn2.j == qubit_idx:
            new_cnot = CNOT(cn2.i, cn1.j)
        elif cn1.j == cn2.i == qubit_idx:
            new_cnot = CNOT(cn1.i, cn2.j)
        else:
            return False

        self._remove_op(meas)
        self._remove_op(cn2)
        self._replace_op(cn1, new_cnot)
        self._remove_op(state)

        return True

    def greedy_bend(self):
        bell_candidates = [o.uid for o in self.ops if isinstance(o, BellState)]
        while bell_candidates:
            bend_at = bell_candidates.pop(-1)
            if self.try_bend_at_bel_state(self._gate_pos[bend_at], push_to_top=False):
                continue
            self.try_bend_at_bel_state(self._gate_pos[bend_at], push_to_top=True)

    def greedy_reduce_nonlocal_cnot(self):
        for i, qubits in enumerate(self.qubit_schedules):
            if len(qubits) == 4:
                self.try_reduce_non_local_cnot(i)


def steane_code():
    h1 = [0, 4, 6]
    cnots_list = [[0, 1], [4, 5], [6, 3], [6, 5], [4, 2], [0, 3], [4, 1], [3, 2], [1, 7], [3, 7], [5, 7]]
    h2 = []
    return Circuit.from_list(7, 8, h1, h2, cnots_list)


def test_manual_bell_bend():
    circuit = steane_code()
    pprint(circuit.ops)
    zx.draw_matplotlib(circuit.to_pyzx_circuit(), figsize=(10, 12))

    circuit.try_bend_at_bel_state(2, False)
    pprint(circuit.ops)
    print(circuit)
    zx.draw_matplotlib(circuit.to_pyzx_circuit(), figsize=(10, 12))

    circuit.try_bend_at_bel_state(3, False)
    pprint(circuit.ops)
    zx.draw_matplotlib(circuit.to_pyzx_circuit(), figsize=(10, 12))

    circuit.try_bend_at_bel_state(2, True)
    zx.draw_matplotlib(circuit.to_pyzx_circuit(), figsize=(10, 12))
    pprint(circuit.ops)


def test_greedy_bend(code):
    circuit = code()
    pprint(circuit.ops)
    zx.draw_matplotlib(circuit.to_pyzx_circuit(), figsize=(10, 12))

    circuit.greedy_path_opt()
    pprint(circuit.ops)
    zx.draw_matplotlib(circuit.to_pyzx_circuit(), figsize=(10, 12))

    return circuit


def test_greedy_opt(code):
    circuit = test_greedy_bend(code)

    circuit.greedy_reduce_nonlocal_cnot()
    pprint(circuit.ops)
    zx.draw_matplotlib(circuit.to_pyzx_circuit(), figsize=(10, 12))

    return circuit


if __name__ == "__main__":
    test_greedy_opt(steane_code)

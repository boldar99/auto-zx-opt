"""History-dependent choices, terminal decoding, and bounded quantum execution."""

from itertools import product

import stim

from test_sol_logical_measurement import ROOT, run_guppy_check, syndrome
from test_sol_steane_qec import inverse_encoder


def test_zero_flag_dispatcher_decodes_either_order_at_the_end(run_guppy_check):
    singles = {syndrome(1 << q): 1 << q for q in range(32)}
    cases = []

    def add(length, b1=0, b2=1, d1=4, d2=4, f1=0, f2=0, s1=0, s2=0,
            action=5, cx=0, cz=0):
        cases.append(dict(
            args=dict(length=length, b1=b1, b2=b2, d1=d1, d2=d2,
                      f1=f1, f2=f2, s1=s1, s2=s2),
            action=action, cx=cx, cz=cz,
        ))

    # Empty history must choose a first action without inspecting padded slots.
    add(0, b1=-1, b2=7, d1=2, d2=2, f1=2047, f2=2047, s1=2, s2=2, action=0)
    for basis in (0, 1):
        for s in range(64):
            add(1, b1=basis, b2=1-basis, s1=s, action=1 if basis == 0 else 0)
        for first, second in product(range(64), repeat=2):
            accepted = (first == 0 or first in singles) and (second == 0 or second in singles)
            sx, sz = (first, second) if basis == 0 else (second, first)
            add(2, b1=basis, b2=1-basis, s1=first, s2=second,
                action=4 if accepted else 5,
                cx=singles.get(sx, 0) if accepted else 0,
                cz=singles.get(sz, 0) if accepted else 0)
        for i in range(11):
            add(1, b1=basis, f1=1 << i)
            add(2, b1=basis, b2=1-basis, f2=1 << i)
        add(2, b1=basis, b2=1-basis, f1=1)
        add(1, b1=basis, d1=2)
        add(2, b1=basis, b2=1-basis, d2=2)
        add(2, b1=basis, b2=basis)
    for length in (-1, 3, 4):
        add(length)
    for basis in (-1, 2):
        add(1, b1=basis)
        add(2, b2=basis)

    run_guppy_check('''
import json
from datetime import timedelta
from pathlib import Path
import sys
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, comptime, output
from sol_experiment.dispatchers import HISTORY_CAPACITY, zero_flag_dispatcher


@guppy
def entry(length: int, b1: int, b2: int, d1: int, d2: int,
          f1: int, f2: int, s1: int, s2: int) -> None:
    history = array((0, 0, 0, 0) for _ in range(comptime(HISTORY_CAPACITY)))
    history[0] = (b1, d1, f1, s1)
    history[1] = (b2, d2, f2, s2)
    decision = zero_flag_dispatcher(history, length)
    output("action", decision.action)
    output("cx", decision.correction_x)
    output("cz", decision.correction_z)
    _, _, _, saved_first = history[0]
    _, _, _, saved_second = history[1]
    output("s1", saved_first)
    output("s2", saved_second)


cases = json.loads(Path(sys.argv[1]).read_text())
shots = (entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=1)
         .stabilizer_sim().with_timeout(timedelta(seconds=20))
         .run_per_shot([case["args"] for case in cases]).collated_shots())
assert len(shots) == len(cases)
for shot, case in zip(shots, cases):
    assert shot == {"action": [case["action"]], "cx": [case["cx"]], "cz": [case["cz"]],
                    "s1": [case["args"]["s1"]], "s2": [case["args"]["s2"]]}, (case, shot)
''', cases)


def _extraction_record(distance, flags, checks):
    """Invert the independent Stim detector map for a specified observation."""
    circuit = stim.Circuit.from_file(
        ROOT / f"spiderwarp/assets/circuits/SECircuits/32_20_4_d{distance}.stim"
    )
    detectors = [tuple(t.value for t in op.targets_copy())
                 for op in circuit if op.name == "DETECTOR"]
    pivots = {}
    for bit in range(circuit.num_measurements):
        vector = sum((group.count(bit - circuit.num_measurements) % 2) << i
                     for i, group in enumerate(detectors))
        word = 1 << bit
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector, word
                break
            vector ^= pivots[pivot][0]
            word ^= pivots[pivot][1]
    target = flags | (checks << (11 if distance == 4 else 6))
    word = 0
    while target:
        vector, contribution = pivots[target.bit_length() - 1]
        target ^= vector
        word ^= contribution
    return [bool(word & (1 << bit)) for bit in range(circuit.num_measurements)]


def test_runner_retains_history_dispatches_all_actions_and_defers_correction(run_guppy_check):
    paths = [
        [(0, 4, 1, 63), (1, 2, 3, 31)],
        [(0, 4, 0, 2), (0, 2, 5, 17), (1, 4, 4, 31)],
    ]
    records = [sum((_extraction_record(d, f, s) for _, d, f, s in path), [])
               for path in paths]
    run_guppy_check('''
import json
from datetime import timedelta
from pathlib import Path
import sys
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, comptime, output
from guppylang.std.quantum import qubit, measure, h, cx, x, z
from selene_sim.backends import ClassicalReplay
from sol_experiment.QEC_optimized_steane import (
    make_optimized_steane_circuit, make_optimized_steane_qec_cycle,
)
from sol_experiment.dispatchers import HISTORY_CAPACITY, DispatcherAction, DispatchDecision
from sol_experiment.logical_state_preparation import state_prep_32_20_4_zero_d1

request = json.loads(Path(sys.argv[1]).read_text())
INVERSE = request["inverse"]


@guppy
def history_dispatcher(
    history: array[tuple[int, int, int, int], comptime(HISTORY_CAPACITY)], count: int,
) -> DispatchDecision:
    output("length", count)
    for i in range(count):
        b, d, f, s = history[i]
        output("basis", b)
        output("distance", d)
        output("flags", f)
        output("syndrome", s)
    if count == 0:
        return DispatchDecision(comptime(DispatcherAction.Z4.value), 256, 512)
    _, _, first_flags, _ = history[0]
    if count == 1:
        if first_flags != 0:
            return DispatchDecision(comptime(DispatcherAction.X2.value), 256, 512)
        return DispatchDecision(comptime(DispatcherAction.Z2.value), 256, 512)
    if first_flags == 0 and count == 2:
        return DispatchDecision(comptime(DispatcherAction.X4.value), 256, 512)
    return DispatchDecision(comptime(DispatcherAction.ACCEPT.value), 8, 32)


@guppy
def short_dispatcher(
    history: array[tuple[int, int, int, int], comptime(HISTORY_CAPACITY)], count: int,
) -> DispatchDecision:
    if count == 0:
        return DispatchDecision(comptime(DispatcherAction.X4.value), 256, 512)
    if count == 1:
        return DispatchDecision(comptime(DispatcherAction.Z2.value), 256, 512)
    return DispatchDecision(comptime(DispatcherAction.ACCEPT.value), 8, 32)


@guppy
def never_finish(
    history: array[tuple[int, int, int, int], comptime(HISTORY_CAPACITY)], count: int,
) -> DispatchDecision:
    output("length", count)
    return DispatchDecision(comptime(DispatcherAction.X2.value), 8, 32)


@guppy
def reject(
    history: array[tuple[int, int, int, int], comptime(HISTORY_CAPACITY)], count: int,
) -> DispatchDecision:
    return DispatchDecision(comptime(DispatcherAction.REJECT.value), 8, 32)


@guppy
def accept_immediately(
    history: array[tuple[int, int, int, int], comptime(HISTORY_CAPACITY)], count: int,
) -> DispatchDecision:
    return DispatchDecision(comptime(DispatcherAction.ACCEPT.value), 8, 32)


cycle = make_optimized_steane_qec_cycle(history_dispatcher)
short = make_optimized_steane_qec_cycle(short_dispatcher)
bounded = make_optimized_steane_qec_cycle(never_finish)
rejected = make_optimized_steane_qec_cycle(reject)
immediate = make_optimized_steane_qec_cycle(accept_immediately)


@guppy
def replay_entry() -> None:
    data = array(qubit() for _ in range(32))
    output("accepted", cycle(data))
    output("data", array(measure(q).read() for q in data))


tail = [bool(i % 3) for i in range(32)]
shots = (replay_entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=60)
         .with_simulator(ClassicalReplay(measurements=[r + tail for r in request["records"]]))
         .with_shots(2).with_timeout(timedelta(seconds=20)).run().collated_shots())
for shot, path in zip(shots, request["paths"]):
    prefixes = [record for length in range(1, len(path) + 1) for record in path[:length]]
    assert shot["accepted"] == [1] and shot["data"] == [tail], shot
    assert shot["length"] == list(range(len(path) + 1)), shot
    for field, index in (("basis", 0), ("distance", 1), ("flags", 2), ("syndrome", 3)):
        assert shot[field] == [record[index] for record in prefixes], (field, shot)


@guppy
def bounded_entry() -> None:
    data = array(qubit() for _ in range(32))
    output("accepted", bounded(data))
    output("data", array(measure(q).read() for q in data))


@guppy
def reject_entry() -> None:
    data = array(qubit() for _ in range(32))
    output("accepted", rejected(data))
    output("data", array(measure(q).read() for q in data))


for entry, record, lengths in (
    (bounded_entry, [False] * (33 + 33 + 33 + 32), [0, 1, 2, 3]),
    (reject_entry, [False] * 32, []),
):
    shot = (entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=60)
            .with_simulator(ClassicalReplay(measurements=[record])).run().collated_shots()[0])
    assert shot["accepted"] == [0] and shot.get("length", []) == lengths, shot


@guppy.comptime
def decode(data: array[qubit, 32]) -> None:
    for name, targets in INVERSE:
        if name == "H":
            for q in targets:
                h(data[q])
        else:
            for a, b in zip(targets[::2], targets[1::2]):
                cx(data[a], data[b])


@guppy
def bell(path: int) -> None:
    data = array(qubit() for _ in range(32))
    reference = array(qubit() for _ in range(32))
    state_prep_32_20_4_zero_d1(data)
    state_prep_32_20_4_zero_d1(reference)
    for i in range(32):
        h(data[i])
    for i in range(32):
        cx(data[i], reference[i])
    if path == 0:
        output("accepted", cycle(data))
    elif path == 1:
        output("accepted", short(data))
    else:
        output("accepted", immediate(data))
    # Undo the terminal correction exactly once; extraction correction fields must
    # have had no effect, and both X/Z components must have been applied.
    x(data[3])
    z(data[5])
    for i in range(32):
        cx(data[i], reference[i])
    for i in range(32):
        h(data[i])
    decode(data)
    decode(reference)
    output("data", array(measure(q).read() for q in data))
    output("reference", array(measure(q).read() for q in reference))


shots = (bell.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=92)
         .stabilizer_sim().with_seed(7)
         .run_per_shot([dict(path=p) for p in (0, 1, 2)]).collated_shots())
assert len(shots) == 3
assert all(s["accepted"] == [1] and s["data"] == s["reference"] == [[0] * 32] for s in shots), shots

for distance in (0, 1, 3, None):
    try:
        make_optimized_steane_circuit("X", distance)
    except ValueError:
        pass
    else:
        raise AssertionError(distance)
''', dict(paths=paths, records=records, inverse=inverse_encoder()))

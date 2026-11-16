"""Bounded RUS control flow, failure reporting, and reuse of the data block."""

from test_sol_logical_measurement import run_guppy_check


def test_bounded_rus_stops_at_success_or_retry_limit(run_guppy_check):
    run_guppy_check('''
from datetime import timedelta
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, measure
from selene_sim.backends import ClassicalReplay
from sol_experiment.state_preparation import (
    repeat_until_success, state_prep_32_20_4_zero_d1, state_prep_32_20_4_zero_d3,
)
from sol_experiment.SPAM_experiment import make_spam_experiment, summarize_result


@guppy
def flagged_preparation(data: array[qubit, 32]) -> int:
    state_prep_32_20_4_zero_d1(data)
    a, b = qubit(), qubit()
    flags = 0
    if measure(a).read():
        flags |= 2
    if measure(b).read():
        flags |= 4
    return flags


def make_entry(preparation):
    @guppy
    def entry() -> None:
        data = array(qubit() for _ in range(32))
        flags = preparation(data)
        output("flags", flags)
        output("data", array(measure(q).read() for q in data))
    return entry


# Construct all wrappers before compiling: retry limits must remain independent.
programs = {limit: make_entry(repeat_until_success(flagged_preparation, limit))
            for limit in (0, 1, 4)}
programs["default"] = make_entry(repeat_until_success(flagged_preparation))
for key in (4, 0, "default", 1):
    limit = 4 if key == "default" else key
    # Exercise success on every allowed attempt and exhaustion with a distinct
    # final mask. Replay also checks that no extra measurements are requested.
    sequences = [[2] * failures + [0] for failures in range(limit + 1)]
    sequences.append([2] * limit + [4])
    records = [[bit for mask in masks for bit in (bool(mask & 2), bool(mask & 4))]
               + [False] * 32 for masks in sequences]
    program = programs[key].with_opt_level(OptimizationLevel.Minimal)
    program.compile()
    result = (program.emulator(n_qubits=34)
              .with_simulator(ClassicalReplay(measurements=records))
              .with_shots(len(records)).with_timeout(timedelta(seconds=10)).run())
    shots = result.collated_shots()
    assert [s["flags"][0] for s in shots] == [0] * (limit + 1) + [4], key
    assert all(s["data"] == [[0] * 32] for s in shots)

# A bounded preparation plugs into the existing SPAM factory unchanged.
rus_d3 = repeat_until_success(state_prep_32_20_4_zero_d3, max_retries=1)
program = make_spam_experiment(rus_d3).with_opt_level(OptimizationLevel.Minimal)
failed_attempt = [True] + [False] * 8
records = [failed_attempt + [False] * 9 + [False] * 32,
           failed_attempt * 2 + [False] * 32]
result = (program.emulator(n_qubits=34)
          .with_simulator(ClassicalReplay(measurements=records))
          .with_shots(2).with_timeout(timedelta(seconds=10)).run())
assert result.collated_shots() == [
    {"accepted": [1], "logical_error": [0]},
    {"accepted": [0], "logical_error": [0]},
]
stats = summarize_result(result)
assert stats.ar == 0.5 and stats.ler == 0
''')


def test_bounded_rus_resets_every_data_qubit_before_retry(run_guppy_check):
    run_guppy_check('''
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, measure, cx, x
from sol_experiment.state_preparation import repeat_until_success


@guppy
def reset_sensitive_preparation(data: array[qubit, 32]) -> int:
    # A fresh block always fails and becomes all ones. Any retry on a dirty
    # qubit would instead spuriously succeed, exposing a missing reset.
    dirty = False
    for i in range(32):
        probe = qubit()
        cx(data[i], probe)
        dirty |= measure(probe).read()
        x(data[i])
    output("attempt", 1)
    if dirty:
        return 0
    return 7


def make_entry(limit):
    preparation = repeat_until_success(reset_sensitive_preparation, limit)
    @guppy
    def entry() -> None:
        data = array(qubit() for _ in range(32))
        output("flags", preparation(data))
        output("data", array(measure(q).read() for q in data))
    return entry


programs = {limit: make_entry(limit) for limit in (0, 1, 4)}
for limit, program in programs.items():
    result = (program.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=34)
              .stabilizer_sim().with_shots(2).run())
    for shot in result.collated_shots():
        assert shot["attempt"] == [1] * (limit + 1)
        assert shot["flags"] == [7]
        # The caller receives the last attempted block, without another reset
        # after exhaustion, and still owns it for measurement/disposal.
        assert shot["data"] == [[1] * 32]
''')


def test_bounded_rus_validates_retry_limit(run_guppy_check):
    run_guppy_check('''
from sol_experiment.state_preparation import repeat_until_success, state_prep_32_20_4_zero_d3

for value, expected in ((-1, ValueError), (1.5, TypeError), (True, TypeError),
                        (None, TypeError), ("4", TypeError)):
    try:
        repeat_until_success(state_prep_32_20_4_zero_d3, value)
    except expected:
        pass
    else:
        raise AssertionError((value, expected))
''')

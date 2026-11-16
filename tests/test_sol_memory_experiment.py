"""Memory-cycle counts, trajectory rejection, logical failures and saved rates."""

from test_sol_logical_measurement import run_guppy_check


def test_memory_experiment_runs_real_qec_methods(run_guppy_check):
    run_guppy_check('''
from datetime import timedelta
from guppylang import OptimizationLevel
from sol_experiment.memory_experiment import make_memory_experiment, QEC_METHODS
from sol_experiment.SPAM_experiment import PREPARATION_METHODS, summarize_result

settings = [("d1", "steane_d1", 0), ("d2_RUS", "steane_d3_RUS", 1),
            ("d4_RUS", "steane_d4_RUS", 3)]
programs = [make_memory_experiment(PREPARATION_METHODS[initial], QEC_METHODS[qec], cycles)
            for initial, qec, cycles in settings]
for config, program in zip(settings, programs):
    program = program.with_opt_level(OptimizationLevel.Minimal)
    program.compile()
    result = (program.emulator(n_qubits=66).stabilizer_sim()
              .with_shots(4).with_timeout(timedelta(seconds=20)).run())
    assert result.collated_shots() == [{"accepted": [1], "logical_error": [0]}] * 4, config
    stats = summarize_result(result)
    assert (stats.shots, stats.accepted, stats.logical_errors) == (4, 4, 0)
    assert stats.ar == 1 and stats.ler == 0
''')


def test_memory_cycle_counts_expected_state_and_validation(run_guppy_check):
    run_guppy_check('''
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, x
from sol_experiment.memory_experiment import make_memory_experiment
from sol_experiment.logical_measurement import LOGICAL_Z_MASKS
from sol_experiment.state_preparation import state_prep_32_20_4_zero_d1
from sol_experiment.SPAM_experiment import summarize_result


@guppy
def initial(data: array[qubit, 32]) -> int:
    output("initialization", 1)
    return state_prep_32_20_4_zero_d1(data)


@guppy
def logical_flip(data: array[qubit, 32]) -> bool:
    # This valid logical X word flips logical Z information on every invocation.
    x(data[0])
    x(data[2])
    x(data[8])
    x(data[10])
    output("cycle", 1)
    return True


programs = {cycles: make_memory_experiment(initial, logical_flip, cycles)
            for cycles in (0, 1, 2, 3)}
for cycles in (3, 0, 2, 1):
    result = (programs[cycles].with_opt_level(OptimizationLevel.Minimal)
              .emulator(n_qubits=34).stabilizer_sim().with_shots(2).run())
    for shot in result.collated_shots():
        assert shot["initialization"] == [1]
        assert shot.get("cycle", []) == [1] * cycles
        assert shot["accepted"] == [1]
        assert shot["logical_error"] == [cycles % 2]
    assert summarize_result(result).ler == cycles % 2

# Support an explicit nonzero expected computational-basis target.
expected = sum(((0x505 & mask).bit_count() % 2) << i
               for i, mask in enumerate(LOGICAL_Z_MASKS))
assert expected != 0
program = make_memory_experiment(initial, logical_flip, 1, expected_logicals=expected)
result = (program.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=34)
          .stabilizer_sim().with_shots(2).run())
assert summarize_result(result).ler == 0

for cycles, expected_mask, exception in (
    (-1, 0, ValueError), (True, 0, TypeError), (1.5, 0, TypeError),
    (0, -1, ValueError), (0, 1 << 20, ValueError), (0, True, TypeError),
):
    try:
        make_memory_experiment(initial, logical_flip, cycles, expected_logicals=expected_mask)
    except exception:
        pass
    else:
        raise AssertionError((cycles, expected_mask, exception))
''')


def test_memory_rejects_whole_trajectories_and_saves_rates(run_guppy_check):
    run_guppy_check('''
from datetime import timedelta
import json
from pathlib import Path
import sys
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, measure
from selene_sim.backends import ClassicalReplay
from sol_experiment.memory_experiment import make_memory_experiment, save_results
from sol_experiment.state_preparation import state_prep_32_20_4_zero_d1
from sol_experiment.SPAM_experiment import PREPARATION_METHODS, summarize_result


@guppy
def controlled_qec(data: array[qubit, 32]) -> bool:
    probe = qubit()
    raised = measure(probe).read()
    output("cycle", 1)
    return not raised


@guppy
def failed_initial(data: array[qubit, 32]) -> int:
    return 7


def replay(program, records):
    return (program.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=66)
            .with_simulator(ClassicalReplay(measurements=records)).with_shots(len(records))
            .with_timeout(timedelta(seconds=10)).run())


program = make_memory_experiment(state_prep_32_20_4_zero_d1, controlled_qec, 3)
logical_word = [bool(0x505 & (1 << q)) for q in range(32)]
records = [
    [False] * 3 + [False] * 32,           # complete success
    [True],                             # first QEC cycle rejects
    [False, False, True],                # final QEC cycle rejects
    [False] * 3 + [True, True] + [False] * 30,  # terminal syndrome rejects
    [False] * 3 + logical_word,          # accepted logical failure
]
result = replay(program, records)
shots = result.collated_shots()
assert [s["accepted"] for s in shots] == [[1], [0], [0], [0], [1]]
assert [s["logical_error"] for s in shots] == [[0], [0], [0], [0], [1]]
assert [len(s["cycle"]) for s in shots] == [3, 1, 3, 3, 3]
stats = summarize_result(result)
assert (stats.shots, stats.accepted, stats.logical_errors) == (5, 2, 1)
assert stats.ar == 0.4 and stats.ler == 0.5

# Initialization rejection requests neither QEC nor terminal readout results.
program = make_memory_experiment(failed_initial, controlled_qec, 3)
result = replay(program, [[]])
assert result.collated_shots() == [{"accepted": [0], "logical_error": [0]}]
rejected_stats = summarize_result(result)
assert rejected_stats.ar == 0 and rejected_stats.ler is None

# A bounded RUS initial preparation can also exhaust before the memory begins.
program = make_memory_experiment(PREPARATION_METHODS["d3_RUS"], controlled_qec, 3)
result = replay(program, [([True] + [False] * 8) * 5])
assert result.collated_shots() == [{"accepted": [0], "logical_error": [0]}]

configuration = {"initial_state_prep": "d1", "qec_method": "controlled",
                 "num_cycles": 3, "seed": 7}
path = Path(sys.argv[1]).parent / "reports" / "nested" / "memory.json"
saved = save_results(path, stats, configuration)
assert saved.is_absolute()
record = json.loads(saved.read_text())
assert record["configuration"] == configuration
assert (record["shots"], record["accepted"], record["logical_errors"]) == (5, 2, 1)
assert record["AR"] == 0.4 and record["LER"] == 0.5
assert record["experiment"] == "memory" and record["created_at_utc"].endswith("+00:00")
save_results(path, rejected_stats, configuration)
record = json.loads(path.read_text())
assert record["AR"] == 0 and record["LER"] is None
assert record["logical_errors"] == record["accepted"] == 0
''')

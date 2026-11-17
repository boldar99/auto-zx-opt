"""Reference-code checks and compiled, destructive readout with injected errors."""

from itertools import combinations
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = json.loads((ROOT / "spiderwarp/assets/qeccs/misc/32_20_4.json").read_text())
CHECKS = [sum(b << q for q, b in enumerate(row)) for row in CODE["H_x"]]
LOGICALS = [sum(b << q for q, b in enumerate(row)) for row in CODE["L_x"]]
SINGLES = [1 << q for q in range(32)]
DOUBLES = [(1 << a) | (1 << b) for a, b in combinations(range(32), 2)]


def parities(word, masks):
    return [(word & mask).bit_count() % 2 for mask in masks]


def syndrome(word):
    return sum(b << i for i, b in enumerate(parities(word, CHECKS)))


@pytest.fixture
def run_guppy_check(tmp_path):
    python = Path(os.environ.get(
        "SPIDERWARP_SOL_PYTHON", ROOT / "local_sol_example/.venv/bin/python"
    ))
    if not python.is_file():
        pytest.skip("Separately installed Guppy 1.x environment unavailable")

    def run(source, payload=None):
        runner = tmp_path / "check_readout.py"
        runner.write_text(source)
        request = tmp_path / "cases.json"
        request.write_text(json.dumps(payload))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [str(python.absolute()), str(runner), str(request)], cwd=ROOT,
            env=env, text=True, capture_output=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    return run


def test_tables_match_reference_and_reject_ambiguous_cosets(run_guppy_check):
    tables = json.loads(run_guppy_check('''
import json
from sol_experiment.decoding import (
    Z_CHECK_MASKS, LOGICAL_Z_MASKS, CORRECTIONS, POST_SELECT,
)
print(json.dumps([Z_CHECK_MASKS, LOGICAL_Z_MASKS, CORRECTIONS, POST_SELECT]))
'''))
    checks, logicals, corrections, rejected = tables
    assert checks == CHECKS
    assert logicals == LOGICALS

    # Independent unique-logical-class policy over all errors through weight 2.
    classes = [set() for _ in range(64)]
    for error in [0] + SINGLES + DOUBLES:
        classes[syndrome(error)].add(tuple(parities(error, LOGICALS)))
    assert rejected == [len(group) != 1 for group in classes]
    assert sum(not value for value in rejected) == 33
    for s, correction in enumerate(corrections):
        if rejected[s]:
            assert correction == 0
        else:
            assert correction.bit_count() <= 1
            assert syndrome(correction) == s
            assert tuple(parities(correction, LOGICALS)) in classes[s]


def test_compiled_decoder_and_transversal_measurement(run_guppy_check):
    clean_words = [0] + CHECKS + LOGICALS
    assert all(syndrome(word) == 0 for word in clean_words)
    classical = [(word, error) for word in clean_words for error in [0] + SINGLES]
    classical += [(0, error) for error in DOUBLES]
    quantum = [(0, error) for error in [0] + SINGLES + DOUBLES]
    quantum += [(word, error) for word in LOGICALS for error in (0, 1, 1 << 31)]

    def case(word, error):
        accepted = error.bit_count() <= 1
        return dict(
            word=word, error=error,
            raw=[bool((word ^ error) & (1 << q)) for q in range(32)],
            accepted=int(accepted), syndrome=syndrome(word ^ error),
            correction=error if accepted else 0,
            logicals=parities(word, LOGICALS) if accepted else [0] * 20,
        )

    run_guppy_check('''
from datetime import timedelta
import json
from pathlib import Path
import sys

from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, x
from selene_sim.backends import ClassicalReplay
from sol_experiment.decoding import decode_z_readout
from sol_experiment.logical_measurement import logical_z_meas
from sol_experiment.logical_state_preparation import state_prep_32_20_4_zero_d3
from sol_experiment.experiment_SPAM import make_spam_experiment, summarize_result


@guppy
def decode_entry(raw: array[bool, 32]) -> None:
    result = decode_z_readout(raw)
    output("accepted", result.accepted)
    output("syndrome", result.syndrome)
    output("correction", result.correction)
    output("logicals", result.logicals)
    output("raw", raw)


@guppy
def measure_entry(word: int, error: int) -> None:
    data = array(qubit() for _ in range(32))
    flags = state_prep_32_20_4_zero_d3(data)
    # Each reference normalizer word prepares a valid, nonzero logical state.
    # X errors immediately before Z measurement model readout-bit flips.
    for q in range(32):
        if ((word ^ error) & (1 << q)) != 0:
            x(data[q])
    result = logical_z_meas(data)
    output("flags", flags)
    output("accepted", result.accepted)
    output("syndrome", result.syndrome)
    output("correction", result.correction)
    output("logicals", result.logicals)


cases = json.loads(Path(sys.argv[1]).read_text())
for mode, entry in (("classical", decode_entry), ("quantum", measure_entry)):
    records = cases[mode]
    args = ([dict(raw=record["raw"]) for record in records] if mode == "classical"
            else [dict(word=record["word"], error=record["error"]) for record in records])
    emulator = (entry.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=34)
                .stabilizer_sim().with_seed(7).with_timeout(timedelta(seconds=30)))
    result = emulator.run_per_shot(args)
    shots = result.collated_shots()
    assert len(shots) == len(records)
    for shot, expected in zip(shots, records):
        for key in ("accepted", "syndrome", "correction", "logicals"):
            assert shot[key] == [expected[key]], (mode, key, expected, shot)
        if mode == "classical":
            assert shot["raw"] == [expected["raw"]]  # decoding preserves the input
        else:
            assert shot["flags"] == [0]
    print(mode, len(records), "cases passed", flush=True)

# The example's acceptance must combine preparation and readout acceptance.
experiment = make_spam_experiment(state_prep_32_20_4_zero_d3)
example = experiment.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=34)
result = example.stabilizer_sim().with_shots(4).run()
for shot in result.collated_shots():
    assert shot == {"accepted": [1], "logical_error": [0]}
stats = summarize_result(result)
assert (stats.shots, stats.accepted, stats.logical_errors) == (4, 4, 0)
assert stats.ar == 1 and stats.ler == 0

# Independently force preparation failure, readout failure, then both failures.
raised_flags = [True] + [False] * 8
bad_readout = [True, True] + [False] * 30
records = [raised_flags + [False] * 32, [False] * 9 + bad_readout,
           raised_flags + bad_readout]
result = (example.with_simulator(ClassicalReplay(measurements=records))
          .with_shots(3).with_timeout(timedelta(seconds=10)).run())
shots = result.collated_shots()
assert all(s == {"accepted": [0], "logical_error": [0]} for s in shots)
assert summarize_result(result).ar == 0
assert summarize_result(result).ler is None

# An accepted nonzero logical word is one block failure, regardless of how
# many of its logical bits are one. The same word with a prep flag is rejected.
logical_bits = [bool(cases["logical_error_word"] & (1 << q)) for q in range(32)]
records = [[False] * 41, [False] * 9 + logical_bits,
           raised_flags + logical_bits, [False] * 9 + bad_readout]
result = (example.with_simulator(ClassicalReplay(measurements=records))
          .with_shots(4).with_timeout(timedelta(seconds=10)).run())
assert result.collated_shots() == [
    {"accepted": [1], "logical_error": [0]},
    {"accepted": [1], "logical_error": [1]},
    {"accepted": [0], "logical_error": [0]},
    {"accepted": [0], "logical_error": [0]},
]
stats = summarize_result(result)
assert (stats.shots, stats.accepted, stats.logical_errors) == (4, 2, 1)
assert stats.ar == stats.ler == 0.5
''', dict(classical=[case(*pair) for pair in classical],
          quantum=[case(*pair) for pair in quantum], logical_error_word=LOGICALS[0]))


def test_spam_statistics_and_empty_acceptance(run_guppy_check):
    run_guppy_check('''
from contextlib import redirect_stdout
from io import StringIO
from hugr.qsystem.result import QsysResult
from sol_experiment.experiment_SPAM import summarize_result, print_summary

# Rejected shots never enter the LER numerator, even if an external result
# supplies an error bit for them.
result = QsysResult([
    [("accepted", 1), ("logical_error", 0)],
    [("accepted", 1), ("logical_error", 1)],
    [("accepted", 0), ("logical_error", 1)],
    [("accepted", 0), ("logical_error", 0)],
])
stats = summarize_result(result)
assert (stats.shots, stats.accepted, stats.logical_errors) == (4, 2, 1)
assert stats.ar == stats.ler == 0.5
text = StringIO()
with redirect_stdout(text):
    print_summary(stats)
assert text.getvalue().splitlines() == [
    "AR:  50.000000% (2/4 shots accepted)",
    "LER: 50.000000% (1/2 accepted blocks have a logical error)",
]

for result in (QsysResult(), QsysResult([[("accepted", 0), ("logical_error", 0)]])):
    stats = summarize_result(result)
    assert stats.accepted == stats.logical_errors == 0
    assert stats.ler is None
    assert stats.ar == (0 if stats.shots else None)
    text = StringIO()
    with redirect_stdout(text):
        print_summary(stats)
    assert "LER: undefined (no accepted shots)" in text.getvalue()
''')


def test_spam_method_factories_bind_distinct_preparations(run_guppy_check):
    run_guppy_check('''
from datetime import timedelta
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit
from selene_sim.backends import ClassicalReplay
from sol_experiment.experiment_SPAM import PREPARATION_METHODS, make_spam_experiment
from sol_experiment.logical_state_preparation import state_prep_32_20_4_zero_d1

# Build every program before compiling any of them. Each must retain its own
# preparation, even though the returned functions share the name experiment.
programs = {name: make_spam_experiment(method)
            for name, method in PREPARATION_METHODS.items()}
flag_counts = {"d1": 0, "d2": 6, "d3": 9, "d4": 11}
for name in (*reversed(tuple(PREPARATION_METHODS)), "d4"):
    program = programs[name].with_opt_level(OptimizationLevel.Minimal)
    program.compile()  # Requires an executable entrypoint with no arguments.
    emulator = program.emulator(n_qubits=34).with_timeout(timedelta(seconds=10))
    result = emulator.stabilizer_sim().with_shots(2).run()
    assert result.collated_shots() == [{"accepted": [1], "logical_error": [0]}] * 2

    # Replay requires exactly this many measurements. This catches accidentally
    # compiling a different preparation, which ideal acceptance alone cannot.
    base = name.split("_")[0]
    count = flag_counts[base]
    records = [[False] * (count + 32)]
    expected = [{"accepted": [1], "logical_error": [0]}]
    if count:
        failed_attempt = [False] * (count - 1) + [True]
        attempts = 5 if name.endswith("_RUS") else 1
        records.append(failed_attempt * attempts + [False] * 32)
        expected.append({"accepted": [0], "logical_error": [0]})
        if attempts > 1:
            records.append(failed_attempt * (attempts - 1) + [False] * (count + 32))
            expected.append({"accepted": [1], "logical_error": [0]})
    result = (emulator.with_simulator(ClassicalReplay(measurements=records))
              .with_shots(len(records)).run())
    assert result.collated_shots() == expected, name

# The factory also accepts a custom regular @guppy function, without adding
# any runtime method selector or changing the shared SPAM measurement logic.
@guppy
def custom_preparation(data: array[qubit, 32]) -> int:
    state_prep_32_20_4_zero_d1(data)
    return 1

custom = make_spam_experiment(custom_preparation)
result = (custom.with_opt_level(OptimizationLevel.Minimal).emulator(n_qubits=34)
          .stabilizer_sim().with_shots(2).run())
assert result.collated_shots() == [{"accepted": [0], "logical_error": [0]}] * 2
''')

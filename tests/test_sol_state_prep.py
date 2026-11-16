"""Compilation, encoded-state preservation, and preparation retry ownership."""

import os
from pathlib import Path
import subprocess

import pytest
import stim

from spiderwarp.utils import load_state_prep_circuit


@pytest.mark.parametrize("distance", range(1, 5))
def test_operation_descriptions_match_reference(distance):
    pytest.importorskip("guppylang", minversion="1")
    from sol_experiment import state_preparation as state_prep

    circuit = stim.Circuit()
    for name, targets in getattr(
        state_prep, f"state_prep_ops_32_20_4_zero_d{distance}"
    )():
        circuit.append(name, targets)
    assert circuit == load_state_prep_circuit("misc", f"32_20_4_zero_d{distance}")


@pytest.mark.parametrize("name", [
    *(f"state_prep_32_20_4_zero_d{d}" for d in range(1, 5)),
    "state_prep_32_20_4_zero", "prepare_zero_rus",
])
def test_preparation_compiles(name):
    pytest.importorskip("guppylang", minversion="1")
    from sol_experiment import state_preparation as state_prep

    getattr(state_prep, name).compile_function()


def test_compiled_states_flags_and_retry_in_sol_environment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    python = Path(os.environ.get(
        "SPIDERWARP_SOL_PYTHON", root / "local_sol_example/.venv/bin/python"
    ))
    if not python.is_file():
        pytest.skip("Separately installed Guppy 1.x Sol environment unavailable")

    # Decode with the inverse of the independent reference encoder. Measuring
    # all-zero afterwards checks the full pure state, including relative phases.
    encoder = stim.Circuit()
    for op in load_state_prep_circuit("misc", "32_20_4_zero_d1"):
        if op.name == "RX":
            encoder.append("H", op.targets_copy())
        elif op.name != "R":
            encoder.append(op)
    inverse = [(op.name, [q.value for q in op.targets_copy()]) for op in encoder.inverse()]
    runner = tmp_path / "check_state_prep.py"
    runner.write_text(f"INVERSE = {inverse!r}\n" + '''
from guppylang import guppy, OptimizationLevel
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit, h, cx, measure
from selene_sim.backends import ClassicalReplay
from datetime import timedelta
from sol_experiment.state_preparation import (
    state_prep_32_20_4_zero, prepare_zero_rus,
)


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
def roundtrip(distance: int, rus: bool) -> None:
    if rus:
        data = prepare_zero_rus(distance)
        flags = 0
    else:
        data = array(qubit() for _ in range(32))
        flags = state_prep_32_20_4_zero(data, distance)
    decode(data)
    bits = array(measure(q).read() for q in data)
    output("flags", flags)
    output("decoded", bits)


emulator = (
    roundtrip.with_opt_level(OptimizationLevel.Minimal)
    .emulator(n_qubits=34).with_seed(7).with_timeout(timedelta(seconds=10))
)
for distance in range(1, 5):
    for rus in (False, True):
        print("Checking state", distance, rus, flush=True)
        result = emulator.stabilizer_sim().with_shots(8).run(distance=distance, rus=rus)
        for shot in result.collated_shots():
            assert shot["flags"] == [0], (distance, rus, shot)
            assert shot["decoded"] == [[0] * 32], (distance, rus, shot)

# Replay distinguishes every output bit, even though ideal flags are all zero.
records = [[i == bit for i in range(11)] + [False] * 32 for bit in range(11)]
print("Checking measurement bit order", flush=True)
result = emulator.with_simulator(ClassicalReplay(measurements=records)).with_shots(11).run(
    distance=4, rus=False,
)
assert [s["flags"][0] for s in result.collated_shots()] == [1 << i for i in range(11)]

# Force a rejection then an acceptance. Exactly twelve verification results
# and the terminal readout must be consumed, within the same 34-qubit limit.
records = [[True] + [False] * 5 + [False] * 6 + [False] * 32]
print("Checking forced retry", flush=True)
result = emulator.with_simulator(ClassicalReplay(measurements=records)).with_shots(1).run(
    distance=2, rus=True,
)
assert result.collated_shots()[0]["decoded"] == [[0] * 32]
print("Verified all four states, measurement bit order, and a forced RUS retry")
''')
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [str(python.absolute()), str(runner)], cwd=root, env=env,
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

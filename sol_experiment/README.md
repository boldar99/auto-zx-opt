# Sol experiments

Guppy implementations of state preparation, syndrome extraction, decoding, and
memory experiments for the **[[32, 20, 4]] CSS code**. A block contains 32 physical
data qubits encoding 20 logical qubits. The experiment entry points compile a
Guppy program to HUGR and simulate it on Sol through Selene/Narsil with the
Helios alpha noise model.

## Running experiments

This folder is an independent uv project. [`pyproject.toml`](pyproject.toml)
declares its dependencies and [`uv.lock`](uv.lock) records the resolved versions.
Use Python 3.12 or newer and run the commands below from the **repository root**.

The runtime dependency is Quantinuum's
[local-sol-example](https://github.com/Quantinuum/local-sol-example), distributed
as `local-sol` and imported as `sol.sim`. uv installs it directly from GitHub at
the revision declared in `pyproject.toml`, together with its Sol runtime packages.
Guppy and Stim are also declared directly; pytest is in the default `dev` group.

Before syncing, configure GitHub SSH access to that repository and the Quantinuum
JFrog credentials described in its
[setup instructions](https://github.com/Quantinuum/local-sol-example#setting-up-the-environment).
The authenticated package index is configured in this project's `pyproject.toml`;
credentials belong in your local authentication configuration.

```sh
uv sync --project sol_experiment --locked
```

This creates `sol_experiment/.venv` and installs the experiment package in editable
mode, including its circuit/code assets. From inside the `sol_experiment` folder,
the equivalent setup command is `uv sync --locked`.

After changing dependencies in `pyproject.toml`, run `uv lock --project
sol_experiment` and commit both project files. The commands below use the existing
lockfile with `--frozen`.

Compare preparation methods with a SPAM (state preparation and measurement)
experiment. Each method gets the requested number of shots:

```sh
uv run --project sol_experiment --frozen python -m sol_experiment.experiment_SPAM \
    --methods d1 d2_RUS d3_RUS d4_RUS --shots 100 --processes 4 --seed 7
```

Run one memory cycle with d3 RUS initialization and d3 RUS Steane ancillas:

```sh
uv run --project sol_experiment --frozen python -m sol_experiment.experiment_memory \
    --initial-prep d3_RUS --qec-method steane_d3_RUS \
    --num-cycles 1 --basis Z --shots 100 --processes 4 --seed 7 \
    --output out/memory/steane_d3_rus_Z_1cycle.json
```

Use `--qec-method optimized_steane` for the optimized extraction circuits,
`--basis X` for a logical-plus memory experiment, or `--num-cycles 0` for an
initialization/readout baseline. Both entry points support `--help`.

The memory CLI defaults are `d3_RUS` initialization, `steane_d3_RUS` QEC,
**10 cycles**, basis `Z`, 2,500 shots, and eight processes. The SPAM CLI defaults
to methods `d1 d2_RUS d3_RUS d4_RUS`, 1,000 shots per method, and eight processes.

## Module layout

| File | Responsibility |
| --- | --- |
| [circuit_descriptions.py](circuit_descriptions.py) | Shared plain Python preparation/extraction descriptions and detector groups, with no Guppy or Stim imports. |
| [logical_state_preparation.py](logical_state_preparation.py) | Encoded-zero Guppy preparations, bounded RUS wrappers, and `PREPARATION_METHODS`. |
| [QEC_steane.py](QEC_steane.py) | Conventional Steane extraction using a supplied encoded-zero ancilla preparation; factories for a single basis or a complete cycle. |
| [QEC_optimized_steane.py](QEC_optimized_steane.py) | Raw optimized d4/d2 extraction circuits, detector metadata, and the dispatcher-driven cycle runner. |
| [optimized_steane_dispatchers.py](optimized_steane_dispatchers.py) | Action names, dispatcher decisions, history capacity, and the default zero-flag policy. |
| [optimized_steane_fault_catalogue.py](optimized_steane_fault_catalogue.py) | Offline single-fault enumeration using the shared descriptions and Stim FlipSimulator, with compact Python/JSON records. |
| [decoding.py](decoding.py) | Check/logical masks, lookup tables, decoder result types, and classical syndrome/logical-readout decoding. |
| [logical_measurement.py](logical_measurement.py) | Destructive transversal Z measurement of a data block, followed by classical decoding. |
| [circuit_utils.py](circuit_utils.py) | Compile-time emission of operation lists, including ancilla allocation, reset, and measurement lifetimes. |
| [measurement_utils.py](measurement_utils.py) | Compile-time helpers for packing bits and detector parities into integer masks. |
| [experiment_SPAM.py](experiment_SPAM.py) | Preparation/readout experiments and shared acceptance/logical-error statistics. |
| [experiment_memory.py](experiment_memory.py) | Initialization, repeated QEC, terminal readout, CLI method registration, and JSON result output. |

Ordinary Python functions describe circuits and bind factories before compilation.
`@guppy.comptime` helpers emit gates and classical expressions into the program.
Outcome-dependent retries, acceptance decisions, and corrections execute in Guppy
at runtime. The quantum program does not interpret Python operation lists.

## State preparation and retries

Every preparation borrows a fresh `array[qubit, 32]`, prepares encoded zero, and
returns an integer flag mask. Zero means successful verification. The caller
retains the data block on either outcome.

| Single-attempt method | Verification bits | Peak live qubits, including the prepared block |
| --- | ---: | ---: |
| `d1` | 0 | 32 |
| `d2` | 6 | 34 |
| `d3` | 9 | 34 |
| `d4` | 11 | 34 |

The labels describe preparation-circuit distances; the underlying code is always
[[32, 20, 4]]. `d1` has no verification and always returns zero flags.

`PREPARATION_METHODS` also contains `d2_RUS`, `d3_RUS`, and `d4_RUS`.
`repeat_until_success(prepare_state, max_retries=4)` makes one initial attempt
and up to four retries, resetting all 32 block qubits between failures. It stops
at the first zero flag mask; a nonzero return means all five attempts failed.
These attempts belong to one experimental shot.

## QEC and decoding

The conventional cycle and default optimized dispatcher run **Z checks followed
by X checks**. Z checks correct X errors, and X checks correct Z errors. Accepted
corrections are applied as physical Pauli gates to the borrowed data block.

The conventional implementation allocates a verified 32-qubit ancilla for each
basis. For Z checks it converts the ancilla to encoded plus, couples data to
ancilla, and measures the ancilla in Z. For X checks it couples ancilla to data
and measures the ancilla in X. The first ancilla is released before the second
is prepared; the supplied verified preparations need at most 66 live qubits
including the persistent data. Only ancilla preparation is retried. A rejected
syndrome ends the trajectory.

The optimized implementation offers four raw actions: `Z4`, `X4`, `Z2`, and `X2`.
The d4/d2 circuits require at most 60/51 live qubits including data. A Z-check
variant conjugates the X-check extraction with transversal H on data and restores
the data basis before returning. `make_optimized_steane_circuit(basis, distance)`
returns only `(flags, syndrome)`, leaving acceptance and decoding to a dispatcher.

`make_optimized_steane_qec_cycle(dispatcher_function=zero_flag_dispatcher)` binds
the classical policy before compilation. The runner calls
`dispatcher_function(history, history_length)` with length zero to choose the
first action, then again after each extraction:

- Each history entry is `(basis, distance, flags, syndrome)`, with `Z_BASIS=0`
  and `X_BASIS=1`.
- History is a three-entry Guppy array; only the first `history_length` entries
  are valid. It starts empty for every cycle and must be preserved by the
  dispatcher. Experimental runtime tuple lists do not compile in this stack.
- The dispatcher returns `DispatchDecision(action, correction_x, correction_z)`.
  Use the `DispatcherAction` members `Z4`, `X4`, `Z2`, or `X2` to request an
  extraction, embedded with `comptime(DispatcherAction.X4.value)` and so on.
  `DispatcherAction.REJECT` ends the cycle; `DispatcherAction.ACCEPT` applies both
  correction masks once and ends successfully. Correction fields on other
  actions are ignored. Both terminal actions also work on an empty history.
- `HISTORY_CAPACITY = 3` in `dispatchers.py` fixes the history storage for all
  dispatchers. After three extractions, a terminal decision is still honored;
  requesting another extraction rejects without accessing beyond the array.

The default zero-flag dispatcher chooses `Z4` on an empty history and rejects
raised flags immediately. It accepts either `Z4` or `X4` as the first recorded
extraction and requests the dual d4 extraction. After the pair, it decodes both
stored syndromes and rejects if either is ambiguous. Z-check syndromes supply
the X correction and X-check syndromes the Z correction, in either order. No
syndrome correction is applied before final acceptance. Custom dispatchers can
choose a different first action, path, or correction rule; the runner adds no
flag/syndrome rejection policy of its own.

`DispatcherAction` is a Python `IntEnum` embedded as integer constants: installed Guppy
1.1.4 supports `@guppy.enum` constructors but not variant matching or equality.

The shared decoder in `decoding.py` accepts the zero syndrome and the 32 distinct
single-bit syndromes, returning a zero- or one-qubit correction. It rejects the
remaining 31 ambiguous syndromes. This corrects every single flipped readout bit
and rejects every double flip; higher-weight errors can share accepted syndromes.
These are syndrome/readout guarantees, not a guarantee of correcting every
fault throughout a noisy QEC circuit.

`decode_syndrome` takes six packed check bits. `decode_readout_syndrome` computes
those checks from 32 raw bits without decoding logical observables.
`decode_z_readout` additionally applies the classical correction and returns
20 logical outcomes. The self-dual code has identical X/Z check supports, so the
syndrome decoder serves both extraction bases. Bit `i` of an integer mask denotes
qubit `i`, check `i`, or flag `i`, according to the mask's role.

## Offline single-fault catalogue

[`optimized_steane_fault_catalogue.py`](optimized_steane_fault_catalogue.py) is
specialized for the four [[32, 20, 4]] extractions. It uses the same Python circuit
descriptions as the Guppy emitters and tracks errors with `stim.FlipSimulator`.
The local Stim assets serve as independent test references. Guppy and SpiderWarp
are not imported by the catalogue generator.

Run from the repository root after syncing the project:

```sh
uv run --project sol_experiment --frozen python -m sol_experiment.optimized_steane_fault_catalogue \
    --output out/optimized_steane_fault_catalogue.json
```

The default command generates all four catalogues. Use `--bases Z --distances 4`
for just `Z_d4`. The Python API exposes the records and reconstructs operations
when needed for inspecting a fault:

```python
from sol_experiment.optimized_steane_fault_catalogue import (
    build_fault_catalogue, extraction_operations, load_catalogues,
)

catalogue = build_fault_catalogue("Z", 4)
fault = catalogue.faults[0]
operations, detectors = extraction_operations("Z", 4)
operation = operations[fault.operation_index]
compatible = catalogue.group_by_observation()  # (flags, syndrome) -> list of faults
saved = load_catalogues("out/optimized_steane_fault_catalogue.json")
```

Each `FaultResponse` has just six fields:

- `operation_index` and `pauli`: the zero-based elementary gate location and
  ordered Pauli factors, including any identity factor of a two-qubit fault.
  Measurement faults occur before M; all other faults occur after the operation.
- `flags` and `syndrome`: detector-outcome changes, in detector-row order.
- `data_x` and `data_z`: the physical output Pauli on the data, with bit `q`
  identifying qubit `q`. Phases are ignored and there is no stabilizer reduction.

The model includes all 15 nonidentity two-qubit Pauli alternatives, X/Y/Z faults
after one-qubit gates, reset flips, and readout flips. `RX` and `MX` are expanded
into separate R/H and H/M operations, matching the Guppy emitter. The Z variants
include their initial/final data Hadamards as fault locations. Idle, transport,
and leakage faults and probability weights are outside this catalogue.

| Catalogue | Single-fault alternatives |
| --- | ---: |
| `X_d2` | 2,010 |
| `X_d4` | 2,584 |
| `Z_d2` | 2,202 |
| `Z_d4` | 2,776 |

Duplicate and invisible fault responses are retained. The no-fault case is the
all-zero response and is not included among the single-fault rows. No incoming
errors, history-dependent corrections, or acceptance decisions are added here.

The saved JSON simply maps action names such as `Z_d4` to their fault records.
Operations are reconstructed from the shared descriptions. Regenerate the file
after editing a description. A later history/decision-table builder can consume
the Python records or JSON and embed integer tables at Guppy compile time.

## Acceptance and logical errors

Each shot emits `accepted` and `logical_error`. A memory shot accepts only if
initialization succeeds, every requested QEC cycle accepts, and terminal readout
accepts. Rejection skips the remaining cycles and readout, but the shot remains
in the attempted-shot count. SPAM acceptance combines preparation verification
with decoded terminal-readout acceptance.

- **AR** = accepted shots / attempted shots.
- **LER** = accepted shots with any wrong logical bit / accepted shots.

LER is an endpoint error probability for the entire 20-logical-qubit block over
the requested trajectory. It is not a per-qubit or per-cycle rate. Rejected shots
contribute neither logical errors nor accepted shots to LER. LER is undefined
when no shots accept.

SPAM and Z-basis memory target all logical zeros. X-basis memory applies
transversal H after initialization and before terminal Z measurement, testing
the all-plus logical state through the same readout code.

## Results and simulation settings

Both CLIs print AR and conditional block LER. The memory CLI also saves JSON to
`out/memory/` relative to the working directory, or to `--output`. Reports include
the configuration, seed, counts (`shots`, `accepted`, `logical_errors`), and
fractional rates (`AR`, `LER`). Undefined rates are written as `null`. SPAM prints
its results without saving a JSON report.

The runner is `sol.sim` from the installed
[local-sol-example](https://github.com/Quantinuum/local-sol-example) dependency.
Initialization, QEC operations, and terminal measurement all use
`qsystem_model_alpha`; both CLIs select Stim and set `coherent_dephasing=False`.
The memory runner currently compiles with `OptimizationLevel.Default` and uses
`n_qubits=66`; SPAM uses `OptimizationLevel.Minimal` and `n_qubits=34`.
The runtime's ion-pool size affects transport with qubit reuse; these arguments
also restrict that pool, independently of the circuit's peak live-qubit count.

Memory passes `leakage_repumping=True` to the wrapper, which sets the noise-model
option `leak2depolar=True`. That wrapper argument does not itself enable Narsil
repumping: the runtime is constructed separately with its default settings.
SPAM leaves this wrapper argument at its default `False`.

These experiments include noisy preparation and terminal readout/postselection.
When comparing with older SpiderWarp benchmarks, also match the noise model,
experiment boundaries, compilation settings, and ion pool.

## Extending and checking the implementation

A custom preparation has the interface
`prepare(data: array[qubit, 32]) -> int`, returning its verification mask. Add it
to `PREPARATION_METHODS` to expose it to both CLIs and to conventional Steane QEC,
or pass it directly to `make_spam_experiment` or `make_steane_qec_cycle`.

A custom complete QEC method has the interface
`cycle(data: array[qubit, 32]) -> bool`, preserving the borrowed data and applying
its own accepted corrections. Pass it to `make_memory_experiment` or register it
in `QEC_METHODS`. The memory factory also takes the initial preparation, cycle
count, and basis.

Run the standalone catalogue tests using the project's development dependencies:

```sh
uv run --project sol_experiment --frozen python -m pytest -q tests/test_sol_optimized_steane_fault_catalogue.py
```

The broader repository tests can be run from an environment containing
SpiderWarp's test dependencies with `python -m pytest -q tests/test_sol_*.py`.
Compiled Guppy checks use `sol_experiment/.venv/bin/python` by default;
`SPIDERWARP_SOL_PYTHON` can select another compatible interpreter. They cover
encoded-state preservation, bounded retries, detector parities, decoder
corrections/rejections, encoded Bell preservation, and whole-trajectory memory
accounting. They check implementation behavior rather than estimating noisy
production AR/LER.

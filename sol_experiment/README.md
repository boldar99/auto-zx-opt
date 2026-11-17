# Sol experiments

Guppy implementations of state preparation, syndrome extraction, decoding, and
memory experiments for the **[[32, 20, 4]] CSS code**. A block contains 32 physical
data qubits encoding 20 logical qubits. The experiment entry points compile a
Guppy program to HUGR and simulate it on Sol through Selene/Narsil with the
Helios alpha noise model.

## Running experiments

Run commands from the **repository root**, using the separate
[`local_sol_example`](../local_sol_example/README.md) environment. It supplies
Python 3.12+, Guppy 1.x, and the Sol runtime packages. Its README describes the
authenticated package-index setup.

```sh
uv sync --frozen --project local_sol_example
```

Compare preparation methods with a SPAM (state preparation and measurement)
experiment. Each method gets the requested number of shots:

```sh
uv run --project local_sol_example --no-sync python -m sol_experiment.experiment_SPAM \
    --methods d1 d2_RUS d3_RUS d4_RUS --shots 100 --processes 4 --seed 7
```

Run one memory cycle with d3 RUS initialization and d3 RUS Steane ancillas:

```sh
uv run --project local_sol_example --no-sync python -m sol_experiment.experiment_memory \
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
| [logical_state_preparation.py](logical_state_preparation.py) | Encoded-zero preparation descriptions, their Guppy implementations, bounded RUS wrappers, and `PREPARATION_METHODS`. |
| [QEC_steane.py](QEC_steane.py) | Conventional Steane extraction using a supplied encoded-zero ancilla preparation; factories for a single basis or a complete cycle. |
| [QEC_optimized_steane.py](QEC_optimized_steane.py) | Optimized d4 extraction circuit, flag/syndrome detector metadata, and correction factories. |
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

Both QEC implementations run **Z checks followed by X checks**. Z checks correct
X errors, and X checks correct Z errors. Accepted corrections are applied as
physical Pauli gates to the borrowed data block.

The conventional implementation allocates a verified 32-qubit ancilla for each
basis. For Z checks it converts the ancilla to encoded plus, couples data to
ancilla, and measures the ancilla in Z. For X checks it couples ancilla to data
and measures the ancilla in X. The first ancilla is released before the second
is prepared; the supplied verified preparations need at most 66 live qubits
including the persistent data. Only ancilla preparation is retried. A rejected
syndrome ends the trajectory.

The optimized implementation uses a fixed extraction circuit with named flag
and syndrome detector groups, requiring at most 60 live qubits including data.
Its Z-check variant conjugates the X-check extraction with transversal H on data
and restores the data basis before returning. Any raised flag rejects; otherwise
the shared syndrome decoder decides acceptance. This extraction is not retried.

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

The runner is [`local_sol_example/sol/sim.py`](../local_sol_example/sol/sim.py).
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

Run the focused tests from an environment containing SpiderWarp's test
dependencies:

```sh
python -m pytest -q tests/test_sol_*.py
```

Compiled Guppy checks use `local_sol_example/.venv/bin/python` by default;
`SPIDERWARP_SOL_PYTHON` can select another compatible interpreter. They cover
encoded-state preservation, bounded retries, detector parities, decoder
corrections/rejections, encoded Bell preservation, and whole-trajectory memory
accounting. They check implementation behavior rather than estimating noisy
production AR/LER.

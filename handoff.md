# SpiderWarp project handoff

Last updated: 2026-11-12

## Steane RUS comparison (2026-11-12)

The user requested conventional Steane QEC with fixed d4 RUS preparation and a
lower-distance retry option. On clarification they requested **both** ladder
scopes, so three baselines are implemented in `spiderwarp/steane_qec.py`:

- `d4`: retry d4 until all preparation verification bits are zero.
- `ladder-ancilla`: discard flagged ancillas and descend d4 -> d2 -> d1;
  restart at d4 for every fresh ancilla, including syndrome confirmation reads.
- `ladder-cycle`: retain that reduced level through all reads and both bases;
  restart at d4 for each new QEC cycle.

All use the original `FTStatePrep/misc/32_20_4_zero_d{1,2,4}.stim` assets,
encoded-zero controls for X extraction and Hadamard-dual encoded-plus targets
for Z extraction. Retry happens before coupling to the data. Each CSS syndrome
is measured until two consecutive readings agree, Z first, then X. There is
no retry limit or data rejection. Data memory noise accrues through every
serial preparation, failed attempt, discard measurement and readout. The input
and final decoder remain ideal; cumulative frames persist between cycles.
Steane requires 66 qubits (32+32+2), versus the adaptive 60-qubit cap, and has
632 CNOT-equivalent gates on its noiseless path. These are simulation baselines,
not independently certified adaptive policies. The existing adaptive circuits,
controller, noise defaults and saved production counts are unchanged.

Standalone Guppy and compiled/reloaded HUGR exist in:

- `build/steane_rus_d4/`
- `build/steane_rus_ladder_ancilla/`
- `build/steane_rus_ladder_cycle/`

Each has `protocol.json`, `export.json`, `steane_se_protocol.py` and
`steane_se_protocol.hugr`. `build-steane --mode ... --out ... --emit hugr`
creates these; the existing `compile`, `inspect`, `run`, `sweep` and `native`
commands accept these bundles. The standalone emitter is `_steane_qec_guppy.py`.

The matched production sweep is **complete**: 32 configurations and 9,329,148
trajectories / 13,529,148 executed cycles per Steane method; 27,987,444 new
trajectories in total. It used floor 100k/cap 1M, 10k-shot batches, inverse-square
allocation, eight workers and seed 20261112. The command was
`python -u -m spiderwarp.se_comparison --jobs 8`, log
`build/steane_comparison.log`. Results are in
`benchmarks/steane_rus_<mode>_100k/` (hyphens replaced by underscores).
All `run.json` files are complete and contain exact per-method resume commands.
A real d4 resume scheduled zero new shots and preserved every batch. Run
`python -m spiderwarp.se_comparison --plot-only` to audit matched counts/settings
and regenerate `benchmarks/steane_comparison/comparison.csv`, `validation.json`,
`resources.json`, and four combined PDF/PNG figures. The final audit has passed: all
noise, seeds, configuration membership, target counts and Wilson intervals match;
all source/HUGR hashes and regenerated sources match, and all three libraries
reload. The comparison retains tight linear survival and logarithmic normalized
LER; CNOT cost is also plotted.

`notebooks/32_20_4_steane_comparison.ipynb` loads these results and links the
source/binary artifacts. Matching cells were appended to the existing demo.
Both notebooks’ comparison cells were executed and their table/figure outputs
are saved. All four final PNG figures were visually checked; matching vector
PDFs are saved. Cycle axes explicitly label 1, 3, 5, 7, 10.

Validation: the full regression collection passed 164 tests. Ten further new
cases passed in the final 44-test focused run, followed by four additional native
retry regressions and two final plot-layout regressions. The focused files are
`tests/test_steane_qec.py`, `tests/test_se_experiment.py`, and
`tests/test_se_protocol_simulation.py`. Coverage includes retry scopes, confirmation,
frame composition, encoded-Bell preservation with split CSS errors, discarding
corrupted flagged ancillas, actual exported controller paths/decoder inputs,
comparison plots, persistence, and native execution in both bases.
All three permanent `*_controls/` datasets have 1,000 noiseless five-cycle shots
per basis, all accepted with zero errors. Native three-cycle HUGR controls have
eight shots per basis/method with split incoming X1/Z17 errors, all correct;
their hashes are in `benchmarks/steane_comparison/controls.json`. Native noise
scope remains separate from the reference sweeps. Four extra native programs
forced d4 and d2 preparation flags to reach d1 in both ladder scopes/bases,
checking dynamic discard/reallocation and cumulative frames. All 16 diagnostic
shots passed; `forced_retry_controls.json` explicitly distinguishes these
artificial flag checks from production noise data.

At exact default hardware noise, the conditional **block** LERs are:

| Method | N=1 logical 0 / + | N=10 logical 0 / + | Mean CNOTs at N=1, 0 / + |
|---|---:|---:|---:|
| Current adaptive | 1.785% / 3.635% | 20.841% / 42.019% | 444.7 / 444.4 |
| Steane RUS d4 | 0.169% / 2.396% | 2.425% / 22.510% | 742.7 / 742.4 |
| Steane ladder per ancilla | 0.347% / 2.428% | 3.862% / 21.987% | 718.5 / 719.1 |
| Steane ladder per cycle | 0.977% / 2.223% | 9.920% / 20.735% | 679.9 / 680.7 |

All Steane trajectories accept by construction (unbounded RUS/confirmation),
versus adaptive ten-cycle survival 90.510% / 90.390%. Full counts and Wilson
intervals are in `benchmarks/steane_comparison/comparison.csv`. At low physical
p the fixed-d4 version has the lowest logical-0 LER; the cycle ladder reduces
CNOT cost and logical-+ LER, while increasing logical-0 LER. The current adaptive
method uses the fewest qubits and CNOTs. Physical-p curves cross at higher noise;
LER comparisons remain conditional on acceptance. No policy certificate is
claimed for the conventional RUS baselines.

## Resume checkpoint

- The three Steane RUS comparisons above are complete, including standalone
  Guppy/HUGR, matched production counts, controls, the final audit, and populated
  comparison notebook/figures. Use `python -m spiderwarp.se_comparison --plot-only`
  to regenerate the combined views without sampling.
- The build/load/simulate workflow is implemented and has been run. The current
  verified protocol bundle, standalone Guppy source, and compiled/reloaded HUGR
  are in `build/se_32_20_4/`. The controller remains the verified 589-CNOT,
  60-qubit adaptive policy.
- The completed production dataset is `benchmarks/se_100k/`: 32 configurations,
  9,329,148 attempted trajectories, 100k shots at every repetition point, and
  up to one million at low physical p. Repetitions use the exact
  `default_noise_model()`; physical-p sweeps scale that model. Keep seed
  20261112 and the saved `run.json` command when resuming.
- The latest plot preference is **linear survival with a tight data-dependent
  range**, approximately 0.895–1.0 for this dataset. Do not restore a logarithmic
  survival axis or force 0–1 limits. Regenerate with
  `python -m spiderwarp.se_experiment plot benchmarks/se_100k`.
- Noise-source assessment is complete; the data and confidence intervals are in
  `analysis/32_20_4_noise_assessment/comparison.csv`, with methodology in
  `assessment.json` and stage/depth/idle counts in `resources.json`.
- **Scope remains assessment only for the proposed reuse change.** The user
  explicitly said not to implement it. No alternative adaptive reuse circuit,
  allocation, or policy was built. The recommendation is to preserve ZX/CNOT optimization and
  consider depth/live-idle minimization under the existing 60-qubit cap, rather
  than minimizing qubit count. d1/d2-only changes are unlikely to remove most
  of the observed logical error rate.
- New comparison validation passed: a 164-test full regression collection,
  44 focused tests, four additional native retry regressions, and two final
  plot-layout checks. `git diff --check` passed. Details and native control
  counts are in `benchmarks/steane_comparison/validation.json`.

## Noise-source and reuse assessment (2026-11-12)

Survival has been changed back to a tight linear y-axis (confidence intervals
plus 8% padding, clipped to the probability domain). Existing 100k-run figures
were regenerated without resampling. The physical-error and normalized-LER
plots retain logarithmic axes.

The user requested assessment, not implementation, of reduced d1/d2 reuse under
the 60-qubit budget. No circuits, allocations, optimizer, or policy were changed.
`analysis/32_20_4_noise_assessment/` contains fixed-circuit noise ablations,
`comparison.csv`, `assessment.json`, and `resources.json`. Baselines are copied
from the completed production run. One-cycle cases use 100k attempted shots per
basis; ten-cycle confirmation cases use 20k, with the existing 100k baseline.
Stage-specific perfect-memory probes use 100k per basis. All include the same
initial-data idle convention and ideal experiment boundaries.

Interpretation: the cycle plot shows P_L(N)/N for the entire 20-logical-qubit
block, conditional on surviving all cycles. It is neither a per-logical-qubit
rate nor a first-failure hazard. Noise-removal experiments assess sensitivity
of the existing circuit, not the performance of a proposed allocation.

One-cycle conditional block LER (logical 0 / logical +): full 1.785% / 3.635%;
without two-qubit noise 0.206% / 0.982%; without memory 1.460% / 2.019%; without
initialization noise 1.065% / 2.880%. At ten cycles, removing two-qubit noise
gives 3.215% / 14.421%, whereas removing memory gives 16.778% / 23.951%
(full 20.841% / 42.019%). Two-qubit-related faults give the largest sensitivity;
linear memory is substantial, especially for logical +. These changes are not
additive error attribution: mixed faults and postselection matter.

At the default noise, d4 accounts for about 90% of extraction executions and
90.5–90.7% of live idle exposure. Removing memory only in d1/d2 changes one-cycle
LER to 1.735% / 3.375%; the logical-0 change is within sampling uncertainty.
Removing memory only in d4 gives 1.376% / 2.099%. These are diagnostic proxies,
not bounds or predictions for a rescheduled circuit.

Current d1 uses 42 qubits, 23 ancilla reset lifetimes, and 79 CNOTs: zero reuse
would need 32+23=55 qubits, within budget. d2 uses 51 qubits, 33 ancilla lifetimes,
and 121 CNOTs: zero reuse needs 65 qubits, so at least five lifetime merges are
necessary at 60 (versus fourteen today). Current d2 noise-model depth is 92
layers in X / 86 in Z, versus 83 for d4, despite its lower CNOT count. This
supports investigating less reuse, keeping the optimized quantum gate graph.
However extra wires help only if they reduce data waits/live ancilla exposure;
the right objective is depth/idle cost subject to 60 qubits, rather than minimum
qubit count or minimum reuse count alone. d1/d2-only changes are a secondary
improvement, given their measured contribution. No alternative was built.

## Higher-statistics default-noise sweeps (2026-11-12)

The latest requested run is `benchmarks/se_100k/`. Its repetition sweep uses
`default_noise_model()` exactly (p_2q=8e-4), while its physical-error sweep
scales every default rate together with p_2q. The preset now has 32 configurations:
11 physical p values in both bases and five default-noise cycle counts in both
bases. Plot membership is explicit, so the extra default-rate N=1 point does
not accidentally enter the 11-point physical-error grid.

`num_samples` in `se_sweep.py` allocates fixed production budgets proportional
to 1/(N*p²), with a 100,000-shot floor and one-million-shot cap for this run.
It plans 9,329,148 total trajectories. This approximately balances relative
error bars in the low-noise regime; the floor and cap bound cost. It does not
stop on observed errors. CLI defaults are now 100k shots, 10k-shot batches,
inverse-square allocation for `sweep`, and fixed allocation for single `run`.
Both allocation and per-task targets are persisted and resume correctly.

`plot` defaults to the default model's p_2q for repetition/survival plots.
Survival uses a linear y-axis fitted to the confidence intervals with 8% padding
(clipped to the probability domain), as subsequently requested. For the
historical pilot described below, use `--cycle-p 0.001` instead. The current
run's exact settings and command are
in `benchmarks/se_100k/run.json`; its log is `build/se_100k.log`.

Targeted checks pass: 17 tests cover the allocator bounds/scaling, exact default
noise selection, variable-budget resume, frame/noise sampling, and plot axes.
The run is complete: all 9,329,148 trajectories and 13,391,955 executed cycles
are saved. `run.json` is marked complete and `validation.json` records the
noise/target/interval checks. A resume scheduled zero new shots. Every
repetition point has exactly 100k attempted shots; the lowest three p values
have one million per basis. The final PDF/PNG figures have been regenerated
and visually checked. Survival was subsequently regenerated on a tight linear
axis without resampling; LER plots retain their logarithmic axes.

At the exact default noise model:

| Cycles | Logical 0 failures / accepted | Logical + failures / accepted |
|---|---:|---:|
| 1 | 1,777 / 99,575 (1.7846%) | 3,618 / 99,527 (3.6352%) |
| 10 | 18,863 / 90,510 (20.8408%) | 37,981 / 90,390 (42.0190%) |

Ten-cycle survival is 90.510% and 90.390%, respectively. At p=1e-4 the
million-shot physical sweep has 376 / 999,993 logical-0 failures and
794 / 999,990 logical-+ failures. Complete Wilson intervals are in summary.csv.

## Saved-protocol experiment workflow (2026-11-12)

`python -m spiderwarp.se_experiment` now provides `build`, `compile`, `inspect`,
`run`, `sweep`, `plot`, and optional `native` commands. See README for usage.

- `build --out build/se_32_20_4 --emit guppy` synthesizes/verifies once, saves a
  reloadable bundle, and emits source without importing Guppy. `compile` adds
  HUGR separately. `export=False, output_dir=...` now saves a bundle without
  triggering compilation; the previous warning about this combination is stale.
- `AdaptiveSEProtocol.load(path)` validates file hashes, decoder identity,
  readout widths, graph structure, and termination without fault enumeration or
  policy synthesis. `verify=True` additionally reruns the independent checker.
  `se_protocol_io.py` stores versioned JSON, content-addressed circuits/code,
  and compressed numeric arrays. Loaded transitions remain read-only array
  views instead of allocating a Python dictionary entry per transition.
- `ProtocolSampler` reuses noisy circuits and ideal input/decoder construction
  between batches. It keeps the same physical data block across cycles and
  carries the cumulative returned frame. It stops on rejection, without retry,
  and ideal-decodes only at the end. All executed extraction labels are noisy.
- `se_sweep.py` provides deterministic batch seeds, bounded process workers,
  atomic count CSVs, a single-writer lock, task fingerprints, restart/deduplication,
  aggregate confidence intervals, and cycle/resource/rejection diagnostics.
  Fingerprints include simulator/decoder sources, Stim/NumPy versions, and
  architecture. Keep the seed when resuming; shot targets and jobs may change.
- p denotes p_2q; other default hardware rates scale by p/8e-4. The new sampler
  includes data idles before their first gate in each extraction, which the
  historical scheduler omitted. It retains scheduled internal/trailing idle
  noise and zero classical/inter-cycle delay. Its explicit model name is
  `stim-adaptive-initial-data-idles-v1`. Historical `estimate_protocol_ler`
  behavior stays the default; opt into this convention with
  `cycles=N, include_initial_idles=True`.

Actual artifacts are in `build/se_32_20_4/`: `protocol.json`, the 5.6 MB
compressed policy, complete standalone `steane_se_protocol.py`, and compiled,
reloaded `steane_se_protocol.hugr` (8,589,785 bytes). `export.json` binds the
source/HUGR hashes to the bundle. The reloaded bundle regenerates source
byte-for-byte identical to the previous verified 589-CNOT export. The current
Guppy 0.21.16 / HUGR 0.16.0 environment compiled the full library in about
131 seconds, including reload; no table-representation rewrite was necessary.

The reduced reference sweep has been run: 11 p values from 1e-4 to 1e-2 at
N=1, and N=1,3,5,7,10 at p=1e-3, in both bases (30 distinct configurations).
`benchmarks/se_small/` holds 1,000 attempted trajectories per configuration,
seed 20261112, task manifests, batch CSVs, `summary.csv`, `diagnostics.json`,
`config.json`, `run.json`, and PDF/PNG figures. Worker batch times sum to about
22.7 seconds for 30,000 trajectories / 69,706 executed cycles. Re-running the
same shot target scheduled zero new work. These are pilot estimates:

| p=1e-3 | Logical 0 failures / accepted | Logical + failures / accepted |
|---|---:|---:|
| N=1 | 22 / 994 | 68 / 992 |
| N=10 | 232 / 856 | 462 / 846 |

The plots match `FT-circuit-construction/713/notebooks/visuals.ipynb` (the
referenced `visualize.ipynb` filename does not exist). They show p_L/p^2 vs p,
p_L/N vs N, and survival, with Wilson intervals and zero-failure upper endpoints.
LER remains conditional block LER across 20 logical qubits, separately in each
basis. p_L/N is an endpoint normalization, not a first-failure hazard.

`benchmarks/se_controls/` additionally holds 1,000 noiseless five-cycle shots
per basis (seed 20261112). All 2,000 accepted with zero logical failures.

Validation: the full regression run had 142 passes and one stale installation
test reference to the removed `opt_zero_32_20_4` asset. Both that test and the
README example now use `32_20_4_zero_d4`. The installation test and all ten new
experiment tests pass on the focused rerun (11 passed). This includes native
compilation/execution, initial data idles, physical frame composition, rejected
trajectory accounting, atomic resume/deduplication, parallel reproducibility,
and source-only export provenance. `git diff --check` passes.

Optional direct serialized-HUGR execution is implemented in `se_native.py`.
It compiles a runnable wrapper with state preparation, repeated framed calls,
acceptance/frame outputs and final measurements, then runs Selene/Stim.
The full-size three-cycle logical-0 wrapper ran eight noiseless shots, all
accepted with zero failures. A separate 1,000-shot p=1e-3 run accepted 928 and
had 292 logical failures; its metadata is in
`build/se_native_32_20_4/native_results.json`. This is explicitly a DIFFERENT
noise model: Selene's built-in gate noise also affects preparation/final
readout and omits scheduled memory, so do not attribute that LER to the primary
ideal-boundary experiment. There is no interpreter fallback. `[native]` pins
the tested Guppy/HUGR/Selene/compiler stack; the primary workflow does not need
these optional dependencies.


## Current adaptive implementation (2026-11-12)

The sections below this update describe earlier implementations and experiments.
The current default is the history-based distance ladder in
`spiderwarp/adaptive_steane_se.py` and `_se_adaptive.py`; the old fixed recovery
controller is available only as `policy_mode="legacy"`.

- Start with d4 Z-SE, retaining every flag and syndrome.
- Compute remaining faults from the joint compatible histories. Two possible
  future faults require at least d4; one requires d2; zero permits fault-free d1.
- Choose the next basis separately. No flag or syndrome is discarded, and fresh
  recovery flags enter another update rather than automatically rejecting.
- Stop with one common safe Pauli frame, or reject only when no compatible
  history requires acceptance. Frames are returned, not applied physically.
- Count every one of the 15 two-qubit Pauli alternatives as one physical fault.
  CSS boundary filters share those joint histories and the same event budget.

For [[32,20,4]], mandatory acceptance covers r+s<=1 and also clean-input s=2.
Logical preservation is required through r+s<=1. Accepted order-two histories
satisfy the internal-fault-count CSS filter without a two-fault logical guarantee.
The arbitrary-input EC A check remains present through one internal fault; in
particular accepted (r,s)=(2,0) outputs are codewords. See README for the full table.

The planner minimizes worst-case CNOT-equivalent count within its library, peak
qubit limit, and ladder. Independent verification recomputes transitions, checks
terminal frames/rejections, and checks termination. Guppy/HUGR returns acceptance
and two frame masks. `export=False` permits synthesis without Guppy.

The earlier claim that current raw d4 inputs pass the legacy verifier is stale:
the exact gate model finds an unflagged one-fault primary syndrome 32 that the
legacy standard decoder rejects. A regression now requires that diagnostic,
and the adaptive controller must recover the corresponding physical history.

A stronger promise of logical correctness for every accepted two-fault history
is impossible with these noisy output boundaries. The optional stronger contract
has a concrete no-go diagnostic and is not used by the default protocol.

### Verified result and export status

On the finalized 32_20_4 d1/d2/d4 assets, the default contract and 60-qubit
limit give a verified minimum of 589 CNOT-equivalent gates, at most four
extractions, and 60 peak qubits. Bound 588 is infeasible in this action library
and distance ladder. The independent checker covers all 4,096 arbitrary-input
syndrome sectors and over 14 million abstract history transitions.

The original standalone source and report are in
`build/32_20_4_adaptive_protocol/`. Full-size compilation and reload are now
complete; use `build/se_32_20_4/` for the new bundle, source and HUGR. The
executable three-cycle wrapper also ran on Selene/Stim; see the newer workflow
section above for its separate noise scope.

### Running the current adaptive protocol with hardware noise

The noisy experiment is in `notebooks/32_20_4_protocol_demo.ipynb`, beginning
with **Estimate one-cycle block LER**. Its implementation is
`spiderwarp/se_protocol_simulation.py::estimate_protocol_ler`. The later cells
show rejection reasons, executed paths and mean CNOT cost, individual noise
sources, the complementary logical basis, and JSON sampling diagnostics.

Simulation runs the adaptive Stim interpreter and incorporates the returned
Pauli frame in ideal final decoding. It does not require Guppy or a compiled
HUGR package. Build with `export=False`; supplying `output_dir` now saves a
simulation bundle without compiling. When using the notebook for simulation
only, skip **Load the exported HUGR**. Keep its local `output_dir` variable for
inspection files and sampling diagnostics.

Runnable setup and sampling, using the finalized optimized SE assets:

```python
import stim
from spiderwarp.csscode import CSSCode
from spiderwarp.utils import get_project_root
from spiderwarp.optimized_steane_se import optimized_steane_SE_orchestrator
from spiderwarp.se_protocol_simulation import estimate_protocol_ler

code = CSSCode.load_code("misc", "32_20_4")
assets = get_project_root() / "assets/circuits/SECircuits"
circuits = {
    d: stim.Circuit.from_file(assets / f"32_20_4_d{d}.stim")
    for d in (1, 2, 4)
}
protocol = optimized_steane_SE_orchestrator(
    code, circuits, peak_qubits=60, export=False, progress=print,
)

for state in ("0", "+"):
    result = estimate_protocol_ler(
        protocol, prepared_state=state, shots=1_000, seed=20261015,
    )
    print(state, "LER:", result.ler,
          "95% interval:", result.ler_interval_95,
          "discard rate:", result.discard_rate)
```

If `protocol` already exists in the kernel, only the sampling loop is needed.
Start with 1,000 shots; the notebook's main run uses 100,000, with 10,000-shot
diagnostic comparisons. The environment used for the latest smoke check was
`/opt/homebrew/Caskroom/miniforge/base/envs/qec/bin/python`.

With no explicit `noise_model`, `default_noise_model()` supplies probabilities
`p_1q=2.5e-5`, `p_2q=8e-4`, `p_meas=1e-6`, `p_init=8e-4`, and
`p_linear_mem=1.26e-5`; coherent memory and leakage rates are zero. Pass
`noise_model=CustomErrorModel(...)` to select another model, or
`CustomErrorModel()` for a noiseless control (import it from
`spiderwarp.noise_model`). Preparation of the input logical state and final
decoding are ideal. Scheduled memory noise is included during the extractions;
classical branch latency and subsequent HUGR backend transformations are not.
This sampling model therefore differs from the exact gate-fault certificate.

LER is the probability that any of the 20 logical readouts fails, conditional
on acceptance. Discards are reported separately. Logical `0` tests X/Y errors;
logical `+` tests Z/Y errors, so run both bases, especially with Z-biased memory
noise. These are separate basis-specific rates, not an arbitrary-state LER.

Latest smoke check (2026-11-12): the current verified 589-CNOT controller ran
100 shots per basis and noise case, with seed 20261015. Both noiseless runs
accepted all shots with zero logical failures. Under default noise, both runs
accepted all 100 shots; logical failures were 1 for `0` and 6 for `+`. These are
small execution checks, not precise performance estimates. The older large-shot
numbers in the historical sections below describe the legacy controller and
must not be attributed to the current adaptive policy.

## Project goal

SpiderWarp optimizes fault-tolerant quantum gadgets using fault-equivalent ZX rewrites. The intended guarantee is stronger than ordinary unitary equivalence: rewrites must preserve the gadget's relevant behaviour under circuit faults, including flag, syndrome, and logical-error information.

The current development case is the `[[32,20,4]]` code in `notebooks/32_20_4.ipynb`. The working pipeline is:

1. Construct a state-preparation or syndrome-extraction circuit in Stim.
2. Convert the circuit to a PyZX graph and then to `CoveredZXGraph`.
3. Apply ZX/path-cover rewrites.
4. Extract an ordered circuit DAG.
5. Reuse and compress ancilla qubits.
6. Reconstruct a Stim circuit while preserving the measurement map.
7. Annotate flag and syndrome measurements as detectors.
8. Verify fault tolerance and derive an adaptive follow-up policy.

## Current headline status

- Hadamard gates, Hadamard edges, and CZ gates are supported throughout the relevant ZX import/extraction path.
- The non-canonical uncovered-edge failure in `build_circuit_dag` has been fixed.
- The notebook now constructs a combined flag-and-syndrome extraction circuit and verifies it against an ideal encoded-zero input.
- One-fault correction plus two-fault detection-or-safe-correction passes for the current in-memory syndrome-extraction circuit.
- Concrete zero-, one-, and two-fault histories can be grouped by observed flag pattern.
- A per-flag flow planner now chooses among immediate correction, a 1-FT follow-up SE, a non-FT follow-up SE, and discard.
- Immediate-correction plans include a concrete Pauli correction and the effective residual state and weight for every compatible fault history.
- The focused regression suite currently passes: **23 tests passed**.

## Implemented changes

### Hadamard and CZ support

The main implementation is in `spiderwarp/path_cover.py` and `spiderwarp/stim_utils.py`.

- `CoveredZXGraph` accepts both `SIMPLE` and `HADAMARD` edges.
- Standard degree-two PyZX H-boxes are normalized to typed Hadamard edges without mutating the caller's graph.
- Edge types are retained by graph copying, identity insertion/removal, and the relevant rewrite operations.
- A Hadamard edge on a covered path extracts as an `H` gate.
- A Hadamard edge between two Z spiders extracts as `CZ`.
- A Hadamard edge between two X spiders extracts as `XCX`.
- Opposite-colour simple edges continue to extract as CNOT-like interactions.
- Stim-to-PyZX conversion now accepts `H`, `CZ`, and `XCX`.
- Identity removal composes edge types by Hadamard parity instead of silently turning every result into a simple edge.

Regression coverage is in `tests/test_path_cover_hadamard.py`. It checks CZ extraction, path Hadamards, explicit H-box normalization, Stim round trips, reset/measurement preservation, and Hadamard parity through identity removal.

### Circuit DAG and qubit reuse

`spiderwarp/qubit_reuse.py::build_circuit_dag` now performs extraction preparation on a deep copy of the covered graph. It inserts temporary identities needed to canonicalize non-extractable edges before obtaining the total operation ordering. This fixes the previous error:

```text
Cannot extract non-canonical uncovered edge ... with type EdgeType.SIMPLE
```

Because the normalization is applied to a copy, DAG construction does not mutate the optimized graph owned by the caller.

### Combined syndrome-extraction circuit

The final section of `notebooks/32_20_4.ipynb` now:

- constructs the optimized SE circuit;
- applies qubit reuse and compression;
- reconstructs the circuit with the new-to-old measurement map;
- combines flag information with the syndrome measurements;
- assigns detector coordinate prefix `0` to flags and `1` to syndrome information;
- builds an ideal encoded-zero input for deterministic, state-specific checking;
- runs the FT and fault-order verifiers; and
- derives and prints the minimal follow-up flow.

The exported circuit is at `spiderwarp/assets/circuits/SECircuits/32_20_4.stim`. The notebook now writes this asset directly after constructing `flagged_syndrome_circuit`; the checked-in asset has been regenerated and byte-for-byte compared with that pipeline output. Regenerate it and rerun verification whenever the optimization or measurement-remapping pipeline changes.

## Fault-tolerance verification

The new and extended functionality is in `spiderwarp/fault_tolerance_verification.py`.

### Ideal encoded inputs

The verification functions accept `ideal_input_circuit`. The input is prepended noiselessly, and noise is inserted only on the gadget being checked. This makes state-specific SE detectors deterministic without counting ideal-state preparation as part of the noisy gadget.

For this notebook, use:

```python
prepared_state="0"
ideal_input_circuit=ideal_encoded_zero
```

Do not use `prepared_state="arbitrary"` for the present circuit. That mode prepares an encoded Bell input and checks preservation of an arbitrary logical state. The current SE construction is state-specific, so arbitrary-state observables can be non-deterministic even when encoded-zero verification is valid.

### One-correct/two-detect-or-correct property

`verify_fault_order_detect_or_correct(...)` checks:

- every zero- or one-fault flag-and-syndrome symptom has a unique correction class; and
- every two-fault history either produces a new symptom that may be discarded or safely reuses the correction assigned to the matching low-order symptom.

It returns `True` on success. On failure it returns a deterministic Stim counterexample circuit, matching the style of `verify_ftsp`. The counterexample combines the colliding one- and two-fault histories; for distance four this is a three-fault undetectable-logical witness.

For the current in-memory `[[32,20,4]]` SE circuit, the result is:

```text
X errors: one-fault correction passes;
          852 two-fault signatures discard, 44 safely reuse a correction.
Z errors: one-fault correction passes;
          654 two-fault signatures discard, 73 safely reuse a correction.
Overall result: True
```

These counts are deduplicated detector-error-model signatures, not counts of concrete physical fault locations.

### Concrete flag-pattern analysis

`classify_flag_patterns_by_fault_order(...)` expands detector-error mechanisms into concrete circuit fault locations using Stim's error explanations. It returns a `FlagPatternFaultAnalysis` for each observed flag pattern, containing:

- zero-, one-, and two-fault histories;
- whether each fault came from the incoming data block or the SE circuit;
- the full stabilizer syndrome and tracked logical signature;
- the possible total fault orders for the flag pattern;
- a minimum set of ideal stabilizer measurements that distinguishes one from two faults, when one exists;
- a witness when even the full syndrome cannot distinguish the orders; and
- a witness when equal symptoms require incompatible corrections.

This analysis deliberately counts concrete fault-location histories, so its counts are much larger than the deduplicated signature counts printed by `verify_fault_order_detect_or_correct`.

Fault-order ambiguity is not automatically a correctness failure. The same flag and full syndrome may be reachable with one and two faults while both histories still admit the same safe correction. The distinguishability verifier checks correction classes; the order classifier separately asks whether the numerical fault order can be inferred.

## Adaptive follow-up flow

`plan_minimal_followup_by_flag(...)` implements the current relaxed Gottesman-style acceptance policy.

The accepted residual weights are:

| Incoming faults | Faults during first SE | Required state after correction |
|---:|---:|---|
| 0 | 0 | weight 0 |
| 1 | 0 | weight 0 |
| 0 | 1 | weight at most 1 |
| total faults 2 | any split | discard or weight at most 1 |

For every flag pattern, the planner searches for a concrete Pauli correction that satisfies the appropriate bound for every compatible zero-, one-, and two-fault history. It produces one of four actions:

1. `correct now; no follow-up SE`
2. `run 1-FT follow-up SE`
3. `run non-FT follow-up SE`
4. `discard`

The decision rule is:

- Correct immediately if one correction satisfies every compatible history.
- Otherwise, if a one-fault history is possible, use a 1-FT follow-up SE.
- Otherwise the pattern is two-fault-only. Use a non-FT follow-up if at least one follow-up syndrome is safely correctable, and discard ambiguous follow-up outcomes.
- Discard immediately if no syndrome outcome is safely correctable.

For the current five-flag `[[32,20,4]]` SE analysis, the 21 reachable flag patterns split as follows:

| Action | Number of flag patterns |
|---|---:|
| Correct now | 5 |
| Run 1-FT follow-up SE | 10 |
| Run non-FT follow-up SE | 6 |
| Discard immediately | 0 |

The all-zero flag pattern has possible orders `{0, 1, 2}` and requires a 1-FT follow-up. Across every immediate-correction plan, the maximum effective residual weight observed was one.

The notebook entry point is:

```python
se_flag_orders = classify_flag_patterns_by_fault_order(
    circuit=flagged_syndrome_circuit,
    H_x=H_x,
    H_z=H_z,
    L_x=L_x,
    L_z=L_z,
    flag_detector_indices=range(num_flags),
    ideal_input_circuit=ideal_encoded_zero,
    prepared_state="0",
)

se_followup_flow = plan_minimal_followup_by_flag(
    se_flag_orders,
    H_x,
    H_z,
    L_x,
    L_z,
    prepared_state="0",
)

print_followup_flow(se_followup_flow, explain_corrections=False)
```

Set `explain_corrections=True` to print each immediate Pauli correction and an aggregation of the resulting effective residual errors by incoming-fault count, SE-fault count, Pauli product, residual weight, and number of compatible histories. Here, “final state” means the effective Pauli residual represented modulo the tracked stabilizer/logical information, not a full state vector.

To avoid very large output while debugging one branch:

```python
pattern = next(
    pattern
    for pattern, plan in se_followup_flow.items()
    if plan.correction_without_followup is not None
)
print_followup_flow(
    {pattern: se_followup_flow[pattern]},
    explain_corrections=True,
)
```

## Correctness scope and current limitations

1. **CSS-separated fault model.** X- and Z-type single-Pauli faults are analyzed separately. Mixed X/Z histories and joint Y combinations are not yet enumerated as a single combined process.
2. **State-specific result.** The current result proves the properties needed for an encoded-zero state. It is not an arbitrary-logical-state error-correction proof.
3. **Follow-up circuit composition is not yet proved end-to-end.** The planner chooses the minimum required kind of branch circuit. The concrete 1-FT and non-FT follow-up circuits must still be supplied and verified under incoming errors and new faults during the follow-up.
4. **Correction representatives.** The GF(2) solver returns a valid Pauli representative and prefers the lightest representative among its generated candidates, but it is not a global minimum-weight decoder over every stabilizer-equivalent physical correction.
5. **Two verification abstraction levels.** The fast verifier works with unique DEM mechanisms/signatures. The detailed flag analysis expands these into concrete physical fault locations. Their roles are different, but consolidating them behind one shared representation would reduce the chance of future semantic drift.
6. **Runtime.** Full concrete classification and planning for the current notebook takes roughly tens of seconds, compared with a few seconds for the focused unit tests.

## Recommended next steps

1. Materialize the two follow-up branches selected by the planner:
   - a 1-FT SE circuit for branches where one fault remains possible;
   - a cheaper non-FT SE circuit for two-fault-only branches.
2. Add an adaptive composition verifier that injects incoming errors, faults during the first SE, and faults during the selected follow-up, then checks the residual-weight/discard contract end-to-end.
3. Decide whether the target theorem is encoded-zero preparation only or arbitrary-state syndrome extraction; the latter requires the Bell-input verification mode and a circuit that preserves arbitrary encoded states.
4. Extend the concrete history enumerator to mixed Pauli faults if the intended noise model requires them.
5. Consider moving the flow-planning data classes and algorithms out of the already-large `fault_tolerance_verification.py` into a focused module.

## Tests and reproducibility

Run the focused suite with the project environment:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/zxlive/bin/python -m pytest -q \
  tests/test_fault_tolerance_verification.py \
  tests/test_path_cover_hadamard.py \
  tests/test_qubit_reuse.py
```

Latest result:

```text
23 passed in 3.72s
```

The full suite has previously had an unrelated `tests/test_qecc.py` failure when the optional `qecc` dependency was unavailable.

## Working-tree notes

The repository is currently a work in progress, not a clean commit. In particular:

- `tests/test_fault_tolerance_verification.py`, `tests/test_path_cover_hadamard.py`, and `tests/test_qubit_reuse.py` are currently untracked and must be added before committing.
- Notebook output has been cleared, and the generated `notebooks/circ.svg` is removed and ignored.
- The renamed `32_20_4_zero_d2.stim` input and regenerated SE asset are modified.
- `.DS_Store` and `__pycache__` are now ignored.

## Adaptive SE builder implementation (2026-10-15)

`spiderwarp.optimized_steane_se.optimized_steane_SE_orchestrator(code, circuits,
*, output_dir=None)` now optimizes distance-labelled canonical X-SE gadgets,
constructs dual gadgets, synthesizes recovery decisions, verifies a complete
cycle, and exports standalone Guppy source plus a composable HUGR package.
The last notebook cell invokes this interface. The public module also provides
the reference Stim interpreter, results, and verification exceptions.

The agreed cycle is Z primary at code distance, then (if unflagged) standard
decoding/correction and X primary at code distance. A flagged primary discards
its own syndrome, runs dual-basis recovery, rejects raised recovery flags, and
returns immediately after recovery decoding. Recovery after the first primary
does not proceed to the second primary. One fault budget covers the whole cycle.

Recovery synthesis now adds an extraction of the discarded primary component
if no single dual recovery can satisfy CSS EC A/B. It still ends the cycle after
that recovery branch and retains the clean-path primary ordering. Both fresh
syndromes are combined before correction; raised flags at either stage reject.
RecoveryPlan.cleanup_distance records this optional second recovery stage.

New internals are `_se_circuit.py` (measurement provenance, ZX/reuse optimization,
encoded-Bell validation, exact mixed-Pauli response propagation), `_se_policy.py`
(shared-budget synthesis/checking), and `_se_guppy.py` (source/HUGR export).
Standard decoding uses the repository's `lut_decoder.LutDecoder` at build time.
The current exact implementation supports identical H_x/H_z, code distances
2–5, and at most 63 data qubits/flags/checks in Guppy's integer-mask emitter.
Basis rotations introduced by Guppy lowering are included before fault analysis.

The user explicitly clarified that the metric is CSS weight max(w_X, w_Z),
since the components decouple. X1 Z17 and X7 Z23 therefore have CSS weight one,
and the earlier total-Pauli-weight failure claims are withdrawn. EC A bounds
accepted residuals by r errors in each component for r <= t regardless of
incoming errors. EC B requires acceptance and preservation of the ideal-decoded
logical state when incoming CSS weight plus internal order <= t. Higher bounded
histories through floor(d/2) use the internal-fault r-filter as well, without a
correction-radius cap. Incoming CSS-weight-one states include split X/Z support.

EC A is checked on every incoming syndrome sector (4,096 for [[32,20,4]]),
using one representative per normalizer class and allowing r <= t new internal
faults independently of incoming weight. Logical Paulis do not affect branch
observations or output syndrome; this follows from the validated linear gadget
response. Logical classes remain intact in the separate EC B histories.
Enumeration currently supports up to 65,536 incoming syndrome sectors.

The complete optimized policy now passes CSS EC A/B: 17,893,758 checked
transitions, 1,831,353 accepted and 1,281,618 discarded contexts. All 8,844
reachable recovery prefixes use d1; 324 additionally need d1 extraction in the
primary basis (162 per primary basis). This is necessary even with CSS weight:
four Z-only histories leave I, Z18, Z0, Z0 Z18 with the same retained recovery
observations; there is no common one-error Z correction. The new extraction
resolves this ambiguity. Four physical prefixes and the proof are saved in
`analysis/32_20_4_protocol/why_css_recovery_cleanup.txt`.
The current source/HUGR are restored in `build/32_20_4_protocol/`, using 136
shared decoders and reduced shared integer decision graphs. The export uses
conditional Guppy functions, with no external LUT or JSON sidecar. Earlier
exports remain in the superseded subdirectory for comparison.

Use raw measurement parities when checking deterministic witnesses or flag
signs: Stim's detector sampler subtracts its own reference, hiding constant
Paulis/raised flags. `_se_circuit.sample_annotation_parities` handles this.

Validation includes unit regressions for the r-filter/logical distinction,
physical fault-response cross-checks, shared-budget recovery synthesis on all
three real inputs, standalone Guppy compilation and HUGR reload. The existing
installation test needs an installed checkout for its isolated interpreter.
The current environment has Guppy 1.1.3 and HUGR 0.18.8.

Current physical checks pass for all 1,089 incoming CSS-weight-at-most-one
classes, 1,001 distinct single-fault responses in each primary, 160 total-order-
two cases, 64 faults in cleanup stages, and 128 arbitrary-input EC A cases.
All 3,443 runs match full code/logical-Bell signatures. The installation test
still needs an installed checkout for its isolated interpreter.

Latest CSS validation: 25 protocol tests pass; full suite 78 passed and one
existing isolated-installation failure. Fresh standalone source/HUGR reload
passes, as do all 492,544 generated decoder cases and 8,844 recovery routes.

Native execution also passes: eight Selene/Stim shots each exercise the clean
path with split X1/Z17 incoming errors, the first-primary recovery plus cleanup,
and the second-primary recovery plus cleanup with an incoming Z0 and an internal
fault. All acceptance bits and independently predicted code/logical-Bell states
match in all three cases. Current compiled decision helpers: 1,255.

## Optional verification and noisy protocol demo

`optimized_steane_SE_orchestrator` now defaults to `verify=False`. Construction
still analyzes faults to select recovery stages and corrections, but independent
gadget/terminal checks are optional. Use `verify=True`, or call
`verify_se_protocol(protocol)` / `protocol.verify()` later. The callable checks
the supplied policy without replacing it and clears a previous successful report
on failure. Exported source records `VERIFIED_AT_BUILD`.

`notebooks/32_20_4_protocol_demo.ipynb` exports/reloads Guppy/HUGR and estimates
basis-specific block LER with ideal logical `0` or `+` input and ideal final
`LutDecoder` correction. It uses the adaptive Stim interpreter.
`estimate_protocol_ler` accepts unverified protocols and defaults to the hardware
model in `noise_model.py`; `apply_noise` handles each executed extraction and
physical correction block, including scheduled memory noise. Classical branch
latency is not included. `default_noise_model()` centralizes the default rates;
pass `noise_model=CustomErrorModel(...)` to override them.

Validation: all 32 protocol/simulation tests pass. The updated notebook executed
with verification disabled and the default noise model. For 2,000 logical-0
shots it accepted 1,925, discarded 75, and found 63 logical failures: conditional
block LER 3.273% (95% Wilson interval 2.566–4.165%).

## Guppy version compatibility

The user's `error-correction` environment has Guppy 0.21.6 / HUGR 0.14.1,
whereas `zxlive` has Guppy 1.1.3 / HUGR 0.18.8. The older Guppy has no
`with_minimal_opt()` and `measure` returns bool directly. Generated source now
contains a version-compatible `measure_bit` helper and `compile_se_cycle()`;
the latter uses minimal optimization when available and otherwise calls
`compile_function()` directly. Standalone source remains independent of
SpiderWarp. The optional dependency now accepts Guppy >=0.21.6,<1.2.
Restart notebook kernels after changing repository/dependency versions. The demo
prints Python/Guppy information and preserves the user's 100,000-shot setting.

Compatibility validation: 30 targeted tests pass on Guppy 1.1.3. A small
protocol compiles, reloads and samples on 0.21.6. The complete demo also ran
successfully on 0.21.6 / Stim 1.15.0: 100,000 logical-0 shots, 76,197 accepted,
1,086 logical failures, conditional block LER 1.425% (95% interval
1.344–1.512%), discard rate 23.803%. This fresh optimization used 60 peak SE
qubits and left verification disabled, as requested. Resource choices can vary
between optimization runs/environments; these results are not the same gadget
schedule as the earlier 68-qubit demonstration.

## Readable Guppy export (2026-10-15)

The exporter now uses Guppy `ExtractionResult`, `Correction`, and `RecoveryRoute`
structs, compile-time circuit batches over target tuples, and immutable lookup
arrays. One masked-decision interpreter replaces the generated decision-helper
functions and recovery-selection Boolean expressions. Corrections use two loops
over the data array, retaining the original X-before-Z order. Circuit batches
combine adjacent identical gates only; preparations and destructive measurements
retain every ancilla lifetime and operation ordering.

`spiderwarp/_se_guppy_template.py.in` contains the readable standalone program;
`_se_guppy.py` supplies metadata, circuit definitions, and shared numeric tables.
The template is included in package data. The saved source and HUGR still need
no SpiderWarp installation or decoder sidecar at runtime. Public entry points
remain `se_cycle`, `postselected_se_cycle`, and `compile_se_cycle`; extraction
and decoder helpers now return the named structs instead of tuples.

Both existing exports were refreshed using their original emitted gate schedules
and decoder decisions, without rerunning the stochastic ZX optimization:

- `build/32_20_4_protocol_demo/steane_se_protocol.py`: 27,771 -> 4,329 lines.
- `build/32_20_4_protocol/steane_se_protocol.py`: 23,216 -> 3,599 lines.

The demo HUGR is saved with Guppy 0.21.6 / HUGR 0.14.1 for the user's
`error-correction` environment; the earlier verified export uses Guppy 1.1.3 /
HUGR 0.18.8. Recompile the standalone source when moving a binary package between
incompatible HUGR format versions.

The bulk of each file is now bounded-width numeric lookup data after the circuit
definitions and adaptive control flow. Original recovery routes, correction
values, explicit rejections, and ordered quantum operations were compared with
the refreshed exports: 447,738 decoder cases for the demo and 492,816 for the
earlier verified export (including invalid-input probes), plus all 8,844 recovery
routes and all six gadget schedules in each export.

Validation: 34 protocol/simulation tests pass; the full 32-qubit demo source
compiles and its HUGR reloads on Guppy 0.21.6 as well as 1.1.3. Native Selene/Stim
execution passes eight small corrected-Bell shots and eleven full-size
decoder/routing cases, including accepted and rejected inputs. An isolated wheel
check confirms the template ships and can build/reload standalone Guppy without
the repository on the import path. New regressions check exact circuit expansion,
measurement parities, compact decision tables, every real decoder entry, and
recovery routing, including unknown prefixes.

## Circuit inspection and postselection investigation (2026-10-15)

The user asked to inspect every protocol circuit in the demo notebook and
investigate its 23.80% discard rate. The notebook now embeds three input and six
optimized circuit views (expandable, native-size, scrollable SVGs plus full
annotated Stim source). It saves circuit files under the export's `circuits/`
directory and shows selected primary/recovery/cleanup roles, gate counts, inferred
depth, and live idle units. Helpers are in `se_protocol_inspection.py`.

`estimate_protocol_ler` now returns aggregate `ProtocolDiagnostics` (stage
executions and flags, terminal rejection causes, and executed paths), plus
`discard_rate` and `discard_interval_95`. Primary flags are counted independently
of rejected shots. The notebook shows rejection/flag tables, noise-source
ablations on fixed circuits, a complementary logical-basis check, and saves
`simulation_diagnostics.json`. The user's 100,000-shot setting is preserved.

Recovery selection retains its original `recovery_objective="qubits"` default.
It tries single dual-basis extractions before any recovery/cleanup pairs; within
each group it ranks local peak qubits, operation count, and distance labels.
It does not optimize a hardware-noise discard/LER estimate. The optional
`recovery_objective="gates"` preference swaps the first two cost terms for BOTH
single and paired candidates. `protocol.with_recovery_objective("gates")`
resynthesizes decisions and compiles a separate export on the exact same circuits,
without rerunning ZX optimization. Its independent verification is optional.

The updated notebook executed end to end in `error-correction` (Guppy 0.21.6,
Stim 1.15.0) and EXACTLY reproduced the original result: 76,197/100,000 accepted,
23.803% discards, 1,086 logical-0 failures, conditional block LER 1.4253%.
Recovery/cleanup flags account for 18,993 rejected shots (79.79% of discards).
The 10,000-shot noise ablations give: default 23.97% discards, no memory 4.20%,
memory only 14.37%, two-qubit only 2.05%, quarter linear-memory rate 8.09%.
The last case changes the noise model for sensitivity analysis; it is not a
physical optimization or a changed hardware default.

The shorter-recovery preference gives 5.43% discards but increases conditional
logical-0 LER to 3.590% (20,000 shots on the same notebook circuits). A separate
100,000-shot check gives 5.408% discards / 3.593% conditional logical-0 LER.
It passes independent EC A/B checks on all 4,096 input syndrome sectors
(21,698,682 checked transitions) and all six circuit instruments pass encoded-Bell
validation. This bounded-fault result does NOT guarantee correct decoding of
every accepted two-fault history or better all-orders noisy performance.
The current default is deliberately unchanged. The user asked what "verified
recovery variant" and "current policy" meant; both were explained, and no
preference for changing the default was provided.

Memory noise also makes the logical-+ block LER much larger: the notebook's
10,000-shot complementary check gives 21.465% conditional LER. Separate
fixed-circuit 100,000-shot checks give roughly 22.49% (current exported-table
reference) and 27.57% (gates-preference reference). Always show both bases when
assessing an improvement; logical 0 is insensitive to much of the Z-biased noise.

The investigation is saved in `analysis/32_20_4_postselection/results.json` with
sampling settings, circuit fingerprints/resources, rejection counts, and proof
scope. Six baseline `.stim` snapshots are beside it. The original source/HUGR
are archived under the demo export's `baseline_before_postselection_analysis/`.
There is a relevant existing abstraction limit: shared Guppy decoder tables may
fill syndromes absent from an individual RecoveryPlan. These completions are
equivalent within the verified fault budget but can differ from `protocol.run`
on higher-order noisy histories. A reconstructed emitted-program reference
therefore gave 22.338% discards, whereas the authoritative notebook Stim-policy
rerun reproduces 23.803%. This is recorded separately in
`export_reference_checks.json`; do not confuse those baselines or claim that
decoder sharing preserves all-orders stochastic behavior.

Validation: 47 focused protocol, simulation, circuit-inspection, and existing
inspection tests pass. All notebook code cells execute; nine circuit views are
embedded. No hardware noise defaults or default acceptance policy were changed.

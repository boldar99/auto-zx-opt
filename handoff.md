# SpiderWarp project handoff

Last updated: 2026-11-14

## Overnight 5K results completed (2026-11-14)

The standard Adaptive/Steane run completed at 06:58 Europe/London on November
14, about 8h40m after launch at 22:18 on November 13. No sampler remains active.
All 20 configurations reached 5,000 attempts: 100,000 total, 400 contiguous
250-shot batches. Raw batch totals match every combined table's accepted,
logical-error, entangling-gate and accepted-duration count. Validation reports
matched engine/noise/seed/budgets and completion. Results are in
`benchmarks/sol_v2_overnight_5k/comparison.csv`, with `summary.png`/`summary.pdf`
providing the acceptance/LER overview and 95% Wilson intervals.

At one cycle, acceptance is about 74.6% Adaptive versus 92% Steane. Conditional
block LER is 2.06% versus 2.02% in logical 0 and 0.48% versus 1.47% in logical
+. At ten cycles Adaptive has only 54/5,000 (0) and 32/5,000 (+) accepted, with
10/54 and 4/32 logical failures (18.52% and 12.50% LER). Steane has 895/5,000
and 848/5,000 accepted, with 88/895 and 98/848 failures (9.83% and 11.56% LER).
Adaptive's ten-cycle intervals are wide: 10.38–30.84% (0), 4.97–28.07% (+).
These are conditional 20-logical-qubit block error probabilities over the full
requested trajectory, not per-qubit or per-cycle rates. Steane has a large
acceptance advantage; Adaptive's LER advantage at several shorter/intermediate
points trades away survivors and is not uniform across cycle counts/bases.

Accepted-trajectory entangling costs average about 315–317 per cycle Adaptive
versus 323–326 Steane; simulated native durations are roughly 1.32 s versus
1.13 s per accepted cycle. Those are modeled protocol durations, distinct from
the actual sampling wall time above. The 60/66-qubit limits remain enforced.

## Overnight standard/Steane launcher (2026-11-13)

The user requested starting the job and reduced the budget to **5,000 attempts
per point** (100,000 total). That run is now launched in a detached session,
coordinator PID 52303, log `sol_overnight_5k.log`, results
`benchmarks/sol_v2_overnight_5k/`, PID record `build/sol_overnight_5k.pid`.
All four initial native workers are active; task manifests and requests confirm
5,000 shots per point, overriding the launcher's 10K default. macOS's idle-sleep
assertion is active. Steane runs after the adaptive grid. The full 10K run was
not launched. Resume this run with the launcher plus
`--shots 5000 --results benchmarks/sol_v2_overnight_5k`.

`analysis/run_sol_overnight.sh` uses the installed `qec` Python and macOS
`caffeinate -i`. Its defaults are:
standard `adaptive_history` plus `steane_fixed_ps`, fixed alpha defaults,
repumping on, compact v2 input, both bases, cycles 1/3/5/7/10, 10,000 attempts
per point (200,000 total), four workers, batch size 250 and seed 20261113.
Default results: `benchmarks/sol_v2_overnight_10k/`. Run with
`nohup ./analysis/run_sol_overnight.sh >> sol_overnight.log 2>&1 &`.
The Sol comparator now accepts multiple values for `--only`, keeping both
methods in one combined table. A new focused CLI regression and the backend
dispatch regression both pass; the complete small native grid validates too.

The timing grid `build/sol_overnight_timing/` has 20 attempts per point (400
total), seed 739102, with two 10-shot batches per configuration. Warm second
batch times and four-worker scheduling project 16.44 hours for 10K per point;
allow roughly 16–20 hours with substantial small-pilot uncertainty. The
projection is in `timing_projection.json` beside its validated comparison.
For an 8–10-hour budget, append `--shots 5000 --results
benchmarks/sol_v2_overnight_5k` to the launcher. Repeating the identical command
resumes committed ranges. These are attempted-shot budgets; accepted counts
at longer cycles are substantially smaller. No precision claim follows from
the 20-attempt pilot's acceptance/error estimates.

## Sol sampling efficiency v2 (2026-11-13)

The current backend is `sol-helios-alpha-stim-ideal-boundaries-v2`, with default
results under `benchmarks/sol_v2/` or `benchmarks/sol_v2_comparison/`. v1 results
remain intact and cannot be mixed into v2 result trees. The user currently wants
fixed default noise and a sweep of repeated QEC cycles, without a noise-scale grid.

The compact input prefix uses a systematic CSS encoder. Logical 0 is the uniform
superposition of the H_x row span; logical + uses the span of H_x and L_x. RREF
gives independent Hadamard pivot controls and nonpivot CNOT targets. For
[[32,20,4]], either basis needs 90 CNOTs (versus the previous generic tableau
encoder's 431 for logical 0). Synchronization greedily covers all data qubits
with known stabilizers: one full-support X check suffices, replacing 32 checks
and 316 CNOTs with one check and 32 CNOTs. Four independent tableau-state tests
cover both bases of the 32-qubit and seven-qubit codes.

`--input-sync full` retains the original generic encoder and full-generator
prefix. Compact mode preserves the ideal quantum input, but may change the
starting ion layout and random draw consumption. This is a versioned native
experiment setup, not a claim of identical v1 noisy trajectories.

Sampling now launches one Python worker and one native Selene process for each
unfinished configuration. That process runs the complete remaining global shot
range and streams atomic batch files to the parent for decoding/commit. Small
batches no longer restart Python, reload the HUGR or reinitialize native plugins.
The parent reuses its protocol/ideal decoder, and a thread pool only orchestrates
the isolated workers. `--jobs` still caps concurrent native configurations.
Default batch size is 250 and controls commit granularity. On failure, process
groups are stopped and committed ranges remain resumable.

Seed mode is explicitly `default`, recorded with each batch's fixed
`stream_seed` and global `[start,start+shots)` range. Selene 0.3.7's default
uses PCG indexed by global shot ID; the quoted increment-by-one seed PSA applies
to its `legacy` mode. The actual plugin seeds were audited in
`build/sol_seed_audit.json`. Batch size, worker count and resumption do not change
per-shot outputs. The new noisy regression compares full records under those
changes; a deliberate partial-worker failure proves committed ranges are not
replayed. Independent replicas require a different user `--seed`.

The controlled 40-shot, one-cycle logical-0 benchmark with repumping and fixed
alpha noise, excluding compilation, is `build/sol_efficiency_benchmark.json`:

| Execution | Wall seconds |
|---|---:|
| Original preparation, worker restart every four shots | 25.878 |
| Original preparation, persistent native run | 17.507 |
| Optimized preparation, persistent native run | 8.953 |
| Original preparation, one 40-shot batch | 17.092 |

This is 2.89x faster than the small-batch baseline, and 1.91x faster than the
large-batch baseline. Original-prefix persistent execution matches all original
per-shot outputs/metrics. All variants' noisy native entangling counts match
their executed protocol paths. These paired performance cases deliberately
reuse the same shot IDs and are not independent scientific datasets.

Validation: 43 focused tests pass. All seven saved methods pass both-basis,
three-cycle native controls with incoming X on qubit 0 and Z on qubit 17:
28/28 accepts, zero logical failures, exact native/circuit gate counts, and
60/66 live-qubit limits (`build/sol_optimized_controls/validation.json`). Extra
standard controls at zero and ten cycles pass in both bases, with multiple
committed batches per native process (`build/sol_optimized_endpoint_controls/`).
The old likelihood-test failure documented below remains outside this change.

Full-state checkpointing, shared cycle-prefix sampling and cached transport
channels are not part of v2: the supplied native API does not expose the complete
runtime/model/simulator snapshots needed to implement those faithfully. Every
shot still executes Narsil, PECOS and Stim, with fresh per-shot state/seeds.

## Initial Sol / Helios-alpha backend v1 (2026-11-13)

`python -m spiderwarp.se_comparison` now defaults to the Sol backend. Historical
comparison commands in the sections below need `--backend stim-reference` to
select their original engine. Their benchmark files have not been rewritten.
`se_experiment run/sweep` and `native` retain their reference/control behavior;
the new `se_experiment sol` command runs one bundle or a grid with resumable
fixed attempted-shot budgets. Sol results have a separate schema/directories.

```sh
uv sync --frozen --project local_sol_example
python -m spiderwarp.se_experiment sol build/se_postselected_32_20_4 \
    --noise-scales 1 --cycles 1 3 --shots 100 --batch-size 50 \
    --results benchmarks/sol_standard
python -m spiderwarp.se_comparison --cycles 1 3 --shots 100 --jobs 2
```

`spiderwarp/se_sol.py` exports the existing frozen controller and performs ideal
final decoding/accounting. `_sol_worker.py` runs by filename in the example's
separate Python environment (override with `--sol-python` or
`SPIDERWARP_SOL_PYTHON`). It explicitly uses minimal Guppy optimization,
Eldarion targeting Sol, QtmPlatformPlugin, Narsil, PECOS `qsystem_model_alpha`,
Stim and 192 ions. `coherent_dephasing=False` and `leak2depolar=True` are fixed,
documented approximations. `--repump` separately enables repumping in Narsil;
it is off by default. The runtime adapter enforces the 60/66 live-qubit limits.
`noise_scale=1` means alpha defaults; this is not the old Pauli `p_2q` axis.

`_sol_boundary.c` is a small descriptor-ABI adapter around the installed Narsil
and PECOS libraries, built with `CC` on macOS/Linux. It supplies ideal preparation
and final readout while advancing PECOS clocks and preserving native ion mapping.
Both the noisy and zero-scale PECOS instances consume every runtime batch;
the inactive instance uses a sink simulator. Using PECOS for ideal replay matters
because Narsil emits opaque QMC commands as well as ordinary gates.

Do not remove the ideal input synchronization: this Narsil version defers
unitaries across local barriers; global barriers and the supplied utility's
drain/get_tc calls are unsupported. The wrapper nondestructively measures the
known input stabilizers with one temporary ancilla and verifies their expected
values, including the incoming Pauli. Reading those futures forces all input
preparation to finish while noise is suppressed. Reserved sleep values switch
the adapter's region without adding actual sleep. Accepted native entangling
counts match the protocol CX/CZ counts, proving preparation/readout gates stay
outside the noisy region for these exports.

Task identities record source/protocol hashes, resolved model parameters,
runtime/compiler versions, native library hashes and boundary settings. HUGR and
native executables are hash-checked and cached. Fixed task seeds plus global
shot offsets preserve individual results across batch sizes and worker counts.
Rejected shots contribute none of their earlier cycles to accepted-only LER,
gate or runtime sums. Per-shot records/metrics are optional with `--retain-shots`.

The probability controllers remain frozen under their original training model;
their 10%/25% thresholds are not recalibrated for Sol. Existing FT certificates
retain their abstract-circuit scope. The old `noise_model.py` remains available
for reference simulation, likelihood synthesis and scheduling helpers.

Validation records are under `build/sol_final_controls/`, with the reusable
driver `analysis/sol_backend_controls.py`. All 28 configurations (seven methods,
both bases, one/three cycles) passed with two shots each and incoming X on qubit
0 plus Z on qubit 17: 56 accepts, zero logical failures, exact native/circuit
entangling counts, and 60/66-qubit limits. All recorded source hashes match the
final implementation and all methods share one engine identity. The native pilot data are
`build/sol_noisy_pilot/` (120 standard attempts, 1/3 cycles, both bases) and
`build/sol_scheduled_pilot/` (60 shorter-d2 attempts, one cycle, both bases).
Both pilots enable repumping and exercise acceptance/rejection. These are smoke
budgets, not replacement production estimates. In the pilots, d4→d2 native
durations are identical for standard and shorter-d2: 1.235245 s in basis 0 and
1.224472 s in basis +. The former unit-layer depth improvement must not be
presented as a demonstrated Sol timing benefit.

The focused Sol/persistence/cost/Steane suite passed 35 tests. An additional
comparison/scheduling/likelihood selection passed 19 tests and exposed one
unchanged legacy failure: `test_full_size_likelihood_uses_complete_logical_paulis`
expects omitted probability <0.05 but obtains 0.2879573642. The likelihood
implementation, test and legacy noise sources were not modified by this work.
The stored old benchmark engines also already differ from the current tracked
`noise_model.py` source hash; no legacy simulations were resumed or resampled.

## Compact Guppy export (2026-11-13)

The adaptive exporter now compares shared direct correction tables against a
standard syndrome decoder plus exact history-conditioned XOR patches. The
choice uses the complete constant payload, because per-table savings can lose
sharing elsewhere. All six current bundles choose the patch-capable form;
individual states still use direct decoding where its local representation is
smaller. Acceptance remains a separate exact total function. Corrections may
omit rejected keys only because acceptance is checked before using them.

`spiderwarp/_se_decisions.py` implements shared XOR-offset decision diagrams.
The emitter also folds acceptance corrections into incoming edges, pools
repeated controller fields and frames, and omits unused extraction circuits.
CLI exports contain readable `steane_se_protocol.py` plus a hash-checked,
compressed `steane_se_tables_<hash>.json.zlib` companion. Both are needed to
compile from Python; the compiled HUGR is self-contained and performs no file
loading or decompression. The existing `protocol.export_guppy(...)` API retains
its single-file format. `write_adaptive_source(protocol, path)` writes the split
format; `inline_adaptive_source(source, path)` resolves it for audits/tools.

| Method | Source lines before → after | HUGR bytes before → after |
|---|---:|---:|
| Standard r+s≤1 | 1,981 → 867 | 163,673 → 122,347 |
| Shorter d2 r+s≤1 | 1,981 → 867 | 163,674 → 122,302 |
| V₂ | 230,903 → 867 | 12,141,871 → 8,682,610 |
| Probability 10% | 67,473 → 867 | 4,054,219 → 2,832,414 |
| Probability 25% | 85,065 → 867 | 4,927,612 → 3,411,731 |
| No flags | 832 → 627 | 92,956 → 71,213 |

The line counts exclude companion data; compressed table bytes are separately
reported in `analysis/32_20_4_bounded_comparison/guppy_compaction.json`. The HUGR
reductions are real representation savings, not merely moving source data.
This work does not establish a runtime speedup or a minimal-size decoder.

All six HUGRs compile/reload. The independent emitted-body audit checks all
**1,607,075 stored transitions**, exact frame masks and shared targets. A
symbolic count of accepted keys proves that every absent outcome rejects.
Validation totals **88 targeted tests passed across runs**; the final regression
run passed 11 tests and the final API/sidecar run passed 19 tests.
Native standard-base controls pass in both bases: noiseless three-cycle CSS-one
inputs accept correctly, and gate-noise smoke runs execute both acceptance and
rejection. Protocol manifests, all production engine source fingerprints,
quantum schedules, and benchmark datasets are unchanged. The source-only
export updates intentionally stay outside `adaptive_steane_se.py` so the
conservative existing simulation fingerprint and zero-shot resumes remain valid.

Reproduce without synthesis or sampling:

```sh
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/compact_guppy.py
# Or one bundle:
python -m spiderwarp.se_experiment compile build/se_postselected_32_20_4
```

Logs are under `build/guppy_compaction/`. The export audit and comparison
finalizer understand companion data. A stale legacy test expecting the optional
arbitrary-input EC-A count was corrected to the already-established total-order
contract; no verifier or input-domain behavior changed.

## Shorter-d2 scheduling extension (2026-11-13)

The user authorized a new **r+s≤1-only** method using shorter d2 schedules
under the 60-qubit limit. The method is `adaptive_scheduled`, displayed as
**Adaptive shorter d2 PS(r+s≤1)**. Its bundle is
`build/adaptive_scheduled_32_20_4/`, protocol hash
`e6530c6fedd15fef528307f63e7354d999aa0bafea679d908e81cd01a5335a23`.
Results are in `benchmarks/adaptive_scheduled_100k/`.

`spiderwarp/se_scheduling.py` expands reused ancilla lifetimes and evaluates
feasible allocations on the existing reuse frontier with a hard 60-qubit cap.
It minimizes production-scheduled data idles, then depth, ancilla idles and
qubits. This is a frontier search, not a depth-optimality claim. Only d2 gadgets
change; all standard controller nodes, frames, accepted records and d4 circuits
are preserved. No V₂ or probability variant uses the new schedules.

| Gadget | Qubits before → after | Layers before → after | Data idle units before → after | CX/CZ |
|---|---:|---:|---:|---:|
| X_d2 | 51 → 60 | 92 → 45 | 2,912 → 1,408 | 121 |
| Z_d2 | 51 → 60 | 86 → 43 | 2,720 → 1,344 | 121 |

The user asked why depth matters for linear memory. In the current code,
`DEPOLARIZE2(p_2q)` after each two-qubit gate is separate from memory.
`IDLE(gap)` produces `Z_ERROR(min(0.5, gap*p_linear_mem))`, with gaps inferred
from parallel scheduled layers. Each data qubit has one interaction per
extraction, so total data idles are `32*(depth-1)`, not a fixed amount per CNOT.
The noise implementation is unchanged.

Build and verification are complete. All 2,371 accepted records match the
standard base; single-fault response sets match; the independent FT/guard checks
pass. Standalone Guppy/HUGR compiles and reloads. Physical and native controls
pass in both bases, including all CSS-one inputs and single-fault primary
replays. Gate-only independent noise-channel distributions match exactly after
removing memory. Tests: 31 targeted checks passed, followed by 6 allocation
checks after a measurement-provenance refinement. Logs use `build/scheduled_`.
Authoritative extension records are `scheduled_build_validation.json` and
`scheduled_controls.json` in the bounded comparison study.

The matched 32-configuration production sweep is complete: **9,329,148 new
attempts**, 100,000 floor / 1,000,000 cap, seed 20261112, eight workers, about
445 seconds. The final audit passed: **224 comparison rows, 28 default endpoints,
448 raw task/batch files**, zero additional shots on resume, and all 384 original
raw files plus the 69 independent Steane files preserved. All six figure pairs
were regenerated and visually inspected; both comparison notebooks were
refreshed and executed. See `build/scheduled_finalize.log`. The comparison now
has seven methods and 65,304,036 attempts in total.

At default noise after ten cycles (100,000 attempts each):

| Method | 0 failures / accepted | 0 LER | + failures / accepted | + LER | 0 acceptance | + acceptance |
|---|---:|---:|---:|---:|---:|---:|
| Standard r+s≤1 | 100 / 36,034 | 0.2775% | 238 / 36,057 | 0.6601% | 36.034% | 36.057% |
| Shorter d2 r+s≤1 | 87 / 37,360 | 0.2329% | 190 / 37,324 | 0.5091% | 37.360% | 37.324% |

The shorter-d2 95% Wilson LER intervals are 0.1888–0.2871% for 0 and
0.4418–0.5865% for +. Both LER and AR point estimates improve against the
standard base. The logical-0 LER intervals overlap, so its reduction is not
conclusive at this shot budget. One-cycle shorter-d2 results are 118 / 94,122
(0.1254% LER, 94.122% AR) for 0 and 138 / 94,004 (0.1468% LER, 94.004% AR) for +.
A deterministic rebuild from the final source matches the sampled protocol hash.

Reproduce the extension only:

```sh
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/build.py --only adaptive_scheduled
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/controls.py --only adaptive_scheduled
PYTHONPATH=. python -m spiderwarp.se_comparison --only adaptive_scheduled --jobs 8
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/finalize.py
```

The following section records the earlier standard-base migration. Its five
adaptive methods and counts describe that completed migration; the shorter-d2
method is an additional method and dataset.

## Standard-base migration completed

The user's latest instruction makes the flag-conditioned rule the sole standard
meaning of **PS(r+s≤1)**. The previous history-only implementation and duplicate
flag-recovery method are removed. All five adaptive methods have been rebuilt on
the corrected base and rerun from empty simulation directories. Do not reuse old
adaptive data or restore the former all-single-fault acceptance requirement.

Canonical rule, implemented in `spiderwarp/se_postselection.py`:

- No raised flags: interpret measured Z_d4/X_d4 syndromes as incoming CSS errors;
  accept and correct iff max(w_X,w_Z)≤1, after removing the cumulative Pauli frame.
- Raised flags: use fault-history recovery, retaining propagated hook corrections
  of weight greater than one. Accepted histories are logically protected through
  total order one; some unflagged internal single faults postselect.

The canonical base accepts 1,089 quiet syndrome pairs and 1,282 flagged records,
2,371 total. It is `build/se_postselected_32_20_4/`, hash
`84bc3315c7b9365c882ecb2d77197fe63e42c43f65f6c94322745026d0f765f6`.
The raw `build/se_32_20_4/` program is a recovery-synthesis template, not a
plotted PS implementation. `se_flag_recovery.py` and its special build/finalizer
entry points have been removed; the behavior is now in the standard compiler.

V₂ and the epsilon methods preserve accepted records of this corrected base.
Base-rejected records must satisfy their additional rescue criterion. V₂ now
includes the remaining fault budget in any extra d2 after a record that still
has a single-fault explanation. Probability mandatory records number 2,371,
not the old 2,375; other records are analytically scored by the MAP confidence
criterion. Thresholds remain 10% and 25%, with a frozen default-noise two-response
channel model including scheduled memory and an incoming iid prior.

Data reset is recorded in
`analysis/32_20_4_bounded_comparison/standard_reset.json`. It removed 60 obsolete
paths containing 1,349 files, including every previous adaptive batch, old
comparison outputs and `analysis/32_20_4_history_attribution/` with its notebook.
No old adaptive data are archived. The 69 files in the independent canonical
Steane dataset are retained with hashes. All adaptive methods were freshly
sampled with 32 configurations, seed 20261112, 100,000 floor / 1,000,000 cap and
9,329,148 attempts each. New adaptive attempts total 46,645,740; comparison total
with retained Steane is 55,974,888.

Active methods and bundles/results:

| Method | Bundle under build | Results under benchmarks |
|---|---|---|
| adaptive_history | se_postselected_32_20_4 | se_postselected_100k |
| adaptive_order_two | adaptive_order_two_32_20_4 | adaptive_order_two_100k |
| adaptive_probability_10 | adaptive_probability_10_32_20_4 | adaptive_probability_10_100k |
| adaptive_probability_25 | adaptive_probability_25_32_20_4 | adaptive_probability_25_100k |
| adaptive_no_flags | adaptive_css_no_flags_32_20_4 | adaptive_css_no_flags_100k |
| steane_fixed_ps | steane_single_d3_fixed_ps | steane_single_d3_fixed_ps_100k_accepted |

Final validation is complete: **65 tests passed, 12 skipped**. Physical/native
controls accept/correct all noiseless CSS-one inputs for both base methods.
Each standard d4 extraction accepts 1,003 of 1,005 single-fault response classes
correctly and rejects two unflagged classes; all 771 flagged classes accept
correctly. All 1,282 flagged record paths/frames match the recovery template.
The remaining-budget regression exhausts possible single faults in the extra
V₂ measurement after each of the four base-rejected single-fault records.

V₂ controls physically replay 6,000 bounded-fault histories, including 1,524
additional accepted cases that correct exactly. Of 3,233 measurement-flip pairs,
429 are additional exact rescues. All **1,045,904 emitted V₂ transitions** match
the saved graph. Each probability threshold has 1,000 independent posterior
checks; the 10% domain is nested in 25%, with identical shared corrections.
All five standalone HUGR programs compile and reload.

The final audit covers **192 comparison rows, 24 default endpoints, 384 raw
task/batch files**, accepted-only gate/count sums, fresh batch timestamps, all
69 unchanged independent Steane files, and zero-new-shot adaptive resumes.
All six PNG/PDF figure pairs were regenerated and visually inspected. Both
comparison notebooks are executed and embed the current PNGs. Logs use
`build/standard_`; the completed final audit is `build/standard_finalize.log`.
Authoritative records are `standard_reset.json`, `standard_base_controls.json`,
`standard_raw_hashes.json`, `standard_validation.json`, the current build/control
records and `sampling_validation.json` in `analysis/32_20_4_bounded_comparison/`.
Additional obsolete logs and an orphan old policy file were also removed; the
reset manifest records them. No prior adaptive simulation data are archived.

At default noise p₂q=0.0008 after ten cycles (100,000 attempts per point):

| Method | 0 LER | + LER | 0 acceptance | + acceptance |
|---|---:|---:|---:|---:|
| r+s≤1 | 0.2775% | 0.6601% | 36.034% | 36.057% |
| r+s≤1 or V₂ | 0.4783% | 1.0385% | 39.721% | 39.671% |
| r+s≤1 or p_err≤10% | 0.4282% | 0.8427% | 39.236% | 38.806% |
| r+s≤1 or p_err≤25% | 0.4993% | 1.2539% | 45.261% | 45.059% |
| f=0, r_syn≤1 | 0.0167% | 0.3142% | 5.991% | 6.047% |
| Steane d3 RUS | 0.0617% | 0.3989% | 89.180% | 88.988% |

The zero-flag logical-0 ten-cycle point has only one failure in 5,991 accepted
shots; its 95% Wilson interval is 0.0029–0.0945%. Avoid treating that
small-count estimate as precise. Exact counts and all intervals are in the
comparison notebook and `benchmarks/steane_comparison/default_endpoints.csv`.

Current derived protocol hashes:

- adaptive_order_two: `9498a1814055a22120af233da990f2ad70407b6464441541608bb29c187bd5a8`.
- adaptive_probability_10: `05568b8fd5d2a8bcc21d83a41607f63e63abad46bd0b6568abc7d35257963949`.
- adaptive_probability_25: `541d24edafd0c72b755c49d97825ed8dfb222a9bde866df45bf181bf7f623d80`.
- adaptive_no_flags: `0475d94b4983ba212d8830c2bf5f496b7e39b68a8f158c4d45e3e833b2d531bc`.

Reproduce with the qec runtime from the repository root:

```sh
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/build.py
PYTHONPATH=. python -m spiderwarp.se_comparison --jobs 8
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/order_two_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/probability_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/finalize.py
```

Python: `/opt/homebrew/Caskroom/miniforge/base/envs/qec/bin/python`. The production
simulator, decoder and noise sources are unchanged; the independent Steane engine
remains compatible. Completed production resumes sample zero additional shots.

Accepted-trajectory statistics remain mandatory: rejected trajectories contribute
no LER or gate cost, including earlier completed cycles. Acceptance uses all
attempts. Allowed paths are d4→d4, d4→d2 and d4→d4→d2, with 312/277/433 CX/CZ,
60 adaptive qubits and no d1/third d4. Steane uses 66 qubits and 288 gates before
preparation retries. Input preparation and final decoding are ideal and happen
once per trajectory; production includes scheduled memory noise.

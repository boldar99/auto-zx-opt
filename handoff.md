# SpiderWarp project handoff

Last updated: 2026-11-13

## Probability confidence rescues completed

Two new methods are displayed: **Adaptive PS(r+s≤1 or p_err≤10%)** and
**Adaptive PS(r+s≤1 or p_err≤25%)**. Implementation:
`spiderwarp/se_probability.py`; weighted two-channel support is in
`spiderwarp/se_likelihood.py`.

Every order-one-compatible record retains the original acceptance, quantum
routing and exact Pauli frame. On other complete records, choose the MAP final
data-error Pauli class modulo stabilizers and apply its inverse correction.
Here **p_err = 1 − P(MAP class | complete record)**. This counts any wrong Pauli
class, including a remaining correctable data error; it is a sufficient criterion
for logical correction. The threshold applies only to additional records.

The default hardware noise, including scheduled memory, and incoming iid Pauli
prior at p₂q per data qubit are frozen before production and reused through the
physical-noise sweep. The incoming prior resets each cycle; it is not a physical
operation or a propagated belief. Inference conditions on at most **two nonzero
response channels**, preserving physical-site multiplicity, exclusive alternatives
and stabilizer degeneracy. Epsilon is an estimated correction-confidence cutoff,
not an all-orders bound or endpoint LER target. Omitted higher-event mass is
reported in the model metadata.

All **637,391** positive-probability additional records in this expansion were
enumerated analytically, without discovery sampling or ideal output labels.
The 10% and 25% variants add **294,509** and **346,138** records respectively,
plus the same **2,375** mandatory order-one records. The original parent selects
any third reading; other supported prefixes complete two readings. Allowed paths
remain d4→d4, d4→d2 and d4→d4→d2. The 10% acceptance domain is a subset of 25%,
and common records have identical corrections (34,064 paired graph states checked).

Bundles: `build/adaptive_probability_10_32_20_4/` and
`build/adaptive_probability_25_32_20_4/`. Protocol hashes:

- 10%: `0db6bd03938e2295a1a0e75433858d7713fde76596e123f70ebd6ab56b1c3eaf`.
- 25%: `08f58e6d201b0cb04429d0cc6d21ba6bb6e8c626106fdabd801e08108b6a535e`.

The graphs have 32,214 / 36,572 nodes; independent FT verification checks
4,738,198 / 4,809,438 transitions. HUGRs compile/reload at 4,053,522 / 4,925,760
bytes. The emitted lookup bodies match all 242,149 / 312,810 stored extraction
transitions, including sampled missing-edge rejection. Independent pointwise
posterior calculations verify 1,000 additional records per threshold. The
reference and compiled controllers also match noisy repeated-cycle sampling.

Both new sweeps are complete: **9,329,148 attempts each**, 32 matching
configurations, 100,000 floor / 1,000,000 cap, seed 20261112, 10,000 batches.
New attempts total **18,658,296**; the six-method comparison totals
**55,974,888**. Results are `benchmarks/adaptive_probability_10_100k/` and
`benchmarks/adaptive_probability_25_100k/`. The original four methods are reused.

At default noise after ten cycles, 10% accepts **39.060% (0) / 38.736% (+)**,
with conditional LER **0.4173% / 1.0455%**. At 25%, acceptance is
**45.108% / 45.185%**, with LER **0.4855% / 1.4717%**. History PS remains
**36.127% / 36.021%** acceptance and **0.2796% / 0.8106%** LER. The probability
variants improve yield with a higher conditional LER. Accepted-cycle gate costs
are approximately 317 (10%) and 316.6 (25%); rejected trajectories remain excluded.

Validation: **79 tests pass**, 12 retired-artifact skips. Authoritative new
records in `analysis/32_20_4_bounded_comparison/` are
`probability_10_build.json`, `probability_25_build.json`,
`probability_controls.json`, `probability_raw_hashes.json` and
`probability_validation.json`. The final audit checks 192 comparison rows,
24 endpoints, 384 raw files, six PNG/PDF pairs and two executed notebooks.
All 256 original raw files are preserved and completed-run resumes add zero shots.
Logs in `build/` use the `probability_` prefix.

Reproduce these methods with:

```sh
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/build_probability.py
PYTHONPATH=. python -m spiderwarp.se_comparison --only adaptive_probability_10 --jobs 4
PYTHONPATH=. python -m spiderwarp.se_comparison --only adaptive_probability_25 --jobs 4
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/probability_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/finalize_probability.py
```

## Full order-two uniqueness rescue completed

The V₂ extension added **Adaptive PS(r+s≤1 or V₂)**, implemented in
`spiderwarp/se_order_two.py`. It preserves all order-one actions/corrections.
Otherwise it accepts only if every compatible r+s≤2 history has one final data
Pauli class modulo stabilizers. All (r,s)=(2,0), (1,1), (0,2) cases are included;
two internal faults must occupy distinct physical locations. Parent-rejected
records are considered. Allowed completed paths remain d4→d4, d4→d2 and
d4→d4→d2. For an unresolved order-two d4/d4 record, the d2 basis is chosen to
separate the most uniquely correctable outcome groups, without probabilities.

Exact uniqueness uses the existing gate/reset/readout certificate model, with
no implicit idle faults. Production includes scheduled memory noise. Only the
additional rescues are certified through order two; order-two histories sharing
mandatory order-one records are not given a new correction guarantee.

Bundle `build/adaptive_order_two_32_20_4/`; protocol hash
`c1886b03bf70e35791d080b5f7b0279871bbbcb9134e9193a841ebd9e9f589be`.
The saved graph has 276,616 nodes; the independent FT verifier checks 5,504,611
transitions. All order-one regression decisions/frames match the parent.
The compact standalone Guppy export elides redundant terminal nodes and packs
control fields; all 1,045,488 stored transitions match the executable Python
lookup bodies. The HUGR compiles/reloads at 12,203,168 bytes. Existing smaller
protocol exports remain identical. The first oversized export attempt is
retained in `build/order_two_export.log`; the successful compact export is in
`build/order_two_export_compact.log`.

Results: `benchmarks/adaptive_order_two_100k/`, **9,329,148 attempts**, 32 matching
configurations, 100,000 floor / 1,000,000 cap, seed 20261112, batches of 10,000.
At that stage the four-method comparison totaled **37,316,592 attempts**. The original three
normal-shot cohorts are reused without resimulation. At default noise and ten
cycles, V₂ increases acceptance to 39.564% (0) / 39.665% (+), but conditional LER
also increases to 0.4398% / 1.2202%, versus history PS 0.2796% / 0.8106%.
This is a yield/error tradeoff; finite-order uniqueness is not an all-orders
high-confidence guarantee under production noise.

Physical controls inject 6,000 bounded-fault histories. All 1,523 additional
accepted cases return exactly to the codespace with logical state preserved.
A separate first-two-readings audit enumerates 3,233 pairs of measurement flips:
429 are additional exact rescues (408 both in the first reading, 1 both in the
second, 20 across readings). These are physical-pair counts, not probabilities.
The current targeted suite has **56 passing tests, 12 retired-artifact skips**.
All 256 task/batch files are hashed in `order_two_raw_hashes.json`; all four
completed-run resume checks add zero shots and preserve those hashes.
Six PNG/PDF pairs and both comparison notebooks include the fourth method.

Authoritative new records in `analysis/32_20_4_bounded_comparison/`:
`order_two_build_validation.json`, `order_two_controls.json`,
`order_two_export_audit.json`, `order_two_resume_validation.json`,
`order_two_raw_hashes.json`, `order_two_validation.json`, and the refreshed
`sampling_validation.json`. The final audit covers 128 rows, 16 endpoints,
256 raw files, six visually checked plot pairs and two executed notebooks.
Logs in `build/` use the `order_two_` prefix.

Reproduce the new method alone with:

```sh
PYTHONPATH=. python -m spiderwarp.se_experiment build-order-two build/se_32_20_4 --out build/adaptive_order_two_32_20_4 --emit hugr
PYTHONPATH=. python -m spiderwarp.se_comparison --only adaptive_order_two --jobs 8
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/order_two_controls.py
PYTHONPATH=. python -m spiderwarp.se_comparison --plot-only
```

## Normal-shot extension completed

The requested normal sample size is restored: **100,000 floor / 1,000,000 cap**,
**9,329,148 attempts per method**, **27,987,444** across the three existing methods.
All 32 configurations have matching budgets. Every half-shot batch was retained
and extended over disjoint shot ranges; there are no duplicated shots. Circuits,
policies, noise, seed and accepted-only statistics are unchanged.

Normal results are `benchmarks/se_postselected_100k/`,
`benchmarks/adaptive_css_no_flags_100k/` and
`benchmarks/steane_single_d3_fixed_ps_100k_accepted/`. All original half-shot
folders and the separate original `steane_single_d3_fixed_ps_100k/` dataset
(which lacks accepted gate counters) remain preserved.

The figures and both notebooks show the completed normal-budget cohort. The
zero-flag default-endpoint Wilson intervals are 28–44% narrower than before.
Its ten-cycle logical-0 estimate still has only four failures, so substantial
uncertainty remains: **0.0666% LER, 95% CI 0.0259–0.1711%**. Logical + has
**0.3123% LER, 95% CI 0.2000–0.4873%**, from 19 failures. See
`zero_flag_uncertainty.json` for the exact old/new counts and intervals.

`normal_shot_promotion.json` records the completed extension, original hashes
and preserved prefixes. `verify_normal_shots.py` verifies all 192 normal raw
files, preserved original data, contiguous shot ranges, counts, saved notebook
outputs and zero-new-shot resumes. Three allocation/resume tests pass in
`build/normal_shots_tests.log`. Logs are `normal_shots_history.log`,
`normal_shots_no_flags.log`, `normal_shots_steane.log`,
`normal_shots_presentation.log` and `normal_shots_validation.log` in `build/`.

This promotion is scoped to the three methods recorded in
`normal_shot_promotion.json`; independently added experiments are not included
in its sample totals or prefix audit.

## Authoritative fault-count correction

The user clarified that one internal fault remains **s=1** even if it propagates
to multiple data errors. With no incoming error, **r=0, s=1, r+s=1**. Flags and
adaptive follow-up identify the propagated pattern and allow its correction.
Neither its Pauli weight nor its raw syndrome weight is the number of faults.

The previous extra full-adaptive CSS guard incorrectly capped propagated
correction/raw syndrome weight. It rejected valid single internal faults and
was not the requested interpretation of r_syn. That implementation and curve
are retired. The corrected `compile_css_candidate(parent, 'css')` uses the same
fault-aware order-one history guard as `postselect_order_one(parent)`.

**The two corrected full-adaptive modes are exactly the same protocol.** Their
protocol hash, standalone Guppy source and HUGR binary are identical. Only
**Adaptive PS(r+s≤1)** is displayed; the valid existing history benchmark is
reused. There is no need to resimulate an identical circuit/controller.

Canonical bundle: `build/se_postselected_32_20_4/`. Corrected compatibility alias:
`build/adaptive_css_32_20_4/`. Both have protocol hash
`62da3175473f60675125c331b7cc2ac9c5bdee5ed71e23f7fe04e8da439cfe44`.
The common HUGR is 163,329 bytes.

The incorrect guard bundle is preserved at
`build/retired_raw_syndrome_guard_32_20_4/`, and its raw results at
`benchmarks/retired/adaptive_raw_syndrome_guard_50k/`. These are excluded from
current comparisons. Their original hash was
`1cc835f035794eac3e89819b2d84699199964d41d506607108e4143543534c99`.
See `retired_raw_syndrome_guard.json` and
`raw_hashes_before_fault_count_fix.json` in the current study directory.

## Executed adaptive protocol and scope

Allowed completed paths are **d4→d4→d2**, **d4→d4**, and **d4→d2**.
Start at Z_d4. If its flags are zero, measure X_d4; otherwise use X_d2.
A second-d4 flag or unresolved correction can use one Z_d2 or X_d2 follow-up,
selected from compatible histories. There is no d1, no third d4 and no fourth
reading. Early rejected prefixes can be shorter.

Incoming CSS order is **r=max(w_X,w_Z)**; s counts internal faulty locations.
For history PS, analyze **r+s≤2** for detection and reject when no
**r+s≤1** explanation remains. V₂ adds the uniqueness rescues described above.
The full adaptive mode guarantees acceptance/logical correctness through total
order one, including single faults whose identified correction has weight two.
Higher-order histories indistinguishable from low-order records cannot be
rejected by an oracle. In particular, a two-input-error syndrome can sometimes
also arise from one internal readout fault; an independent raw-weight cap would
sacrifice the single-fault acceptance guarantee again.

The internal parent is `build/se_32_20_4/` and is not a plotted baseline.
`FTContract` has correctable=1, logical_order=1, fault_order=2,
accept_clean_through=1, accepted_filter_order=1 and input_scope='total-order'.
Its verification covers 3,947,446 abstract transitions, three maximum readings,
433 worst-case CX/CZ and 60 peak qubits. The earlier hidden-order-two residual
constraint and two-total-reading cap are not used.

## Separate zero-flag experiment

The requested “PS(s=0 and r_syn≤1), meaning no flags raised” experiment is
labelled **Adaptive PS(f=0, r_syn≤1)**. Here f=0 is observable; it does not prove
that the actual internal-fault count is zero. This stricter experiment requires
all measured flags zero and direct-readout CSS uniqueness through component
weights ≤2. It retains all noiseless incoming CSS-one errors, including X and Z
on different qubits, but can reject unflagged internal faults. It does not claim
the full adaptive method's acceptance guarantee for every single internal fault.

Its only accepted path is **Z_d4→X_d4**, exactly **312 CX/CZ per cycle**; no d2
or d1 is executed. Its existing program and benchmark are unchanged by the
fault-count correction. Bundle `build/adaptive_css_no_flags_32_20_4/`; protocol
hash `131e76ee8a48727bc49c57fe1e91192578cd7627931903ad0644f87982874c88`;
HUGR 92,948 bytes.

## Current comparison and statistics

| Displayed method | Bundle | Results |
|---|---|---|
| Adaptive PS(r+s≤1) | `build/se_postselected_32_20_4/` | `benchmarks/se_postselected_100k/` |
| Adaptive PS(r+s≤1 or V₂) | `build/adaptive_order_two_32_20_4/` | `benchmarks/adaptive_order_two_100k/` |
| Adaptive PS(r+s≤1 or p_err≤10%) | `build/adaptive_probability_10_32_20_4/` | `benchmarks/adaptive_probability_10_100k/` |
| Adaptive PS(r+s≤1 or p_err≤25%) | `build/adaptive_probability_25_32_20_4/` | `benchmarks/adaptive_probability_25_100k/` |
| Adaptive PS(f=0, r_syn≤1) | `build/adaptive_css_no_flags_32_20_4/` | `benchmarks/adaptive_css_no_flags_100k/` |
| Steane d3 RUS PS(r_syn≤1) | `build/steane_single_d3_fixed_ps/` | `benchmarks/steane_single_d3_fixed_ps_100k_accepted/` |

Every method in this cohort has 32 matching configurations, a 100,000 floor,
1,000,000 cap, 10,000 batches, seed 20261112 and inverse-square allocation:
**9,329,148 attempts per method**, **55,974,888 attempts** total. The corrected
alias is not counted as a separate method. The initial 4,664,576 samples per
original method are preserved as exact batch prefixes; V₂ and both probability variants use fresh sweeps.

**LER and gate costs exclude rejected trajectories entirely**, including earlier
completed cycles of a shot that later rejects. Cost is accepted_cnots / accepted
/ N. Acceptance rate is accepted / attempted to show yield. Raw all-attempt counts
remain for auditing. Noisy discarded ancillas inside an eventually accepted
Steane RUS trajectory still contribute to that accepted trajectory's cost.

Adaptive d4/d2 gadgets include coupling and cost 156/121 CX/CZ, giving completed
path costs **433, 312 and 277**. Mean accepted-cycle cost lies between 277 and 433.
The zero-flag experiment always costs 312 per accepted cycle. Steane prepares
d3 ancillas by RUS until all preparation flags are zero, then reads the data once
per CSS basis. Its d3 preparation file is `FTStatePrep/misc/32_20_4_zero_d3.stim`
(112 CX/CZ). Two preparations plus two 32-CX couplings cost 288 before retries;
peak qubits are 66. No descending, ladder or repeated-per-basis Steane is used.
The valid original fixed 100k-floor dataset is preserved separately.

Production includes scheduled memory noise, ideal input preparation once and
ideal final decoding once. LER is the conditional endpoint block error across
20 logical outputs. LER/N is an endpoint normalization, not a first-failure
hazard. Plots use 95% Wilson intervals from actual accepted counts. Adaptive
methods use circles and Steane squares.

## Default-noise endpoints, p₂q=0.0008

Each row has 100,000 attempted trajectories. Costs condition on full acceptance.

| Method | Input | N | Failures / accepted | LER (%) | AR (%) | CX/CZ / accepted cycle |
|---|---|---:|---:|---:|---:|---:|
| Adaptive PS(r+s≤1) | + | 1 | 194 / 93,596 | 0.2073 | 93.596 | 317.83 |
| Adaptive PS(r+s≤1) | 0 | 1 | 147 / 93,649 | 0.1570 | 93.649 | 318.10 |
| Adaptive PS(r+s≤1) | + | 10 | 292 / 36,021 | 0.8106 | 36.021 | 317.20 |
| Adaptive PS(r+s≤1) | 0 | 10 | 101 / 36,127 | 0.2796 | 36.127 | 317.14 |
| Adaptive PS(r+s≤1 or V₂) | + | 1 | 204 / 94,351 | 0.2162 | 94.351 | 318.21 |
| Adaptive PS(r+s≤1 or V₂) | 0 | 1 | 173 / 94,379 | 0.1833 | 94.379 | 318.32 |
| Adaptive PS(r+s≤1 or V₂) | + | 10 | 484 / 39,665 | 1.2202 | 39.665 | 317.60 |
| Adaptive PS(r+s≤1 or V₂) | 0 | 10 | 174 / 39,564 | 0.4398 | 39.564 | 317.71 |
| Adaptive PS(r+s≤1 or p_err≤10%) | + | 1 | 216 / 94,322 | 0.2290 | 94.322 | 317.94 |
| Adaptive PS(r+s≤1 or p_err≤10%) | 0 | 1 | 204 / 94,511 | 0.2158 | 94.511 | 317.92 |
| Adaptive PS(r+s≤1 or p_err≤10%) | + | 10 | 405 / 38,736 | 1.0455 | 38.736 | 316.93 |
| Adaptive PS(r+s≤1 or p_err≤10%) | 0 | 10 | 163 / 39,060 | 0.4173 | 39.060 | 317.03 |
| Adaptive PS(r+s≤1 or p_err≤25%) | + | 1 | 234 / 95,314 | 0.2455 | 95.314 | 318.01 |
| Adaptive PS(r+s≤1 or p_err≤25%) | 0 | 1 | 194 / 95,398 | 0.2034 | 95.398 | 317.76 |
| Adaptive PS(r+s≤1 or p_err≤25%) | + | 10 | 665 / 45,185 | 1.4717 | 45.185 | 316.64 |
| Adaptive PS(r+s≤1 or p_err≤25%) | 0 | 10 | 219 / 45,108 | 0.4855 | 45.108 | 316.61 |
| Adaptive PS(f=0, r_syn≤1) | + | 1 | 63 / 75,808 | 0.0831 | 75.808 | 312.00 |
| Adaptive PS(f=0, r_syn≤1) | 0 | 1 | 28 / 75,728 | 0.0370 | 75.728 | 312.00 |
| Adaptive PS(f=0, r_syn≤1) | + | 10 | 19 / 6,084 | 0.3123 | 6.084 | 312.00 |
| Adaptive PS(f=0, r_syn≤1) | 0 | 10 | 4 / 6,008 | 0.0666 | 6.008 | 312.00 |
| Steane d3 RUS PS(r_syn≤1) | + | 1 | 51 / 99,284 | 0.0514 | 99.284 | 315.59 |
| Steane d3 RUS PS(r_syn≤1) | 0 | 1 | 39 / 99,312 | 0.0393 | 99.312 | 315.28 |
| Steane d3 RUS PS(r_syn≤1) | + | 10 | 355 / 88,988 | 0.3989 | 88.988 | 315.54 |
| Steane d3 RUS PS(r_syn≤1) | 0 | 10 | 55 / 89,180 | 0.0617 | 89.180 | 315.54 |

Zero-flag postselection lowers the sampled adaptive LER but reduces acceptance
to approximately 75.8% for one cycle and 6.0% for ten cycles. Conditioning gate
costs on acceptance does not account for this yield penalty. The ten-cycle
endpoints now have 4 and 19 failures; use their saved confidence intervals.

## Validation and reproduction

The earlier fault-count fix had **34 passing tests**, with 12 skips for retired research
artifacts. Its regression replays all 1,005 single-fault response classes for
each d4 position and both bases: **4,020 accepted and logically correct cases**,
including flagged histories requiring syndrome-weight-two corrections.
Guppy and HUGR byte identity independently establish equivalence of the full
adaptive aliases. No duplicate benchmark was generated.

Active physical controls cover 4,356 CSS-one inputs and 8,040 first/second-gadget
single-fault response replays across the history and zero-flag methods. History
accepts/corrects all of its single-fault cases; 771 second-d4 classes per basis
use an accepted third d2. The zero-flag experiment accepts/corrects 232 of 1,005
classes per gadget/basis and rejects 773. Active native controls have 32 clean
three-cycle shots and 400 noisy gate-only shots. Gate-only native controls are
execution checks; their noise model differs from production memory noise.

All 192 comparison rows, 24 endpoints, raw batch counts, Wilson intervals and
accepted-stage/path gate reconstructions pass the audit. All six PNG/PDF pairs
were visually checked and both notebooks contain the current six-method
results. The original three-method promotion audit below remains separately scoped. All 192 normal-budget task/batch files are hashed. The 50k batches are
verified as exact prefixes of the extended data; the half-shot folders remain
byte-identical. The 64 previously retired raw files remain preserved.
Completed-run resume checks add zero shots.
Fixed Steane protocol/source/HUGR/export hashes are preserved. The user's
separate `notebooks/32_20_4.ipynb` is untouched by this work.

The study is `analysis/32_20_4_bounded_comparison/`. Authoritative records:
`fault_count_fix_validation.json`, `validation.json`, `sampling_validation.json`,
`build_validation.json`, `controls.json`, `raw_hashes.json` and
`retired_raw_syndrome_guard.json`. Latest logs in `build/` include the normal-shot logs above and
`fault_count_fix_build.log`, `fault_count_fix_tests.log` and
`fault_count_fix_presentation.log`. Prior no-flag build/production/control logs
remain under `no_flags_*.log`.

```sh
PYTHONPATH=. python -m spiderwarp.se_experiment build --paired --distances 2 4 --out build/se_32_20_4 --emit none
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/build.py
PYTHONPATH=. python -m spiderwarp.se_comparison --jobs 8
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/controls.py
PYTHONPATH=. python -m spiderwarp.se_comparison --plot-only
```

`build.py --only adaptive_syndrome` rebuilds the corrected compatibility alias.
Its benchmark is the canonical history result; it is deliberately absent from
the displayed method registry. Earlier Adaptive Baseline and obsolete pilots,
wrong two-reading runs, incorrect/descending Steane variants remain excluded.

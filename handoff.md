# SpiderWarp project handoff

Last updated: 2026-11-13

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

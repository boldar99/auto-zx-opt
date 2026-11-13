# SpiderWarp project handoff

Last updated: 2026-11-13

## Matched-budget comparison and experiment review (2026-11-13)

**Current update: sampling extensions in progress; final count/figure audits pending.**
This section supersedes earlier active-comparison descriptions that included
0.1x Steane PS curves. The user asked for a summary of everything tried, better
names, equal samples in the plots, promising untried ideas, and this handoff update.
The chosen error assumption remains **each CSS component weight <=2**, modulo
stabilizers. The data code is [[32,20,4]], with Steane-style extraction; it is
not the seven-qubit Steane code. Keep the per-cycle Steane ladder and failed-EC-
filter aggressive-recovery ablation excluded from the main analysis.

### Display names and dataset identity

Names describe the decision rule. RUS retries a discarded ancilla before data
coupling; PS discards a data trajectory. The descending ancilla sequence is
**d4 -> d2 -> d1, reset to d4 for each fresh ancilla**, including confirmation
reads. All display names come from `spiderwarp/se_comparison_names.py`.
Protocol IDs, circuit assets, source/HUGR hashes and bundle paths are unchanged.

| Display name | Previous label / ID | Full results directory |
|---|---|---|
| **Adaptive baseline** | Adaptive original | `benchmarks/se_100k/` |
| **Adaptive history PS** | Adaptive strict PS | `benchmarks/se_postselected_100k/` |
| **Adaptive syndrome PS** | Adaptive CSS PS / css | `benchmarks/adaptive_css_100k/` |
| **Adaptive verified recovery** | Adaptive CSS rescue PS / css-recover-safe | `benchmarks/adaptive_css_recover_safe_100k/` |
| **Steane d4 RUS** | Steane RUS d4 | `benchmarks/steane_rus_d4_100k/` |
| **Steane descending RUS** | Steane ladder / ancilla | `benchmarks/steane_rus_ladder_ancilla_100k/` |
| **Steane d4 RUS + syndrome PS** | Steane d4 PS | `benchmarks/steane_css_ps_d4_100k/` |
| **Steane descending RUS + syndrome PS** | Steane ladder / ancilla PS | `benchmarks/steane_css_ps_ladder_ancilla_100k/` |

Adaptive plots append `(60 qubits)` to these names; Steane plots append
`(66 qubits)`. The selected adaptive controllers still use the same quantum
circuits, at most four extractions and a 589-CNOT worst-case bound. No circuit
reuse, allocation or scheduling optimization was introduced in this update.

### Equal attempted samples at every plotted configuration

All eight displayed methods must have **32 matching configurations and
9,329,148 attempted trajectories each**: 100k floor, 1M cap, inverse-square
allocation, seed 20261112. Counts are compared **per configuration**, including
p, N, basis, full noise profile, sweep role and seed; matching totals alone is
insufficient. Every default-noise repetition point has **100,000 attempts per
basis/method**. Accepted counts intentionally differ because acceptance rate is
measured. Wilson 95% intervals use each method's actual accepted/failure counts.

Only the two Steane syndrome-PS datasets were undersampled. The extension script
`analysis/32_20_4_syndrome_uniqueness/full_steane_ps.py` creates separate `_100k`
folders, reuses every archived `_10k` task and batch, and adds **8,396,230 attempts
per mode** (16,792,460 total). Each contains 932,918 old plus 8,396,230 new shots.
Original 1k batches are preserved; additional batches use up to 10k shots with
deterministic task offsets. `seed_provenance.json` preserves old run/validation
metadata and all archived raw hashes. The saved sampler engine identity matches.
The archived quick directories and every older full dataset remain unchanged.

`se_comparison` always requires equal budgets. It defaults to the full Steane PS
folders and refuses incomplete/unequal runs instead of silently including the
quick data. `--steane-postselected-results` is the explicit-path option; the old
`--quick-postselected-results` flag is an alias subject to the same count check.
The lower-level `save_comparison(...require_equal_budgets=False)` remains only
for explicit historical/diagnostic work; canonical plots never use that escape.

The main table contains **256 count rows** (8 x 32). Both notebooks expose an
Attempts column in their 32-row default-noise endpoint tables, five main figure
pairs and the focused LER/acceptance tradeoff (four adaptive + two Steane PS).
Figures state equal attempts, and the focused figure states 100,000 per point.
Archived pilots are discovery evidence, not extra curves in the main comparison.

### What has actually been tried

1. **Original adaptive and Steane controls.** The original verified adaptive
   controller, d4-only ancilla RUS, descending per-ancilla RUS, and historical
   per-cycle descending RUS were fully benchmarked. The user excluded the
   per-cycle variant. Steane confirms two consecutive syndromes per CSS basis
   and includes persistent data memory during every serial preparation/retry.
2. **Flag-based rejection and risk diagnostics.** Zero d4 flags, zero flags
   everywhere, two flagged extractions, sums of local minimum fault orders and
   learned raw-mask risk thresholds were tested with discovery plus independent
   holdout records. They gave worse LER/yield tradeoffs than joint flag/syndrome
   history inference. For example, zero d4 flags retains about 66% at N=1 with
   LER 0.0772% / 0.7383% on the independent holdout. Conditional risk given a
   raised d4 flag in an accepted original trajectory is 5.0757% / 9.5649%.
   These are conditional associations, not isolated causal fault probabilities.
   See `analysis/32_20_4_branch_assessment/` for counts, intervals and patterns.
3. **History PS.** Reject when no observation-compatible explanation with
   r+s<=1 exists. r is the maximum CSS component order; s counts faulty
   locations, not raised flag bits. This uses the full raw flag/syndrome
   record and is the strongest measured low-LER adaptive rule.
4. **Syndrome uniqueness / Steane data PS.** Exhaustive enumeration proves
   none of this code's weight-two CSS-component syndromes is uniquely
   correctable: all 31 have 16 stabilizer-inequivalent classes. Per component,
   accept identity plus the 32 one-error syndromes; reject the rest after
   adjusting for the cumulative frame. Both Steane RUS families were tested
   with that data rejection, first at 0.1x and now extended to full budgets.
5. **Ten adaptive pilots.** Fixed 10k attempts per logical basis at N=1/N=10,
   400,000 discovery shots total, seed 2026111301. Full production uses independent
   seed 20261112. The pilot N=10 results are below; they are not the final
   production LERs. Adapter pilot resource costs are not production gate counts.

| Pilot name | Internal ID | N=10 conditional LER, 0 / + | Acceptance, 0 / + |
|---|---|---|---|
| Syndrome PS | `css` | 11.039% / 22.255% | 33.97% / 33.97% |
| Syndrome PS + flagged-history check | `css-flag` | 0.199% / 1.127% | 15.09% / 15.09% |
| Syndrome PS + X/Z reversal | `css-flag-reversed` | 0.129% / 1.482% | 15.52% / 15.52% |
| Syndrome PS + final d2 checks | `css-flag-check` | 1.604% / 0.535% | 1.87% / 1.87% |
| Separate CSS histories | `css-separated` | 0.207% / 1.519% | 14.48% / 14.48% |
| Separate CSS histories + frame update | `css-separated-retarget` | 0.448% / 2.073% | 17.85% / 17.85% |
| Bounded recovery | `css-recover` | 0.685% / 3.493% | 45.23% / 45.23% |
| Exact recovery without the EC filter constraint | `css-recover-exact` | 0.608% / 2.017% | 36.20% / 36.20% |
| Bounded recovery without d1 | `css-recover-no-d1` | 0.518% / 2.507% | 36.69% / 36.69% |
| Verified recovery | `css-recover-safe` | 0.481% / 1.951% | 35.37% / 35.37% |

The syndrome baseline and verified recovery were promoted to full 9,329,148-shot
sweeps. The aggressive exact-recovery rule also has a full research sweep but
fails the original arbitrary-input EC filter and is excluded. Verification
caught record Z_d4(flags136,syndrome50), X_d1(flags0,syndrome57), history(-1,1,3648).
Verified recovery repairs this by preserving the original frame syndrome;
it independently verifies the **same strict-PS contract** as history PS.
Other pilots were not promoted. In particular, two full final d2 checks gave
only ~1.9% ten-cycle acceptance, and simple X/Z reversal had no consistent gain.

At full production/default noise, history PS has N=10 LER **0.09027% / 0.91315%**,
AR **25.480% / 25.516%**; verified recovery gives **0.37422% / 1.98086%**,
AR **35.808% / 35.641%**. It yields about 40% more accepted outputs at higher
conditional error, not universal dominance. Syndrome PS gives
**9.92418% / 22.85487%**, AR **33.897% / 33.879%**, so is inferior here.
Full counts and intervals for all methods are in the regenerated notebooks,
`benchmarks/steane_comparison/comparison.csv`, and
`analysis/32_20_4_adaptive_improvements/default_endpoints.csv`.

The full-budget Steane syndrome-PS endpoint results (100,000 attempts per
basis/point) are:

| Method | Cycles | Failure / accepted, 0 / + | Conditional LER, 0 / + | Acceptance, 0 / + |
|---|---|---|---|---|
| Steane d4 RUS + syndrome PS | 1 | 107/97628 / 216/97601 | 0.10960% / 0.22131% | 97.628% / 97.601% |
| Steane d4 RUS + syndrome PS | 10 | 138/76439 / 1830/76715 | 0.18054% / 2.38545% | 76.439% / 76.715% |
| Steane descending RUS + syndrome PS | 1 | 267/97746 / 266/97719 | 0.27316% / 0.27221% | 97.746% / 97.719% |
| Steane descending RUS + syndrome PS | 10 | 562/76766 / 2059/76820 | 0.73209% / 2.68029% | 76.766% / 76.820% |

Steane d4 syndrome PS retains much more yield than history PS with higher
ten-cycle conditional LER. Neither dominates the LER/acceptance frontier.
Descending RUS has higher ten-cycle LER than d4 RUS here; the full intervals
are in the summaries. These new counts supersede archived quick-run estimates.

6. **Noise-source and correctness investigations.** Channel ablations implicate
   both two-qubit faults and memory, with nonadditive interactions. Original
   N=1 phase LER was 3.635%; memory off gives 2.019%, two-qubit faults off 0.982%,
   and initialization faults off 2.880%. Most measured memory effect was during
   d4. These ablations are not realizable schedule improvements. Independent
   low-order physical replay found no original low-order logical bug; it also
   found silent two-fault logical witnesses with clean records. Verified recovery's
   stronger contract has 7,333,909 checked transitions over all 4,096 arbitrary
   input syndrome sectors. The aggressive ablation's filter failure is documented
   separately; do not generalize its disqualification to the original controller.

### Ranked ideas not yet implemented or sampled for this code

These are hypotheses, not established gains. Freeze selection before full
production, use independent holdout records and retain the declared fault
contract. An actual fault count is never an allowed runtime input.

1. **Noise-weighted logical-risk postselection over complete histories.**
   Weight actual gate/reset/readout/memory mechanisms, sum degeneracy within
   stabilizer/logical classes, and admit or reject using estimated logical risk
   (or the likelihood gap between classes). Protect mandatory low-order
   acceptance/filter cases, and tune additional rescues on separate data.
   This is my first classical experiment: no extra gates, direct LER/AR control.
   Under the current mandatory one-fault acceptance contract, thresholds mainly
   choose extra rescues. Rejecting mandatory histories needs an explicitly
   relaxed acceptance contract; do not promise a lower error floor for free.
   Raw flag-mask threshold learning *was* tried; calibrated full-history
   logical-class likelihoods were not. General precedent:
   [Smith, Brown & Bartlett, exclusive decoders](https://arxiv.org/abs/2405.03766).
2. **Retain residual-error uncertainty across cycles.** Data and the hard Pauli
   frame currently persist, while compatibility inference restarts. Carry a
   small belief/set of likely residual errors into a sliding two-cycle decoder.
   Existing later readings may resolve timing ambiguity without extra qubits;
   delayed rejection can cost more gates, so measure accepted error and total
   work together. Temporal decoding precedent:
   [AlphaQubit](https://www.nature.com/articles/s41586-024-08148-8).
3. **Targeted verification chosen by expected information gain.** On rare risky
   branches, select an X/Z check only if competing hypotheses predict different
   current syndromes. Phase-heavy paths are a natural target. This differs from
   the already-failed blanket pair of final d2 rounds. Choose enough verification
   to preserve mandatory acceptance under a single fault, and formally verify
   every changed path. Ordinary stabilizer-only checks can never separate
   already identical current syndromes from different logical classes; do not
   suggest a lookahead can resolve that algebraic ambiguity. Related work:
   [Adaptive Syndrome Extraction](https://arxiv.org/abs/2502.14835).
4. **Schedule/circuit optimization for accepted logical faults and idle time.**
   Optimize gate ordering and protection against measured malignant pairs and
   memory exposure rather than CNOT count alone. Reverify under the 60-qubit
   cap and resimulate the actual schedule. Simple X/Z reversal was tested;
   noise-aware internal rescheduling/resynthesis was not. This is my first
   circuit-level experiment. Related scheduling work:
   [AlphaSyndrome](https://arxiv.org/abs/2601.12509). No performance numbers
   from other codes are predictions for [[32,20,4]].
5. **Force just one distance drop with observable rejection.** The user's earlier
   d4 -> d1-only / d4 -> d2-only controllers remain unimplemented. Existing paths
   include some such transitions but do not impose either as a whole policy.
   Synthesize them with ambiguity-based rejection and reverify; d1 has no flags,
   so postselecting every *actual* subsequent fault is not an observable rule.

A generator-basis change alone is only a syndrome relabeling and cannot improve
transversal Steane cost; it would require actual verified circuit resynthesis.
Likewise spin echo does not remove the current stochastic Markovian memory
channel. Neither should be advertised as a free simulated improvement.

### Reproduction, provenance and validation for this update

Use `/opt/homebrew/Caskroom/miniforge/base/envs/qec/bin/python` (qec env) from the
repository root. The extension script is resumable and preserves archived pilots.

```sh
python analysis/32_20_4_syndrome_uniqueness/full_steane_ps.py --mode d4 --jobs 8
python analysis/32_20_4_syndrome_uniqueness/full_steane_ps.py --mode ladder-ancilla --jobs 8
python analysis/32_20_4_adaptive_improvements/finalize.py
```

`audit_full_steane_ps.py` independently checks full target counts, intervals,
cycle accounting, unchanged sampler identity, source regeneration/HUGR reload,
archived hashes and raw-prefix reuse, followed by real zero-work resumes.
Circuit controls are inherited by explicit provenance from the unchanged
archived controls; this update does not claim a new native noise-model LER.
The finalizer checks old datasets, refreshes count audits/five main figure pairs,
the focused tradeoff pair and all tagged comparison outputs in both notebooks.
The notebooks preserve earlier expensive demo outputs. Seven focused budget,
resume and plotting regression tests pass (`build/matched_comparison_tests.log`);
the pre-existing Matplotlib/NumPy scalar-conversion deprecation warnings remain.
Previous adaptive 74+21 regressions and physical/native/export audits remain
valid for the unchanged protocols and are recorded separately as prior tests.

## Adaptive CSS PS improvement study: complete (2026-11-13)

Historical checkpoint before equal-budget Steane PS extension; see newest section.

The user authorized adaptive improvements, pilot selection for low LER / decent
acceptance, and original-size production under **CSS component weights <=2**.
All work is complete. Keep the per-cycle Steane ladder excluded. Original raw
production data and quantum circuit assets remain unchanged; no reuse or
allocation optimization was introduced.

Ten fixed candidate rules were piloted on 10k shots per logical basis at N=1
and N=10: **400,000 discovery shots**, seed 2026111301. Results and definitions
are in `analysis/32_20_4_adaptive_improvements/`, including `all_pilots.csv`,
`pilot*.{csv,json}`, `selection_report.json`, and reproducible scripts. No
runtime decision uses the prepared-state label, hidden fault counts or final
ideal logical readout. Full production uses independent seed 20261112 and
fixed original budgets, rather than stopping on observed failures.

Selected methods:

1. **Adaptive CSS PS** (`css`): preserve the original actions/frames and reject
   when the new input-relative correction syndrome is ambiguous through the
   CSS weight-two bound. This conservative baseline allows single-fault
   postselection. It is worse than the strict history filter in this noise model.
2. **Adaptive CSS rescue PS** (`css-recover-safe`): retain strict order-one
   records. An otherwise rejected record can be rescued only when *all* full
   Pauli explanations with incoming CSS components <=1 and at most one internal
   faulty location identify the same final data class modulo stabilizers.
   The rescue must preserve the original frame syndrome. This retains the
   original arbitrary-input filters and only adds observation-conditioned
   logical/stabilizer frame updates. It provides a useful acceptance/LER tradeoff.

Both full selected sweeps have **32 completed configurations and 9,329,148
attempted shots each** (100k floor, 1M cap, inverse-square allocation, 10k batches,
8 workers, seed20261112). At exact default noise p_2q=0.0008, entries below are
logical 0 / logical +; LER means conditional endpoint block LER, not LER/N.

| Method | Cycles | Conditional LER, 0 / + | Acceptance, 0 / + |
|---|---|---|---|
| Strict history PS | 1 | 0.04510% / 0.41579% | 90.905% / 91.151% |
| Strict history PS | 10 | 0.09027% / 0.91315% | 25.480% / 25.516% |
| CSS-only PS | 1 | 1.30374% / 2.83046% | 92.043% / 92.317% |
| CSS-only PS | 10 | 9.92418% / 22.85487% | 33.897% / 33.879% |
| Verified CSS rescue PS | 1 | 0.09645% / 0.47056% | 92.278% / 92.444% |
| Verified CSS rescue PS | 10 | 0.37422% / 1.98086% | 35.808% / 35.641% |

The rescue yields about 40% more accepted outputs after ten cycles, with higher
conditional error. Strict PS remains the low-LER choice. No universal dominance
is claimed. Each default-noise repetition point has 100,000 attempts per basis.
Raw rescue counts: N=1 89/92278 and 435/92444; N=10 134/35808 and 706/35641.
All Wilson intervals and the N=3/5/7 points are in the summary CSVs/notebooks.

The new methods retain the original **60-qubit cap, 589-CNOT worst-case bound,
at most four extractions, and quantum circuit assets**. No new optimality claim
is made. The verified rescue has exactly the strict-PS contract:
correctable=1, logical_order=1, fault_order=2, accept_clean_through=1. Its
independent checker covers **7,333,909 transitions / all 4,096 arbitrary input
syndrome sectors**. The CSS-only baseline verifies clean mandatory acceptance
and logical preservation of accepted r+s<=1 histories (correctable=0,
logical_order=1, fault_order=1, accept_clean_through=0); 154 of 1,005 representative
first-gadget single-fault responses are postselected in that conservative rule.

Artifacts (standalone Guppy + compiled/reloaded HUGR + ordinary protocol bundle):

- `build/adaptive_css_32_20_4/`: 54,711 nodes; hash
  `08ebce8890734f155dc0724418347b4a7519ec995127e4149dbccb22b9fa11c7`;
  HUGR 6,196,004 bytes; independent contract check 17,227 transitions.
- `build/adaptive_css_recover_safe_32_20_4/`: 104,623 nodes; hash
  `c0254535e1e1468c29a81bfbc0214170c1336b29c2a4e66ff6ceb4001dd3e936`;
  HUGR 4,881,838 bytes; full strict-PS contract verified as above.

Datasets: `benchmarks/adaptive_css_100k/` and
`benchmarks/adaptive_css_recover_safe_100k/`. Each has a complete `run.json`,
local `validation.json`, raw batches/tasks, diagnostics, summary, and standalone
figures. The saved ordinary DAGs execute through the existing sampler/loader;
no new simulation engine or runtime fault enumerator is used.

An aggressive exact-recovery ablation (`css-recover-exact`) was also sampled
at full size: `benchmarks/adaptive_css_recover_exact_100k/` (9,329,148 attempts).
It preserves accepted r+s<=1 logical behavior but **fails the original arbitrary
input CSS filter**: record Z_d4(flags136, syndrome50), X_d1(flags0, syndrome57),
history(-1,1,3648). See `stronger_contract.json`. It is excluded from the main
comparison. Its full source/HUGR and physical controls are retained for research.
Executable-wrapper compilation was stopped after it was disqualified; no native
execution claim is made for this ablation. In total this study added 27,987,444
full-run attempts plus 400,000 pilot attempts. The selected pair accounts for
18,658,296 of the full-run attempts.

Other negative pilots: X/Z role reversal did not improve the overall tradeoff;
extra final d2 checks had only 1.87% ten-cycle acceptance; independent projected
CSS histories and looser rescue rules did not justify promotion. The successful
repair was the syndrome-preserving restriction, not additional quantum rounds.

Validation:

- 74 broader regression tests passed, followed by 21 final candidate tests after
  adding verified rescue. Tests include fresh noisy adapter-vs-compiled equality
  in both bases at N=1/3, ten noiseless cycles, exact compact-vs-full Pauli evidence,
  original-parent preservation, and all 4,096 ideal input syndrome sectors.
- Each selected controller matches the explicit reference rule on **400,000
  previously saved observation records**, including **4,000 nonzero incoming-frame
  checks per method**. New logical error rates come from fresh quantum sampling;
  they are not inferred from old boolean logical-failure traces.
- All 1,089 CSS-order-one physical inputs per basis/method accept correctly.
  The verified rescue accepts/corrects all 1,005 distinct first-gadget single-fault
  response cases per basis and passes 251 additional input-one-plus-fault rescue
  cases per basis. Four thousand noiseless five-cycle reference shots across the
  selected pair all accept correctly.
- Native controls: CSS baseline has 8 clean split-input shots over three cycles
  and a 100-shot noisy branch check (64 accepted). Verified rescue has 8 clean
  split-input shots over one native cycle and a 100-shot noisy branch check
  (66 accepted). Clean controls have zero logical errors. Repeated frame behavior
  is also covered by reference tests and 400k-trace replay. Native gate noise
  includes preparation/readout and excludes memory; it is not the production model.
- Source regeneration, hashes and serialized HUGR reload passed. Every stored
  edge/frame was checked: 1,194,645 CSS edges and 307,039 verified-rescue edges,
  plus representative missing-observation rejection at 21,651 / 21,315 nodes.
- Real full-size resumes schedule zero new shots; raw task/batch hashes are
  unchanged. Counts, cycle accounting, matched noise/configurations/seeds, fixed
  budgets and 95% Wilson intervals are checked. Old tracked raw datasets plus
  the prior strict-PS and Steane-PS raw hashes remain unchanged.

The canonical comparison now contains **eight methods** (six full-budget methods,
plus two clearly labeled 0.1x Steane PS runs). It has 256 count rows and five
PNG/PDF figure pairs in `benchmarks/steane_comparison/figures/`, all visually
checked. The per-cycle ladder and failed-contract ablation are excluded.
Both notebooks have refreshed 32-row endpoint tables, five comparison figures,
and the new four-panel **LER-versus-acceptance** figure at N=1 and N=10:
`analysis/32_20_4_adaptive_improvements/figures/acceptance_error_tradeoff.{png,pdf}`.
Survival remains linear and data-fitted; LER/N remains an endpoint normalization,
not a first-failure hazard. Do not infer ten-cycle AR from independent N=1 AR.

Reproduce/resume with the qec Python environment:

```sh
python analysis/32_20_4_adaptive_improvements/production.py --variant css --jobs 8
python analysis/32_20_4_adaptive_improvements/production.py --variant css-recover-safe --jobs 8
python -m spiderwarp.se_comparison --plot-only
python analysis/32_20_4_adaptive_improvements/report.py
```

`finalize.py` performs all completed-data audits/zero-work resumes, regenerates
figures/report, and refreshes tagged comparison cells in both notebooks. It
preserves earlier heavy demo outputs. Other audit scripts are `controls.py`,
`audit_exports.py`, and `replay.py`. Logs are `build/adaptive_*` and the final
study certificate is `analysis/32_20_4_adaptive_improvements/validation.json`.
Matplotlib's existing NumPy scalar-conversion deprecation warnings remain in
boundary plotting tests. No work remains from this request.

## CSS syndrome uniqueness and quick Steane data postselection (2026-11-13)

The user asked for 0.1x sampling, correction only when unique modulo stabilizers,
and removal of **Steane ladder / cycle** from the analysis. They explicitly
selected **CSS component weights <=2**, rather than total Pauli weight <=2.
The per-cycle ladder remains available in code and its historical raw data is
preserved, but it is excluded from the default comparison, notebook tables,
artifact tables and plotted curves. Default Steane comparison modes are now
`d4` and `ladder-ancilla`.

Exhaustive results for the exact current code matrices are saved in
`analysis/32_20_4_syndrome_uniqueness/{summary.json,syndromes.csv}`; reproduce with
`python analysis/32_20_4_syndrome_uniqueness/analyze.py`. The calculation checks
stabilizer equivalence against an independently enumerated group of all 4,096
stabilizers, whose minimum nonidentity physical weight is 16.

- Per CSS component, all 31 syndromes requiring weight two have **16 distinct
  logical classes** of weight-two explanations. None is uniquely correctable.
- Example (zero-based indices): X0 X1 and X2 X3 share syndrome 32, while their
  product X0 X1 X2 X3 is a nontrivial logical operator.
- Under the selected CSS bound, all 4,096 joint syndromes are attainable from
  529^2 input patterns. Only 1,089 are unique (identity or one error per
  component); all 3,007 requiring CSS order two are ambiguous. Consequently,
  there is no new syndrome-only CSS-order-two input class to add to the
  optimized controller. Flags/restricted circuit fault histories are extra
  information and must be checked separately; no adaptive contract was widened.
- Under the different total-Pauli-weight<=2 assumption, 2,976 joint syndromes
  requiring physical weight two are unique (mixed Pauli pairs), while 93 are
  ambiguous with 16 logical classes each. This includes mixed pairs that the
  selected CSS model must reject, so do not conflate the two statements.

`SteaneQECProtocol(..., postselection="unique-css-2")` and CLI
`build-steane --postselection unique-css-2` now add data postselection after
two consecutive identical readouts in each CSS basis. The cumulative Pauli
frame is removed from the confirmed syndrome before testing its correction
domain. Ambiguous syndromes reject the data immediately; flagged ancillas
still retry before data coupling. Domains are obtained by checking all
component errors through weight two modulo stabilizers, rather than merely
counting syndrome bits or assuming minimum-weight correction is unique.
Standalone Guppy emits the same acceptance tables and early returns, and
the tables/rule are included in each new protocol manifest hash. Original
non-postselected manifest identities and generated sources remain unchanged.

Both quick runs are complete; reproduce/resume with:

```sh
python analysis/32_20_4_syndrome_uniqueness/quick_steane_ps.py --jobs 8
python -m spiderwarp.se_comparison --plot-only
```

New bundles: `build/steane_css_ps_{d4,ladder_ancilla}/`.
Separate quick data: `benchmarks/steane_css_ps_{d4,ladder_ancilla}_10k/`.
Each uses the same 32 configurations/noise/seed, 10k-shot floor, 100k cap,
inverse-square allocation and 1k-shot batches, totaling 932,918 attempts
(integer rounding makes this slightly above exactly one tenth of 9,329,148).
The comparison auto-includes complete quick datasets and labels their 0.1x
budgets; original full-budget datasets are reused at full precision. The
count audit therefore checks matched noise/seeds/configurations while reporting
unequal shot budgets explicitly. Existing production data must remain unchanged.

Both runs reached all 32 targets, totaling **1,865,836 new attempted shots**.
At exact default hardware noise (p_2q=0.0008), each row below has 10,000 attempts
per basis. Entries list logical 0 / logical +; LER is conditional block error,
not LER divided by N.

| Steane CSS-PS mode | N | Conditional LER, 0 / + | Survival, 0 / + |
|---|---|---|---|
| d4 RUS | 1 | 0.11270% / 0.22550% | 97.600% / 97.560% |
| d4 RUS | 10 | 0.13060% / 2.36302% | 76.570% / 77.020% |
| Ladder / ancilla | 1 | 0.30659% / 0.34768% | 97.850% / 97.790% |
| Ladder / ancilla | 10 | 0.97113% / 2.62065% | 77.230% / 77.080% |

Raw N=1 failure/accepted counts: d4 11/9760 and 22/9756; ladder 30/9785 and
34/9779. N=10: d4 10/7657 and 182/7702; ladder 75/7723 and 202/7708.
Full Wilson intervals and resource counts are in each summary CSV and the
refreshed comparison notebooks. These quick estimates have wider intervals;
for example, d4 N=1 logical-0 LER has a 0.06295–0.20172% interval.

All five comparison PNG/PDF pairs now contain the six retained methods, with
0.1x shots labeled on the two new curves and a shared zero-failure CI marker
guide. Both notebooks have refreshed 24-row endpoint tables and all five
embedded figures; all plots were visually checked. The per-cycle ladder is
absent from active CSV rows, figure models, notebook tables and artifact links.
Historical sections below document past runs and do not describe the current
selected comparison.

Validation is complete. `validate_controls.py` checks all 1,089 CSS-order-one
inputs and all 992 pure weight-two inputs per basis/mode: **4,356 accepted and
correct**, **3,968 rejected**. Four thousand noiseless five-cycle reference
shots all accept correctly. Native three-cycle HUGR controls cover both modes
and bases: 16 split-CSS-input shots accept correctly, while 16 ambiguous-pair
shots reject. Source regeneration/hashes and both serialized HUGRs were checked.
Real resumes schedule zero new shots and preserve all raw task/batch hashes.
New run manifests point to local `validation.json`; the combined validation
links both audits and explicitly records unequal shot budgets. Original raw
production datasets remain unchanged. Engine fingerprints conservatively
change because the optional Steane controller code changed; old artifacts'
non-postselected manifests and generated sources retain their exact identities.

Six initial CSS-PS tests and 59 broader regressions passed; five final focused
budget/plot regressions passed after the last legend update. Matplotlib still
emits NumPy scalar-conversion deprecation warnings in boundary plotting tests.
Logs are `build/steane_css_ps_{10k,tests,regressions,final_plot_tests,controls,final_audit}.log`.
The final audit script is
`analysis/32_20_4_syndrome_uniqueness/finalize.py`; it refreshes notebook outputs
from the already-regenerated comparison figures and performs zero-work resumes.

## Implemented order-one postselection: full benchmark complete (2026-11-12)

The user authorized implementing the history-based method, running the full
benchmark, and updating plots. `spiderwarp/se_postselection.py` now compiles the
guard into a separate ordinary `AdaptiveSEProtocol` decision graph: reject as
soon as no r+s<=1 explanation remains. Quantum circuits and surviving actions/
frames are preserved from `build/se_32_20_4/`. The default/original is retained.
The revised contract has `accept_clean_through=1`, preserving mandatory
acceptance/logical correctness through r+s<=1 and the original CSS filters.
There is no new reuse optimization, fault oracle, or ideal readout in the guard.

The CLI build is complete:

```sh
python -m spiderwarp.se_experiment build-postselected build/se_32_20_4 \
    --out build/se_postselected_32_20_4 --emit hugr
```

The new bundle is `build/se_postselected_32_20_4/`, including standalone Guppy,
compiled/reloaded HUGR (195,837 bytes), `protocol.json`, `export.json`, and
`postselection.json` tying it to its parent. It has 976 nodes (original 83,787),
60 peak qubits, a 589-CNOT worst-case bound, and at most four extractions.
No new optimality claim is made. Reachable extraction labels are d4 and d2;
d1 is pruned before execution. The independent standard checker passed
4,452,310 transitions / all 4,096 arbitrary input syndrome sectors. The
independent full-Pauli-history guard check covers 1,254 states and 3,430
non-reject edges. Ordinary loading/compilation/native/sweep interfaces work
without a new bundle schema or runtime interpreter.

`analysis/32_20_4_postselected_validation/replay_compiled.py` confirms all
400,000 existing discovery/validation observation records reproduce the
analysis's acceptance exactly, preserve surviving frames/actions, and abort
rejected traces no later. It includes 4,000 nonzero incoming-frame checks.
The physical audit passes all 8,208 low-order cases again on the actual new
controller; its JSON is in the same directory. Native split-input controls in
both bases pass; a 100-shot noisy native check exercised both accept/reject
branches (52 accepted). Native noise remains a distinct gate-only model.

The full production run is COMPLETE:

```sh
python -m spiderwarp.se_experiment sweep build/se_postselected_32_20_4 \
    --config benchmarks/se_postselected_100k/config.json \
    --shots 100000 --allocation inverse-square --max-shots 1000000 \
    --batch-size 10000 --jobs 8 --seed 20261112 \
    --results benchmarks/se_postselected_100k
```

Log: `build/se_postselected_100k.log`. Exact settings/targets are in
`benchmarks/se_postselected_100k/run.json`; configuration JSON is copied from
the original production sweep. All 32 configurations reached their prescribed
targets: **9,329,148 attempted shots**, 12,047,242 cycles started, and 10,844,180
cycles completed. Every default-noise repetition point has 100k attempts per
basis; low-noise points have up to one million. A real production resume
scheduled zero additional shots and left all task/raw-batch hashes unchanged.
The original adaptive and three Steane production datasets remain unchanged.

At the exact default hardware model (p_2q=0.0008), the new controller gives:

| Cycles | Basis | Logical failures / accepted | Conditional block LER (95% Wilson) | Survival |
|---|---|---|---|---|
| 1 | 0 | 41 / 90,905 | 0.04510% (0.03325–0.06118%) | 90.905% |
| 1 | + | 379 / 91,151 | 0.41579% (0.37606–0.45971%) | 91.151% |
| 10 | 0 | 23 / 25,480 | 0.09027% (0.06016–0.13542%) | 25.480% |
| 10 | + | 233 / 25,516 | 0.91315% (0.80360–1.03748%) | 25.516% |

These are failures of any of the 20 logical readouts, conditional on all cycles
accepting, with one ideal final decode. The N=1 original adaptive LER was
1.7846% / 3.6352%; the new rule reduces it by about 39.6x / 8.7x. Survival is
the tradeoff: do not infer N=10 survival from independent N=1 acceptance.
LER/N is an endpoint normalization, not a first-failure hazard. Mean executed
CNOTs per attempted trajectory are 442.51 / 442.62 at N=1 and 2615.16 / 2624.34
at N=10; dividing by requested N includes early aborts and is not the cost per
accepted output.

The complete five-method comparison is regenerated with:

```sh
python -m spiderwarp.se_comparison --plot-only
```

This automatically includes completed `benchmarks/se_postselected_100k/`,
accounts for its rejections and 60-qubit count, and audits matched noise,
seeds, shot budgets and configurations against all four existing datasets.
`benchmarks/steane_comparison/figures/` contains five PNG/PDF pairs:
physical_error, physical_survival, cycles, survival and cnot_cost. Shared
legends leave the data visible; survival remains linear with bounds fitted to
the measured range. Both comparison notebooks have refreshed tables and all
five embedded figures. Standalone PS figures are in its own `figures/`.

Validation is recorded in `benchmarks/se_postselected_100k/validation.json`
and linked from `benchmarks/steane_comparison/validation.json`. The standalone
export audit checks 5,505,024 legal observation inputs, all stored transitions/
frames, and unknown-outcome rejection, regenerates the source identically and
reloads the HUGR. A separate deterministic rebuild reproduces protocol hash
`369e501c3fd7d6bc0fac687fb89c7e76870a44044bafe7ebaafe2519d681e58a`.
The full regression run passed 185 tests; five final plotting/resume regressions
also passed after the final layout/boundary fix. Full logs are
`build/se_postselected_tests.log` and `build/se_postselected_final_plot_tests.log`.
Matplotlib emits existing NumPy scalar-conversion deprecation warnings in the
boundary plotting tests. Silent two-fault logical errors with clean observation
records still exist, as saved in the physical audit; the guard rejects observed
incompatibility with order one, not every actual pair of faults.

## Flag-pattern risk and selective two-order postselection (2026-11-12)

The latest user asked why zero d4 flags differs from zero flags everywhere,
requested flag-pattern -> logical-error probabilities, asked whether something
is incorrect, and suggested always postselecting at two observed faults.

### Clarification of r, s and the two zero-flag filters

`r` is incoming CSS error order at the cycle boundary; `s` counts faulty
locations during the cycle. Observing a syndrome does not set r=1. A nonzero
syndrome with zero d4 flags can be explained by r=1,s=0 or r=0,s=1; the policy
retains compatible histories. Evidence of one spent order permits d2, which
can subsequently raise flags. An actual replayed accepted logical-0 failure is:

```text
action     flags    raw syndrome
Z_d4       0        0
X_d4       0        0
Z_d4       0        9
X_d2       36       11
```

Before X_d2, its order-at-most-one explanations have (r,s)=(0,1), rather than
(1,0): the syndrome changed during QEC. This path satisfies "all d4 flags zero"
and violates "all flags zero". The default contract can accept r=1,s=1, but
logical preservation is guaranteed only through r+s<=1, not at r+s=2.

### Discovery, independent validation and saved data

`analysis/32_20_4_branch_assessment/patterns.py` exactly replays the same 100k
production shots per basis as discovery, then samples **100k fresh independent
default-noise N=1 shots per basis** for validation. All 20 discovery batches
again match saved production counts. Seed lists and the protocol hash are in
`patterns_assessment.json`; the four compressed trace files preserve observed
flags/syndromes and outcomes. New validation shots are not appended to production.

Outputs:

- `pattern_risk.csv`: discovery and validation risks for action/mask, stage/mask,
  and full flag-record patterns, including sample counts and Wilson intervals.
- `flag_pattern_summary.csv`: validation action/mask table with both bases side
  by side, hexadecimal masks, LSB-first bits, bit counts, minimum local fault
  order, accepted counts, failures and confidence intervals.
- `conditional_risk.csv`: risks given raised d4 flags and given no order-0/1
  explanation. All logical risks are conditional on original policy acceptance.
- `postselection_policies.csv`: discovery/validation LER and survival for each
  fixed or learned pruning rule. One common rule is used in both logical bases.
- `figures/postselection_tradeoff.{png,pdf}` and
  `figures/common_d4_mask_risk.{png,pdf}`: visually checked tradeoff and mask-risk
  figures. Mask risk counts each trajectory once per encountered action/mask;
  rows for different masks can overlap and must not be summed as disjoint groups.

In the independent validation set, a raised d4 flag anywhere in an accepted
cycle gives logical-0 risk **1,703/33,552 = 5.0757%** (95% CI 4.8459–5.3158%)
and logical-+ risk **3,185/33,299 = 9.5649%** (9.2536–9.8854%). No raised d4
flag gives **51/66,026 = 0.0772%** / **489/66,230 = 0.7383%**. These agree with
the earlier discovery estimates 5.1243% / 9.3335% versus 0.1027% / 0.7339%.

Example common d4 masks, using independent validation (failures / accepted):

| Action / hexadecimal mask | Logical 0 | Logical + |
|---|---:|---:|
| Z_d4 / 0x4f2 | 6/164 = 3.66% | 30/174 = 17.24% |
| X_d4 / 0x428 | 32/206 = 15.53% | 11/173 = 6.36% |
| Z_d4 / 0x120 | 5/274 = 1.82% | 39/292 = 13.36% |
| Z_d4 / 0x008 | 31/803 = 3.86% | 87/853 = 10.20% |

Individual rarer masks can show 50–100% empirical risk with only 5–10 samples;
use the counts and intervals. The aggregate action/mask risk depends on stage
and earlier syndrome/flag context. Blacklisting raw masks alone loses that
information. Learned blacklists were fit on discovery only: at least 100
accepted discovery cases per basis, thresholding the maximum of the two bases'
Wilson lower bounds. Threshold-family comparisons remain exploratory.

### Highest-potential measured rule: reject when r+s<=1 is ruled out

`LowOrderEvidence` tracks explanations of the measured raw flags/syndromes
with r+s<=1. It uses only observations available to the controller: **no hidden
noise-event counts or final ideal/logical readout**. Reject once no such
explanation remains. This means observable evidence for order at least two,
or an observation outside that bounded model, rather than an exact physical
fault counter. Incoming r=1 and clean-input s=1 explanations are both retained.

| Postselection rule, independent validation | Logical-0 failures / accepted (LER) | Logical-+ failures / accepted (LER) | Survival 0 / + |
|---|---:|---:|---:|
| Original acceptance | 1,754/99,578 (1.7614%) | 3,674/99,529 (3.6914%) | 99.578% / 99.529% |
| Every d4 flag zero | 51/66,026 (0.0772%) | 489/66,230 (0.7383%) | 66.026% / 66.230% |
| Flags alone certify >=2 internal faults | 842/95,805 (0.8789%) | 2,342/95,872 (2.4428%) | 95.805% / 95.872% |
| Full history has no r+s<=1 explanation | **43/90,848 (0.0473%)** | **381/90,825 (0.4195%)** | **90.848% / 90.825%** |
| Learned raw-mask blacklist, 5% threshold | 1,207/86,643 (1.3931%) | 2,509/86,721 (2.8932%) | 86.643% / 86.721% |

The full-history rule's LER intervals are 0.0351–0.0637% / 0.3795–0.4637%.
Rejected originally accepted histories without an order-0/1 explanation have
conditional risks 1,711/8,730 = 19.599% / 3,293/8,704 = 37.833%. The theoretical
history rule was not fitted to logical outcomes. Its discovery results were
0.0616% / 0.4291% at 90.901% / 90.895% survival; the holdout supports the gain.

The weaker flag-only evidence rule sums the minimum internal fault order for
each raw mask across extractions. Two raised **bits** do not imply two faulty
locations. A physical one-location witness in Z_d4 raises mask 0x006 (two bits):
the two-qubit Pauli alternative Z58 X59 at operation offset 159. All 15
nonidentity alternatives at a two-qubit gate count as one event. A second
flagged extraction does require another internal event because validated
fault-free gadget flags are independent of incoming data errors; individual
multi-bit masks may also certify two events if no one-fault response produces
that mask. Full syndrome history adds substantially more information.

### Audit and next implementation step

`audit_patterns.py` checked the compact syndrome-state compatibility calculation
against the full Pauli-history engine on **2,000 distinct prefixes**. It also
physically replayed **3,015 distinct single-fault responses per logical basis**
across every stage of the clean path, and **1,089 incoming CSS-weight-at-most-one
cases per basis**: **8,208 cases**, all accepted, logically correct, and kept
by the proposed order-0/1 compatibility rule. No low-order discrepancy was found.
This does not replace a full certificate for a new exported controller.

`patterns_audit.json` also saves physical **silent two-fault logical-error
witnesses** in both bases. X1 X2 (or Z1 Z2) after their last gates in the final
Z_d4 leaves every observation equal to the clean path but fails ideal logical
decoding. Thus even perfect observable postselection cannot catch every actual
two-fault history while accepting the clean record. This is consistent with
the existing logical-order-one contract, not evidence of a new single-fault bug.

The highest-priority next controller is the full-history rejection rule above,
keeping the current circuits, 60-qubit cap and existing one-fault acceptance/
logical guarantees. The current contract **mandates clean-input s=2 acceptance**;
relaxing `accept_clean_through` from 2 to 1 is the natural contract to investigate
for this rule. That contract change alone does not force rejection: the
history-based PS predicate must also be explicitly enforced. Rebuild/check/export
that postselected controller and
measure N=10 yield/LER and early-abort CNOT cost. This turn measured posthoc
pruning only: **no new protocol/Guppy/HUGR or repeated-cycle results are claimed**.
The more aggressive zero-flag modes require further acceptance relaxation.

## Adaptive improvement proposals and flag-path diagnosis (2026-11-12)

The user proposed restricting recovery to **d4 -> d1** or **d4 -> d2**, with
postselection on further fault indications, and asked about keeping only
trivial d4 flags. They also asked which mechanisms primarily cause the current
high LER. The restricted controllers are **proposals to evaluate**, not newly
built or certified protocols. This update adds a diagnostic replay only; the
current adaptive controller, circuits, allocations, exports and production
counts are unchanged. The earlier restriction against implementing an adaptive
qubit-reuse change still applies.

### Measured effect of flag postselection

`analysis/32_20_4_branch_assessment/replay.py` replays the **existing** default-
noise N=1 production batches, using their exact saved per-batch seeds. All 20
batches reproduce acceptance and logical-error counts exactly: 100k original
shots per basis. These are the same observations, not 200k independent new
samples. Reproduce with:

```sh
/opt/homebrew/Caskroom/miniforge/base/envs/qec/bin/python \
    analysis/32_20_4_branch_assessment/replay.py
```

`assessment.json` saves the matched batch counts/seeds, input hashes, group
statistics, paths, versions and interpretation; `comparison.csv` contains the
counts and 95% Wilson intervals. This is postselection on the existing policy's
trajectories and returned frames. Original policy rejections remain rejected.
It is not a forced d4-only or restricted-ladder experiment, and it does not
estimate gate savings from aborting a rejected trajectory earlier.

| Keep condition | Logical-0 failures / accepted (LER) | Logical-+ failures / accepted (LER) | Survival 0 / + |
|---|---:|---:|---:|
| Current adaptive acceptance | 1,777 / 99,575 (1.7846%) | 3,618 / 99,527 (3.6352%) | 99.575% / 99.527% |
| Additionally, every executed d4 has zero flags | 68 / 66,224 (0.1027%) | 484 / 65,949 (0.7339%) | 66.224% / 65.949% |
| Additionally, every extraction has zero flags | 53 / 66,194 (0.0801%) | 484 / 65,929 (0.7341%) | 66.194% / 65.929% |
| Only existing d4-only paths, with all flags zero | 48 / 65,960 (0.0728%) | 477 / 65,673 (0.7263%) | 65.960% / 65.673% |

The d4-zero-flag filter therefore gives a large conditional-LER improvement,
at the cost of rejecting roughly a third of attempts. Its LER intervals are
0.0810–0.1301% (logical 0) and 0.6716–0.8020% (logical +); use CSV values as
authoritative if quoting additional precision. Repeated-cycle survival has
**not** been measured for these filters. The one-cycle yield loss may compound
strongly across cycles; do not present these as new N=10 results or as ancilla
RUS with free retries of the data block.

### Where failures concentrate

On accepted paths with at least one raised d4 flag, the LER is 5.1243%
(logical 0) / 9.3335% (logical +), compared with 0.1027% / 0.7339% when no d4
flag is raised. Those flagged-d4 paths contain **1,709/1,777 = 96.17%** of
logical-0 failures and **3,134/3,618 = 86.62%** of logical-+ failures. This is
path association, not identification of the physical locations of the faults.

Blaming only d1 recovery would miss most failures: d1-using paths contain
13.45% / 17.74% of failures. Paths using d2 but no d1 contain 40.35% / 20.45%;
d4-only paths contain 46.20% / 61.80%. Simply pruning existing paths containing
d1 gives LER 1.5660% / 3.0332% at survival 98.214% / 98.113%; pruning d2 gives
1.2818% / 3.2928% at 81.913% / 81.632%. **These pruning proxies do not implement
the user's forced d4 -> d2 or d4 -> d1 proposals.**

Particularly harmful observed path classes are saved in `assessment.json`:
for logical 0, `Z4;X4!` contributes 754 failures and `Z4!;X2!` contributes 474;
for logical +, `Z4;X4;Z4!` contributes 1,072, and `Z4!` contributes 445.
`!` means a nonzero raw flag mask at that extraction, not a number of physical
faults. Audit these terminal decisions by their full flag/syndrome histories.

### Noise channel evidence and next experiments

The existing fixed-circuit ablations in
`analysis/32_20_4_noise_assessment/comparison.csv` identify two-qubit noise as
the strongest measured sensitivity, with Z-biased memory substantial for
logical +:

| Noise case | N=1 logical 0 / + | N=10 logical 0 / + |
|---|---:|---:|
| Full default model | 1.785% / 3.635% | 20.841% / 42.019% |
| Two-qubit noise removed | 0.206% / 0.982% | 3.215% / 14.421% |
| Memory noise removed | 1.460% / 2.019% | 16.778% / 23.951% |
| Initialization noise removed | 1.065% / 2.880% | not measured |

N=1 modified cases have 100k attempts per basis; N=10 modified cases have
20k, versus the saved 100k full-noise baseline. These sensitivities are not an
additive attribution: mixed faults and changed postselection interact.

About 90% of executions and 90.5–90.7% of live-idle exposure occur in d4.
Removing memory only in d1/d2 barely changes logical-0 LER and reduces logical-+
LER from 3.635% to 3.375%. Exposure dominance does not prove that the same
fraction of logical failures originates at d4 fault locations.

Recommended evaluation order:

1. Implement explicit postselected modes for zero d4 flags and zero flags at
   every stage; reproduce the replay's N=1 survivor statistics, then measure
   N=10 survival, LER and actual early-abort CNOT cost. Keep these separate from
   the current certified acceptance policy.
2. Build the two requested restricted ladders, allowing at most the specified
   distance drop and defining rejection from **observable** flags/syndromes or
   compatible histories. Do not use a simulator's hidden physical fault count.
   A d1 gadget has no flags: additional faults there can be silent, so d4 -> d1
   needs a justified history restriction or another diagnostic to implement
   "postselect on further faults." A raised flag alone does not justify d1
   under the current distance ladder; d1 requires zero remaining fault budget.
3. Split two-qubit-noise ablations by d4 versus recovery stages, and then by
   data-touching versus ancilla-only gates. This distinguishes physical fault
   origin from the branch associations measured here.
4. Hold circuits fixed and test extra syndrome confirmation, first/final basis
   ordering under Z-biased memory, and noise-weighted choices among permitted
   terminal frames. The current policy minimizes worst-case CNOT count; it
   does not minimize stochastic LER. Conventional Steane also confirms each
   CSS syndrome twice consecutively, a control not isolated by the comparison.
5. Consider depth/live-idle optimization under the existing 60-qubit cap after
   these controls. Current adaptive d2 is 92 layers in X / 86 in Z versus 83
   for d4, so a lower distance label does not guarantee a shorter noisy wait.

Stricter postselection changes the acceptance contract: the current checker
requires acceptance through r+s<=1 and for clean-input s=2. Rejecting flagged
single-fault or two-fault histories cannot retain that exact certificate.
Keep guarantees about accepted outputs distinct from guarantees of acceptance.
The current contract guarantees logical preservation through total order one;
accepted order-two histories need not preserve the logical state. This is a
plausible contributor to the gap, not a fault-pair attribution established by
the replay. Zero flags do not certify that no faults occurred.

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

- The newest matched-budget section supersedes the earlier mixed-budget checkpoint
  below. Use the centralized new display names and full `_100k` Steane PS datasets.
  Equal attempts are checked per configuration; accepted counts differ by AR.
- Prior analysis had eight methods: original adaptive, strict PS, new CSS-only
  PS, verified CSS rescue PS, d4 Steane, per-ancilla Steane, and both Steane CSS-PS
  quick runs. The new adaptive selected pair each has 9,329,148 full-budget shots.
  The per-cycle ladder and the failed-contract aggressive-rescue ablation are
  excluded. CSS component weights <=2 is the chosen input model. Strict PS has
  the lowest LER; verified rescue trades higher LER for ~40% more accepted outputs
  at N=10 while preserving the strict-PS contract, 60-qubit cap and 589-CNOT bound.
  The full study is complete; see the newest section and its validation/report.
- Latest result: the full-history r+s<=1 postselected controller is implemented,
  independently verified, and exported as standalone Guppy/HUGR in
  `build/se_postselected_32_20_4/`. Its complete 32-configuration/9,329,148-shot
  benchmark is `benchmarks/se_postselected_100k/`. At default noise it gives
  N=1 LER **0.04510% / 0.41579%**, survival **90.905% / 91.151%**; at N=10,
  LER **0.09027% / 0.91315%**, survival **25.480% / 25.516%**. See the newest
  section for raw counts, Wilson intervals, guarantees and all validation.
- The latest user proposals and exact flag-path replay are documented above.
  Strict d4 -> d1 / d4 -> d2 controllers are pending experiments. Rejecting
  raised d4 flags on existing N=1 trajectories gives LER 0.1027% / 0.7339%
  at survival 66.224% / 65.949%; repeated-cycle behavior is not yet measured.
- The three historical Steane RUS comparisons above are complete, including standalone
  Guppy/HUGR, matched production counts, controls, the final audit, and populated
  comparison notebook/figures. Use `python -m spiderwarp.se_comparison --plot-only`
  to regenerate the current eight-method views without sampling; the per-cycle
  ladder is excluded by default.
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
  range**: approximately 0.895–1.0 for the original alone, and 0.2–1.0 for the
  combined plot including PS. Do not restore a logarithmic
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
- Prior adaptive-PS validation passed: 185 full regression tests and five final focused
  plotting/resume tests, plus reference/native controls, the independent FT
  and guard checks, 400k exact trace replays, 8,208 physical low-order cases,
  4,000 incoming-frame checks and the complete standalone export audit.
  Details are in the two production/comparison validation JSON files. Prior
  Steane-specific native retry and export checks are retained in the comparison
  validation history.

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

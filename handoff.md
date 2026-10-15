# SpiderWarp project handoff

Last updated: 2026-09-27

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

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
- The renamed `t1_zero_32_20_4.stim` input and regenerated SE asset are modified.
- `.DS_Store` and `__pycache__` are now ignored.

# SpiderWarp 🕷️🌀

**SpiderWarp** is the first automated quantum circuit optimizer designed specifically for **Fault-Tolerant Quantum Gadgets**. 

Unlike standard tools that only preserve logical unitaries, SpiderWarp uses **Distance-Preserving Rewrites** to optimize CSS-code based Clifford circuits.
It "warps" the circuit topology—reducing CNOT and ancilla counts via the OCM (Only Connectivity Matters) rule—while strictly ensuring that the gadget's error-correcting properties and fault-distance remain intact.

## 🚀 Key Features

* **Topological Morphing:** Automatically "bends" wires and fuses spiders to eliminate redundant hardware overhead.
* **Zero-Metadata Inference:** No manual tagging required. SpiderWarp automatically identifies:
    * **Qubit Roles:** Distinguishes between persistent Data qubits and lifecycle-limited Ancillas.
    * **Measurement Types:** Categorizes outcomes into Syndromes (stabilizers) or Flags (high-weight fault catchers) by analyzing the parity-check graph.
* **Multi-Language Support:** Seamlessly ingest and optimize gadgets from **OpenQASM 3.0** and **Stim**.
* **Strict FT-Preservation:** Every rewrite is verified against the gadget’s fault-signature, ensuring that a weight-1 fault never "warps" into an uncorrectable logical error.

---

## 🛠 How it Works

SpiderWarp treats FT gadgets as dynamic spacetime volumes rather than static gate lists.

### 1. Unified Qubit & Measurement Logic
The library eliminates the "Labeling Burden." It parses your circuit and derives roles from the **Detector Error Model (DEM)**:
* **Ancillas** are identified by their local initialization-to-measurement footprint.
* **Syndromes** are identified as measurements participating in temporal parity checks (Detectors).
* **Flags** are identified as local parity checks designed to catch circuit-level "hooks."

### 2. The OCM Warp Rule
By mapping the gadget to a ZX-diagram, SpiderWarp applies the **Outer Commutation Markup (OCM)** rule. This allows for the "bending" of wires—effectively commuting gates through the stabilizer boundary—to cancel out CNOTs that are logically redundant but physically expensive.



---

## Installation

Use Python 3.10 or newer. In the environment you use to run the notebooks,
install SpiderWarp from the repository root:

```sh
python -m pip install -e .
```

The editable install keeps imports connected to this checkout as you change the
code, regardless of the notebook's working directory. The circuit and code
assets are also included in regular installations.

For a notebook that already has a running kernel, run this once in a separate
cell, replacing the path with your checkout's absolute path:

```python
%pip install -e "/absolute/path/to/AutomatedSteane"
```

Restart the kernel after installation, then run the notebook from the beginning.
`%pip` installs into the notebook's own Python environment. If imports still fail,
check `import sys; print(sys.executable)` in the notebook and select the environment
where you installed SpiderWarp. No `sys.path` changes are needed.

Optional dependencies can be installed with `python -m pip install -e ".[notebooks]"`
for Jupyter, `".[test]"` for tests, or `".[mqt]"` for the MQT synthesis notebooks.

## 💻 Usage

```python
from spiderwarp.csscode import CSSCode
from spiderwarp.utils import load_state_prep_circuit
from spiderwarp.stim_utils import steane_se_from_stim_state_prep

code = CSSCode.load_code("misc", "32_20_4")
state_prep = load_state_prep_circuit("misc", "32_20_4_zero_d4")
syndrome_extraction = steane_se_from_stim_state_prep(
    state_prep, se_basis="Z", n=code.n
)
print(syndrome_extraction.num_qubits)
```

## Adaptive and Steane QEC comparison

The comparison entry point now defaults to **Sol/Narsil with the Helios alpha
noise model**. It uses the local stack in `local_sol_example`, including actual
ion transport, rather than the inferred unit-layer noise in `noise_model.py`.
The existing Stim datasets below retain their original model and results.

Keep the Sol environment separate from SpiderWarp's existing `qec` environment:

```sh
uv sync --frozen --project local_sol_example
# Run this with the Python environment where SpiderWarp is installed:
python -m spiderwarp.se_experiment sol build/se_postselected_32_20_4 \
    --noise-scales 1 --cycles 1 3 --shots 100 --batch-size 50 \
    --results benchmarks/sol_v2_standard
# All seven saved methods, with matched fixed attempted-shot budgets:
python -m spiderwarp.se_comparison --cycles 1 3 --shots 100 --jobs 2
```

For the standard Adaptive PS(r+s≤1) and Steane methods together, the macOS
launcher uses the installed `qec` Python, keeps idle sleep off, and runs fixed
alpha noise with repumping, both bases and cycles 1, 3, 5, 7, 10:

```sh
nohup ./analysis/run_sol_overnight.sh >> sol_overnight.log 2>&1 &
tail -f sol_overnight.log
```

It requests 10,000 **attempts per point** (200,000 total), four concurrent
configurations and checkpoints every 250 attempts. Results go to
`benchmarks/sol_v2_overnight_10k/`; the combined `comparison.csv` is written after
both methods finish. Rerunning the same command resumes committed ranges.
Leave the Mac plugged in with its lid open. Set `SPIDERWARP_PYTHON` to override
the launcher's SpiderWarp interpreter.

A 400-attempt pilot on this machine projects roughly 16–20 hours for this grid;
that is a small-sample estimate, not a completion guarantee. For about an
8–10-hour budget, use a separate 5K result tree:

```sh
nohup ./analysis/run_sol_overnight.sh --shots 5000 \
    --results benchmarks/sol_v2_overnight_5k >> sol_overnight_5k.log 2>&1 &
```

The comparison CLI also accepts multiple methods directly with
`--only adaptive_history steane_fixed_ps`. Accepted samples can be far fewer
than attempted shots at longer cycle counts; LER precision depends on the
accepted and logical-error counts.

The example's authenticated package index setup is described in
[`local_sol_example`](local_sol_example/README.md). Its existing
`uv.lock` defines the runtime versions. By default the worker uses that
directory's `.venv/bin/python`; override it with `--sol-python /absolute/path`
or `SPIDERWARP_SOL_PYTHON`. macOS/Linux and a C compiler (`CC`, default `cc`)
are required for the small Selene boundary adapter. Private runtime packages
are imported only by the separate worker.

`--noise-scales` uses PECOS's overall scaling control: 1 means the installed
alpha defaults, and 0 is noiseless. It is not the old Pauli-error probability
`p_2q`; PECOS's `p2` is average infidelity. Parameter scales must also satisfy
PECOS's channel constraints; unsupported values fail rather than being clipped.
For explicit parameters use `--model-overrides params.json` on `se_experiment sol`.

Sol execution uses Stim with `coherent_dephasing=False` and
`leak2depolar=True`. These are explicit approximations: quadratic dephasing is
stochastic and leakage becomes depolarization. `--repump` separately enables
repumping in Narsil; it is off by default. The simulator has the full 192-ion
pool while the adapter enforces each program's 60/66 live-qubit limit.

Preparation and final readout/decoding are ideal once per trajectory. The v2
wrapper prepares the CSS state using a systematic binary encoder (90 CNOTs for
either basis of [[32,20,4]]). To flush lazy preparation in this alpha stack, it
nondestructively reads a deterministic set of known stabilizers covering every
data qubit. One full-support X check suffices for this code, reducing the
synchronization from 32 measurements/316 CNOTs to one measurement/32 CNOTs.
It verifies the expected outcome, including any incoming Pauli. PECOS's clocks
and Narsil's ion mapping advance through these ideal operations. All extraction operations, persistent data
waiting, ancilla preparation and Steane retries in the protocol region use the
hardware noise model. Corrections remain classical Pauli frames.

The optimized prefix preserves the logical input but can change the starting
ion layout and random draw consumption. It has a new v2 engine identity and
separate default results directories (`benchmarks/sol_v2*`); v1 samples cannot
be merged with it. Use `--input-sync full` to retain the original generic
tableau encoder and complete synchronization prefix for controls.

Task identities include the exact
source, protocol, compiler/runtime versions, shared-library hashes, resolved
noise parameters and boundary settings. Disjoint global shot indices preserve
results across changes to batch size and worker count. Seed mode is explicitly
`default`; every committed batch records its fixed `stream_seed`, seed mode and
disjoint global `[start, start+shots)` range. Independent replicas need a different
`--seed`; changing the batch size does not create independent samples.
Compiled HUGR and native executables are hash-checked and cached. `--retain-shots` keeps
the measured histories, frames and runtime metrics for replay/audits.

Each unfinished configuration uses one Python worker and one native Selene
invocation for all its remaining shots. The worker streams atomic batch files
to the parent, which commits their decoded counts as they arrive. `--batch-size`
(default 250) controls checkpoint granularity, not process restarts or reseeding.
On failure, committed ranges stay intact and resume starts at the first
uncommitted global shot. `--jobs` limits concurrent configurations; native
runtimes remain isolated processes.

The Sol `summary.csv` reports `noise_scale`, conditional LER/95% Wilson intervals,
acceptance, and accepted-only circuit/native gate costs and runtime duration.
`*_native_rzz` counts native RPP/RZZ entangling gates, before noise insertion;
it excludes ideal input preparation and final readout. Divide accepted mean
costs/duration by the requested cycle count for per-cycle figures. A shot that
rejects contributes none of its earlier cycles to those accepted-only sums.
Sol tables use a separate schema from the historical `p`-axis plotting workflow.

The 10%/25% controllers are evaluated as frozen policies trained under the old
noise model. Their thresholds are not Sol-calibrated confidence bounds.
Existing fault certificates likewise retain their abstract-circuit scope.

Executable controls for all seven saved bundles:

```sh
PYTHONPATH=. python analysis/sol_backend_controls.py
python -m pytest -q tests/test_se_sol.py
```

**PS(r+s≤1)** uses one standard flag-conditioned rule. If no flags are raised,
interpret the measured Z_d4/X_d4 syndromes as incoming CSS errors and accept and
correct iff r=max(w_X,w_Z)≤1. Raised flags use the fault-history recovery
procedure, including propagated hook corrections of weight greater than one.
The cumulative incoming Pauli frame is removed before syndrome decoding.
Some unflagged single internal faults postselect; every noiseless incoming
CSS-one input accepts correctly. Accepted histories through total order one are
independently verified for logical preservation.

All adaptive methods share this base:

- **Adaptive PS(r+s≤1)**.
- **Adaptive shorter d2 PS(r+s≤1)** uses the same controller and corrections,
  with d2 ancilla reuse relaxed to use the full 60-qubit budget. X_d2 decreases
  from 92 to 45 scheduled layers and Z_d2 from 86 to 43; both retain 121 CX/CZ.
  The d4 circuits are identical. Linear memory noise scales with scheduled
  idle duration, which decreases even though the number of gates is unchanged.
  These layer counts describe the retained Stim scheduler. Sol recompiles and
  schedules transport itself; a native timing improvement is not implied.
- **Adaptive PS(r+s≤1 or V₂)** preserves accepted base records and adds records
  only when all compatible r+s≤2 histories agree on one final Pauli class modulo
  stabilizers. Any additional d2 measurement includes the remaining fault budget.
- **Adaptive PS(r+s≤1 or p_err≤10%)** and **25%** preserve accepted base records.
  Other records use the MAP Pauli correction and accept when its estimated error
  probability meets epsilon. The frozen default-noise posterior includes memory,
  an incoming iid Pauli prior and at most two nonzero response channels. Epsilon
  is a correction-confidence estimate, not an all-orders endpoint LER bound.
- **Adaptive PS(f=0, r_syn≤1)** restricts the same base by immediately rejecting
  every raised flag.
- **Steane d3 RUS PS(r_syn≤1)** is the independent reference, with one data
  syndrome reading per CSS basis and flagged ancilla-preparation retries.

Allowed adaptive paths are d4→d4, d4→d2 and d4→d4→d2. There is no d1 or third d4.
Adaptive methods use 60 peak qubits and 312, 277 or 433 CX/CZ gates respectively;
Steane uses 66 qubits and 288 gates before preparation retries.

LER and gate cost include only fully accepted trajectories. Cost is
`accepted_cnots / accepted / N`, including all preparation retries within an
accepted Steane trajectory. Earlier cycles of later-rejected shots are excluded.
Acceptance uses all attempted trajectories; intervals are 95% Wilson.

The previous history-only base, duplicate flag-recovery method, simulation data
and attribution study have been removed. The original five adaptive methods were rerun
from empty datasets with 32 matching configurations, seed 20261112 and
100,000 floor / 1,000,000 cap: 9,329,148 attempts each. The independent Steane
results are retained. The shorter-d2 method adds a fresh dataset with the same
32 configurations and shot budgets. These retained Stim results include scheduled memory noise, with ideal
preparation and final decoding once per trajectory.

```sh
# The paired program supplies the recovery template, not a plotted baseline.
python -m spiderwarp.se_experiment build --paired --distances 2 4 \
    --out build/se_32_20_4 --emit none
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/build.py
PYTHONPATH=. python -m spiderwarp.se_comparison --backend stim-reference --jobs 8
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/order_two_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/probability_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/finalize.py
# Build or sample only the shorter-d2 method.
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/build.py --only adaptive_scheduled
python -m spiderwarp.se_comparison --backend stim-reference --only adaptive_scheduled --jobs 8
# Refresh figures and notebooks without sampling.
python -m spiderwarp.se_comparison --backend stim-reference --plot-only
```

The [comparison notebook](notebooks/32_20_4_steane_comparison.ipynb) and
[protocol demo](notebooks/32_20_4_protocol_demo.ipynb) embed counts and all six
figure pairs. Sources, HUGRs, controls and raw-batch audits are linked from
[the study](analysis/32_20_4_bounded_comparison/).

## Compact Guppy exports

Adaptive CLI exports separate the readable circuit/controller module
(`steane_se_protocol.py`) from immutable decoder data
(`steane_se_tables_<hash>.json.zlib`). Keep those two files together to compile
the module. They require Guppy and Python's standard library, with no runtime
dependency on SpiderWarp, NumPy, or Stim. The compiled `.hugr` is self-contained:
table loading and decompression happen only when Python compiles the source.
This follows Guppy's [compile-time constant model](https://docs.quantinuum.com/guppy/language_guide/comptime.html).

The exporter compares two exact representations of the certified policy:
shared correction functions, and a standard syndrome decoder plus XOR patches
conditioned on the measurement history. It selects the smaller complete table
payload, rather than assuming that patches always save space. The base decoder
alone never grants acceptance: the original acceptance rule is checked first.
Flags constrain possible faults; they do not identify a unique physical fault.

Both representations share decision subgraphs and correction bases, factor
constant XOR offsets, omit correction values on rejected outcomes, and pool
repeated controller fields. A single accepting state replaces duplicated
acceptance nodes. Circuit operations, frame conventions, accepted records,
protocol hashes, and Monte Carlo data are preserved. The larger V₂ and
probability policies still encode substantial history-dependent information;
splitting source and data does not by itself shrink HUGR or guarantee faster
execution. The measured HUGR reductions come from the representation changes.

Recompile one existing bundle without synthesis or sampling:

```sh
python -m spiderwarp.se_experiment compile build/se_postselected_32_20_4
```

Rebuild all six adaptive exports and check every stored transition, exact frame,
and rejection of every absent outcome:

```sh
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/compact_guppy.py
```

The measured sizes and audit counts are in
[`guppy_compaction.json`](analysis/32_20_4_bounded_comparison/guppy_compaction.json).
For tools that need one expanded source string, use
`generate_adaptive_source(protocol)` or
`inline_adaptive_source(source, source_path)` from `spiderwarp._se_adaptive_guppy`.
The existing `protocol.export_guppy(...)` Python API also retains its single-file
format. Use `write_adaptive_source(protocol, path)` for a split export from Python.

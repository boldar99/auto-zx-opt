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
state_prep = load_state_prep_circuit("misc", "opt_zero_32_20_4")
syndrome_extraction = steane_se_from_stim_state_prep(
    state_prep, se_basis="Z", n=code.n
)
print(syndrome_extraction.num_qubits)
```

## Adaptive Steane syndrome-extraction programs

The default builder retains every flag and syndrome and synthesizes a verified
adaptive decision graph. It starts with d4 Z-SE for `[[32,20,4]]`; the next basis
and protection level depend on the complete compatible-history set.

```python
import stim
from spiderwarp.csscode import CSSCode
from spiderwarp.utils import get_project_root
from spiderwarp.optimized_steane_se import optimized_steane_SE_orchestrator

code = CSSCode.load_code("misc", "32_20_4")
assets = get_project_root() / "assets/circuits/SECircuits"
circuits = {
    d: stim.Circuit.from_file(assets / f"32_20_4_d{d}.stim")
    for d in (1, 2, 4)
}
protocol = optimized_steane_SE_orchestrator(
    code, circuits,
    peak_qubits=60,
    output_dir="build/32_20_4_adaptive_protocol",
    progress=print,
)
print(protocol.verification)
```

Inputs are finalized X-SE circuits; integer keys are their protection labels.
The builder derives Z duals for identical `H_x`/`H_z`. Set `optimize=True` to
apply the existing ZX boundary-bend/reuse pipeline before policy synthesis.
Alternatively, `build_adaptive_se_protocol` in `spiderwarp.adaptive_steane_se`
accepts named `SEGadget` variants, including multiple circuits at the same label.

The minimum eligible protection follows the remaining internal-fault budget:

| Further faults possible in the compatible histories | Minimum circuit label |
|---:|---:|
| 2 | d4 |
| 1 | d2 |
| 0 | d1 |

The data-code distance stays four. Raised flags certify expenditure through the
histories, rather than by counting flag bits. d2 transitions include the possible
new fault; d1 transitions are fault-free. Both bases remain available. A common
safe correction ends the cycle immediately; a flag does not automatically reject
the data or discard a syndrome.

The fault model includes all 15 nonidentity Pauli alternatives at each two-qubit
gate as **one event**, X/Y/Z after one-qubit gates, reset flips, and measurement
flips. The X and Z boundary constraints share the same joint histories and fault
count. Guppy basis rotations are lowered before enumeration. Implicit idle
locations and backend-specific gate transformations are outside this exact model.

For `r` incoming CSS error order and `s` internal faulty locations, the default
`[[32,20,4]]` contract is:

| (r,s) | Requirement |
|---|---|
| (0,0), (1,0) | Accept, zero residual error, preserve logical state. |
| (0,1) | Accept, preserve ideal-decoded logical state, at most one residual error per CSS component. |
| (0,2) | Accept, at most two residual errors per component; no logical guarantee at this order. |
| (1,1) | Reject, or accept with at most one residual error per component. |
| (2,0) | Reject, or accept with zero residual error. |

The recovery condition additionally checks arbitrary incoming syndrome sectors
through one internal fault. Its bound is the **internal** fault count. Bounds are
modulo stabilizers where logical preservation is required and relative to some
codeword otherwise. A fixed component decoder makes logical tests unambiguous.
`FTContract` exposes these orders explicitly. The stronger
`FTContract.correct_or_reject(code)` request can raise
`FTContractInfeasibleError` with physical silent-terminal-fault witnesses; it is
not the default contract.

Synthesis first finds a certified ladder policy, then minimizes worst-case total
CNOT-equivalent entangling gates under `peak_qubits`. Preparation, coupling, and
all diagnostic extractions count. CX/CZ/XCX each cost one entangling unit.
Optimality is relative to this library, ladder, fault model, and terminal-frame
interface. The search report gives the cost and its lower bound. Use
`minimize=False` for a feasible policy without a minimality claim, or configure
`max_cnot_cost` and `max_search_states` to bound the search.

Adaptive construction always runs an independent policy check that recomputes
transitions and tests the supplied leaves without resynthesizing them. Calling
`protocol.verify()` checks the existing graph again, including the distance
ladder and termination. The old `verify` keyword controls only legacy mode.

`protocol.run(tableau_simulator)` borrows the existing Stim data block and returns
`ProtocolResult(accepted, ..., frame_x, frame_z)`. Corrections are **classical
Pauli-frame masks**; physical correction gates are not applied. Retain the frame
when composing further computation. Pass an existing frame back with
`protocol.run(simulator, frame_x=x, frame_z=z)`; the returned frame includes it.
The generated Guppy `se_cycle(data)` returns
`CycleResult(accepted, frame_x, frame_z)` with the same semantics.
Use `se_cycle_with_frame(data, x, z)` to compose an existing frame.
`postselected_se_cycle(data)` exits on rejection and otherwise returns that result.
The source and HUGR embed all decision tables; there is no runtime decoder sidecar.
Reload the HUGR with `hugr.package.Package.from_bytes(path.read_bytes())`.
Saving also writes `verification.json` and the exact lowered action circuits,
including hashes that tie the report to the exported files.

Guppy is optional for analysis: pass `export=False` to build and run the Stim
reference without it. Install `python -m pip install -e ".[guppy]"` to export.
`protocol.export(output_dir)` verifies and compiles later. The current exact
implementation supports up to two faults, 65,536 input syndrome sectors, and
63 data qubits or combined flag/check bits in the Guppy emitter.

`estimate_protocol_ler` in `spiderwarp.se_protocol_simulation` incorporates the
returned frame in ideal final decoding. It reports conditional LER and discards
separately for logical `0` or `+` inputs. Its default hardware noise includes
scheduled memory noise, so sampling and the exact gate-fault certificate have
different explicitly reported scopes. Classical branch latency is not modeled.
The [demo notebook](notebooks/32_20_4_protocol_demo.ipynb) shows circuit inspection,
export/reload, both logical bases, and noise-source diagnostics.

The earlier fixed primary/recovery implementation remains available explicitly
with `policy_mode="legacy"`. Its bool-returning Guppy interface and physical
corrections are preserved for reproducibility. The current SE assets expose its
previously documented correctable one-fault rejection; it is not the default.

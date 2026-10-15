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

Install `python -m pip install -e ".[guppy]"` in a Python version supported by
Guppy (currently Python 3.12 or newer). The protocol builder accepts X-stabilizer
SE gadgets keyed by circuit distance and derives the primary distance and fault
budgets from the code:

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
    code, circuits, output_dir="build/32_20_4_protocol"
)
```

The short [protocol demo notebook](notebooks/32_20_4_protocol_demo.ipynb) shows
distance-labelled inputs, Guppy/HUGR export and reload, and noisy LER estimation.
`spiderwarp.se_protocol_simulation.estimate_protocol_ler` runs the adaptive Stim
reference policy with `prepared_state="0"` or `"+"`. It reports the corresponding
basis-specific block LER after ideal final decoding, conditional on acceptance,
along with the discard rate and a 95% interval. By default it uses
`noise_model.py`'s `apply_noise`, including the default memory noise, on each
executed extraction and correction block. Input preparation and final decoding
are ideal; classical branch latency is not modeled. Pass a `CustomErrorModel`
through `noise_model=` to override the default.

The builder expands reused ancilla lifetimes, applies ZX/path-cover optimization,
reuses ancillas, remaps measurement parities, and constructs dual-basis gadgets.
It then synthesizes recovery choices and modified decoding from exact circuit
fault responses. Standard decoding uses `spiderwarp.lut_decoder.LutDecoder` at
construction time. The exported source and HUGR contain the decoder/control-flow
decisions, without a separate LUT or JSON control-flow file.

The standalone Guppy program uses named `ExtractionResult`, `Correction`, and
`RecoveryRoute` structs. Circuit instructions are batched over target tuples
inside `@guppy.comptime` functions, preserving gate order and ancilla lifetimes.
Shared decoder and recovery decisions are stored as immutable arrays and read
by one masked-lookup loop, rather than thousands of generated helper functions.
The adaptive cycle and circuit definitions precede the generated lookup data.
The same source supports Guppy 0.21.6 and 1.1.x.

Verification is disabled by default. Pass `verify=True` to the builder, or call
`verify_se_protocol(protocol)` (also available as `protocol.verify()`) separately.
The callable checks the existing gadgets, recovery choices and corrections; it
does not replace them with a newly synthesized policy. It returns a
`VerificationReport` and raises on failure. Unverified protocols can be exported,
executed and sampled. Fault analysis required to construct adaptive decisions
still runs during synthesis. Exported source records `VERIFIED_AT_BUILD`.

A cycle runs a maximum-distance Z primary. With no flags it decodes, corrects,
and runs the maximum-distance X primary. A flagged primary discards its syndrome,
runs the selected dual-basis recovery, and ends the cycle. Any raised recovery
flag rejects the block. The secondary recovery decision retains the first
accepted syndrome, and the analysis uses one shared incoming/internal fault
budget across the complete cycle, including physical corrections.

If a dual-basis recovery alone cannot satisfy the CSS conditions, the builder
adds an extraction of the discarded primary component before decoding. It tries
all single-recovery candidates first. This extra stage runs only on affected
flagged branches, rejects raised flags, and combines the two fresh syndromes.
The clean Z-then-X path is unchanged.

`protocol.run(tableau_simulator)` executes the reference policy on an existing
Stim data block and returns a `ProtocolResult`; callers must honor `accepted`.
Import `se_cycle(data) -> bool` from the generated Python file to compose it
with Guppy, or use `postselected_se_cycle(data)` to terminate rejected shots.
The `.hugr` file is a composable function package, not a state-preparation
entrypoint. Reload it with `hugr.package.Package.from_bytes(path.read_bytes())`.
Call the generated module's `compile_se_cycle()` to compile it across supported
Guppy versions. It uses `with_minimal_opt()` when available (Guppy 1.1.3), and
direct `compile_function()` on older Guppy (tested with 0.21.6). Backend gate
transformations require their own fault analysis. The generated measurement
helper also handles both the older boolean API and the newer `Measurement.read()`
API. Restart the notebook kernel after updating SpiderWarp or Guppy, then run all
cells from the beginning.

Verification preserves arbitrary logical information using an encoded Bell
input and retains Pauli classes modulo stabilizers, including logical Paulis.
The enumerator includes all 15 two-qubit Pauli faults, mixed X/Y/Z histories,
preparation/measurement flips, and single-qubit/correction-gate faults. Implicit
idle noise is not added. The current implementation supports identical
`H_x`/`H_z`, code distances 2–5 (up to two faults), and up to 63 data qubits,
flags, or checks in Guppy's integer-mask emitter.

Verification uses CSS weight `max(w_X, w_Z)` and checks Gottesman's EC A and
EC B with separate X/Z filters. Thus X1 Z17 and X7 Z23 each have CSS weight one.
For `s` incoming CSS errors, `r` internal faults, and
`t = floor((d-1)/2)`:

- EC A: with `r <= t`, every accepted output is at most `r` errors from some
  codeword in each CSS component, regardless of the incoming errors. Input errors do not enlarge
  this residual allowance.
- EC B: when `s+r <= t`, the cycle must accept and its ideal-decoded logical
  state must be preserved.

Postselection outside correctable-order histories is allowed. EC A covers all
incoming syndrome sectors; logical equivalence of these representatives follows
from the linear Pauli responses and encoded-Bell instrument validation. The
current exact implementation supports up to 65,536 input syndrome sectors.
Additional bounded histories through `floor(d/2)` also use the internal-fault
`r`-filter, without capping `r` at the correction radius: two internal faults
can leave two errors per component. Incoming CSS-weight-one states include an
X and Z on different qubits, and are both corrected in a fault-free cycle.
See Gottesman's [EC conditions](https://arxiv.org/pdf/0904.2557#page=21) and
[separate CSS filter discussion](https://perimeterinstitute.ca/personal/dgottesman/QECC2007/Sols6.pdf).

The optimized [[32,20,4]] policy verifies under these conditions. All recovery
stages select d1. Of 8,844 reachable recovery prefixes, 324 need a second d1
extraction to resolve the discarded component; 8,520 use only the dual recovery.
The reason for the extra extraction is documented with four physical Z-only
histories in `analysis/32_20_4_protocol/why_css_recovery_cleanup.txt`.
The old X1 Z17 and X7 Z23 examples are allowed CSS-weight-one residuals; their
earlier total-Pauli-weight failure claims are withdrawn. Current certified Guppy
and HUGR files are in `build/32_20_4_protocol/`; older exports remain archived.

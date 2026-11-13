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

The [[32,20,4]] adaptive methods have these allowed completed distance paths:

- **d4 → d4 → d2**: use one d2 follow-up after a second-d4 flag or an unresolved correction;
- **d4 → d4**: stop once a safe correction is determined;
- **d4 → d2**: raised flags in the first d4 select d2 directly.

There is **no d1 and no third d4**. The first reading is Z_d4, the second X_d4
or X_d2; the follow-up is Z_d2 or X_d2 selected from compatible histories.
A rejected trajectory may stop at an earlier prefix.

Incoming CSS component order is `r=max(w_X,w_Z)` and `s` counts internal faulty
locations. Analyze **r+s≤2** for detection, and **post-select if no r+s≤1
explanation remains**. Correction/residual guarantees cover **r+s≤1**; hidden
order-two histories are not an additional correction demand. The history method
independently verifies mandatory acceptance and correction through order one.
An internal fault remains one fault after propagation to multiple data qubits;
its identified correction is allowed to have weight greater than one. The former
extra raw syndrome/correction-weight guard was incorrect for this full adaptive
mode and is retired. Its corrected fault-aware interpretation is identical to
the history policy, so only one such curve is displayed.
All adaptive methods accept and correct every noiseless CSS-weight-at-most-one input
in the physical controls. The earlier all-nonzero-input rejection was a modeling
error and its benchmark was removed.

The comparison contains:

- **Adaptive PS(r+s≤1)**.
- **Adaptive PS(r+s≤1 or V₂)** preserves every order-one decision and correction.
  Additional records accept only when all compatible histories through order two
  have one final data Pauli class modulo stabilizers. This includes incoming-two,
  incoming-one plus one internal fault, and two distinct internal faulty locations.
  Parent-rejected records are considered. A d4→d4 record may use one d2 in the
  basis that separates the most uniquely correctable outcomes. The certificate
  uses the existing gate/reset/readout fault model; production includes memory
  noise. It does not promise correctness of every accepted order-two history.
- **Adaptive PS(r+s≤1 or p_err≤10%)** and **Adaptive PS(r+s≤1 or p_err≤25%)**
  retain all order-one histories and their original corrections. Other records
  use the most probable final Pauli class modulo stabilizers as their correction;
  `p_err = 1 - P(MAP class | complete record)`. The default-noise model includes
  scheduled memory and an incoming iid Pauli prior, with at most two nonzero
  response channels. All positive-probability records in that expansion are
  enumerated without discovery sampling or output labels. The model and prior
  are frozen before production and reused throughout the physical-noise sweep.
  Epsilon is a per-correction estimate, not an all-orders or endpoint LER bound.
- **Adaptive PS(f=0, r_syn≤1)**, with immediate rejection of any raised flag
  and a direct-readout CSS uniqueness guard. This is the observable “no internal faults” proxy;
  unflagged faults can still pass. It corrects incoming CSS-one inputs, including
  split X/Z errors. Its only accepted path is d4→d4, at exactly 312 CX/CZ.
  This separate direct-readout guard can reject unflagged internal faults;
  it does not have the full adaptive method's single-fault acceptance guarantee.
- **Steane d3 RUS PS(r_syn≤1)**, which retries flagged ancilla preparations
  before coupling and measures the data syndrome once per CSS basis.

Adaptive Baseline, d1 and the retired V/likelihood experiments are excluded.
V₂ is the new full order-two uniqueness analysis on the current controller.
The previous two-total-reading datasets and derived figures were
removed. Current runs use the normal matching shot budgets, extending the valid
half-shot samples without duplicating their completed batches.

**LER and gate-cost statistics exclude rejected data trajectories.** Gate cost
is `accepted_cnots / accepted / N`, including all preparation retries inside
accepted Steane trajectories. Even the early cycles of a later-rejected shot
are excluded. Acceptance rate remains accepted/attempted to show yield. Raw
all-attempt counts remain only for bookkeeping and audit.

Finalized adaptive d4/d2 extractions include coupling and cost 156/121 CX/CZ.
Its completed paths therefore cost **433, 312 or 277**; an accepted-cycle mean
cannot be below **277**. Peak qubits are 60. Steane uses d3 preparation
`FTStatePrep/misc/32_20_4_zero_d3.stim` (112 CX/CZ), two 32-CX couplings, and
66 peak qubits: 288 gates before preparation retries.

Each method uses the same 32 configurations, noise, seed 20261112 and inverse-square
allocation: **100,000 floor / 1,000,000 cap**, **9,329,148 attempts per method**.
Wilson intervals use accepted counts; circles mark adaptive methods and squares
Steane. Production includes scheduled memory noise with ideal input/final decoding.

```sh
python -m spiderwarp.se_experiment build --paired --distances 2 4 \
    --out build/se_32_20_4 --emit none
python analysis/32_20_4_bounded_comparison/build.py
python -m spiderwarp.se_comparison --jobs 8
python analysis/32_20_4_bounded_comparison/controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/order_two_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/probability_controls.py
# Refresh saved figures and notebook outputs without sampling.
python -m spiderwarp.se_comparison --plot-only
```

The [comparison notebook](notebooks/32_20_4_steane_comparison.ipynb) and
[protocol demo](notebooks/32_20_4_protocol_demo.ipynb) embed count tables and all
six figure pairs. Guppy/HUGR artifacts, independent verification, physical/native
controls and raw-batch gate audits are linked from
[the study](analysis/32_20_4_bounded_comparison/). Legacy arbitrary-input research
contracts remain explicit and are not used for this comparison.

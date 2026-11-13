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
and attribution study have been removed. All five adaptive methods are rerun
from empty datasets with 32 matching configurations, seed 20261112 and
100,000 floor / 1,000,000 cap: 9,329,148 attempts each. The independent Steane
results are retained. Production includes scheduled memory noise, with ideal
preparation and final decoding once per trajectory.

```sh
# The paired program supplies the recovery template, not a plotted baseline.
python -m spiderwarp.se_experiment build --paired --distances 2 4 \
    --out build/se_32_20_4 --emit none
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/build.py
PYTHONPATH=. python -m spiderwarp.se_comparison --jobs 8
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/order_two_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/probability_controls.py
PYTHONPATH=. python analysis/32_20_4_bounded_comparison/finalize.py
# Refresh figures and notebooks without sampling.
python -m spiderwarp.se_comparison --plot-only
```

The [comparison notebook](notebooks/32_20_4_steane_comparison.ipynb) and
[protocol demo](notebooks/32_20_4_protocol_demo.ipynb) embed counts and all six
figure pairs. Sources, HUGRs, controls and raw-batch audits are linked from
[the study](analysis/32_20_4_bounded_comparison/).

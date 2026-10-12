import subprocess
import sys


def test_notebook_imports_and_assets_outside_checkout(tmp_path):
    """An installed package must work without the checkout on sys.path."""
    script = '''
import sys

class NoSpiderCSS:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "spidercss" or fullname.startswith("spidercss."):
            raise ModuleNotFoundError("SpiderWarp must not require SpiderCSS")

sys.meta_path.insert(0, NoSpiderCSS())

from spiderwarp.csscode import CSSCode
from spiderwarp.utils import load_state_prep_circuit
from spiderwarp.stim_utils import steane_se_from_stim_state_prep, qasm_str_to_stim_circuit
from spiderwarp.path_cover import CoveredZXGraph
from spiderwarp.fault_tolerance_verification import verify_ftsp
from spiderwarp.path_cover_metrics import metric_hardware_qubits_resource_scheduled
from spiderwarp.resource_scheduling import plan_resource_aware_reuse

code = CSSCode.load_code("misc", "32_20_4")
assert (code.n, code.k, code.d) == (32, 20, 4)
state_prep = load_state_prep_circuit("misc", "opt_zero_32_20_4")
se = steane_se_from_stim_state_prep(state_prep, se_basis="Z", n=code.n)
assert se.num_qubits >= code.n
assert se.num_measurements > code.n
# Exercise the ancilla API required by the optimization cells, too.
graph = CoveredZXGraph.from_stim(se)
assert graph.G.number_of_nodes() > 0
qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'
assert qasm_str_to_stim_circuit(qasm).num_qubits == 1
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

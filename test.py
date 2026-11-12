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
        protocol,
        prepared_state=state,
        shots=1_000,
        seed=20261015,
    )
    print(
        state,
        "LER:", result.ler,
        "95% interval:", result.ler_interval_95,
        "discard rate:", result.discard_rate,
    )
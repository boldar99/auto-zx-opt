"""A late rejection excludes all earlier cycles of that shot from gate stats."""
import csv
from types import SimpleNamespace

import numpy as np
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.noise_model import CustomErrorModel
from spiderwarp.optimized_steane_se import ProtocolResult
from spiderwarp.se_protocol_simulation import ProtocolSampler
from spiderwarp.se_plotting import plot_results


def test_late_rejection_excludes_the_whole_trajectory():
    code = CSSCode.load_code('MQT', '7_1_3')
    # Two CNOTs cancel exactly, but still cost two gates on each executed cycle.
    gadget = SimpleNamespace(circuit=stim.Circuit('CX 0 1 0 1'), basis='Z', distance=4)
    acceptance = iter([True, True, True, True, False, False, True, True, True])

    def run(simulator, **frames):
        simulator.do_circuit(gadget.circuit)
        keep = next(acceptance)
        return ProtocolResult(keep, (('g', 'Z', 4, 0, 0),), (), '' if keep else 'scripted rejection')

    protocol = SimpleNamespace(code=code, required_qubits=code.n, gadgets={'g': gadget}, policy=True, run=run)
    result = ProtocolSampler(protocol, noise_model=CustomErrorModel(), cycles=3).sample(4, 17)
    d = result.diagnostics
    assert result.accepted == 2 and result.logical_errors == 0
    assert d.cycles_started == 9 and d.total_cnots == 18
    assert d.accepted_cnots == 12
    assert d.accepted_stage_counts == {('g', 'Z', 4): 6}
    assert d.accepted_cnots / result.accepted / result.cycles == 2
    assert d.total_cnots / result.shots / result.cycles == 1.5


def test_resource_plot_uses_only_gates_from_accepted_trajectories(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    directories = []
    for method in ('adaptive', 'steane'):
        directory = tmp_path / method
        directory.mkdir()
        directories.append(directory)
        rows = []
        for basis in ('0', '+'):
            for cycles, accepted in ((1, 85), (10, 10)):
                cost = 312 if cycles == 1 else 300
                rows.append(dict(protocol_hash=method, engine_hash='engine', prepared_state=basis,
                    p=.0008, cycles=cycles, shots=100, accepted=accepted, logical_errors=1,
                    total_cnots=30600 if cycles == 1 else 120000,
                    accepted_cnots=accepted*cycles*cost, noise='{"p_2q":0.0008}', sweeps='["cycles"]'))
        with (directory / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    seen = []

    def inspect(figure, path, **kwargs):
        if path.stem == 'cnot_cost':
            for ax in figure.axes:
                line = next(line for line in ax.lines if line.get_label() == 'adaptive')
                np.testing.assert_allclose(line.get_ydata(), [312, 300])
                assert 'accepted completed cycle' in ax.get_ylabel()
            seen.append(path)

    monkeypatch.setattr(Figure, 'savefig', inspect)
    plot_results(directories[0], tmp_path / 'figures', compare=directories[1:], labels=['adaptive', 'steane'])
    assert len(seen) == 2

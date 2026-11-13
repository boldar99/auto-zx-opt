"""Audit counts and regenerate the matched scientific figures/notebooks."""
import base64
from collections import Counter
import csv
import hashlib
import html
import inspect
from io import StringIO
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hugr.package import Package

from spiderwarp.se_comparison import save_comparison, save_resources
from spiderwarp.se_comparison_names import METHODS, METHOD_COLORS, LEGEND_NOTE
from spiderwarp.se_plotting import plot_results
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import _wilson_interval
from importlib import import_module

study = import_module('analysis.32_20_4_bounded_comparison.study')
ROOT, OUT, write = study.ROOT, study.OUT, study.write
COMPARISON = ROOT / 'benchmarks/steane_comparison'


def counts_html(rows):
    text = ['<table><tr><th>Method</th><th>Basis</th><th>Cycles</th><th>Attempts</th>'
            '<th>Failures / accepted</th><th>Conditional block LER (95% Wilson)</th>'
            '<th>Acceptance</th><th>Mean CX/CZ / accepted completed cycle</th></tr>']
    for row in rows:
        shots, accepted, errors = (int(row[k]) for k in ('shots', 'accepted', 'logical_errors'))
        low, high = _wilson_interval(errors, accepted)
        ler = (f'{100*errors/accepted:.4f}% ({100*low:.4f}–{100*high:.4f}%)'
               if accepted else 'undefined (0 accepted)')
        cost = (f'{int(row["accepted_cnots"])/accepted/int(row["cycles"]):.2f}'
                if accepted else 'undefined')
        values = [row['method'], row['prepared_state'], row['cycles'], f'{shots:,}',
                  f'{errors:,} / {accepted:,}',
                  ler, f'{100*accepted/shots:.3f}%', cost]
        text.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>')
    return ''.join(text) + '</table>'


def tradeoff(rows):
    by = {(r['method'], r['prepared_state'], int(r['cycles'])): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    for i, cycles in enumerate((1, 10)):
        for j, basis in enumerate(('0', '+')):
            ax = axes[i, j]
            for key, method in METHODS.items():
                row = by[method.label, basis, cycles]
                shots, accepted, errors = (int(row[k]) for k in ('shots', 'accepted', 'logical_errors'))
                x = 100*accepted/shots
                lowx, highx = (100*v for v in _wilson_interval(accepted, shots))
                lowy, highy = (100*v for v in _wilson_interval(errors, accepted))
                y = 100*errors/accepted
                zero = errors == 0
                ax.errorbar(x, highy if zero else y,
                            xerr=[[x-lowx], [highx-x]],
                            yerr=None if zero else [[y-lowy], [highy-y]],
                            fmt='v' if zero else method.marker, mfc='none' if zero else METHOD_COLORS[key],
                            color=METHOD_COLORS[key], capsize=3, markersize=6, label=method.plot_label)
            ax.set_yscale('log')
            ax.set_title(f'Logical {basis}; N={cycles}')
            ax.set_xlabel('Acceptance of all requested cycles (%)')
            ax.set_ylabel('Conditional block LER (%)')
            ax.grid(True, which='both', ls='--', alpha=.35)
            ax.margins(x=.12, y=.18)
            lo, hi = ax.get_xlim()
            ax.set_xlim(max(0, lo), min(100, hi))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .065), ncol=2, fontsize=8.7)
    fig.suptitle('Logical error versus acceptance at default hardware noise (p₂q=0.0008)')
    attempts = int(rows[0]['shots'])
    fig.text(.5, .012, f'{attempts:,} attempts per point for every method; 95% Wilson intervals. Any of 20 logical readouts may fail.', ha='center', fontsize=9)
    fig.text(.5, .04, LEGEND_NOTE, ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .26 if len(METHODS) > 6 else .23, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(COMPARISON / 'figures' / ('acceptance_error_tradeoff.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def code(source, tag=None):
    return dict(cell_type='code', id=hashlib.sha256(source.encode()).hexdigest()[:8],
                metadata={'tags': [tag]} if tag else {}, execution_count=None,
                source=source.splitlines(keepends=True), outputs=[])


def markdown(source):
    return dict(cell_type='markdown', id=hashlib.sha256(source.encode()).hexdigest()[:8],
                metadata={}, source=source.splitlines(keepends=True))


def capture(cells):
    import IPython.display as ipd
    namespace = {}
    original = ipd.display
    try:
        for number, cell in enumerate(cells, 1):
            if cell['cell_type'] != 'code':
                continue
            outputs = []

            def display(*objects, **kwargs):
                for obj in objects:
                    if isinstance(obj, ipd.HTML):
                        data = {'text/html': obj.data, 'text/plain': '<matched counts>'}
                    elif isinstance(obj, ipd.Image):
                        data = {'image/png': base64.b64encode(obj.data).decode(), 'text/plain': '<saved scientific figure>'}
                    else:
                        raise TypeError(type(obj))
                    outputs.append(dict(output_type='display_data', data=data, metadata={}))
            ipd.display = display
            exec(compile(''.join(cell['source']), '<bounded comparison notebook>', 'exec'), namespace)
            cell['outputs'], cell['execution_count'] = outputs, number
    finally:
        ipd.display = original


def notebooks(shot_floor, shot_cap, total_attempts):
    intro = '''# Adaptive and Steane QEC: standard flag-conditioned postselection

The meaning of **PS(r+s≤1)** is shared by every active adaptive method:

- If no flags are raised, interpret the Z_d4 and X_d4 readings as incoming
  CSS errors. Accept and apply their direct correction iff max(w_X,w_Z)≤1.
- Raised flags use the fault-history recovery procedure, including propagated
  hook corrections of weight greater than one. The cumulative incoming Pauli
  frame is removed before interpreting a reading.

The base has 1,089 accepted quiet syndrome pairs and 1,282 accepted flagged
records. It accepts every noiseless CSS-one input. Some unflagged internal
single faults are postselected; they are not exceptions to the incoming-syndrome
rule. Logical preservation of accepted histories is independently checked
through total order one. No additional circuit-distance claim is made.

The six displayed methods are the standard base; **r+s≤1 or V₂**;
**r+s≤1 or p_err≤10%**; **r+s≤1 or p_err≤25%**;
**f=0, r_syn≤1**; and **Steane d3 RUS PS(r_syn≤1)**.
The previous history-only implementation and separate flag-recovery curve are
removed, together with their prior simulation data.

**V₂** preserves every accepted base path and correction. Other records accept
only when all compatible histories through r+s≤2 agree on one final Pauli class
modulo stabilizers. An optional d2 follow-up includes any remaining fault budget,
including for a base-rejected record with a possible single-fault explanation.
The certificate covers the gate/reset/readout model, without implicit idles.

**Probability rescues** preserve every accepted base path and correction.
For other records, choose the MAP final data Pauli class modulo stabilizers and
accept if p_err=1−P(MAP class | record)≤epsilon. The 10% and 25% domains are nested.
The frozen default-noise model includes scheduled memory and an incoming iid
Pauli prior, conditioned on at most two nonzero response channels. Physical-site
multiplicity, exclusive alternatives and stabilizer degeneracy are retained.
All positive-probability records are scored analytically; no discovery shots or
ideal output labels select the controller. Epsilon is an estimated correction
confidence cutoff, not an all-orders endpoint LER bound. The incoming prior is
reset for inference each cycle and adds no physical operation.

**The zero-flag method** restricts the same standard base by immediately rejecting
any raised flag. Accepted cycles are exactly Z_d4→X_d4, at 312 CX/CZ per cycle.

Allowed completed adaptive paths are d4→d4, d4→d2 and d4→d4→d2. There is no d1
or third d4. Each adaptive uses 60 peak qubits, with completed-cycle costs
312, 277 or 433 CX/CZ respectively. Steane uses 66 qubits and 288 gates before
its noisy ancilla-preparation retries. It reads each data CSS syndrome once.

All five adaptive datasets are fresh reruns after deleting the previous data.
The independent Steane dataset is retained byte-for-byte. Every method uses
32 matching configurations, SHOT_FLOOR floor / SHOT_CAP cap and TOTAL_ATTEMPTS
attempts, with inverse-square allocation and seed 20261112.

LER and gate cost condition on fully accepted trajectories; earlier cycles of
a later-rejected shot are excluded. Cost is accepted_cnots/accepted/N. Acceptance
uses all attempts. LER is a 20-logical-qubit endpoint block error; LER/N is a
normalization, rather than a measured first-failure hazard. Intervals are 95%
Wilson. Production includes scheduled memory with ideal preparation and final
decoding once per trajectory. Circles denote adaptive methods and squares Steane.
'''
    intro = (intro.replace('SHOT_FLOOR', f'{shot_floor:,}')
             .replace('SHOT_CAP', f'{shot_cap:,}').replace('TOTAL_ATTEMPTS', f'{total_attempts:,}'))
    artifacts = '## Guppy and HUGR artifacts\n\n| Method | Standalone Guppy | Compiled HUGR | Counts |\n|---|---|---|---|\n'
    for key, method in METHODS.items():
        folder = study.BUNDLES[key]
        artifacts += f'| {method.name} | [source](../build/{folder}/steane_se_protocol.py) | [HUGR](../build/{folder}/steane_se_protocol.hugr) | [counts](../benchmarks/{method.results}/summary.csv) |\n'
    artifacts += '\nProtocol/source/binary hashes, bounded verification, physical/native controls and sampling audits are recorded in [the study](../analysis/32_20_4_bounded_comparison/).\n'
    table_source = '''from pathlib import Path
import csv
import html
from spiderwarp.se_protocol_simulation import _wilson_interval
from IPython.display import display, HTML, Image
root = Path.cwd()
if not (root / 'benchmarks').exists():
    root = root.parent
'''
    table_source += inspect.getsource(counts_html) + '''
rows = list(csv.DictReader((root / 'benchmarks/steane_comparison/default_endpoints.csv').open()))
display(HTML(counts_html(rows)))
'''
    image_source = '''for name in ('physical_error', 'cycles', 'survival', 'cnot_cost', 'physical_survival'):
    display(Image(filename=str(root / 'benchmarks/steane_comparison/figures' / (name + '.png'))))
'''
    tradeoff_source = "display(Image(filename=str(root / 'benchmarks/steane_comparison/figures/acceptance_error_tradeoff.png')))\n"
    for name in ('32_20_4_steane_comparison', '32_20_4_protocol_demo'):
        cells = [markdown(intro), markdown(artifacts), markdown(f'## Default-noise endpoints\n\nEach row has {shot_floor:,} attempted trajectories.\n'),
                 code(table_source, 'qec-comparison'), code(image_source, 'qec-comparison'),
                 code(tradeoff_source, 'qec-comparison'),
                 markdown('Regenerate with `python -m spiderwarp.se_comparison --plot-only`.\nThese notebook cells load saved counts and figures; they do not sample.\n')]
        capture(cells)
        book = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python (qec)', 'language': 'python', 'name': 'python3'},
                    'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)
        write(ROOT / 'notebooks' / (name + '.ipynb'), book)


def main():
    directories = [ROOT / 'benchmarks' / method.results for method in METHODS.values()]
    methods = list(METHODS.values())
    runs = [json.loads((directory / 'run.json').read_text()) for directory in directories]
    assert all(run['status'] == 'complete' for run in runs)
    shot_floor, shot_cap = runs[0]['shot_floor'], runs[0]['max_shots']
    total_attempts = runs[0]['total_target_shots']
    assert all((run['shot_floor'], run['max_shots'], run['total_target_shots']) ==
               (shot_floor, shot_cap, total_attempts) for run in runs)
    save_comparison(directories, [m.label for m in methods], COMPARISON,
                    qubit_counts=[m.qubits for m in methods], always_accept=[False]*len(methods))
    audits = {}
    for key, directory in zip(METHODS, directories):
        bundle = ROOT / 'build' / study.BUNDLES[key]
        protocol = load_protocol(bundle)
        export = json.loads((bundle / 'export.json').read_text())
        source = (bundle / 'steane_se_protocol.py').read_text()
        if METHODS[key].family == 'adaptive':
            from spiderwarp._se_adaptive_guppy import generate_adaptive_source
            assert generate_adaptive_source(protocol) == source
        else:
            from spiderwarp._steane_qec_guppy import generate_source
            assert generate_source(protocol) == source
        binary = (bundle / 'steane_se_protocol.hugr').read_bytes()
        assert hashlib.sha256(source.encode()).hexdigest() == export['source_sha256']
        assert hashlib.sha256(binary).hexdigest() == export['hugr_sha256']
        Package.from_bytes(binary)
        costs = {name: sum(len(op.targets_copy()) // 2 for op in gadget.circuit.flattened()
                           if op.name in {'CX', 'CZ', 'XCX'}) for name, gadget in protocol.gadgets.items()}
        rows = list(csv.DictReader((directory / 'summary.csv').open()))
        for row in rows:
            batches = list(csv.DictReader((directory / 'batches' / (row['task_hash'] + '.csv')).open()))
            assert sum(int(b['shots']) for b in batches) == int(row['shots']) == int(row['target_shots'])
            for field in ('accepted', 'logical_errors', 'total_cnots', 'accepted_cnots', 'cycles_started', 'cycles_completed'):
                assert sum(int(b[field]) for b in batches) == int(row[field])
            stages, accepted_stages = Counter(), Counter()
            for batch in batches:
                stages.update({stage[0]: count for stage, count in json.loads(batch['diagnostics'])['stage_counts']})
                accepted_stages.update({stage[0]: count for stage, count in json.loads(batch['diagnostics'])['accepted_stage_counts']})
            assert sum(count * costs[stage] for stage, count in stages.items()) == int(row['total_cnots'])
            assert sum(count * costs[stage] for stage, count in accepted_stages.items()) == int(row['accepted_cnots'])
            if METHODS[key].family == 'adaptive':
                assert not any('_d1' in stage for stage in stages)
                assert stages['Z_d4'] == int(row['cycles_started'])
                assert stages['X_d4'] <= int(row['cycles_started'])
                assert stages['X_d2'] + stages['Z_d2'] <= int(row['cycles_started'])
                assert sum(stages.values()) <= 3*int(row['cycles_started'])
                n = int(row['accepted']) * int(row['cycles'])
                assert accepted_stages['Z_d4'] == n
                assert 2*n <= sum(accepted_stages.values()) <= 3*n
                path42 = n - accepted_stages['X_d4']
                path442 = accepted_stages['X_d2'] + accepted_stages['Z_d2'] - path42
                path44 = accepted_stages['X_d4'] - path442
                assert min(path42, path442, path44) >= 0
                assert path42 + path442 + path44 == n
                assert 277*path42 + 312*path44 + 433*path442 == int(row['accepted_cnots'])
                assert 277*n <= int(row['accepted_cnots']) <= 433*n
                if key == 'adaptive_no_flags':
                    assert int(row['accepted_cnots']) == 312*n
                    assert stages['X_d2'] == stages['Z_d2'] == 0
                    assert path42 == path442 == 0 and path44 == n
            else:
                n = int(row['accepted']) * int(row['cycles'])
                assert accepted_stages['read_Z'] == accepted_stages['read_X'] == n
                assert accepted_stages['prep_Z_d3'] >= n and accepted_stages['prep_X_d3'] >= n
                assert not any('_d1' in stage for stage in stages)
                assert int(row['accepted_cnots']) >= 288*n
        assert len(rows) == 32 and sum(int(r['shots']) for r in rows) == total_attempts
        audits[key] = dict(configurations=32, attempted_shots=total_attempts, raw_gate_counts_verified=True,
                           batch_counts_verified=True, protocol_hash=rows[0]['protocol_hash'],
                           source_regenerates_identically=True, source_sha256=export['source_sha256'],
                           hugr_reloaded=True, hugr_sha256=export['hugr_sha256'])
    all_rows = list(csv.DictReader((COMPARISON / 'comparison.csv').open()))
    endpoints = [r for r in all_rows if 'cycles' in r['sweeps'] and int(r['cycles']) in (1, 10)]
    assert len(all_rows) == 32*len(METHODS) and len(endpoints) == 4*len(METHODS)
    assert all(int(r['shots']) == shot_floor for r in endpoints)
    buffer = StringIO(newline='')
    writer = csv.DictWriter(buffer, list(endpoints[0]))
    writer.writeheader()
    writer.writerows(endpoints)
    atomic = import_module('spiderwarp.se_protocol_io').atomic_write
    atomic(COMPARISON / 'default_endpoints.csv', buffer.getvalue())
    plot_results(directories[0], COMPARISON / 'figures', compare=directories[1:],
                 labels=[m.plot_label for m in methods], markers=[m.marker for m in methods],
                 colors=list(METHOD_COLORS.values()), physical_survival=True,
                 sample_note=f'Matched shot budgets: {shot_floor:,} floor / {shot_cap:,} cap', legend_note=LEGEND_NOTE,
                 resource_floor=277)
    tradeoff(endpoints)
    save_resources(load_protocol(ROOT / 'build/steane_single_d3_fixed_ps'), COMPARISON)
    notebooks(shot_floor, shot_cap, total_attempts)
    validation = json.loads((COMPARISON / 'validation.json').read_text())
    controls = json.loads((OUT / 'controls.json').read_text())
    base_controls = json.loads((OUT / 'standard_base_controls.json').read_text())
    v2_controls = json.loads((OUT / 'order_two_controls.json').read_text())
    probability_controls = json.loads((OUT / 'probability_controls.json').read_text())
    v2_export = json.loads((OUT / 'order_two_export_audit.json').read_text())
    assert all(c['complete'] for c in (controls, base_controls, v2_controls, probability_controls))
    assert v2_export['complete'] and v2_export['transitions_checked'] > 1000000
    validation.update(
        standard_base_rule='No flags: incoming CSS order <=1; flags: fault-history recovery',
        all_adaptive_methods_use_standard_base=True,
        old_history_only_implementation_removed=True, old_simulation_data_deleted=True,
        adaptive_methods_rerun=5, independent_steane_data_retained=True,
        no_d1=True, max_adaptive_extractions_per_cycle=3,
        allowed_distance_paths=[[4, 4, 2], [4, 4], [4, 2]],
        gate_cost='accepted_cnots / accepted / N; complete accepted trajectories only',
        rejected_trajectories_excluded_from_ler_and_gate_statistics=True,
        adaptive_accepted_cycle_gate_bounds=[277, 433],
        methods_audited=audits, physical_native_controls=controls,
        standard_base_controls=base_controls, order_two_controls=v2_controls,
        order_two_export_audit=v2_export,
        probability_controls=probability_controls,
        notebooks_refreshed=True, six_figure_pairs_generated=True,
        visual_inspection_pending=True)
    write(COMPARISON / 'validation.json', validation)
    write(OUT / 'sampling_validation.json', validation)
    print(f'All {len(all_rows)} rows and {len(endpoints)} endpoints audited; figures and both notebooks refreshed.', flush=True)


if __name__ == '__main__':
    main()

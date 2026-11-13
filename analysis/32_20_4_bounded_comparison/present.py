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
    fig.tight_layout(rect=(0, .23, 1, .96))
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
    intro = '''# Adaptive and Steane QEC: d2 follow-up, accepted-trajectory statistics

The comparison contains **Adaptive PS(r+s≤1)**, **Adaptive PS(r+s≤1 or V₂)**,
**Adaptive PS(r+s≤1 or p_err≤10%)**, **Adaptive PS(r+s≤1 or p_err≤25%)**,
**Adaptive PS(f=0, r_syn≤1)**, and **fixed d3 Steane RUS PS(r_syn≤1)**.
Adaptive Baseline, d1, and the retired V/likelihood experiments are excluded.

The allowed completed adaptive paths are exactly:

- **d4 → d4 → d2**: a d2 follow-up after the second d4 flag or an unresolved correction;
- **d4 → d4**: stop when the two readings determine a safe correction;
- **d4 → d2**: use d2 directly when the first d4 flags are raised.

The first reading is Z_d4 and the second is X_d4 or X_d2. A third reading,
when needed, uses Z_d2 or X_d2 chosen from the compatible history classes.
There is **no d1 and no third d4**. Rejected prefixes can stop sooner.

Incoming CSS order is r=max(w_X,w_Z), and s counts internal faulty locations.
Analyze **r+s≤2** for detection and reject when **no r+s≤1 explanation remains**.
Correction and residual guarantees cover **r+s≤1**. Higher-order histories that
remain indistinguishable from low-order ones are not an extra correction demand.
The history method independently verifies mandatory acceptance/correction of
all order-one histories. One internal fault remains s=1 even when it propagates
to multiple data errors. Flag and follow-up information identify its correction.
There is no additional restriction on the propagated correction's weight.

**V₂** preserves the history controller on every order-one-compatible record.
Other records accept only if every compatible explanation through total order
two agrees on one final data Pauli class modulo stabilizers. The analysis includes
(r,s)=(2,0), (1,1), and (0,2), with two faults required to occupy distinct physical
locations. Parent-rejected records are considered. A d4/d4 record may use one
d2 in the basis that separates the most uniquely correctable outcome groups.
The exact certificate uses the existing gate/reset/readout fault model, without
implicit idle faults; production simulation includes scheduled memory noise.
This certifies the additional rescues through order two in that circuit model,
not every accepted order-two history or the all-orders production LER.

**Probability rescues** also preserve every order-one action and correction.
For each other complete record, they choose the most probable final data-error
Pauli class modulo stabilizers and return its inverse as the correction frame.
Here **p_err = 1 − P(MAP class | complete record)**: the estimated probability
that this correction does not return the original encoded state exactly.
Accept additional records at p_err≤10% or p_err≤25%. This class-error criterion
is a sufficient condition for logical correction; a remaining correctable data
error is still counted as a wrong Pauli class by the selection model.

The probability model includes scheduled memory, gate, reset and measurement
noise, physical-location multiplicities, exclusive alternatives and stabilizer
degeneracy. It is frozen at default hardware noise before production and reused
throughout the physical-noise sweep. The incoming
prior is iid depolarization at p₂q per data qubit, reset each cycle; it is an
inference prior and adds no physical operation. The model conditions on at most
two nonzero response channels. Omitted higher-event mass is recorded, so epsilon
is an estimated per-correction confidence cutoff, not an all-orders bound or a
target endpoint LER. All positive-probability records in that expansion are
enumerated analytically, without discovery sampling or ideal output labels.
The original parent chooses any third reading; other records finish the first
two readings. The two thresholds therefore share the same measurement routing
and correction for common records and have nested acceptance domains.

The former full-adaptive “r_syn≤1” curve incorrectly applied a raw syndrome/
correction-weight guard and rejected some valid single internal faults. That
curve is retired. With incoming order inferred using the complete fault history,
the corrected mode has exactly the same protocol hash as PS(r+s≤1), so its
compatibility alias is not displayed separately. Existing valid history results are reused.

**PS(f=0, r_syn≤1)** adds immediate rejection of every nonzero measured flag
to the direct-readout CSS guard. This separate, stricter experiment does not
promise acceptance of all unflagged internal faults. It is the requested
observable meaning of “s=0”; a fault
can be unflagged. It retains and corrects noiseless incoming CSS-one inputs,
including separate X and Z errors on different qubits. Its accepted path is
always **Z_d4 → X_d4**, costing exactly **312 CX/CZ per cycle**. Flagged
prefixes reject before a d2 follow-up, so no d2 is executed by this variant.

**Rejected data trajectories are excluded from LER and gate-cost statistics.**
The cost is `accepted_cnots / accepted / N`: gates used by accepted full
trajectories divided by their actual N completed cycles. Even the early cycles
of a trajectory later rejected contribute nothing to that statistic. Acceptance
rate remains accepted/attempted to expose yield. Noisily discarded ancillas
inside an eventually accepted Steane RUS trajectory still count toward its cost.

For a completed adaptive cycle, the three paths cost **433, 312, and 277 CX/CZ**
respectively. Thus its accepted-cycle mean cannot be below **277**. A quiet cycle
costs 312. Peak qubits are 60. Steane costs 288 before preparation retries and
uses 66 peak qubits, with one data reading per CSS basis.

Every method uses **32 matching configurations, SHOT_FLOOR attempts at each default-noise
repetition point and a SHOT_CAP low-noise cap**, totaling **TOTAL_ATTEMPTS attempts per
method**. Noise, seed 20261112 and inverse-square allocation match. Accepted
counts differ. LER is conditional endpoint block error across 20 logical readouts;
LER/N is a normalization of that endpoint rate, not a measured first-failure hazard.
95% Wilson intervals use actual accepted counts. Production includes scheduled
memory noise; ideal input and final decoding occur once per trajectory.

Circles mark adaptive methods; squares mark Steane. Open downward triangles
mark zero-failure confidence upper bounds. All sources, HUGRs, controls and
batch/count/gate audits are saved with the study.
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
    validation.update(baseline_excluded=True, old_adaptive_data_deleted=True,
                      adaptive_domain='r+s<=2; history PS requires order-one compatibility; V2 adds exact uniqueness rescues', no_d1=True,
                      max_adaptive_extractions_per_cycle=3, max_adaptive_d4_per_cycle=2,
                      allowed_distance_paths=[[4, 4, 2], [4, 4], [4, 2]],
                      gate_cost='accepted_cnots / accepted / N; complete accepted trajectories only',
                      rejected_trajectories_excluded_from_ler_and_gate_statistics=True,
                      acceptance_rate_denominator='All attempted trajectories',
                      adaptive_accepted_cycle_gate_bounds=[277, 433],
                      no_order_two_rescues=False, methods_audited=audits,
                      order_two_uniqueness_rescue='adaptive_order_two',
                      rescue_certificate_includes_memory=False,
                      propagated_weight_guard_retired=True,
                      full_adaptive_css_mode_identical_to_history=True,
                      notebooks_refreshed=True, six_figure_pairs_generated=True,
                      visual_inspection_pending=True)
    controls = json.loads((OUT / 'controls.json').read_text())
    assert controls['complete']
    validation['physical_native_controls'] = 'controls.json'
    validation['all_css_one_inputs_accepted_and_correct_in_controls'] = all(
        record['accepted_inputs_correct'] == 1089 and record['postselected_inputs'] == 0
        for method in controls['methods'].values() for record in method.values())
    rescue_controls = json.loads((OUT / 'order_two_controls.json').read_text())
    assert rescue_controls['complete'] and rescue_controls['counts']['rescued_exactly_correct'] > 0
    validation['order_two_physical_controls'] = 'order_two_controls.json'
    validation['additional_rescues_physically_replayed_correct'] = rescue_controls['counts']['rescued_exactly_correct']
    export_audit = json.loads((OUT / 'order_two_export_audit.json').read_text())
    assert export_audit['complete'] and export_audit['transitions_checked'] > 1000000
    validation['order_two_export_transition_audit'] = export_audit
    validation['probability_rescues'] = dict(epsilons=[0.1, 0.25],
        error='Wrong MAP Pauli class modulo stabilizers', mandatory_order_one=True,
        model='Default-noise frozen two-response-channel posterior, including memory',
        all_orders_bound=False, discovery_sampling=False)
    probability_controls = json.loads((OUT / 'probability_controls.json').read_text())
    assert probability_controls['complete'] and probability_controls['nesting']['acceptance_nested']
    validation['probability_controls'] = probability_controls
    write(COMPARISON / 'validation.json', validation)
    write(OUT / 'sampling_validation.json', validation)
    print(f'All {len(all_rows)} rows and {len(endpoints)} endpoints audited; six figure pairs and both notebooks refreshed.', flush=True)


if __name__ == '__main__':
    main()

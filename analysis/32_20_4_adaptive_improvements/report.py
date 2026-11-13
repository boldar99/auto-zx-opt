"""Report held-out full sweeps and the fixed pilot selection study."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spiderwarp.se_protocol_simulation import _wilson_interval
from spiderwarp.se_comparison import save_comparison
from spiderwarp.se_comparison_names import METHODS as NAMED_METHODS, PILOT_NAMES


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METHODS = (
    ('adaptive_baseline', 0),
    ('adaptive_history', 3),
    ('adaptive_syndrome', 6),
    ('adaptive_recovery', 7),
    ('steane_d4_ps', 4),
    ('steane_descending_ps', 5),
)


def main():
    pilots = []
    for name in ('pilot', 'pilot_separated', 'pilot_recover', 'pilot_rescue_refinements', 'pilot_safe_rescue'):
        pilots.extend(csv.DictReader((OUT / (name + '.csv')).open()))
    pilots = [dict(method=PILOT_NAMES[row['variant']], **row) for row in pilots]
    with (OUT / 'all_pilots.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(pilots[0]))
        writer.writeheader()
        writer.writerows(pilots)
    data, endpoint_rows = {}, []
    save_comparison([ROOT / 'benchmarks' / NAMED_METHODS[key].results for key, _ in METHODS],
        [NAMED_METHODS[key].label for key, _ in METHODS], OUT / 'focused_comparison',
        qubit_counts=[NAMED_METHODS[key].qubits for key, _ in METHODS],
        always_accept=[False]*len(METHODS), require_equal_budgets=True)
    for key, color in METHODS:
        method = NAMED_METHODS[key]
        label, name = method.label, method.results
        path = ROOT / 'benchmarks' / name
        assert json.loads((path / 'run.json').read_text())['status'] == 'complete'
        rows = list(csv.DictReader((path / 'summary.csv').open()))
        for row in rows:
            if 'cycles' in json.loads(row['sweeps']) and int(row['cycles']) in (1, 10):
                data[label, row['prepared_state'], int(row['cycles'])] = row
                endpoint_rows.append(dict(method=label, results=name, **row))
    with (OUT / 'default_endpoints.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)

    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for row_index, cycles in enumerate((1, 10)):
        for column, basis in enumerate(('0', '+')):
            ax = axes[row_index, column]
            for key, color in METHODS:
                label = NAMED_METHODS[key].label
                row = data[label, basis, cycles]
                shots, accepted, errors = (int(row[k]) for k in ('shots', 'accepted', 'logical_errors'))
                x, y = 100*accepted/shots, 100*errors/accepted
                xl, xh = (100*v for v in _wilson_interval(accepted, shots))
                yl, yh = (100*v for v in _wilson_interval(errors, accepted))
                ax.errorbar(x, y, xerr=[[x-xl], [max(0, xh-x)]],
                    yerr=[[y-yl], [max(0, yh-y)]], fmt='o', color=colors[color], capsize=3,
                    markersize=6, label=label)
            ax.set_yscale('log')
            ax.set_title(f'Logical {basis}; N={cycles}')
            ax.set_xlabel('Acceptance of all requested cycles (%)')
            ax.set_ylabel('Conditional block LER (%)')
            ax.grid(True, which='both', linestyle='--', alpha=.35)
            ax.margins(x=.10, y=.15)
            bottom, top = ax.get_xlim()
            ax.set_xlim(max(0, bottom), min(100, top))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3, fontsize=9)
    fig.suptitle('Logical error versus acceptance at default hardware noise (p₂q=0.0008)')
    fig.text(.5, .012, '100,000 attempts per point for every method; 95% Wilson intervals. Failure of any of 20 logical readouts.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / ('acceptance_error_tradeoff.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)

    stronger = json.loads((OUT / 'stronger_contract.json').read_text())
    selection = dict(pilot_candidates=len({r['variant'] for r in pilots}),
        pilot_attempted_shots=sum(int(r['shots']) for r in pilots),
        pilot_seed=2026111301, independent_production_seed=20261112,
        pilot_names=PILOT_NAMES,
        comparison_names={key: method.name for key, method in NAMED_METHODS.items()},
        equal_attempted_shots_per_configuration=True,
        steane_ps_full_budget_extension='../../benchmarks/steane_css_ps_<mode>_100k/seed_provenance.json',
        production_variants=['css', 'css-recover-safe'],
        budgets='32 fixed configurations, 100k floor / 1M cap, inverse square, 9,329,148 attempts per new full sweep',
        selected_css_assumption='Each CSS component weight <=2; stabilizer equivalence, not total Pauli weight <=2',
        reference='Adaptive history PS remains the low-LER adaptive reference',
        disqualified_aggressive_rescue=dict(variant='css-recover-exact', reason=stronger,
            dataset='benchmarks/adaptive_css_recover_exact_100k',
            use='Experimental ablation only; excluded from the main comparison because it fails the original arbitrary-input filter'),
        safe_rescue='Only rescue a higher-order observation record when incoming CSS<=1 plus one internal fault explanations have one exact Pauli class, and its correction has the same syndrome as the original parent frame',
        negative_pilots=['CSS role reversal', 'Extra final d2 checks', 'Independent projected CSS histories'],
        guarantees='Safe rescue preserves the strict-PS fault contract; the plain CSS filter allows one-fault postselection',
        interpretation='LER is conditional on acceptance; endpoint LER/N is not a first-failure hazard. No one-cycle independence assumption for N=10 acceptance.',
        selection_frozen_before_production=True,
        endpoint_results=endpoint_rows)
    (OUT / 'selection_report.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(f'Reported {selection["pilot_candidates"]} pilot candidates, {selection["pilot_attempted_shots"]:,} discovery shots, and full independent sweeps.')


if __name__ == '__main__':
    main()

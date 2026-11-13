"""Separate seeded pilot samples for choosing CSS adaptive improvements."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import json
from pathlib import Path

from spiderwarp.se_adaptive_experiments import AdaptiveCSSCandidate, SeparatedCSSCandidate, CSSRecoverCandidate, VARIANTS
from spiderwarp.se_protocol_io import load_protocol
from spiderwarp.se_protocol_simulation import ProtocolSampler


def sample(task):
    variant, basis, cycles = task
    parent = load_protocol('build/se_32_20_4')
    protocol = (CSSRecoverCandidate(parent, variant) if variant.startswith('css-recover') else
                SeparatedCSSCandidate(parent, variant) if variant.startswith('css-separated') else
                AdaptiveCSSCandidate(parent, variant))
    result = ProtocolSampler(protocol, prepared_state=basis, cycles=cycles, collect_paths=False).sample(10000, 2026111301)
    low, high = result.ler_interval_95
    return dict(variant=variant, basis=basis, cycles=cycles, shots=result.shots,
        accepted=result.accepted, logical_errors=result.logical_errors,
        ler=result.ler, low=low, high=high, survival=result.acceptance_rate,
        mean_cnots=result.diagnostics.total_cnots/result.shots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS[:4]))
    parser.add_argument('--output', default='pilot')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    tasks = [(v, b, n) for v in args.variants for n in (1, 10) for b in ('0', '+')]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(sample, tasks):
            rows.append(row)
            print(row, flush=True)
    with (out / (args.output + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / (args.output + '.json')).write_text(json.dumps(dict(seed=2026111301,
        shots_per_point=10000, configurations=tasks, results=rows,
        selection='Discovery only; selected candidates get independent production samples',
        rules={v: ('Full-Pauli incoming CSS<=1 / internal-fault<=1 rescue; exact and syndrome-preserving restrictions named by the variant'
                   if v.startswith('css-recover') else
                   'Independent projected CSS-component histories, common safe frame'
                   if v.startswith('css-separated') else
                   'Final input-relative correction syndrome must be unique through CSS weight two; optional flagged-history guard, role reversal or final d2 checks')
               for v in args.variants},
        acceptance_inputs='Measured flags/syndromes and existing Pauli frame only; no prepared-state label or final ideal readout',
        pilot_resource_note='Adapters decide at the parent endpoint; compiled graphs can abort rejected branches earlier. Pilot CNOT counts are not used for selection.',
        no_oracle=True), indent=2) + '\n')

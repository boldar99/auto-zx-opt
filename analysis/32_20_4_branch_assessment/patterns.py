"""Flag-mask risk, observable two-fault filters, and independent holdout checks.

Discovery exactly replays the saved 100k default-noise N=1 shots per basis.
Validation uses 100k fresh shots per basis. Learned masks use discovery only.
No controller, circuit, production dataset or exported program is changed.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import csv
import gzip
from io import StringIO
import json
import multiprocessing
from pathlib import Path

import numpy as np

from spiderwarp.noise_model import CustomErrorModel, default_noise_model
from spiderwarp.se_protocol_io import atomic_write, digest, load_protocol, read_manifest, versions
from spiderwarp.se_protocol_simulation import ProtocolSampler, _NoisySimulator, _wilson_interval

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRODUCTION = ROOT / "benchmarks/se_100k"
PROTOCOL = ROOT / "build/se_32_20_4"
THRESHOLDS = (.01, .02, .05, .10, .20, .50)
MIN_TRAIN_ACCEPTED = 100


class LowOrderEvidence:
    """Raw-observation compatibility with r+s<=1, independent of policy frames.

    Ideal gadgets preserve the physical data Pauli and their input flags are
    zero. Future observations depend only on the data syndrome. Thus merging
    equal (r,s,data-syndrome) histories is exact for this compatibility test;
    it makes no claim about their untracked logical Pauli or posterior weight.
    """

    def __init__(self, protocol):
        self.protocol = protocol
        self.names = sorted(protocol.gadgets)
        space = protocol.policy.decoder.space
        self.space = space
        decoder = protocol.policy.decoder
        self.x_width, self.x_mask = decoder.x_width, decoder.x_mask
        initial = set()
        for sx in decoder.classes["X"]:
            for sz in decoder.classes["Z"]:
                r = max(decoder.weights["X"][sx], decoder.weights["Z"][sz])
                if r <= 1:
                    initial.add((r, 0, sx | (sz << self.x_width)))
        self.initial = frozenset(initial)
        self.responses, self.flag_orders = {}, {}
        for aid, name in enumerate(self.names):
            g = protocol.gadgets[name]
            # Check assumptions used by the syndrome-only compatibility test.
            for q in range(2 * protocol.code.n):
                flags, _, output = g.unpack(g.input_effect(1 << q))
                assert flags == 0 and output == space.reduce(1 << q)
            responses = defaultdict(set)
            for effect in g.single_fault_effects:
                f, s, data = g.unpack(effect)
                responses[f, s].add(space.pauli_syndrome(data))
            self.responses[aid] = responses
            sites = defaultdict(set)
            for location, effects in g.responses[1]:
                for effect in effects:
                    sites[effect & ((1 << g.num_flags)-1)].add(location)
            orders = {0: 0, **{f: 1 for f in sites if f}}
            for f, left in sites.items():
                for other, right in sites.items():
                    if len(left) > 1 or len(right) > 1 or left != right:
                        orders.setdefault(f ^ other, 2)
            self.flag_orders[aid] = orders

    @lru_cache(maxsize=None)
    def step(self, state, aid, flags, syndrome):
        if not state:
            return state
        g = self.protocol.gadgets[self.names[aid]]
        following = set()
        for r, s, signature in state:
            readout = signature & self.x_mask if g.basis == "Z" else signature >> self.x_width
            if flags == 0 and syndrome == readout:
                following.add((r, s, signature))
            if r + s == 0:
                for delta in self.responses[aid].get((flags, syndrome ^ readout), ()):
                    following.add((0, 1, signature ^ delta))
        return frozenset(following)


def count_update(counter, accepted, failed):
    counter["encountered"] += 1
    counter["accepted"] += int(accepted)
    counter["logical_errors"] += int(failed)


def counts_row(counts):
    a, e = counts["accepted"], counts["logical_errors"]
    lo, hi = _wilson_interval(e, a)
    return dict(counts, p_error_given_accepted=e/a if a else None, low=lo, high=hi,
                p_original_accept= a/counts["encountered"] if counts["encountered"] else None)


def sample_basis(task_path):
    metadata = json.loads(Path(task_path).read_text())
    task, basis = metadata["task"], metadata["task"]["prepared_state"]
    production_batches = list(csv.DictReader((PRODUCTION / "batches" / (Path(task_path).stem + ".csv")).open()))
    protocol = load_protocol(PROTOCOL)
    sampler = ProtocolSampler(protocol, noise_model=CustomErrorModel(**task["noise"]),
                              prepared_state=basis, cycles=1, collect_paths=False)
    evidence = LowOrderEvidence(protocol)
    ids = {name: i for i, name in enumerate(evidence.names)}
    all_stats, examples = {}, []
    for split in ("discovery", "validation"):
        stats = defaultdict(Counter)
        trace_path = OUT / f"{split}_{'zero' if basis == '0' else 'plus'}.jsonl.gz"
        seeds = []
        with gzip.open(trace_path, "wt") as stream:
            for batch_index, production in enumerate(production_batches):
                seed = (int(production["seed"]) if split == "discovery" else
                        int(digest(dict(study="flag-pattern-holdout-v1", basis=basis, batch=batch_index))[:15], 16))
                shots = int(production["shots"])
                seeds.append(dict(seed=seed, shots=shots))
                rng = np.random.default_rng(seed)
                accepted = errors = 0
                for shot_index in range(shots):
                    sim = sampler.experiment.initial.copy(seed=int(rng.integers(0, 2**63)))
                    result = protocol.run(_NoisySimulator(sim, sampler.circuits, sampler.noise, sampler.corrections))
                    failed = bool(result.accepted and sampler.experiment.logical_failure(
                        sim, frame_x=result.frame_x, frame_z=result.frame_z))
                    accepted += result.accepted
                    errors += failed
                    observations = [[ids[name], flags, syndrome] for name, _, _, flags, syndrome in result.measurements]
                    state = evidence.initial
                    flag_order = 0
                    unique = set()
                    prior_rs = []
                    for step, (aid, f, s) in enumerate(observations, 1):
                        prior_rs.append(sorted({(r, spent) for r, spent, _ in state}))
                        state = evidence.step(state, aid, f, s)
                        # Missing flag masks are outside the two-fault model;
                        # both evidence filters reject rather than claiming an
                        # exact physical event count for those observations.
                        flag_order += evidence.flag_orders[aid].get(f, 3)
                        unique.add(("action_mask", aid, f))
                        unique.add(("stage_mask", step, aid, f))
                    history = tuple((aid, f) for aid, f, _ in observations)
                    unique.add(("flag_history", history))
                    for key in unique:
                        count_update(stats[key], result.accepted, failed)
                    record = [batch_index, shot_index, int(result.accepted), int(failed), observations,
                              flag_order, int(bool(state))]
                    stream.write(json.dumps(record, separators=(",", ":"))+"\n")
                    no_d4_flag = not any(f and protocol.gadgets[evidence.names[aid]].distance == 4
                                         for aid, f, _ in observations)
                    lower_flag = any(f and protocol.gadgets[evidence.names[aid]].distance < 4
                                     for aid, f, _ in observations)
                    if split == "discovery" and no_d4_flag and lower_flag and len(examples) < 6:
                        examples.append(dict(accepted=result.accepted, failed=failed,
                            observations=[dict(action=evidence.names[aid], flags=f, syndrome=s,
                                               low_order_rs_before=rs)
                                          for (aid, f, s), rs in zip(observations, prior_rs)]))
                if split == "discovery":
                    assert (accepted, errors) == (int(production["accepted"]), int(production["logical_errors"]))
                print(f"{basis} {split}: {sum(s['shots'] for s in seeds):,} shots", flush=True)
        all_stats[split] = dict(seeds=seeds, patterns=[(key, dict(value)) for key, value in stats.items()])
    return dict(basis=basis, names=evidence.names,
                flag_orders={aid: orders for aid, orders in evidence.flag_orders.items()},
                widths={i: protocol.gadgets[name].num_flags for i, name in enumerate(evidence.names)},
                distances={i: protocol.gadgets[name].distance for i, name in enumerate(evidence.names)},
                splits=all_stats, examples=examples)


def csv_save(path, rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue())


def analyze(results):
    names = results[0]["names"]
    widths, distances = results[0]["widths"], results[0]["distances"]
    risks, training = [], {}
    for result in results:
        basis = result["basis"]
        for split, data in result["splits"].items():
            for key, counts in data["patterns"]:
                kind = key[0]
                aid, flag, step = (key[1], key[2], None) if kind == "action_mask" else (
                    (key[2], key[3], key[1]) if kind == "stage_mask" else (None, None, None))
                history = key[1] if kind == "flag_history" else None
                pattern = (";".join(f"{names[a]}:0x{f:03x}" for a, f in history) if history is not None
                           else f"{names[aid]}:0x{flag:03x}")
                risks.append(dict(split=split, prepared_state=basis, kind=kind, pattern=pattern,
                    step=step, action=names[aid] if aid is not None else None,
                    flag_mask=flag, mask_hex=f"0x{flag:03x}" if flag is not None else None,
                    bits_lsb_first="".join(str(flag >> i & 1) for i in range(widths[aid])) if aid is not None else None,
                    set_bits=flag.bit_count() if flag is not None else None,
                    min_internal_order_for_mask=result["flag_orders"][aid].get(flag) if aid is not None else None,
                    **counts_row(counts)))
                if split == "discovery" and kind == "action_mask" and flag:
                    training.setdefault((aid, flag), {})[basis] = counts_row(counts)
    learned = {}
    for threshold in THRESHOLDS:
        bad = {key for key, bases in training.items()
               if set(bases) == {"0", "+"} and min(b["accepted"] for b in bases.values()) >= MIN_TRAIN_ACCEPTED
               and max(b["low"] for b in bases.values()) > threshold}
        learned[f"learned_bad_masks_{threshold:g}"] = bad
    policy_rows = []
    for result in results:
        for split in ("discovery", "validation"):
            counts = defaultdict(Counter)
            trace = OUT / f"{split}_{'zero' if result['basis'] == '0' else 'plus'}.jsonl.gz"
            total = 0
            with gzip.open(trace, "rt") as stream:
                for line in stream:
                    _, _, accepted, failed, observations, flag_order, compatible = json.loads(line)
                    keys = {(aid, f) for aid, f, _ in observations}
                    rules = dict(baseline=True,
                        all_d4_flags_zero=not any(f and distances[aid] == 4 for aid, f, _ in observations),
                        all_flags_zero=not any(f for _, f, _ in observations),
                        reject_two_flagged_extractions=sum(bool(f) for _, f, _ in observations) < 2,
                        reject_flag_evidence_ge2=flag_order < 2,
                        reject_no_order01_explanation=bool(compatible))
                    rules.update({name: not bool(keys & bad) for name, bad in learned.items()})
                    for name, keep in rules.items():
                        if keep:
                            count_update(counts[name], accepted, failed)
                    total += 1
            for name, values in sorted(counts.items()):
                a = values["accepted"]
                lo, hi = _wilson_interval(a, total)
                policy_rows.append(dict(split=split, prepared_state=result["basis"], rule=name,
                    total_shots=total, **counts_row(values), survival=a/total,
                    survival_low=lo, survival_high=hi))
    csv_save(OUT / "pattern_risk.csv", risks)
    csv_save(OUT / "postselection_policies.csv", policy_rows)
    record = dict(methodology="100k saved production discovery shots per basis; 100k independent validation shots per basis",
        protocol_hash=read_manifest(PROTOCOL)[1]["protocol_hash"], versions=versions(),
        minimum_training_accepted_per_basis=MIN_TRAIN_ACCEPTED, risk_thresholds=THRESHOLDS,
        learned_rule="Reject a nonzero action/mask if worst-basis discovery Wilson lower bound exceeds threshold; same rule for both bases",
        logical_metric="P(any logical readout wrong | original policy accepted and rule keeps trajectory)",
        selection_inputs="Only measured extraction flags/syndromes; no hidden noise events or final ideal/logical readout",
        compatibility_rule="Reject if no raw-observation explanation with r+s<=1; physical data syndromes tracked, no hidden fault counts",
        flag_evidence_rule="Sum minimum internal orders for raw masks across separate gadgets; reject sum>=2 or out-of-model masks",
        caveats=["Rules prune current accepted paths; no forced switching or new certificate",
                 "At least two flagged bits need not mean two faulty locations",
                 "Order bounds apply to the exact gate-fault model; unflagged higher-order faults can remain indistinguishable",
                 "Rare masks have wide intervals; learned masks use discovery only; threshold-family comparison is exploratory",
                 "Both bases use one common rejection rule; their errors are not a measured joint arbitrary-state LER"],
        learned_bad_masks={name: [dict(action=names[aid], flag_mask=f, mask_hex=f"0x{f:03x}")
                                 for aid, f in sorted(bad)] for name, bad in learned.items()},
        seeds={result["basis"]: {split: data["seeds"] for split, data in result["splits"].items()} for result in results},
        no_d4_flags_with_lower_flag_examples={r["basis"]: r["examples"] for r in results},
        policy_results=policy_rows)
    atomic_write(OUT / "patterns_assessment.json", json.dumps(record, indent=2)+"\n")
    for row in policy_rows:
        if row["split"] == "validation":
            print(row, flush=True)


if __name__ == "__main__":
    tasks = []
    for path in sorted((PRODUCTION / "tasks").glob("*.json")):
        task = json.loads(path.read_text())["task"]
        if task["cycles"] == 1 and task["noise"] == vars(default_noise_model()):
            tasks.append(str(path))
    assert len(tasks) == 2
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(sample_basis, tasks))
    analyze(results)

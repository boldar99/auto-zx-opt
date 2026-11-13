"""Cross-check the evidence classifier and replay physical single faults."""
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import gzip
import json

from patterns import LowOrderEvidence, OUT, PROTOCOL
from spiderwarp._se_adaptive import CompatibleSet, FTContract, HistoryEngine
from spiderwarp._se_circuit import CircuitFault
from spiderwarp.se_protocol_io import atomic_write, load_protocol
from spiderwarp.se_protocol_simulation import _StateExperiment


def main(protocol_path=PROTOCOL, output_dir=OUT):
    protocol = load_protocol(protocol_path)
    evidence = LowOrderEvidence(protocol)
    space = protocol.policy.decoder.space
    engine = HistoryEngine(space, FTContract(1, 1, fault_order=1, accept_clean_through=1), protocol.policy.decoder)
    initial = engine.initial()
    initial = CompatibleSet(initial.logical, (0, 0), engine.sectors)
    checked = set()

    @lru_cache(maxsize=None)
    def compare(prefix):
        if not prefix:
            full, compact = initial, evidence.initial
        else:
            full_parent, compact_parent = compare(prefix[:-1])
            aid, f, s = prefix[-1]
            action = protocol.policy.actions[evidence.names[aid]]
            full = engine.transition(full_parent, action, (f, s)).get((f, s),
                CompatibleSet(frozenset(), (0, 0), engine.sectors))
            compact = evidence.step(compact_parent, aid, f, s)
        projected = frozenset((r, spent, space.pauli_syndrome(p)) for r, spent, p in full.logical)
        assert compact == projected, prefix
        checked.add(prefix)
        return full, compact

    for basis in ("zero", "plus"):
        with gzip.open(OUT / f"discovery_{basis}.jsonl.gz", "rt") as stream:
            for line in stream:
                observations = json.loads(line)[4]
                compare(tuple(tuple(v) for v in observations))
                if len(checked) >= 2000:
                    break
    print(f"Syndrome-only classifier matches full Pauli-history engine on {len(checked)} prefixes", flush=True)

    circuits = {}
    audit_counts = {}
    one_event_many_flags = None
    silent_pairs = []
    for basis in ("0", "+"):
        experiment = _StateExperiment(protocol.code, protocol.required_qubits, basis)
        clean = protocol.run(experiment.initial.copy(seed=17))
        single_count = incoming_count = 0

        def physical_run(stage=None, circuit=None, incoming_x=0, incoming_z=0):
            simulator = experiment.initial.copy(seed=17)
            for q in range(protocol.code.n):
                if incoming_x >> q & 1:
                    simulator.x(q)
                if incoming_z >> q & 1:
                    simulator.z(q)

            class Inject:
                index = 0
                def current_measurement_record(self):
                    return simulator.current_measurement_record()
                def do_circuit(self, ideal):
                    simulator.do_circuit(circuit if self.index == stage else ideal)
                    self.index += 1

            result = protocol.run(Inject())
            failed = result.accepted and experiment.logical_failure(
                simulator, frame_x=result.frame_x, frame_z=result.frame_z)
            state = evidence.initial
            order = 0
            for name, _, _, f, s in result.measurements:
                aid = evidence.names.index(name)
                state = evidence.step(state, aid, f, s)
                order += evidence.flag_orders[aid].get(f, 3)
            return result, bool(failed), bool(state), order

        for stage, (name, _, _, _, _) in enumerate(clean.measurements):
            gadget = protocol.gadgets[name]
            for effect in gadget.single_fault_effects:
                key = name, effect
                faults = gadget.fault_witness(effect, 1)
                if key not in circuits:
                    circuits[key] = gadget.with_faults(faults)
                result, failed, compatible, order = physical_run(stage, circuits[key])
                assert result.accepted and not failed and compatible and order <= 1, (basis, stage, effect)
                single_count += 1
                if one_event_many_flags is None and result.measurements[stage][3].bit_count() > 1:
                    one_event_many_flags = dict(action=name, flags=result.measurements[stage][3],
                        set_bits=result.measurements[stage][3].bit_count(), faults=[asdict(f) for f in faults])
            print(f"basis {basis}, stage {stage+1}: all {len(gadget.single_fault_effects)} single-fault responses pass", flush=True)
        for x in [0] + [1 << q for q in range(protocol.code.n)]:
            for z in [0] + [1 << q for q in range(protocol.code.n)]:
                result, failed, compatible, order = physical_run(incoming_x=x, incoming_z=z)
                assert result.accepted and not failed and compatible and order == 0
                incoming_count += 1

        # A concrete invisible output-boundary pair: observations remain clean
        # but ideal final decoding gives a wrong logical readout. No flag or
        # syndrome-only postselection rule can reject this record alone.
        last_name = clean.measurements[-1][0]
        gadget = protocol.gadgets[last_name]
        last = {q: i for i, (_, qs) in enumerate(gadget.operations) for q in qs if q < protocol.code.n}
        fault_effects = {fault: effect for effect, faults in gadget._fault_witnesses.items() for fault in faults}
        component = "X" if basis == "0" else "Z"
        for error, physical in space.component_errors(component, 2).items():
            if not protocol.policy.decoder.logical(error):
                continue
            sites = {}
            for letter, q in space.product(physical):
                sites.setdefault(last[q], []).append((letter, q))
            faults = tuple(CircuitFault(i, tuple(product)) for i, product in sorted(sites.items()))
            if len(faults) != 2 or any(f not in fault_effects for f in faults):
                continue
            response = 0
            for f in faults:
                response ^= fault_effects[f]
            if gadget.unpack(response) != (0, 0, error):
                continue
            result, failed, compatible, order = physical_run(len(clean.measurements)-1, gadget.with_faults(faults))
            assert result.measurements == clean.measurements and result.accepted and failed and compatible and order == 0
            silent_pairs.append(dict(prepared_state=basis, action=last_name, faults=[asdict(f) for f in faults],
                                     accepted=True, logical_failure=True, observations_equal_clean=True))
            break
        assert len(silent_pairs) == (1 if basis == "0" else 2)
        audit_counts[basis] = dict(single_fault_responses=single_count, incoming_css_weight_at_most_one=incoming_count,
                                  unexpected_rejections=0, logical_failures_in_these_low_order_cases=0)
    record = dict(full_history_engine_prefix_matches=len(checked), physical_low_order_checks=audit_counts,
                  one_fault_multiple_flag_bits=one_event_many_flags, silent_two_fault_witnesses=silent_pairs,
                  conclusion="No low-order discrepancy found. Invisible two-fault logical errors remain possible under the existing contract.")
    atomic_write(Path(output_dir) / "patterns_audit.json", json.dumps(record, indent=2)+"\n")
    print(record, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.protocol, args.out)

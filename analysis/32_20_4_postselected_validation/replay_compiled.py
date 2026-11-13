"""Compare the compiled graph with all 400k previously analyzed observations."""
from collections import Counter
import gzip
import json
from pathlib import Path

from spiderwarp.se_protocol_io import atomic_write, load_protocol, read_manifest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRACES = ROOT / "analysis/32_20_4_branch_assessment"
OLD = ROOT / "build/se_32_20_4"
NEW = ROOT / "build/se_postselected_32_20_4"


def execute(protocol, observations, frame_x=0, frame_z=0, shift_observations=False):
    policy = protocol.policy
    space = policy.decoder.space
    frame = frame_x | (frame_z << protocol.code.n)
    initial_frame = frame
    index, consumed = policy.root, 0
    names = sorted(protocol.gadgets)
    while True:
        node = policy.nodes[index]
        if node.kind == "reject":
            return False, 0, 0, consumed
        if node.kind == "accept":
            mask = (1 << protocol.code.n)-1
            correction = frame ^ node.frame
            return True, correction & mask, correction >> protocol.code.n, consumed
        if node.kind == "frame":
            frame ^= node.frame
            index = node.next
            continue
        aid, flags, syndrome = observations[consumed]
        assert names[aid] == node.action, (names[aid], node.action)
        basis = "X" if protocol.gadgets[node.action].basis == "Z" else "Z"
        if shift_observations:
            syndrome ^= space.component_syndrome(initial_frame, basis)
        syndrome ^= space.component_syndrome(frame, basis)
        consumed += 1
        index = node.children.get((flags, syndrome), 0)


def main():
    old, new = load_protocol(OLD), load_protocol(NEW)
    assert {n: str(g.circuit) for n, g in old.gadgets.items()} == {n: str(g.circuit) for n, g in new.gadgets.items()}
    totals = {}
    for split in ("discovery", "validation"):
        for basis, suffix in (("0", "zero"), ("+", "plus")):
            counts = Counter()
            with gzip.open(TRACES / f"{split}_{suffix}.jsonl.gz", "rt") as stream:
                for line in stream:
                    _, _, accepted, failed, observations, _, compatible = json.loads(line)
                    expected = execute(old, observations)
                    assert expected[0] == bool(accepted)
                    actual = execute(new, observations)
                    assert actual[0] == bool(accepted and compatible)
                    assert actual[3] <= expected[3]
                    if actual[0]:
                        assert actual == expected
                        counts["accepted"] += 1
                        counts["logical_errors"] += failed
                    else:
                        counts["earlier_abort"] += actual[3] < expected[3]
                    counts["shots"] += 1
                    if counts["shots"] <= 1000:
                        shifted = execute(new, observations, frame_x=1, frame_z=1 << 17,
                                          shift_observations=True)
                        assert shifted[0] == actual[0] and shifted[3] == actual[3]
                        if actual[0]:
                            assert shifted[1:3] == (actual[1] ^ 1, actual[2] ^ (1 << 17))
                        counts["external_frame_checks"] += 1
            totals[split + "/" + basis] = dict(counts)
            print(split, basis, dict(counts), flush=True)
    record = dict(parent_protocol_hash=read_manifest(OLD)[1]["protocol_hash"],
                  protocol_hash=read_manifest(NEW)[1]["protocol_hash"],
                  circuits_identical=True, all_400k_decisions_match=True,
                  surviving_frames_and_actions_identical=True, totals=totals)
    atomic_write(OUT / "replay_compiled.json", json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()

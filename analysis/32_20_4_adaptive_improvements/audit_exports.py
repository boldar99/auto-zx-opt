"""Check every stored edge/frame and representative missing outcomes in exports."""
import json
from pathlib import Path
import sys

from spiderwarp._se_adaptive_guppy import generate_adaptive_source
from spiderwarp.se_protocol_io import load_protocol, read_manifest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from test_adaptive_se_policy import test_real_export_preserves_every_stored_transition
from test_optimized_steane_se import generated_python


if __name__ == '__main__':
    output = {}
    for tag in ('css', 'css_recover_exact', 'css_recover_safe'):
        directory = Path('build') / ('adaptive_' + tag + '_32_20_4')
        protocol = load_protocol(directory)
        test_real_export_preserves_every_stored_transition(protocol)
        exported = generated_python(generate_adaptive_source(protocol))
        ids = {old: new for new, old in enumerate(i for i, node in enumerate(protocol.policy.nodes) if node.kind != 'frame')}
        checked = 0
        for index, node in enumerate(protocol.policy.nodes):
            if node.kind != 'extract':
                continue
            kind, _, root, *_ = exported['POLICY_NODES'][ids[index]]
            gadget = protocol.gadgets[node.action]
            keys = {f | (s << gadget.num_flags) for f, s in node.children}
            key = 0
            while key in keys:
                key += 1
            if key >= 1 << gadget.num_detectors:
                continue
            if kind == 2:
                assert exported['lookup_control'](root, key) == ids[0]
            else:
                assert not exported['lookup'](exported['TERMINAL_DECODERS'][root][0], key)
            checked += 1
        output[tag] = dict(protocol_hash=read_manifest(directory)[1]['protocol_hash'],
            all_stored_transitions_and_frames_match=True,
            stored_edges=sum(len(n.children) for n in protocol.policy.nodes),
            missing_observation_checks=checked, missing_observations_reject=True)
        print(tag, output[tag], flush=True)
        (Path(__file__).resolve().parent / 'export_audit.json').write_text(json.dumps(output, indent=2) + '\n')

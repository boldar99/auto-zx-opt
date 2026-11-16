"""Re-emit existing certified bundles and audit every decoder transition.

No synthesis, sampling, or protocol-manifest changes. Run from the repository
root with PYTHONPATH=. and the qec Python environment.
"""
import argparse
import json
from importlib import import_module
from pathlib import Path
import time

from spiderwarp.se_experiment import emit_protocol
from spiderwarp.se_protocol_io import load_protocol, read_manifest, atomic_write, versions

study = import_module('analysis.32_20_4_bounded_comparison.study')
checks = import_module('analysis.32_20_4_bounded_comparison.probability_controls')
REPORT = Path(__file__).with_name('guppy_compaction.json')


def sizes(directory):
    source = (directory / 'steane_se_protocol.py').read_text()
    return dict(source_lines=len(source.splitlines()), source_bytes=len(source.encode()),
                hugr_bytes=(directory / 'steane_se_protocol.hugr').stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=[name for name in study.BUNDLES if name != 'steane_fixed_ps'])
    args = parser.parse_args()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else dict(methods={})
    report['versions'] = versions()
    for name, bundle in study.BUNDLES.items():
        if name == 'steane_fixed_ps' or (args.only and name != args.only):
            continue
        directory = study.ROOT / 'build' / bundle
        before = report['methods'].get(name, {}).get('before', sizes(directory))
        manifest = (directory / 'protocol.json').read_bytes()
        protocol = load_protocol(directory)
        started = time.monotonic()
        emit_protocol(protocol, directory, 'hugr')
        source_path = directory / 'steane_se_protocol.py'
        exported = checks.emitted_bodies(source_path.read_text(), source_path)
        audit = checks.audit_export(protocol, exported)
        assert (directory / 'protocol.json').read_bytes() == manifest
        # The sidecar filename is content-addressed; account for the one used
        # by this module rather than unrelated files from older exports.
        import ast
        names = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse(source_path.read_text()).body
                 if isinstance(node, ast.Assign) and node.targets[0].id == 'TABLES_FILE'}
        after = sizes(directory)
        after['table_bytes'] = (directory / names['TABLES_FILE']).stat().st_size
        report['methods'][name] = dict(
            before=before, after=after, audit=audit,
            protocol_hash=read_manifest(directory)[1]['protocol_hash'],
            manifest_unchanged=True, elapsed_seconds=time.monotonic() - started,
            selected=exported['DECODER_BASELINE_MODE'],
            direct_table_bytes=exported['DIRECT_TABLE_BYTES'],
            patched_table_bytes=exported['PATCHED_TABLE_BYTES'],
            decision_nodes=exported['N_DECISION_NODES'],
            policy_nodes=exported['N_POLICY_NODES'],
        )
        atomic_write(REPORT, json.dumps(report, indent=2) + '\n')
        print(name, json.dumps(report['methods'][name]), flush=True)


if __name__ == '__main__':
    main()

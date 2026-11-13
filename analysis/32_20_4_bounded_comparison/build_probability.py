"""Build both noise-frozen MAP confidence controllers without discovery shots."""
import argparse
from dataclasses import asdict
from importlib import import_module

from spiderwarp.se_probability import compile_probability_rescues
from spiderwarp.se_protocol_io import load_protocol, read_manifest
from spiderwarp.se_experiment import emit_protocol

study = import_module('analysis.32_20_4_bounded_comparison.study')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit', choices=('none', 'guppy', 'hugr'), default='hugr')
    args = parser.parse_args(argv)
    parent_path = study.ROOT / 'build' / study.BUNDLES['adaptive_history']
    parent = load_protocol(parent_path)
    results = compile_probability_rescues(parent, progress=lambda message: print(message, flush=True))
    # Save both ordinary controllers before optional native compilation so
    # reference production sampling can proceed independently.
    for epsilon, (protocol, provenance) in results.items():
        key = 'adaptive_probability_' + str(round(100 * epsilon))
        path = study.ROOT / 'build' / study.BUNDLES[key]
        protocol.save_bundle(path)
        provenance.update(protocol_hash=read_manifest(path)[1]['protocol_hash'],
            parent_protocol_hash=read_manifest(parent_path)[1]['protocol_hash'],
            verification=asdict(protocol.verification))
        study.write(path / 'probability.json', provenance)
        study.write(study.OUT / ('probability_' + str(round(100 * epsilon)) + '_build.json'), provenance)
        print('Saved', key, provenance['protocol_hash'], flush=True)
    for epsilon, (protocol, _) in results.items():
        key = 'adaptive_probability_' + str(round(100 * epsilon))
        emit_protocol(protocol, study.ROOT / 'build' / study.BUNDLES[key], args.emit)


if __name__ == '__main__':
    main()

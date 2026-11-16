"""Exact partial decoders, shared offsets, and portable compile-time tables."""
import importlib.util
from pathlib import Path
import random
import shutil
import sys

import pytest

from spiderwarp._se_decisions import XorDecisions


def test_xor_decisions_preserve_total_and_partial_functions():
    rng = random.Random(20261113)
    decisions = XorDecisions()
    for width in range(9):
        table = {k: rng.getrandbits(63) for k in range(1 << width) if rng.random() < .4}
        total = decisions.build(table, width)
        partial = decisions.build(table, width, partial=True)
        for key in range(1 << width):
            assert decisions.evaluate(total, key) == table.get(key, 0)
            if key in table:
                assert decisions.evaluate(partial, key) == table[key]
    # A translated function reuses every decision, including its subtrees.
    table = {0: 3, 1: 7, 4: 2, 6: 9}
    first = decisions.build(table, 20, partial=True)
    count = len(decisions.nodes)
    second = decisions.build({k: v ^ 123 for k, v in table.items()}, 20, partial=True)
    assert first[0] == second[0] and first[1] ^ second[1] == 123
    assert len(decisions.nodes) == count


@pytest.mark.parametrize('table,width', [({-1: 0}, 3), ({8: 0}, 3), ({0: -1}, 3)])
def test_decision_bounds_are_checked(table, width):
    with pytest.raises(ValueError):
        XorDecisions().build(table, width)


def test_portable_sidecar_compiles_and_rejects_tampering(tmp_path):
    pytest.importorskip('guppylang')
    from hugr.package import Package
    from spiderwarp.adaptive_steane_se import build_adaptive_se_protocol
    from spiderwarp._se_adaptive_guppy import (
        inline_adaptive_source, generate_adaptive_source, write_adaptive_source,
    )
    from test_adaptive_se_policy import bell_actions
    from test_optimized_steane_se import generated_python

    code, _, actions = bell_actions()
    protocol = build_adaptive_se_protocol(code, actions, first_action='Z', export=False)
    path = tmp_path / 'original' / 'steane_se_protocol.py'
    write_adaptive_source(protocol, path)
    source = path.read_text()
    assert 'spiderwarp' not in source
    assert generated_python(inline_adaptive_source(source, path))['POLICY_NODES'] == (
        generated_python(generate_adaptive_source(protocol))['POLICY_NODES'])
    # Import the saved module itself, so this exercises its standard-library
    # loader rather than just the repository's inlining/compilation helper.
    destination = tmp_path / 'relocated'
    shutil.copytree(path.parent, destination)
    copied = destination / path.name
    spec = importlib.util.spec_from_file_location('_portable_se_test', copied)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert Package.from_bytes(module.compile_se_cycle().to_bytes())
    finally:
        sys.modules.pop(spec.name, None)
    tables = destination / module.TABLES_FILE
    tables.write_bytes(tables.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        inline_adaptive_source(source, copied)


def test_default_decoder_and_patches_both_preserve_the_policy():
    from importlib import import_module
    from pprint import pformat
    from spiderwarp._se_adaptive_guppy import _generate_adaptive_source
    from spiderwarp.se_protocol_io import load_protocol
    from test_optimized_steane_se import generated_python
    path = Path(__file__).resolve().parents[1] / 'build/se_postselected_32_20_4'
    if not path.exists():
        pytest.skip('Full-size integration bundle not available')
    protocol = load_protocol(path)
    audit = import_module('analysis.32_20_4_bounded_comparison.probability_controls').audit_export
    for baseline in (False, True):
        source = _generate_adaptive_source(protocol, baseline=baseline, table_writer=lambda tables:
            '\n\n'.join(f'{name} = {pformat(value)}' for name, value in tables.items()))
        result = audit(protocol, generated_python(source))
        assert result['transitions_checked'] == 2546
        assert result['all_absent_outcomes_checked']

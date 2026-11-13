"""Regression for the out-of-scope 2+1 history and two-primary read count."""
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from spiderwarp._se_adaptive import CompatibleSet, ExtractionAction, FTContract, HistoryEngine
from spiderwarp.se_protocol_io import load_protocol
from test_adaptive_se_policy import real_actions


@pytest.fixture(scope='module')
def engine():
    _, space, gadgets = real_actions()
    engine = HistoryEngine(space, FTContract(1, 1, fault_order=2, accept_clean_through=1))
    return engine, {name: ExtractionAction(name, g) for name, g in gadgets.items()}


def test_total_order_initial_state_remembers_incoming_two_budget(engine):
    evidence, _ = engine
    initial = evidence.initial()
    assert Counter(r for r, _, _ in initial.logical) == {0: 1, 1: 1088, 2: 3007}
    assert initial.filters == (0, 0, 0)
    assert all(evidence.contract.remaining(h) == 2 - h[0] for h in initial.logical)
    assert evidence.contract.remaining((2, 0, 3)) == 0


def test_two_quiet_readings_stop_within_total_order_two(engine):
    evidence, actions = engine
    initial = evidence.initial()
    first = evidence.transition(initial, actions['Z4'], (0, 0))[(0, 0)]
    second = evidence.transition(first, actions['X4'], (0, 0))[(0, 0)]
    assert evidence.safe_frame(first) is None
    assert evidence.safe_frame(second) == 0
    assert all(r + s <= 2 for r, s, _ in second.logical)
    # Mixed input-one/fault-one cases must remain, rather than accidentally
    # disappearing with the old arbitrary-input optimization.
    assert sum((r, s) == (1, 1) for r, s, _ in second.logical) == 104
    assert all(evidence.decoder.weights[b][evidence.space.component_syndrome(error, b)] <= s
               for _, s, error in second.logical for b in ('X', 'Z'))


def test_legacy_arbitrary_contract_is_explicitly_distinct(engine):
    evidence, actions = engine
    legacy = HistoryEngine(evidence.space, replace(evidence.contract, input_scope='arbitrary'))
    first = legacy.transition(legacy.initial(), actions['Z4'], (0, 0))[(0, 0)]
    second = legacy.transition(first, actions['X4'], (0, 0))[(0, 0)]
    assert legacy.safe_frame(second) is None
    assert legacy.safe_frame(CompatibleSet(second.logical, (0, 0, 0), legacy.sectors)) == 0


@pytest.mark.parametrize('folder', ('se_postselected_32_20_4',))
def test_rebuilt_controllers_never_execute_three_d4s(folder):
    path = Path(__file__).resolve().parents[1] / 'build' / folder
    if not (path / 'protocol.json').exists():
        pytest.skip('Rebuilt integration bundle not yet available')
    protocol = load_protocol(path)
    assert protocol.policy.contract.input_scope == 'total-order'
    assert protocol.policy.contract.fault_order == 2
    nodes = protocol.policy.nodes

    @lru_cache(None)
    def count(index):
        node = nodes[index]
        if node.kind in {'accept', 'reject'}:
            return 0
        if node.kind == 'frame':
            return count(node.next)
        return int(protocol.gadgets[node.action].distance == 4) + max(
            (count(child) for child in node.children.values()), default=0)

    assert count(protocol.policy.root) == 2
    assert protocol.policy.contract.accepted_filter_order == 1
    assert protocol.verification.max_extractions == 3
    assert protocol.verification.worst_case_cnots == 433
    index, zero = protocol.policy.root, []
    while nodes[index].kind not in {'accept', 'reject'}:
        node = nodes[index]
        if node.kind == 'frame':
            index = node.next
        else:
            zero.append(node.action)
            index = node.children.get((0, 0), 0)
    assert zero == ['Z_d4', 'X_d4'] and nodes[index].kind == 'accept'

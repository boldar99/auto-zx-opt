import pytest
import stim

from spiderwarp.path_cover import CoveredZXGraph


def measured_graph():
    graph = CoveredZXGraph.from_stim(
        stim.Circuit('RX 0\nR 1\nCX 0 1\nM 1\nMX 0')
    )
    graph.basic_FE_rewrites()
    return graph


def test_boundary_bends_cannot_turn_measurements_into_internal_nodes():
    graph = measured_graph()
    # The two surviving spiders are connected, but merging their singleton
    # paths would lose one of the two original classical outputs.
    assert len(graph.paths) == 2
    assert not list(graph.all_causal_single_boundary_bends())
    optimised = graph.greedy_best_first_boundary_bends(
        lambda g, paths: float(len(paths)), max_evaluations=20
    )
    circuit, mapping = optimised.extract_circuit_with_measurement_map()
    assert circuit.num_measurements == 2
    assert set(mapping) == {0, 1}
    assert set(mapping.values()) == {0, 1}


def test_unlabelled_boundary_bends_are_still_available():
    graph = measured_graph()
    for vertex in graph.G:
        graph.set_measurement_id(vertex, None)
    assert len(list(graph.all_causal_single_boundary_bends())) == 2


def test_extraction_rejects_a_manually_invalid_measurement_cover():
    graph = measured_graph()
    first, second = (path[0] for path in graph.paths.values())
    lost_id = graph.measurement_id(first)
    graph.paths = {0: (first, second)}
    with pytest.raises(ValueError, match=rf'missing original measurement IDs \[{lost_id}\]'):
        graph.extract_circuit_with_measurement_map()


def test_extraction_rejects_duplicate_measurement_ids():
    graph = measured_graph()
    for vertex in graph.G:
        graph.set_measurement_id(vertex, 0)
    with pytest.raises(ValueError, match='measurement IDs must be unique'):
        graph.extract_circuit_with_measurement_map()

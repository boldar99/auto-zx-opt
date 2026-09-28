import networkx as nx
import pyzx as zx

from spiderwarp.path_cover import CoveredZXGraph
from spiderwarp.qubit_reuse import build_circuit_dag, dag_to_circuit


def test_build_circuit_dag_normalises_noncanonical_edges_on_a_copy() -> None:
    graph = nx.Graph()
    graph.add_node(
        0,
        type=zx.VertexType.Z,
        pos=(0, 0),
        qubit_index=0,
        measurement_id=0,
    )
    graph.add_node(
        1,
        type=zx.VertexType.Z,
        pos=(0, -1),
        qubit_index=1,
        measurement_id=1,
    )
    graph.add_edge(0, 1, type=zx.EdgeType.SIMPLE)
    covered = CoveredZXGraph(graph, {0: (0,), 1: (1,)})

    expected_circuit, expected_measurement_map = (
        covered.extract_circuit_with_measurement_map()
    )
    dag = build_circuit_dag(covered)
    actual_circuit, actual_measurement_map = dag_to_circuit(dag)

    assert actual_circuit == expected_circuit
    assert actual_measurement_map == expected_measurement_map
    assert set(covered.G.nodes()) == {0, 1}
    assert covered.edge_type(0, 1) == zx.EdgeType.SIMPLE

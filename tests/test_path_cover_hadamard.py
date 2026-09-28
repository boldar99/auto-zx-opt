import pyzx as zx
import stim
import networkx as nx

from spiderwarp.path_cover import CoveredZXGraph
from spiderwarp.stim_utils import steane_se_from_stim_state_prep


def _assert_same_unitary(actual: stim.Circuit, expected: stim.Circuit) -> None:
    assert actual.to_tableau() == expected.to_tableau()


def test_hadamard_edge_between_z_spiders_extracts_as_cz() -> None:
    circuit = zx.Circuit(2)
    circuit.add_gate("CZ", 0, 1)

    covered = CoveredZXGraph.from_zx_diagram(circuit.to_graph())

    assert sum(
        covered.edge_type(u, v) == zx.EdgeType.HADAMARD
        for u, v in covered.G.edges()
    ) == 1
    _assert_same_unitary(covered.extract_circuit(), stim.Circuit("CZ 0 1"))


def test_hadamard_edge_on_path_extracts_as_h_gate() -> None:
    circuit = zx.Circuit(1)
    circuit.add_gate("H", 0)

    covered = CoveredZXGraph.from_zx_diagram(circuit.to_graph())

    assert sum(
        covered.edge_type(u, v) == zx.EdgeType.HADAMARD
        for u, v in covered.G.edges()
    ) == 1
    _assert_same_unitary(covered.extract_circuit(), stim.Circuit("H 0"))


def test_degree_two_h_box_extracts_as_h_gate_without_mutating_input() -> None:
    diagram = zx.Graph()
    input_vertex = diagram.add_vertex(zx.VertexType.BOUNDARY, qubit=0, row=0)
    h_box = diagram.add_vertex(zx.VertexType.H_BOX, qubit=0, row=1)
    output_vertex = diagram.add_vertex(zx.VertexType.BOUNDARY, qubit=0, row=2)
    diagram.add_edge((input_vertex, h_box))
    diagram.add_edge((h_box, output_vertex))
    diagram.set_inputs((input_vertex,))
    diagram.set_outputs((output_vertex,))

    covered = CoveredZXGraph.from_zx_diagram(diagram)

    assert diagram.type(h_box) == zx.VertexType.H_BOX
    _assert_same_unitary(covered.extract_circuit(), stim.Circuit("H 0"))


def test_from_stim_imports_cz_and_h() -> None:
    original = stim.Circuit("CZ 0 1\nH 0")

    covered = CoveredZXGraph.from_stim(original)

    _assert_same_unitary(covered.extract_circuit(), original)


def test_from_stim_infers_data_prefix_with_unused_lower_qubit() -> None:
    original = stim.Circuit("H 1\nCZ 0 1")

    covered = CoveredZXGraph.from_stim(original)

    _assert_same_unitary(covered.extract_circuit(), original)


def test_from_stim_round_trips_xcx() -> None:
    original = stim.Circuit("XCX 0 1")

    covered = CoveredZXGraph.from_stim(original)

    _assert_same_unitary(covered.extract_circuit(), original)


def test_h_gate_between_reset_and_measurement_is_not_simplified_away() -> None:
    original = stim.Circuit("R 0\nH 0\nM 0")

    covered = CoveredZXGraph.from_stim(original)

    assert covered.extract_circuit() == original


def test_identity_removal_preserves_hadamard_parity() -> None:
    graph = nx.Graph()
    for node, row in enumerate((0, 1, 2)):
        graph.add_node(
            node,
            type=zx.VertexType.Z,
            pos=(row, 0),
            qubit_index=0,
            measurement_id=None,
        )
    graph.add_edge(0, 1, type=zx.EdgeType.SIMPLE)
    graph.add_edge(1, 2, type=zx.EdgeType.HADAMARD)
    covered = CoveredZXGraph(graph, {0: (0, 1, 2)})

    assert covered.remove_id(
        1,
        flow_preserving=False,
        parity_measurement_preserving=False,
    )
    assert covered.edge_type(0, 2) == zx.EdgeType.HADAMARD


def test_steane_syndrome_extraction_uses_the_requested_basis() -> None:
    z_extraction = steane_se_from_stim_state_prep(
        stim.Circuit("R 0"), se_basis="Z", n=1
    )
    x_extraction = steane_se_from_stim_state_prep(
        stim.Circuit("RX 0"), se_basis="X", n=1
    )

    assert z_extraction == stim.Circuit("R 1\nCX 0 1\nM 1")
    assert x_extraction == stim.Circuit("RX 1\nCX 1 0\nMX 1")

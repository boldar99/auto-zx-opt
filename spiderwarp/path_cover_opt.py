import copy
import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.utils import (
    _sorted_pair,
    load_state_prep_circuit,
    steane_se_from_stim_state_prep,
    stim_to_pyzx,
)
from spiderwarp.verify_fault_tolerance import (
    build_css_syndrome_table,
    compute_modified_lookup_table,
    list_to_str_stabs,
)


@dataclass(frozen=True)
class CircuitOperation:
    name: str
    targets: list[int]
    measurement_id: Optional[int] = None


@dataclass
class _MCTSNode:
    paths: dict[int, tuple[int, ...]]
    path_hash: int
    children: set[int]
    unexpanded_moves: Optional[list[dict[int, tuple[int, ...]]]]
    visits: int = 0
    total_reward: float = 0.0


class CoveredZXGraph:
    """
    A NetworkX-backed ZX graph together with a path cover.
    """

    TYPE_COLORS = {
        zx.VertexType.Z: "#66cc66",
        zx.VertexType.X: "#ff6666",
        zx.VertexType.BOUNDARY: "black",
    }

    MEASUREMENT_OPS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
    _STIM_ANNOTATION_OPS = {
        "TICK",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
    }

    def __init__(
        self,
        G: nx.Graph,
        paths: dict[int, tuple[int, ...]],
        num_data_qubits: Optional[int] = None,
    ) -> None:
        self.G = G
        self.paths = paths
        self._num_qubits = (
            num_data_qubits if num_data_qubits is not None else self._infer_num_data_qubits()
        )
        self._validate_node_attributes()

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_stim(cls, circuit: stim.Circuit) -> "CoveredZXGraph":
        """Build a CoveredZXGraph from Stim via PyZX.

        ``stim_to_pyzx`` needs the number of data qubits, which is inferred
        from the circuit by tracking which qubit wires are still live after the
        final measurement/reset/reuse on each wire. A qubit is live if its most
        recent relevant operation is not a destructive measurement. This makes
        the inference robust to qubit reuse: ``M 5; R 5; ...`` leaves qubit 5
        live again, whereas ``...; M 5`` leaves it non-data.

        ``stim_to_pyzx`` constructs a ``zx.Circuit`` by appending Stim
        operations in order. PyZX then turns the circuit into a graph by
        creating vertices in that same order. Therefore, for diagrams produced
        by this importer, terminal non-boundary vertices sorted by ZX vertex ID
        are in the original Stim measurement-sample order.
        """
        num_data_qubits = cls._infer_num_data_qubits_from_stim(circuit)
        diagram = stim_to_pyzx(circuit, num_data_qubits)
        return cls.from_zx_diagram(diagram, num_data_qubits=num_data_qubits)

    @classmethod
    def _infer_num_data_qubits_from_stim(cls, circuit: stim.Circuit) -> int:
        """Infer the number of unmeasured data qubits at the end of a Stim circuit.

        The result is the count of qubit indices whose last relevant operation
        leaves the qubit live. Destructive measurements make a qubit non-live;
        resets, reset-measurements, and later unitary/noisy operations make it
        live again. Annotation instructions such as ``TICK`` and ``DETECTOR``
        are ignored.
        """
        qubit_is_live: dict[int, bool] = {}
        cls._update_live_qubits_from_stim_block(circuit, qubit_is_live)
        return sum(qubit_is_live.values())

    @classmethod
    def _update_live_qubits_from_stim_block(
        cls,
        circuit: stim.Circuit,
        qubit_is_live: dict[int, bool],
    ) -> None:
        for instruction in circuit:
            if cls._is_stim_repeat_block(instruction):
                repeat_count = int(instruction.repeat_count)
                if repeat_count > 0:
                    cls._update_live_qubits_from_stim_block(
                        instruction.body_copy(),
                        qubit_is_live,
                    )
                continue

            name = instruction.name.upper()
            if name in cls._STIM_ANNOTATION_OPS:
                continue

            qubits = cls._stim_instruction_qubits(instruction)
            if name in cls.MEASUREMENT_OPS:
                for qubit in qubits:
                    qubit_is_live[qubit] = False
            else:
                for qubit in qubits:
                    qubit_is_live[qubit] = True

    @staticmethod
    def _is_stim_repeat_block(instruction: stim.CircuitInstruction) -> bool:
        return hasattr(instruction, "body_copy") and hasattr(instruction, "repeat_count")

    @staticmethod
    def _stim_instruction_qubits(instruction: stim.CircuitInstruction) -> list[int]:
        qubits: list[int] = []
        for target in instruction.targets_copy():
            is_qubit_target = getattr(target, "is_qubit_target", False)
            if callable(is_qubit_target):
                is_qubit_target = is_qubit_target()
            if is_qubit_target:
                qubits.append(int(target.value))
        return qubits

    @classmethod
    def from_zx_diagram(
        cls,
        diagram: zx.graph.graph.BaseGraph,
        num_data_qubits: Optional[int] = None,
    ) -> "CoveredZXGraph":
        """Build a CoveredZXGraph from a PyZX diagram.

        Measurement IDs are assigned to terminal non-boundary vertices by
        increasing PyZX vertex ID. For graphs produced by ``stim_to_pyzx``, this
        is exactly the original Stim measurement order because the PyZX circuit
        and graph are constructed operation-by-operation.
        """
        cls._normalise_terminal_hadamards(diagram)

        graph_dict = diagram.to_dict()
        G = nx.Graph()
        qubit_indices: dict[int, float] = {}

        for v_data in graph_dict["vertices"]:
            v_id = v_data["id"]
            row, qubit = v_data["pos"]
            qubit_indices[v_id] = qubit
            G.add_node(
                v_id,
                type=v_data["t"],
                pos=(row, -qubit),
                qubit_index=qubit,
                measurement_id=None,
            )

        for u, v, _ in graph_dict["edges"]:
            G.add_edge(u, v)

        paths = cls._initial_paths_by_qubit_track(G, qubit_indices)
        inferred_num_data_qubits = (
            num_data_qubits if num_data_qubits is not None else cls._infer_num_data_qubits_from_graph(G)
        )
        cls._attach_measurement_ids_by_terminal_vertex_order(G, paths)
        return cls(G, paths, num_data_qubits=inferred_num_data_qubits)

    @staticmethod
    def _normalise_terminal_hadamards(diagram: zx.graph.graph.BaseGraph) -> None:
        apply_h_at = []
        zx.simplify.id_simp(diagram)
        for v in diagram.vertices():
            if diagram.vertex_degree(v) != 1:
                continue
            [neighbor] = list(diagram.neighbors(v))
            edge = list(diagram.edges(v, neighbor))[0]
            if diagram.edge_type(edge) == zx.EdgeType.HADAMARD:
                apply_h_at.append(v)

        for v in apply_h_at:
            zx.simplify.color_change(diagram, v)

    @staticmethod
    def _flipped_xz_type(vertex_type: zx.VertexType) -> zx.VertexType:
        if vertex_type == zx.VertexType.X:
            return zx.VertexType.Z
        if vertex_type == zx.VertexType.Z:
            return zx.VertexType.X
        raise ValueError(f"Expected an X or Z spider, got {vertex_type!r}.")

    @staticmethod
    def _initial_paths_by_qubit_track(
        G: nx.Graph,
        qubit_indices: dict[int, float],
    ) -> dict[int, tuple[int, ...]]:
        nodes_by_qubit: dict[float, list[int]] = defaultdict(list)
        for v in G.nodes():
            nodes_by_qubit[qubit_indices[v]].append(v)

        paths: dict[int, tuple[int, ...]] = {}
        path_id = 0

        for q_index in sorted(nodes_by_qubit):
            nodes_on_track = nodes_by_qubit[q_index]
            nodes_on_track.sort(key=lambda v: G.nodes[v]["pos"][0])

            if not nodes_on_track:
                continue

            current_path: list[int] = [nodes_on_track[0]]
            for curr_node in nodes_on_track[1:]:
                prev_node = current_path[-1]
                if G.has_edge(prev_node, curr_node):
                    current_path.append(curr_node)
                else:
                    paths[path_id] = tuple(current_path)
                    path_id += 1
                    current_path = [curr_node]

            paths[path_id] = tuple(current_path)
            path_id += 1

        return paths

    @classmethod
    def _attach_measurement_ids_by_terminal_vertex_order(
        cls,
        G: nx.Graph,
        paths: dict[int, tuple[int, ...]],
    ) -> None:
        """Attach measurement IDs by increasing terminal ZX vertex ID.

        This is the provenance rule used by ``from_stim``. ``stim_to_pyzx`` appends
        operations to a PyZX circuit in Stim order, and ``Circuit.to_graph()``
        creates gate/measurement vertices in append order. Consequently, the
        terminal non-boundary vertices sorted by their PyZX vertex IDs are exactly
        the original Stim measurements sorted by sample index.
        """
        terminals: list[int] = []
        for path in paths.values():
            if not path:
                continue
            terminal = path[-1]
            if cls.node_type_from_graph(G, terminal) != zx.VertexType.BOUNDARY:
                terminals.append(terminal)

        for measurement_id, terminal in enumerate(sorted(terminals)):
            G.nodes[terminal]["measurement_id"] = measurement_id

    # ---------------------------------------------------------------------
    # Basic graph metadata helpers
    # ---------------------------------------------------------------------

    def _validate_node_attributes(self) -> None:
        required = {"type", "pos", "qubit_index", "measurement_id"}
        for v, data in self.G.nodes(data=True):
            missing = required.difference(data)
            if missing:
                raise ValueError(f"Node {v!r} is missing attributes: {sorted(missing)}")

    def _infer_num_data_qubits(self) -> int:
        return self._infer_num_data_qubits_from_graph(self.G)

    @staticmethod
    def _infer_num_data_qubits_from_graph(G: nx.Graph) -> int:
        return sum(
            data.get("type") == zx.VertexType.BOUNDARY
            for _, data in G.nodes(data=True)
        ) // 2

    @staticmethod
    def node_type_from_graph(G: nx.Graph, v: int) -> zx.VertexType:
        return G.nodes[v]["type"]

    def node_type(self, v: int) -> zx.VertexType:
        return self.G.nodes[v]["type"]

    def node_pos(self, v: int) -> tuple[float, float]:
        return self.G.nodes[v]["pos"]

    def set_node_pos(self, v: int, pos: tuple[float, float]) -> None:
        self.G.nodes[v]["pos"] = pos

    def measurement_id(self, v: int) -> Optional[int]:
        return self.G.nodes[v].get("measurement_id")

    def set_measurement_id(self, v: int, measurement_id: Optional[int]) -> None:
        self.G.nodes[v]["measurement_id"] = measurement_id

    def offset_measurement_ids_by(self, offset: int) -> None:
        for v in self.G.nodes():
            if self.G.nodes[v]["measurement_id"] is not None:
                self.set_measurement_id(v, self.G.nodes[v]["measurement_id"] + offset)

    # ---------------------------------------------------------------------
    # Copying and display
    # ---------------------------------------------------------------------

    def deepcopy(self) -> "CoveredZXGraph":
        return CoveredZXGraph(
            self.G.copy(),
            copy.deepcopy(self.paths),
            num_data_qubits=self._num_qubits,
        )

    def shallow_copy(self) -> "CoveredZXGraph":
        return CoveredZXGraph(
            self.G,
            self.paths,
            num_data_qubits=self._num_qubits,
        )

    def visualize(
        self,
        figsize: tuple[int, int] = (15, 12),
        show_node_ids: bool = True,
        show_measurement_ids: bool = True,
    ) -> None:
        world_line_edges: list[tuple[int, int]] = []
        for path in self.paths.values():
            world_line_edges.extend(zip(path, path[1:]))

        pos = nx.get_node_attributes(self.G, "pos")
        node_colors = [self.TYPE_COLORS[self.node_type(n)] for n in self.G.nodes()]
        labels = self._visualization_labels(show_node_ids, show_measurement_ids)

        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(
            self.G,
            pos,
            node_color=node_colors,
            node_size=250,
            edgecolors="black",
        )
        nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=self.G.edges(),
            edge_color="gray",
            alpha=0.8,
        )
        nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=world_line_edges,
            edge_color="#336699",
            width=2,
            arrows=True,
            arrowstyle="->",
        )
        nx.draw_networkx_labels(
            self.G,
            pos,
            labels=labels,
            font_color="gray",
            font_size=8,
        )
        plt.axis("off")
        plt.show()

    def _visualization_labels(
        self,
        show_node_ids: bool,
        show_measurement_ids: bool,
    ) -> dict[int, str]:
        labels = {}
        for v in self.G.nodes():
            parts = []
            if show_node_ids:
                parts.append(str(v))
            measurement_id = self.measurement_id(v)
            if show_measurement_ids and measurement_id is not None:
                parts.append(f"m{measurement_id}")
            labels[v] = "\n".join(parts) if parts else ""
        return labels

    # ---------------------------------------------------------------------
    # Path-cover utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _paths_hash(paths: dict[int, tuple[int, ...]]) -> int:
        return hash(tuple(sorted(tuple(path) for path in paths.values())))

    def path_hash(self) -> int:
        return self._paths_hash(self.paths)

    def total_hardware_qubits(self) -> int:
        return len(self.paths)

    def _get_path_to_qubit(self) -> dict[int, int]:
        path_ids = sorted(self.paths)
        return {path_id: qubit for qubit, path_id in enumerate(path_ids)}

    def _path_edges(self, paths: dict[int, tuple[int, ...]]) -> set[tuple[int, int]]:
        return {
            _sorted_pair(u, v)
            for path in paths.values()
            for u, v in zip(path, path[1:])
        }

    def _get_uncovered_edges(self, paths: dict[int, tuple[int, ...]]) -> set[tuple[int, int]]:
        all_edges = {_sorted_pair(u, v) for u, v in self.G.edges()}
        return all_edges.difference(self._path_edges(paths))

    def _num_parity_measurement(self, paths: dict[int, tuple[int, ...]]) -> int:
        count = 0
        for v, w in self._get_uncovered_edges(paths):
            if self.node_type(v) == self.node_type(w):
                count += 1
        return count

    def _construct_flow_graph(self, paths: dict[int, tuple[int, ...]]) -> nx.DiGraph:
        constraint_graph = nx.DiGraph()
        constraint_graph.add_nodes_from(self.G.nodes())

        for path_nodes in paths.values():
            for u, v in zip(path_nodes, path_nodes[1:]):
                constraint_graph.add_edge(u, v)
                for neighbor in self.G.neighbors(v):
                    if neighbor != u:
                        constraint_graph.add_edge(u, neighbor)

        return constraint_graph

    def check_causal_flow(self, paths: Optional[dict[int, tuple[int, ...]]] = None) -> bool:
        paths_to_check = self.paths if paths is None else paths
        constraint_graph = self._construct_flow_graph(paths_to_check)
        return nx.is_directed_acyclic_graph(constraint_graph)

    # ---------------------------------------------------------------------
    # Rewrites
    # ---------------------------------------------------------------------

    def _remove_vertex_from_paths(self, v: int) -> None:
        for path_id, path in list(self.paths.items()):
            if v not in path:
                continue

            replacement_path = tuple(node for node in path if node != v)

            if v == path[-1] and len(path) > 1:
                old_measurement_id = self.measurement_id(v)
                if old_measurement_id is not None:
                    self.set_measurement_id(path[-2], old_measurement_id)

            if replacement_path:
                self.paths[path_id] = replacement_path
            else:
                del self.paths[path_id]

    def _purge_vertex(self, v: int) -> None:
        self._remove_vertex_from_paths(v)
        if self.G.has_node(v):
            self.G.remove_node(v)

    def fuse(self, u: int, v: int) -> bool:
        """Fuse same-colour X/Z spiders connected by an edge.

        Vertex `u` is removed and absorbed into `v`.
        """
        if not (
            self.G.has_edge(u, v)
            and self.node_type(u) == self.node_type(v)
            and self.node_type(u) in (zx.VertexType.X, zx.VertexType.Z)
        ):
            return False

        self.G.remove_edge(u, v)
        u_neighbors = list(self.G.neighbors(u))
        for neighbor in u_neighbors:
            self.G.add_edge(neighbor, v)
        self._purge_vertex(u)
        return True

    def _remove_id_preserves_flow(self, v: int) -> bool:
        for path_id, path in list(self.paths.items()):
            if v not in path:
                continue
            new_path = tuple(node for node in path if node != v)
            candidate_paths = copy.copy(self.paths)
            if new_path:
                candidate_paths[path_id] = new_path
            else:
                del candidate_paths[path_id]
            return self.check_causal_flow(candidate_paths)
        return True

    def remove_id(
        self,
        v: int,
        flow_preserving: bool = True,
        parity_measurement_preserving: bool = True,
    ) -> bool:
        if self.G.degree(v) != 2:
            return False

        n1, n2 = list(self.G.neighbors(v))

        flow_check = flow_preserving and not self._remove_id_preserves_flow(v)
        parity_spider_check = (
            parity_measurement_preserving
            and self.node_type(n1) == self.node_type(n2) != self.node_type(v)
            and self.G.degree(n1) != 2
            and self.G.degree(n2) != 2
        )
        if flow_check or parity_spider_check:
            return False

        self.G.add_edge(n1, n2)
        self._purge_vertex(v)
        return True

    def basic_FE_rewrites(self) -> None:
        for v in list(self.G.nodes()):
            if self.G.degree(v) == 1 and self.node_type(v) != zx.VertexType.BOUNDARY:
                [neighbor] = list(self.G.neighbors(v))
                self.fuse(v, neighbor)

        for v in list(self.G.nodes()):
            if self.G.has_node(v):
                self.remove_id(v)

    # ---------------------------------------------------------------------
    # Boundary bends / path-cover optimisation
    # ---------------------------------------------------------------------

    def _vertex_to_path(self, paths: dict[int, tuple[int, ...]]) -> dict[int, int]:
        vertex_to_path = {}
        for path_id, path in paths.items():
            for v in path:
                vertex_to_path[v] = path_id
        return vertex_to_path

    def _boundary_bends(self, paths: dict[int, tuple[int, ...]]):
        vertex_path = self._vertex_to_path(paths)

        for v, w in self.G.edges():
            if v not in vertex_path or w not in vertex_path:
                continue
            v_path_id = vertex_path[v]
            w_path_id = vertex_path[w]
            if v_path_id == w_path_id:
                continue

            v_is_first = v == paths[v_path_id][0]
            v_is_last = v == paths[v_path_id][-1]
            w_is_first = w == paths[w_path_id][0]
            w_is_last = w == paths[w_path_id][-1]

            if (v_is_first or v_is_last) and (w_is_first or w_is_last):
                yield v_path_id, w_path_id, v_is_first, w_is_first

    def _causal_path_bends(
        self,
        paths: dict[int, tuple[int, ...]],
        i: int,
        j: int,
        i_first: bool,
        j_first: bool,
    ) -> Iterator[dict[int, tuple[int, ...]]]:
        match (i_first, j_first):
            case (True, True):
                merged_path_options = [
                    paths[i][::-1] + paths[j],
                    paths[j][::-1] + paths[i],
                ]
            case (True, False):
                merged_path_options = [paths[j] + paths[i]]
            case (False, True):
                merged_path_options = [paths[i] + paths[j]]
            case _:
                merged_path_options = []

        for merged_path in merged_path_options:
            new_paths = copy.copy(paths)
            del new_paths[i]
            new_paths[j] = merged_path
            if self.check_causal_flow(new_paths):
                yield new_paths

    def all_causal_single_boundary_bends(
        self,
        paths: Optional[dict[int, tuple[int, ...]]] = None,
    ) -> Iterator[dict[int, tuple[int, ...]]]:
        current_paths = self.paths if paths is None else paths
        for bend_data in self._boundary_bends(current_paths):
            yield from self._causal_path_bends(current_paths, *bend_data)

    def bfs_causal_boundary_bends(self) -> Iterator["CoveredZXGraph"]:
        yield self
        queue = [self]
        seen = {self.path_hash()}

        while queue:
            current_graph = queue.pop(0)
            for candidate_paths in current_graph.all_causal_single_boundary_bends():
                candidate_graph = current_graph.shallow_copy()
                candidate_graph.paths = candidate_paths
                candidate_hash = candidate_graph.path_hash()
                if candidate_hash not in seen:
                    queue.append(candidate_graph)
                    seen.add(candidate_hash)
                    yield candidate_graph

    def min_ancilla_boundary_bends(self) -> list["CoveredZXGraph"]:
        min_num_qubits = self.total_hardware_qubits()
        best_graphs: list[CoveredZXGraph] = []

        for current_graph in self.bfs_causal_boundary_bends():
            current_num_qubits = current_graph.total_hardware_qubits()
            if current_num_qubits < min_num_qubits:
                min_num_qubits = current_num_qubits
                best_graphs = [current_graph]
            elif current_num_qubits == min_num_qubits:
                best_graphs.append(current_graph)

        return best_graphs

    def best_first_boundary_bends(self, max_evaluations: int = 1000) -> "CoveredZXGraph":
        start_paths = len(self.paths)
        start_parity = self._num_parity_measurement(self.paths)
        pq = [(start_paths, start_parity, 0, self)]
        seen = {self.path_hash()}

        best_graph = self
        min_paths = start_paths
        eval_count = 0
        tie_breaker = 1

        while pq and eval_count < max_evaluations:
            current_path_count, _, _, current_graph = heapq.heappop(pq)
            eval_count += 1

            if current_path_count < min_paths:
                min_paths = current_path_count
                best_graph = current_graph

            for candidate_paths in current_graph.all_causal_single_boundary_bends():
                candidate_graph = current_graph.shallow_copy()
                candidate_graph.paths = candidate_paths
                candidate_hash = candidate_graph.path_hash()
                if candidate_hash in seen:
                    continue

                seen.add(candidate_hash)
                n_paths = len(candidate_graph.paths)
                n_parity = candidate_graph._num_parity_measurement(candidate_graph.paths)
                heapq.heappush(pq, (n_paths, n_parity, tie_breaker, candidate_graph))
                tie_breaker += 1

        return best_graph

    def mcts_boundary_bends(
        self,
        max_iterations: int = 1000,
        rollout_depth: int = 32,
        exploration_weight: float = 1.4,
        parity_weight: float = 1e-3,
        seed: Optional[int] = None,
    ) -> "CoveredZXGraph":
        """Optimise the path cover using Monte Carlo Tree Search.

        Each MCTS state is a valid causal path cover. Actions are exactly the
        causal single-boundary bends generated by
        :meth:`all_causal_single_boundary_bends`. States are deduplicated using
        :meth:`path_hash`, so different bend sequences reaching the same path
        cover share one MCTS node. The search objective is
        lexicographic in spirit: reduce the number of hardware qubits first,
        and use the number of same-colour uncovered edges, i.e. parity
        measurements, as a small tie-breaker.

        Args:
            max_iterations: Number of MCTS iterations to run.
            rollout_depth: Maximum number of random boundary bends per rollout.
            exploration_weight: UCT exploration constant. Larger values explore
                more; smaller values exploit current best branches more.
            parity_weight: Cost contribution of each parity measurement. Keep
                this below ``1`` if hardware-qubit count should dominate.
            seed: Optional seed for reproducible stochastic choices.

        Returns:
            A shallow copy of this graph whose ``paths`` are the best path cover
            found by the search. The original graph is not modified.
        """
        if max_iterations <= 0:
            result = self.shallow_copy()
            result.paths = self.paths
            return result
        if rollout_depth < 0:
            raise ValueError("rollout_depth must be non-negative.")
        if exploration_weight < 0:
            raise ValueError("exploration_weight must be non-negative.")
        if parity_weight < 0:
            raise ValueError("parity_weight must be non-negative.")

        rng = random.Random(seed)

        def cost(paths: dict[int, tuple[int, ...]]) -> float:
            return len(paths) + parity_weight * self._num_parity_measurement(paths)

        def reward(paths: dict[int, tuple[int, ...]]) -> float:
            return -cost(paths)

        def shuffled_moves(paths: dict[int, tuple[int, ...]]) -> list[dict[int, tuple[int, ...]]]:
            moves = list(self.all_causal_single_boundary_bends(paths))
            rng.shuffle(moves)
            return moves

        def rollout(paths: dict[int, tuple[int, ...]]) -> dict[int, tuple[int, ...]]:
            current_paths = paths
            best_rollout_paths = current_paths
            best_rollout_cost = cost(current_paths)

            for _ in range(rollout_depth):
                moves = shuffled_moves(current_paths)
                if not moves:
                    break

                current_paths = rng.choice(moves)
                current_cost = cost(current_paths)
                if current_cost < best_rollout_cost:
                    best_rollout_cost = current_cost
                    best_rollout_paths = current_paths

            return best_rollout_paths

        def child_score(parent_visits: int, child: _MCTSNode) -> float:
            if child.visits == 0:
                return math.inf
            exploitation = child.total_reward / child.visits
            exploration = exploration_weight * math.sqrt(math.log(parent_visits) / child.visits)
            return exploitation + exploration

        root_hash = self.path_hash()
        nodes: list[_MCTSNode] = [
            _MCTSNode(
                paths=self.paths,
                path_hash=root_hash,
                children=set(),
                unexpanded_moves=None,
            )
        ]
        transpositions: dict[int, int] = {root_hash: 0}

        best_paths = self.paths
        best_cost = cost(best_paths)

        for _ in range(max_iterations):
            node_index = 0
            search_path = [node_index]

            # Selection: descend through fully expanded nodes using UCT.
            while True:
                node = nodes[node_index]
                if node.unexpanded_moves is None:
                    # Deduplicate moves at this state by path hash.  Different
                    # boundary-bend descriptions can lead to the same path cover.
                    unique_moves: dict[int, dict[int, tuple[int, ...]]] = {}
                    for move in shuffled_moves(node.paths):
                        unique_moves.setdefault(self._paths_hash(move), move)
                    node.unexpanded_moves = list(unique_moves.values())

                if node.unexpanded_moves or not node.children:
                    break

                node_index = max(
                    node.children,
                    key=lambda child_index: child_score(max(1, node.visits), nodes[child_index]),
                )
                search_path.append(node_index)

            node = nodes[node_index]

            # Expansion: add one previously unexpanded causal bend.  A path cover
            # can be reached through multiple bend sequences, so use a
            # transposition table keyed by _paths_hash instead of creating a
            # duplicate MCTS node.
            while node.unexpanded_moves:
                child_paths = node.unexpanded_moves.pop()
                child_hash = self._paths_hash(child_paths)

                child_index = transpositions.get(child_hash)
                if child_index is None:
                    child_index = len(nodes)
                    transpositions[child_hash] = child_index
                    nodes.append(
                        _MCTSNode(
                            paths=child_paths,
                            path_hash=child_hash,
                            children=set(),
                            unexpanded_moves=None,
                        )
                    )

                if child_index != node_index:
                    node.children.add(child_index)
                    node_index = child_index
                    search_path.append(node_index)
                    node = nodes[node_index]
                    break

            # Simulation: randomly continue bending from the selected/expanded state.
            rollout_paths = rollout(node.paths)
            rollout_cost = cost(rollout_paths)
            rollout_reward = -rollout_cost

            if rollout_cost < best_cost:
                best_cost = rollout_cost
                best_paths = rollout_paths

            # Backpropagation over the actual tree path followed this iteration.
            # This is important because transposition nodes may have many parents.
            for visited_index in search_path:
                visited_node = nodes[visited_index]
                visited_node.visits += 1
                visited_node.total_reward += rollout_reward

        result = self.shallow_copy()
        result.paths = best_paths
        return result

    def mcts_path_opt(
        self,
        max_iterations: int = 1000,
        rollout_depth: int = 32,
        exploration_weight: float = 1.4,
        parity_weight: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        """Mutate ``self.paths`` using :meth:`mcts_boundary_bends`."""
        optimised = self.mcts_boundary_bends(
            max_iterations=max_iterations,
            rollout_depth=rollout_depth,
            exploration_weight=exploration_weight,
            parity_weight=parity_weight,
            seed=seed,
        )
        self.paths = optimised.paths

    def greedy_path_opt(self) -> None:
        current_paths = self.paths

        while True:
            min_pcheck = self._num_parity_measurement(current_paths)
            best_candidate = None

            for candidate_paths in self.all_causal_single_boundary_bends(current_paths):
                candidate_pcheck = self._num_parity_measurement(candidate_paths)
                if candidate_pcheck < min_pcheck:
                    min_pcheck = candidate_pcheck
                    best_candidate = candidate_paths
                    break

            if best_candidate is None:
                break

            current_paths = best_candidate

        self.paths = current_paths

    def _new_node_id(self) -> int:
        return max(self.G.nodes, default=-1) + 1

    @staticmethod
    def _opposite_spider_type(node_type: zx.VertexType) -> zx.VertexType:
        if node_type == zx.VertexType.Z:
            return zx.VertexType.X
        if node_type == zx.VertexType.X:
            return zx.VertexType.Z
        raise ValueError(f"Expected an X/Z spider, got {node_type!r}.")

    def _insert_identity_on_uncovered_edge(
        self,
        u: int,
        v: int,
        identity_type: zx.VertexType,
    ) -> int:
        """Insert an identity node (of the opposite type of u and v) between u and v, placing it on u's path."""
        new_node = self._new_node_id()

        u_pos = self.node_pos(u)
        v_pos = self.node_pos(v)
        new_pos = (
            (u_pos[0] + v_pos[0]) / 2,
            (u_pos[1] + v_pos[1]) / 2,
        )

        self.G.remove_edge(u, v)
        self.G.add_node(
            new_node,
            type=identity_type,
            pos=new_pos,
            measurement_id=None,
        )
        self.G.add_edge(u, new_node)
        self.G.add_edge(new_node, v)

        for path_id, path in self.paths.items():
            if u not in path:
                continue

            if path[0] == u:
                self.paths[path_id] = (new_node,) + path
            elif path[-1] == u:
                self.paths[path_id] = path + (new_node,)
                self.set_measurement_id(new_node, self.measurement_id(u))
                del self.G.nodes[u]["measurement_id"]
                raise ValueError("This implementation should be double checked")
            else:
                raise NotImplementedError("Insertion of new paths not implemented.")
            return new_node

        raise ValueError(f"Could not find path containing {u}.")

    def add_identities_for_same_type_uncovered_edges(self) -> None:
        """Insert identity spiders so extraction never sees same-type uncovered edges."""
        for u, v in list(self._get_uncovered_edges(self.paths)):
            u_type = self.node_type(u)
            v_type = self.node_type(v)

            if u_type != v_type or u_type not in (zx.VertexType.X, zx.VertexType.Z):
                continue

            identity_type = self._opposite_spider_type(u_type)

            self._insert_identity_on_uncovered_edge(u, v, identity_type)

    # ---------------------------------------------------------------------
    # Circuit extraction and measurement provenance
    # ---------------------------------------------------------------------

    def _node_to_qubit(self) -> dict[int, int]:
        path_to_qubit = self._get_path_to_qubit()
        node_to_qubit = {}
        for path_id, path in self.paths.items():
            for node in path:
                node_to_qubit[node] = path_to_qubit[path_id]
        return node_to_qubit

    def _find_total_ordering(self) -> list[CircuitOperation]:
        ordered_operations: list[CircuitOperation] = []
        path_to_qubit = self._get_path_to_qubit()
        node_to_qubit = self._node_to_qubit()
        terminal_nodes = {path[-1] for path in self.paths.values() if path}

        for path_id, path in self.paths.items():
            first_node_type = self.node_type(path[0])
            qubit = path_to_qubit[path_id]
            if first_node_type == zx.VertexType.Z:
                ordered_operations.append(CircuitOperation("RX", [qubit]))
            elif first_node_type == zx.VertexType.X:
                ordered_operations.append(CircuitOperation("R", [qubit]))

        path_edges = self._path_edges(self.paths)
        constraint_graph = self._construct_flow_graph(self.paths)

        for node in list(self.G.nodes()):
            if self.node_type(node) == zx.VertexType.BOUNDARY and constraint_graph.has_node(node):
                constraint_graph.remove_node(node)

        processed_edges: set[tuple[int, int]] = set()

        while constraint_graph.nodes:
            sources = [node for node, degree in constraint_graph.in_degree() if degree == 0]
            if not sources:
                raise ValueError("No solution found: cycle detected in causal-flow constraints.")

            for source in sources:
                constraint_graph.remove_node(source)
                source_qubit = node_to_qubit[source]
                source_type = self.node_type(source)

                neighbors = sorted(
                    self.G.neighbors(source),
                    key=lambda neighbor: int(self.node_type(neighbor) == source_type),
                )

                for neighbor in neighbors:
                    if self.node_type(neighbor) == zx.VertexType.BOUNDARY:
                        continue

                    edge = _sorted_pair(source, neighbor)
                    if edge in path_edges or edge in processed_edges:
                        continue

                    neighbor_qubit = node_to_qubit[neighbor]
                    neighbor_type = self.node_type(neighbor)

                    if source_type != neighbor_type:
                        if source_type == zx.VertexType.Z:
                            ordered_operations.append(
                                CircuitOperation("CNOT", [source_qubit, neighbor_qubit])
                            )
                        else:
                            ordered_operations.append(
                                CircuitOperation("CNOT", [neighbor_qubit, source_qubit])
                            )
                    else:
                        raise ValueError("Cannot extract same-type uncovered edge ...")

                    processed_edges.add(edge)

                if source in terminal_nodes:
                    measurement_id = self.measurement_id(source)
                    if source_type == zx.VertexType.Z:
                        ordered_operations.append(
                            CircuitOperation("MX", [source_qubit], measurement_id)
                        )
                    elif source_type == zx.VertexType.X:
                        ordered_operations.append(
                            CircuitOperation("M", [source_qubit], measurement_id)
                        )

        return ordered_operations

    def extract_circuit(self) -> stim.Circuit:
        circuit, _ = self.extract_circuit_with_measurement_map()
        return circuit

    def extract_circuit_with_measurement_map(self) -> tuple[stim.Circuit, dict[int, int]]:
        if not self.check_causal_flow():
            raise ValueError("Circuit must have causal flow.")

        extraction_graph = self.deepcopy()
        extraction_graph.add_identities_for_same_type_uncovered_edges()
        extraction_graph.visualize()

        circuit = stim.Circuit()
        measurement_map: dict[int, int] = {}
        next_measurement_index = 0

        for operation in extraction_graph._find_total_ordering():
            circuit.append(operation.name, operation.targets)
            if operation.name in self.MEASUREMENT_OPS:
                if operation.measurement_id is not None:
                    for offset in range(len(operation.targets)):
                        measurement_map[next_measurement_index + offset] = operation.measurement_id + offset
                next_measurement_index += len(operation.targets)

        return circuit, measurement_map

    # ---------------------------------------------------------------------
    # Existing analysis helpers
    # ---------------------------------------------------------------------

    def _terminal_measurement_paths(self) -> dict[int, int]:
        return {
            path_id: path[-1]
            for path_id, path in self.paths.items()
            if path and self.node_type(path[-1]) != zx.VertexType.BOUNDARY
        }

    def matrix_transformation_indices(self) -> list[int]:
        indices = []
        for terminal in self._terminal_measurement_paths().values():
            measurement_id = self.measurement_id(terminal)
            if measurement_id is not None and measurement_id < self._num_qubits:
                indices.append(measurement_id)
        return indices

    def measurement_qubit_indices(self) -> list[int]:
        path_to_qubit = self._get_path_to_qubit()
        indices = []
        for path_id, terminal in self._terminal_measurement_paths().items():
            measurement_id = self.measurement_id(terminal)
            if measurement_id is not None and measurement_id < self._num_qubits:
                indices.append(path_to_qubit[path_id] - self._num_qubits)
        return indices

    def flag_qubit_indices(self) -> list[int]:
        path_to_qubit = self._get_path_to_qubit()
        indices = []
        for path_id, terminal in self._terminal_measurement_paths().items():
            measurement_id = self.measurement_id(terminal)
            if measurement_id is not None and measurement_id >= self._num_qubits:
                indices.append(path_to_qubit[path_id] - self._num_qubits)
        return indices


def all_good_FT_opts(
    covered_zx_graph: CoveredZXGraph,
    H_matrix: np.ndarray,
    L_matrix: np.ndarray,
    basis: str,
    d: int,
) -> Iterator[CoveredZXGraph]:
    stabs = list_to_str_stabs(H_matrix)
    decoder_table = build_css_syndrome_table(stabs, d)

    yield covered_zx_graph
    covered_graphs = [covered_zx_graph]
    seen = {covered_zx_graph.path_hash()}

    while covered_graphs:
        current_graph = covered_graphs.pop(0)
        for candidate_paths in current_graph.all_causal_single_boundary_bends():
            candidate_graph = current_graph.shallow_copy()
            candidate_graph.paths = candidate_paths
            circuit = candidate_graph.extract_circuit()
            good = compute_modified_lookup_table(
                circuit,
                H_matrix,
                L_matrix,
                decoder_table,
                candidate_graph.flag_qubit_indices(),
                basis,
                d,
                verbose=True,
            )
            candidate_hash = candidate_graph.path_hash()
            if not good:
                print("BAD")
            if candidate_hash not in seen:
                covered_graphs.append(candidate_graph)
                seen.add(candidate_hash)
                candidate_graph.visualize()
                yield candidate_graph
            else:
                print("PRUNED")


if __name__ == "__main__":
    code_name, circuit_path = "17_1_5", "z"
    # code_name, circuit_path = "25_1_5", "rotated_surface_d5/zero_ft_heuristic_opt"
    # code_name, circuit_path = "15_7_3", "hamming/zero_ft_opt_opt"
    # code_name, circuit_path = "7_1_3", "steane/zero_ft_opt_opt"

    code = CSSCode.load_code("FAO", code_name)
    circuit = load_state_prep_circuit("SAT", circuit_path)
    se = steane_se_from_stim_state_prep(circuit, se_basis="Z", n=code.n)
    covered = CoveredZXGraph.from_stim(se)
    covered.visualize()
    covered.basic_FE_rewrites()
    covered.visualize()
    # optimised = covered.min_ancilla_boundary_bends()[0]
    optimised = covered.mcts_boundary_bends(
        max_iterations=500,
        rollout_depth=16,
        seed=0,
    )
    # optimised = covered.best_first_boundary_bends(max_evaluations=2_000)
    optimised.visualize()
    print("Number of ancilla qubits:", len(optimised.paths) - code.n)
    new_circuit, measurement_map = optimised.extract_circuit_with_measurement_map()
    print(new_circuit)
    print("Measurement map:", measurement_map)
    print()

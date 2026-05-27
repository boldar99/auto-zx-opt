from __future__ import annotations

import copy
import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Iterator, TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spiderwarp.csscode import CSSCode
from spiderwarp.utils import (
    _sorted_pair,
    load_state_prep_circuit,
)
from spiderwarp.stim_utils import steane_se_from_stim_state_prep, stim_to_pyzx, get_circuit_depth, get_num_measurements
from spiderwarp.verify_fault_tolerance import (
    build_css_syndrome_table,
    compute_modified_lookup_table,
    list_to_str_stabs,
)

if TYPE_CHECKING:
    from spiderwarp.path_cover_metrics import PathCostFunction


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


class _OrderingNode:
    def __init__(self, state_hash, remaining, depths, seq, parent=None):
        self.state_hash = state_hash
        self.remaining = remaining      # frozenset of nodes left
        self.depths = depths            # tuple of current qubit depths
        self.seq = seq                  # tuple of extraction order so far
        self.parent = parent
        self.children = []
        self.untried_moves = None       # Populated on expansion
        self.visits = 0
        self.total_reward = 0.0


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
    def from_stim(cls, circuit: stim.Circuit, num_data_qubits = None) -> "CoveredZXGraph":
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
        num_data_qubits = cls._infer_num_data_qubits_from_stim(circuit) if num_data_qubits is None else num_data_qubits
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
            and (self.G.degree(n1) != 2 and self.G.degree(n2) != 2)
        )
        to_boundary = parity_measurement_preserving and zx.VertexType.BOUNDARY in (self.node_type(n1), self.node_type(n2))
        if flow_check or parity_spider_check or to_boundary:
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

        self.add_identities_for_same_type_uncovered_edges()

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

    def exhaustive_optimal_boundary_bends(
        self,
        cost_func: "PathCostFunction"
    ) -> list["CoveredZXGraph"]:
        """
        Exhaustively searches the reachable boundary bend space to find the absolute minimum.
        WARNING: Scales extremely poorly. Do not use for large circuits.
        """
        min_cost = float('inf')
        best_graphs: list["CoveredZXGraph"] = []

        for current_graph in self.bfs_causal_boundary_bends():
            current_cost = cost_func(current_graph, current_graph.paths)

            if current_cost < min_cost:
                min_cost = current_cost
                best_graphs = [current_graph]
            elif current_cost == min_cost:
                best_graphs.append(current_graph)

        return best_graphs

    def greedy_best_first_boundary_bends(
        self,
        cost_func: "PathCostFunction",
        max_evaluations: int = 1000
    ) -> "CoveredZXGraph":
        start_paths = self.paths
        start_cost = cost_func(self, start_paths)

        # Priority queue stores: (cost, tie_breaker, graph)
        pq = [(start_cost, 0, self)]
        seen = {self.path_hash()}

        best_graph = self
        min_cost = start_cost
        eval_count = 0
        tie_breaker = 1

        while pq and eval_count < max_evaluations:
            current_cost, _, current_graph = heapq.heappop(pq)
            eval_count += 1

            if current_cost < min_cost:
                min_cost = current_cost
                best_graph = current_graph

            for candidate_paths in current_graph.all_causal_single_boundary_bends():
                candidate_graph = current_graph.shallow_copy()
                candidate_graph.paths = candidate_paths
                candidate_hash = candidate_graph.path_hash()

                if candidate_hash in seen:
                    continue

                seen.add(candidate_hash)
                c_cost = cost_func(candidate_graph, candidate_graph.paths)
                heapq.heappush(pq, (c_cost, tie_breaker, candidate_graph))
                tie_breaker += 1

        return best_graph

    def mcts_boundary_bends(
        self,
        cost_func: PathCostFunction,
        max_iterations: int = 1000,
        rollout_depth: int = 32,
        exploration_weight: float = 1.4,
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

        rng = random.Random(seed)

        def cost(paths: dict[int, tuple[int, ...]]) -> float:
            return cost_func(self, paths)

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

    def greedy_path_opt(self, cost_func: "PathCostFunction") -> None:
        """
        Steepest-ascent hill climbing. Evaluates all local single boundary bends
        and permanently commits the one that reduces the cost the most.
        Stops when no local bend provides a strict cost improvement.
        """
        current_paths = self.paths

        while True:
            min_cost = cost_func(self, current_paths)
            best_candidate = None

            for candidate_paths in self.all_causal_single_boundary_bends(current_paths):
                candidate_cost = cost_func(self, candidate_paths)

                if candidate_cost < min_cost:
                    min_cost = candidate_cost
                    best_candidate = candidate_paths

            # If no neighbor strictly improved the cost, we have hit a local minimum
            if best_candidate is None:
                break

            current_paths = best_candidate

        self.paths = current_paths

    def optimize_path_extremities(self, max_iterations: int = 100) -> None:
        """
        Greedily searches for valid path-extremity swaps to minimize depth.
        Dynamically inserts identity nodes to resolve same-color uncovered edges,
        routing them to preserve the measurement basis of the original paths.
        """
        try:
            best_depth = get_circuit_depth(self.extract_circuit())
        except Exception:
            best_depth = float('inf')

        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            path_ids = list(self.paths.keys())
            best_swap = None

            for i in range(len(path_ids)):
                for j in range(i + 1, len(path_ids)):
                    p1_id, p2_id = path_ids[i], path_ids[j]
                    p1, p2 = self.paths[p1_id], self.paths[p2_id]

                    if not p1 or not p2:
                        continue

                    # The 4 possible extremity pairs: (p1_is_front, p2_is_front, n1, n2)
                    extremity_pairs = [
                        (True, True, p1[0], p2[0]),
                        (True, False, p1[0], p2[-1]),
                        (False, True, p1[-1], p2[0]),
                        (False, False, p1[-1], p2[-1])
                    ]

                    for p1_is_front, p2_is_front, n1, n2 in extremity_pairs:
                        if not self.G.has_edge(n1, n2):
                            continue

                        # Evaluate both directions of the swap
                        for move_n2_to_p1 in [True, False]:
                            if move_n2_to_p1:
                                src_path, src_is_front = p2, p2_is_front
                                dst_path, dst_is_front = p1, p1_is_front
                                n_move, n_target = n2, n1
                                src_id, dst_id = p2_id, p1_id
                            else:
                                src_path, src_is_front = p1, p1_is_front
                                dst_path, dst_is_front = p2, p2_is_front
                                n_move, n_target = n1, n2
                                src_id, dst_id = p1_id, p2_id

                            if len(src_path) <= 1:
                                continue  # Cannot effectively chop a path of length 1

                            m_adj = src_path[1] if src_is_front else src_path[-2]

                            # If the exposed edge is the same color, we need a buffer identity
                            needs_id = (self.node_type(n_move) == self.node_type(m_adj))
                            new_src = src_path[1:] if src_is_front else src_path[:-1]

                            nodes_to_add = []
                            edges_to_remove = []
                            edges_to_add = []
                            meas_transfers = []  # Tracks (from_node, to_node, value) to preserve syndromes

                            # --- ESSENTIAL CHANGE 1: Preserve src_path's measurement ID ---
                            if not src_is_front:
                                val_src = self.G.nodes[n_move].get("measurement_id")
                                if val_src is not None:
                                    meas_transfers.append((n_move, m_adj, val_src))

                            if needs_id:
                                # Create the buffer identity
                                I_id = max(self.G.nodes()) + 1 if self.G.nodes() else 0
                                I_type = self._opposite_spider_type(self.node_type(n_move))

                                p_move = self.node_pos(n_move)
                                p_adj = self.node_pos(m_adj)
                                I_pos = ((p_move[0] + p_adj[0]) / 2, (p_move[1] + p_adj[1]) / 2)
                                q_idx = self.G.nodes[n_target].get("qubit_index", 0)

                                nodes_to_add.append({'id': I_id, 'type': I_type, 'pos': I_pos, 'qubit_index': q_idx})

                                # Splice it into the graph
                                if self.G.has_edge(n_move, m_adj):
                                    edges_to_remove.append((n_move, m_adj))
                                edges_to_add.extend([(n_move, I_id), (I_id, m_adj)])

                                # Route the identity to the *receiving* path to preserve the src_path color
                                if dst_is_front:
                                    new_dst = (I_id, n_move) + dst_path
                                else:
                                    new_dst = dst_path + (n_move, I_id)

                                    # If capping the back, transfer the measurement ID to the new terminal
                                    old_terminal = dst_path[-1]
                                    val = self.G.nodes[old_terminal].get("measurement_id")
                                    if val is not None:
                                        meas_transfers.append((old_terminal, I_id, val))

                            else:
                                if dst_is_front:
                                    new_dst = (n_move,) + dst_path
                                else:
                                    new_dst = dst_path + (n_move,)

                                    old_terminal = dst_path[-1]
                                    val = self.G.nodes[old_terminal].get("measurement_id")
                                    if val is not None:
                                        meas_transfers.append((old_terminal, n_move, val))

                            # --- 1. Apply Speculative Mutation ---
                            for nd in nodes_to_add:
                                self.G.add_node(nd['id'], type=nd['type'], pos=nd['pos'], qubit_index=nd['qubit_index'])
                            for u, v in edges_to_remove:
                                self.G.remove_edge(u, v)
                            for u, v in edges_to_add:
                                self.G.add_edge(u, v)

                            # --- ESSENTIAL CHANGE 2: Split Clear/Set loops to avoid overwriting ---
                            for frm, to, val in meas_transfers:
                                self.G.nodes[frm]["measurement_id"] = None
                            for frm, to, val in meas_transfers:
                                self.G.nodes[to]["measurement_id"] = val

                            old_src_path = self.paths[src_id]
                            old_dst_path = self.paths[dst_id]
                            self.paths[src_id] = new_src
                            self.paths[dst_id] = new_dst

                            # --- 2. Evaluate Integrity and Physical Depth ---
                            try:
                                flow_graph = self._construct_flow_graph(self.paths)
                                list(nx.topological_sort(flow_graph))
                                current_depth = get_circuit_depth(self.extract_circuit())

                                if current_depth < best_depth:
                                    best_depth = current_depth
                                    best_swap = {
                                        'src_id': src_id, 'dst_id': dst_id,
                                        'new_src': new_src, 'new_dst': new_dst,
                                        'nodes_to_add': nodes_to_add,
                                        'edges_to_remove': edges_to_remove,
                                        'edges_to_add': edges_to_add,
                                        'meas_transfers': meas_transfers
                                    }
                            except Exception:
                                pass

                            # --- 3. Revert Speculative Mutation ---
                            self.paths[src_id] = old_src_path
                            self.paths[dst_id] = old_dst_path

                            # Revert loops split
                            for frm, to, val in meas_transfers:
                                self.G.nodes[to]["measurement_id"] = None
                            for frm, to, val in meas_transfers:
                                self.G.nodes[frm]["measurement_id"] = val

                            for u, v in edges_to_add:
                                self.G.remove_edge(u, v)
                            for u, v in edges_to_remove:
                                self.G.add_edge(u, v)
                            for nd in nodes_to_add:
                                self.G.remove_node(nd['id'])

            # --- 4. Commit the best swap found in this sweep ---
            if best_swap:
                for nd in best_swap['nodes_to_add']:
                    self.G.add_node(nd['id'], type=nd['type'], pos=nd['pos'], qubit_index=nd['qubit_index'])
                for u, v in best_swap['edges_to_remove']:
                    self.G.remove_edge(u, v)
                for u, v in best_swap['edges_to_add']:
                    self.G.add_edge(u, v)

                # Commit loops split
                for frm, to, val in best_swap['meas_transfers']:
                    self.G.nodes[frm]["measurement_id"] = None
                for frm, to, val in best_swap['meas_transfers']:
                    self.G.nodes[to]["measurement_id"] = val

                self.paths[best_swap['src_id']] = best_swap['new_src']
                self.paths[best_swap['dst_id']] = best_swap['new_dst']
                improved = True

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
        """
        Insert an identity node (of the opposite type) between u and v.
        Intelligently places the new node on a path extremity if available,
        otherwise creates a new ancilla path.
        """
        new_node = self._new_node_id()
        u_pos = self.node_pos(u)
        v_pos = self.node_pos(v)

        # Place it visually exactly halfway between the nodes
        new_pos = ((u_pos[0] + v_pos[0]) / 2, (u_pos[1] + v_pos[1]) / 2)

        self.G.remove_edge(u, v)

        # 1. Dynamically discover which node (if any) sits at a path extremity
        adopting_path_id = None
        target_node = None
        is_front = False

        for pid, path in self.paths.items():
            if not path:
                continue

            # Check if u can adopt the node
            if path[0] == u:
                adopting_path_id, target_node, is_front = pid, u, True
                break
            if path[-1] == u:
                adopting_path_id, target_node, is_front = pid, u, False
                break

            # Check if v can adopt the node (Crucial for the Boundary bug)
            if path[0] == v:
                adopting_path_id, target_node, is_front = pid, v, True
                break
            if path[-1] == v:
                adopting_path_id, target_node, is_front = pid, v, False
                break

        # 2. Assign the hardware qubit index
        if adopting_path_id is not None:
            qubit_idx = self.G.nodes[target_node].get("qubit_index", 0)
        else:
            # If both are internal, allocate a new ancilla qubit
            qubit_idx = max((self.G.nodes[n].get("qubit_index", -1) for n in self.G.nodes()), default=-1) + 1

        # 3. Commit the new node and edges to the graph
        self.G.add_node(
            new_node,
            type=identity_type,
            pos=new_pos,
            measurement_id=None,
            qubit_index=qubit_idx,
        )
        self.G.add_edge(u, new_node)
        self.G.add_edge(new_node, v)

        # 4. Integrate into the Path Cover
        if adopting_path_id is not None:
            path = self.paths[adopting_path_id]
            if is_front:
                self.paths[adopting_path_id] = (new_node,) + path
            else:
                self.paths[adopting_path_id] = path + (new_node,)

                # Securely transfer measurement ID if capping the back of a path
                target_meas_id = self.G.nodes[target_node].get("measurement_id")
                if target_meas_id is not None:
                    self.set_measurement_id(new_node, target_meas_id)
                    self.G.nodes[target_node]["measurement_id"] = None
        else:
            # 5. The Ancilla Fix: Generate a new path for deeply internal uncovered edges
            new_path_id = max(self.paths.keys(), default=-1) + 1
            self.paths[new_path_id] = (new_node,)

        return new_node

    def add_identities_for_same_type_uncovered_edges(self) -> None:
        """Insert identity spiders so extraction never sees same-type uncovered edges."""
        for u, v in list(self._get_uncovered_edges(self.paths)):
            u_type = self.node_type(u)
            v_type = self.node_type(v)

            if u_type != v_type and zx.VertexType.BOUNDARY not in (u_type, v_type):
                continue

            if u_type == zx.VertexType.BOUNDARY:
                identity_type = self._opposite_spider_type(v_type)
            elif v_type == zx.VertexType.BOUNDARY:
                identity_type = self._opposite_spider_type(u_type)
            else:
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

        # Track the current depth of each hardware qubit wire
        qubit_depths = {qubit: 0 for qubit in set(node_to_qubit.values())}

        # Initial state preparation
        for path_id, path in self.paths.items():
            first_node_type = self.node_type(path[0])
            qubit = path_to_qubit[path_id]
            if first_node_type == zx.VertexType.Z:
                ordered_operations.append(CircuitOperation("RX", [qubit]))
                qubit_depths[qubit] += 1
            elif first_node_type == zx.VertexType.X:
                ordered_operations.append(CircuitOperation("R", [qubit]))
                qubit_depths[qubit] += 1

        path_edges = self._path_edges(self.paths)
        constraint_graph = self._construct_flow_graph(self.paths)

        for node in list(self.G.nodes()):
            if self.node_type(node) == zx.VertexType.BOUNDARY and constraint_graph.has_node(node):
                constraint_graph.remove_node(node)

        # Precompute Criticality for the ultimate tie-breaker
        successors = {n: list(constraint_graph.successors(n)) for n in constraint_graph.nodes()}
        criticality = {n: 0 for n in constraint_graph.nodes()}
        try:
            rev_topo = list(reversed(list(nx.topological_sort(constraint_graph))))
            for node in rev_topo:
                criticality[node] = max((criticality[child] for child in successors[node]), default=0) + 1
        except nx.NetworkXUnfeasible:
            raise ValueError("No solution found: cycle detected in causal-flow constraints.")

        processed_edges: set[tuple[int, int]] = set()

        # Optimized Depth-aware topological sort
        while constraint_graph.nodes:
            sources = [node for node, degree in constraint_graph.in_degree() if degree == 0]

            best_source = None
            # Score format: (Max Peak Depth, Total Depth Sum Increase, Negative Criticality)
            # We want to minimize all three. Negative criticality means higher criticality is better.
            best_score = (float('inf'), float('inf'), float('inf'))

            for candidate in sources:
                cand_qubit = node_to_qubit[candidate]
                cand_type = self.node_type(candidate)

                # 1. Simulate using the OPTIMAL execution order (Lowest depth targets first)
                simulated_neighbors = sorted(
                    self.G.neighbors(candidate),
                    key=lambda neighbor: (
                        int(self.node_type(neighbor) == cand_type),
                        qubit_depths[node_to_qubit[neighbor]]  # Magic depth-saver
                    )
                )

                sim_source_depth = qubit_depths[cand_qubit]
                total_depth_increase = 0

                for neighbor in simulated_neighbors:
                    if self.node_type(neighbor) == zx.VertexType.BOUNDARY:
                        continue
                    edge = _sorted_pair(candidate, neighbor)
                    if edge in path_edges or edge in processed_edges:
                        continue

                    target_qubit = node_to_qubit[neighbor]
                    target_depth = qubit_depths[target_qubit]

                    # Simulate the depth sync
                    new_depth = max(sim_source_depth, target_depth) + 1

                    # Calculate how much slack we just destroyed
                    total_depth_increase += (new_depth - sim_source_depth) + (new_depth - target_depth)
                    sim_source_depth = new_depth

                if candidate in terminal_nodes:
                    sim_source_depth += 1
                    total_depth_increase += 1

                score = (sim_source_depth, total_depth_increase, -criticality[candidate])

                if score < best_score:
                    best_score = score
                    best_source = candidate

            source = best_source
            constraint_graph.remove_node(source)
            source_qubit = node_to_qubit[source]
            source_type = self.node_type(source)

            # 2. Extract using the OPTIMAL execution order
            neighbors = sorted(
                self.G.neighbors(source),
                key=lambda neighbor: (
                    int(self.node_type(neighbor) == source_type),
                    qubit_depths[node_to_qubit[neighbor]]  # Dynamically routing to low-depth qubits first
                ),
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
                        ordered_operations.append(CircuitOperation("CNOT", [source_qubit, neighbor_qubit]))
                    else:
                        ordered_operations.append(CircuitOperation("CNOT", [neighbor_qubit, source_qubit]))

                    new_depth = max(qubit_depths[source_qubit], qubit_depths[neighbor_qubit]) + 1
                    qubit_depths[source_qubit] = new_depth
                    qubit_depths[neighbor_qubit] = new_depth
                else:
                    raise ValueError("Cannot extract same-type uncovered edge ...")

                processed_edges.add(edge)

            if source in terminal_nodes:
                measurement_id = self.measurement_id(source)
                if source_type == zx.VertexType.Z:
                    ordered_operations.append(CircuitOperation("MX", [source_qubit], measurement_id))
                elif source_type == zx.VertexType.X:
                    ordered_operations.append(CircuitOperation("M", [source_qubit], measurement_id))
                qubit_depths[source_qubit] += 1

        return ordered_operations

    def extract_circuit(self) -> stim.Circuit:
        circuit, _ = self.extract_circuit_with_measurement_map()
        return circuit

    def extract_circuit_with_measurement_map(self) -> tuple[stim.Circuit, dict[int, int]]:
        extraction_graph = self.deepcopy()
        extraction_graph.add_identities_for_same_type_uncovered_edges()

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
    from spiderwarp.path_cover_metrics import LexicographicCost, metric_depth, metric_num_paths, metric_hardware_qubits_exact, metric_spacetime_volume_exact
    from spiderwarp.qubit_reuse import NoReuseStrategy

    # code_dir, code_name, circ_dir, circ_path = "MQT", "17_1_5", "SAT", "cc_4_8_8_d5/zero_ft_heuristic_opt"
    # code_dir, code_name, circ_dir, circ_path = "MQT", "25_1_5", "SAT", "rotated_surface_d5/zero_ft_heuristic_opt"
    code_dir, code_name, circ_dir, circ_path = "MQT", "15_7_3", "SAT", "hamming/zero_ft_opt_opt"
    # code_dir, code_name, circ_dir, circ_path = "MQT", "12_2_4", "SAT", "carbon/zero_ft_opt_opt"
    # code_dir, code_name, circ_dir, circ_path = "MQT", "7_1_3", "SAT", "steane/zero_ft_opt_opt"
    # code_dir, code_name, circ_dir, circ_path = "misc", "32_20_4", "misc", "zero_32_20_4"

    code = CSSCode.load_code(code_dir, code_name)
    circuit = load_state_prep_circuit(circ_dir, circ_path)
    se = steane_se_from_stim_state_prep(circuit, se_basis="Z", n=code.n)
    covered = CoveredZXGraph.from_stim(se)
    covered.visualize()
    covered.basic_FE_rewrites()
    covered.visualize()
    optimised = covered.greedy_best_first_boundary_bends(
        cost_func=metric_hardware_qubits_exact(NoReuseStrategy),
        max_evaluations=10
    )
    optimised.visualize()
    new_circuit, measurement_map = optimised.extract_circuit_with_measurement_map()

    print("Sim Qubits:", metric_hardware_qubits_exact(optimised, optimised.paths))
    print("Circuit Depth:", get_circuit_depth(new_circuit))
    print("Circuit Volume:", metric_spacetime_volume_exact(optimised, optimised.paths))

    perfect_state = stim.Circuit()
    perfect_state.append("RX", range(len(code.H_x[0])))
    for row in code.H_z:  # Note: H_x, not H_z
        targets = []
        support = [i for i, r in enumerate(row) if r == 1]
        for q in support:
            targets.append(stim.target_z(q))
            targets.append(stim.target_combiner())
        if targets:
            targets.pop()
            perfect_state.append("MPP", targets)

    samples = (perfect_state + new_circuit).compile_sampler().sample(10)
    samples = samples[:, code.H_z.shape[0]:]

    num_measurements = get_num_measurements(se)
    flag_indices = [k for k, v in measurement_map.items() if v < num_measurements - code.n]
    flags = samples[:, flag_indices].astype(int)

    print(samples.shape, num_measurements - len(flag_indices))

    print("Flags:", flags, sep="\n", end="\n\n")

    print("measurement_map:", measurement_map)
    syndrome_indices = {k: v - (num_measurements - code.n) for k, v in measurement_map.items() if
                        v >= num_measurements - code.n}
    print("syndrome_indices:", syndrome_indices)
    effective_H = code.H_z[:, list(syndrome_indices.values())]
    raw_syndrome_measurements = samples[:, list(syndrome_indices.keys())]
    syndromes = raw_syndrome_measurements @ effective_H.T % 2

    print("Syndromes:", syndromes, sep="\n")

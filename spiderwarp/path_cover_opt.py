import copy
import heapq
import itertools
import math
import random
from collections import defaultdict
from typing import Iterator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spiderwarp.csscode import GadgetManager, SyndromeGadget
from spiderwarp.utils import _sorted_pair
from spiderwarp.verify_fault_tolerance import list_to_str_stabs, build_css_syndrome_table, compute_modified_lookup_table


class CoveredZXGraph:
    TYPE_COLORS = {
        zx.VertexType.Z: "#66cc66",
        zx.VertexType.X: "#ff6666",
        zx.VertexType.BOUNDARY: "black",
    }

    def __init__(self,
                 G: nx.Graph,
                 pos: dict[int, tuple[float, float]],
                 node_types: dict[int, zx.VertexType],
                 paths: dict[int, list[int]]) -> None:
        self.G = G
        self.pos = pos
        self.node_types = node_types
        self.paths = paths
        self._num_qubits = len([n for n in node_types.values() if n == zx.VertexType.BOUNDARY]) // 2
        self._measurements = {p[-1]: k - self._num_qubits for k, p in paths.items() if
                              node_types[p[-1]] != zx.VertexType.BOUNDARY}

    @classmethod
    def from_zx_diagram(cls, diagram: zx.graph.graph.BaseGraph):
        apply_h_at = []
        zx.simplify.id_simp(diagram)
        for v in diagram.vertices():
            if diagram.vertex_degree(v) == 1:
                e = list(diagram.edges(v, list(diagram.neighbors(v))[0]))[0]
                if diagram.edge_type(e) == zx.EdgeType.HADAMARD:
                    apply_h_at.append(v)
        for v in apply_h_at:
            zx.simplify.color_change(diagram, v)

        # Convert to NetworkX
        graph_dict = diagram.to_dict()
        G = nx.Graph()
        pos = {}
        node_types = {}
        qubit_indices = {}

        for v_data in graph_dict['vertices']:
            v_id = v_data['id']
            G.add_node(v_id)
            row, qubit = v_data['pos']
            pos[v_id] = (row, -qubit)
            node_types[v_id] = v_data['t']
            qubit_indices[v_id] = qubit

        for u, v, _ in graph_dict['edges']:
            G.add_edge(u, v)

        # 1. Group nodes purely by their spatial qubit track
        nodes_by_qubit = defaultdict(list)
        for v in G.nodes():
            nodes_by_qubit[qubit_indices[v]].append(v)

        paths = {}
        path_id = 0

        # 2. Sort tracks to ensure data qubits get the lowest path IDs
        for q_index in sorted(nodes_by_qubit.keys()):
            nodes_on_track = nodes_by_qubit[q_index]

            # Sort chronologically by row (x-coordinate)
            nodes_on_track.sort(key=lambda v: pos[v][0])

            current_path = [nodes_on_track[0]]

            # 3. Traverse track and split lifelines based on actual causal edges
            for i in range(1, len(nodes_on_track)):
                prev_node = nodes_on_track[i - 1]
                curr_node = nodes_on_track[i]

                if G.has_edge(prev_node, curr_node):
                    # Causal flow is unbroken; continue path
                    current_path.append(curr_node)
                else:
                    # Qubit reuse detected! (No edge = broken causal flow)
                    paths[path_id] = current_path
                    path_id += 1
                    current_path = [curr_node]

            # Add the final lifeline for this track
            paths[path_id] = current_path
            path_id += 1

        return cls(G, pos, node_types, paths)

    def copy(self):
        cg = CoveredZXGraph(
            self.G.copy(),
            self.pos.copy(),
            self.node_types,
            copy.deepcopy(self.paths),
        )
        cg._num_qubits = self._num_qubits
        cg._measurements = self._measurements.copy()
        return cg

    def visualize(self, figsize=(15, 12)):
        world_line_edges = []
        for p in self.paths.values():
            world_line_edges += list(zip(p, p[1:]))

        node_colors = [self.TYPE_COLORS[self.node_types[n]] for n in self.G.nodes()]

        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=150, edgecolors='black')
        nx.draw_networkx_edges(self.G, self.pos, edgelist=self.G.edges(),
                               edge_color='gray', alpha=0.8)
        nx.draw_networkx_edges(self.G, self.pos, edgelist=world_line_edges,
                               edge_color='#336699', width=2, arrows=True, arrowstyle='->')
        nx.draw_networkx_labels(self.G, self.pos, font_color='white', font_size=8)
        plt.show()

    def path_hash(self) -> int:
        return hash(
            tuple(sorted(tuple(v) for v in self.paths.values()))
        )

    def _purge_vertex_information(self, v):
        del self.pos[v]
        del self.node_types[v]
        for id, path in list(self.paths.items()):
            if v in path:
                measurement_key_change = (
                        v == path[-1] and len(path) > 1 and v in self._measurements
                )
                if measurement_key_change:
                    self._measurements[path[-2]] = self._measurements[v]
                    del self._measurements[v]
                path.remove(v)
            if len(path) == 0:
                del self.paths[id]

    def fuse(self, u, v):
        if not (self.G.has_edge(u, v)
                and self.node_types[u] == self.node_types[v]
                and self.node_types[u] in (zx.VertexType.X, zx.VertexType.Z)):
            return False

        self.G.remove_edge(u, v)
        u_neighbors = list(self.G.neighbors(u))
        self.G.remove_node(u)
        for n in u_neighbors:
            self.G.add_edge(n, v)

        self._purge_vertex_information(u)

        return True

    def _remove_id_check_flow(self, v):
        paths_copy = self.paths.copy()
        for id, path in list(self.paths.items()):
            if v in path:
                new_path = [p for p in path if p != v]
                paths_copy[id] = new_path
                return self.check_causal_flow(paths_copy)
        return True

    def remove_id(self, v, flow_preserving=True, parity_measurement_preserving=True):
        if self.G.degree(v) != 2:
            return False

        [n1, n2] = list(self.G.neighbors(v))
        self.G.remove_node(v)
        self.G.add_edge(n1, n2)

        flow_check = flow_preserving and not self._remove_id_check_flow(v)
        parity_spider_check = (parity_measurement_preserving and
                               (self.node_types[n1] == self.node_types[n2] != self.node_types[v]) and
                               (self.G.degree(n1) != 2 and self.G.degree(n2) != 2)
                               )

        if flow_check or parity_spider_check:
            self.G.remove_edge(n1, n2)
            self.G.add_node(v)
            self.G.add_edge(n1, v)
            self.G.add_edge(n2, v)
            return False

        self._purge_vertex_information(v)
        return True

    def basic_FE_rewrites(self):
        for v in list(self.G.nodes()):
            if self.G.degree(v) == 1 and self.node_types[v] != zx.VertexType.BOUNDARY:
                [n] = self.G.neighbors(v)
                self.fuse(v, n)

        for v in list(self.G.nodes()):
            self.remove_id(v)

    def _construct_flow_graph(self, paths_dict) -> nx.DiGraph:
        constraint_graph = nx.DiGraph()
        constraint_graph.add_nodes_from(self.G.nodes())

        for path_nodes in paths_dict.values():
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i + 1]
                constraint_graph.add_edge(u, v)

                for neighbor in self.G.neighbors(v):
                    if neighbor != u:
                        constraint_graph.add_edge(u, neighbor)
        return constraint_graph

    def total_hardware_qubits(self):
        return len(self.paths)

    def check_causal_flow(self, paths=None) -> bool:
        paths_to_check = paths if paths is not None else self.paths
        constraint_graph = self._construct_flow_graph(paths_to_check)
        return nx.is_directed_acyclic_graph(constraint_graph)

    def _boundary_bends(self, paths):
        vertex_qubit = {}
        for k, path in paths.items():
            for v in path:
                vertex_qubit[v] = k

        for v, w in self.G.edges():
            if vertex_qubit[v] == vertex_qubit[w]:
                continue

            # Check if start of one path connects to start of another
            v_is_first = v == paths[vertex_qubit[v]][0]
            v_is_last = v == paths[vertex_qubit[v]][-1]
            w_is_first = w == paths[vertex_qubit[w]][0]
            w_is_last = w == paths[vertex_qubit[w]][-1]

            if (v_is_first or v_is_last) and (w_is_first or w_is_last):
                yield vertex_qubit[v], vertex_qubit[w], v_is_first, w_is_first

    def all_causal_single_boundary_bends(self, paths=None) -> Iterator:
        """
        Returns all new paths candidates that are causal
        """
        current_paths = paths or self.paths

        for bend_data in self._boundary_bends(current_paths):
            for bend in self._causal_path_bends(current_paths, *bend_data):
                yield bend

    def bfs_causal_boundary_bends(self) -> Iterator["CoveredZXGraph"]:
        yield self
        covered_graphs = [self]
        seen = {self.path_hash()}

        while len(covered_graphs) > 0:
            current_graph = covered_graphs.pop(0)

            for path in current_graph.all_causal_single_boundary_bends():
                cov_graph = current_graph.copy()
                cov_graph.paths = copy.deepcopy(path)
                p_hash = cov_graph.path_hash()
                if p_hash not in seen:
                    covered_graphs.append(cov_graph)
                    seen.add(p_hash)
                    yield cov_graph

    def min_ancilla_boundary_bends(self) -> list["CoveredZXGraph"]:
        min_num_qubits = self.total_hardware_qubits()
        min_covered_graphs = []
        for current_graph in self.bfs_causal_boundary_bends():
            current_num_qubits = current_graph.total_hardware_qubits()
            if current_num_qubits < min_num_qubits:
                min_num_qubits = current_num_qubits
                min_covered_graphs = [current_graph]
            elif current_num_qubits == min_num_qubits:
                min_covered_graphs.append(current_graph)
        return min_covered_graphs

    def best_first_boundary_bends(self, max_evaluations=1000) -> "CoveredZXGraph":
        """
        Approximates min_ancilla_boundary_bends using a Greedy Best-First Search.
        Prioritizes states with the fewest paths, breaking ties with fewest parity measurements.
        """
        # Priority Queue tuple: (num_paths, num_parity_measurements, tie_breaker, CoveredZXGraph)
        # tie_breaker prevents heapq from trying to compare CoveredZXGraph objects

        start_paths = len(self.paths)
        start_parity = self._num_parity_measurement(self.paths)

        pq = [(start_paths, start_parity, 0, self)]
        seen = {self.path_hash()}

        best_graph = self
        min_paths = start_paths
        eval_count = 0
        tie_breaker = 1

        while pq and eval_count < max_evaluations:
            current_paths, _, _, current_graph = heapq.heappop(pq)
            eval_count += 1

            # Track the global best found so far
            if current_paths < min_paths:
                min_paths = current_paths
                best_graph = current_graph

            # Generate neighbors
            for path_candidate in current_graph.all_causal_single_boundary_bends():
                cov_graph = current_graph.copy()
                cov_graph.paths = copy.deepcopy(path_candidate)
                p_hash = cov_graph.path_hash()

                if p_hash not in seen:
                    seen.add(p_hash)
                    n_paths = len(cov_graph.paths)
                    n_parity = cov_graph._num_parity_measurement(cov_graph.paths)

                    heapq.heappush(pq, (n_paths, n_parity, tie_breaker, cov_graph))
                    tie_breaker += 1

        return best_graph

    def simulated_annealing_bends(self, initial_temp=10.0, cooling_rate=0.95,
                                  max_steps=100) -> "CoveredZXGraph":
        """
        Approximates min_ancilla_boundary_bends using Simulated Annealing.
        Note: Requires an 'un-bend' generator to fully utilize temperature jumps.
        """
        current_graph = self.copy()
        best_graph = current_graph

        current_cost = len(current_graph.paths)
        best_cost = current_cost

        temp = initial_temp

        for step in range(max_steps):
            # Gather all valid neighboring states
            neighbors = []
            for path_candidate in current_graph.all_causal_single_boundary_bends():
                cg = current_graph.copy()
                cg.paths = copy.deepcopy(path_candidate)
                neighbors.append(cg)

            if not neighbors:
                break  # Hit a local minimum and have no 'split' operators to escape

            # Pick a random valid neighbor
            candidate_graph = random.choice(neighbors)
            candidate_cost = len(candidate_graph.paths)

            # Calculate cost difference
            delta = candidate_cost - current_cost

            # If it's better (fewer paths), or we randomly accept a worse state based on temp
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_graph = candidate_graph
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_graph = current_graph

            # Cool down
            temp *= cooling_rate

        return best_graph

    def realign_pos(self) -> None:
        ys = [self.pos[path[0]][1] for path in self.paths.values()]
        ys.sort()

        for path in self.paths.values():
            xs = [self.pos[n][0] for n in path]
            xs.sort()
            # Spread out x coordinates if duplicates exist
            for i in range(len(xs) - 1):
                if xs[i] == xs[i + 1]:
                    xs[i + 1] += 1
            for i in range(len(xs) - 1):
                if xs[i] == xs[i + 1]:
                    xs[i + 1] += 1

            y = min([self.pos[n][1] for n in path])

            for p, x in zip(path, xs):
                self.pos[p] = (x, y)

    def _get_uncovered_edges(self, paths):
        all_edges = set(_sorted_pair(u, v) for u, v in self.G.edges())
        path_edges = []
        for path in paths.values():
            path_edges += list(zip(path[1:], path[:-1]))
        path_edges_set = set(_sorted_pair(u, v) for u, v in path_edges)
        return all_edges.difference(path_edges_set)

    def _num_parity_measurement(self, paths):
        count = 0
        uncovered = self._get_uncovered_edges(paths)
        for e in uncovered:
            v, w = _sorted_pair(*e)
            if self.node_types[v] == self.node_types[w]:
                count += 1
        return count

    def greedy_path_opt(self):
        """
        Iteratively improves the path cover by checking 'bell bends' (merging paths).
        """
        new_paths_candidate = {}
        current_paths = self.paths

        while new_paths_candidate is not None:
            new_paths_candidate = None
            min_pcheck = self._num_parity_measurement(current_paths)

            for new_paths_candidate in self.all_causal_single_boundary_bends(current_paths):
                num_pcheck = self._num_parity_measurement(new_paths_candidate)

                if num_pcheck < min_pcheck:
                    break

            if new_paths_candidate is not None:
                current_paths = new_paths_candidate

        self.paths = current_paths

    def _causal_path_bends(self, paths, i, j, i_first: bool, j_first: bool):
        match (i_first, j_first):
            case (True, True):
                merged_path_options = [paths[i][::-1] + paths[j], paths[j][::-1] + paths[i]]
            case (True, False):
                merged_path_options = [paths[j] + paths[i], ]
            case (False, True):
                merged_path_options = [paths[i] + paths[j], ]
            case _:
                merged_path_options = []

        for merged_path in merged_path_options:
            new_paths = copy.deepcopy(paths)
            del new_paths[i]
            new_paths[j] = merged_path

            if self.check_causal_flow(new_paths):
                yield new_paths

    def _get_path_to_qubit(self):
        qubits = list(self.paths.keys())
        qubits.sort()
        return {q: i for i, q in enumerate(qubits)}

    def _find_total_ordering(self):
        ordered_operations = []

        qubit_to_qubit = self._get_path_to_qubit()

        node_to_qubit = {}
        lasts = set()

        for q, p in self.paths.items():
            for u in p:
                node_to_qubit[u] = qubit_to_qubit[q]
            if self.node_types[p[0]] == zx.VertexType.Z:
                ordered_operations.append(("RX", [qubit_to_qubit[q]]))
            elif self.node_types[p[0]] == zx.VertexType.X:
                ordered_operations.append(("R", [qubit_to_qubit[q]]))
            lasts.add(p[-1])

        path_edges_list = [[_sorted_pair(v1, v2) for v1, v2 in zip(p, p[1:])]
                           for p in self.paths.values()]
        all_path_edges = list(itertools.chain(*path_edges_list))

        constraint_graph = self._construct_flow_graph(self.paths)

        # Remove boundary nodes from constraint calculation
        for n in list(self.G.nodes()):
            if self.node_types[n] == zx.VertexType.BOUNDARY:
                if constraint_graph.has_node(n):
                    constraint_graph.remove_node(n)

        processed_edges = set()

        # Topological sort processing
        while len(list(constraint_graph.nodes())) > 0:
            # Find nodes with in-degree 0
            sources = [n for n, d in constraint_graph.in_degree() if d == 0]

            if not sources:
                raise ValueError("No solution found (cycle detected in flow).")

            for s in sources:
                constraint_graph.remove_node(s)

                neighbors = sorted(self.G.neighbors(s),
                                   key=lambda n: int(self.node_types[n] == self.node_types[s]))

                qs = node_to_qubit[s]
                ts = self.node_types[s]

                for n in neighbors:
                    if self.node_types[n] == zx.VertexType.BOUNDARY:
                        continue

                    edge = _sorted_pair(s, n)
                    if edge in all_path_edges or edge in processed_edges:
                        continue

                    qn = node_to_qubit[n]
                    tn = self.node_types[n]

                    if ts != tn:
                        if ts == zx.VertexType.Z:
                            ordered_operations.append(("CNOT", (qs, qn)))
                        else:
                            ordered_operations.append(("CNOT", (qn, qs)))

                    processed_edges.add(edge)

                if s in lasts:
                    if ts == zx.VertexType.Z:
                        ordered_operations.append(("MX", qs))
                    elif ts == zx.VertexType.X:
                        ordered_operations.append(("M", qs))

        return ordered_operations

    def to_syndrome_measurement_circuit(self) -> SyndromeGadget:
        if not self.check_causal_flow():
            raise ValueError("Circuit must have causal flow.")

        ket_0, ket_plus = [], []
        cnots = []
        bra_0, bra_plus = [], []
        measurements = []

        ops = self._find_total_ordering()
        for op, qubits in ops:
            if op == "PrepZ":
                ket_0.append(qubits)
            elif op == "PrepX":
                ket_plus.append(qubits)
            elif op in ("CNOT", "CX"):
                cnots.append(qubits)
            elif op == "MeasZ":
                bra_0.append(qubits)
                measurements.append(qubits)
            elif op == "MeasX":
                bra_plus.append(qubits)
                measurements.append(qubits)

        measurements.sort()
        # flag_qubits = [measurements[i] for i in self.flag_qubit_indices()]
        flag_qubits = []

        return SyndromeGadget(
            ket_0, ket_plus, cnots, bra_0, bra_plus, flag_qubits
        )

    def extract_circuit(self) -> stim.Circuit:
        if not self.check_causal_flow():
            raise ValueError("Circuit must have causal flow.")
        circuit = stim.Circuit()

        ops = self._find_total_ordering()
        for op_name, qubits in ops:
            circuit.append(op_name, qubits)

        return circuit

    def matrix_transformation_indices(self) -> list:
        lasts = [p[-1] for p in self.paths.values() if self.node_types[p[-1]] != zx.VertexType.BOUNDARY]
        return [self._measurements[v] for v in lasts if self._measurements[v] < self._num_qubits]

    def measurement_qubit_indices(self) -> list:
        lasts = {i: p[-1] for i, p in self.paths.items() if self.node_types[p[-1]] != zx.VertexType.BOUNDARY}
        path_to_qubit = self._get_path_to_qubit()
        return [path_to_qubit[i] - self._num_qubits for i, v in lasts.items() if
                self._measurements[v] < self._num_qubits]

    def flag_qubit_indices(self) -> list:
        lasts = {i: p[-1] for i, p in self.paths.items() if self.node_types[p[-1]] != zx.VertexType.BOUNDARY}
        path_to_qubit = self._get_path_to_qubit()
        return [path_to_qubit[i] - self._num_qubits for i, v in lasts.items() if
                self._measurements[v] >= self._num_qubits]


def all_good_FT_opts(
        covered_zx_graph: CoveredZXGraph,
        H_matrix: np.ndarray,  # Binary Matrix (num_stabilizers x num_data_qubits)
        L_matrix: np.ndarray,  # Binary Matrix (num_logicals x num_data_qubits)
        basis: str,  # "Z" (Measuring Z-stabs) or "X" (Measuring X-stabs)
        d: int
) -> Iterator[CoveredZXGraph]:
    stabs = list_to_str_stabs(H_matrix)
    decoder_table = build_css_syndrome_table(stabs, d)

    yield covered_zx_graph
    covered_graphs = [covered_zx_graph]
    seen = {covered_zx_graph.path_hash()}
    while len(covered_graphs) > 0:
        current_graph = covered_graphs.pop(0)

        for path in current_graph.all_causal_single_boundary_bends():
            cov_graph = current_graph.copy()
            cov_graph.paths = copy.deepcopy(path)
            # cov_graph.insert_empty_spiders()
            circ = cov_graph.extract_circuit()
            good = compute_modified_lookup_table(
                circ,
                H_matrix,
                L_matrix,
                decoder_table,
                cov_graph.flag_qubit_indices(),
                basis,
                d,
                verbose=True,
            )
            p_hash = cov_graph.path_hash()
            if not good:
                print("BAD")
            if p_hash not in seen:
                covered_graphs.append(cov_graph)
                seen.add(p_hash)
                cov_graph.visualize()
                yield cov_graph
            else:
                print("PRUNED")


if __name__ == '__main__':
    # gadgets = QECCGadgets.from_json("circuits/15_7_3.json")

    gadgets = GadgetManager.load_circuit_data(32, 20, 4)
    print(gadgets.non_ft_steane_z_syndrome_extraction.to_stim(_layer_cnots=False))
    diagram = gadgets.non_ft_steane_z_syndrome_extraction.to_pyzx()
    cov_graph = CoveredZXGraph.from_zx_diagram(diagram)
    initial_size = len(cov_graph.paths)
    cov_graph.visualize()
    cov_graph.basic_FE_rewrites()
    cov_graph.visualize()

    for cv in cov_graph.min_ancilla_boundary_bends():
        cv.visualize()
        print("Number of ancilla qubits:", len(cv.paths) - gadgets.code.n)
        print(
            f"H indices: {cv.matrix_transformation_indices()}; "
            f"Measurement indices: {cv.measurement_qubit_indices()}; "
            f"Flag qubits: {cv.flag_qubit_indices()}"
        )
        print(cv.matrix_transformation_indices())
        print(cv.measurement_qubit_indices())
        print(cv.flag_qubit_indices())
        se_circ = cv.to_syndrome_measurement_circuit()
        print(se_circ)
        print(se_circ.to_stim())
        print()

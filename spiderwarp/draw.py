import networkx as nx
from matplotlib import pyplot as plt


def display_digraph(di_graph: nx.DiGraph, figsize=(10, 10)):
    """
    Displays the directed graph, distinguishing between tree edges and cycle closures.
    """

    if nx.is_directed_acyclic_graph(di_graph):
        plt.figure(figsize=figsize)
        # 1. Calculate the hierarchical layers mathematically
        for layer, nodes in enumerate(nx.topological_generations(di_graph)):
            for node in nodes:
                # 2. Explicitly assign the layer attribute to each node
                di_graph.nodes[node]["layer"] = layer

        # 3. Use the newly created "layer" attribute for the layout
        # align="horizontal" makes it a top-down tree.
        # (Remove align="horizontal" if you prefer left-to-right)
        pos = nx.multipartite_layout(di_graph, subset_key="layer", align="vertical")
    else:
        plt.figure(figsize=(10, 8))
        # Kamada-Kawai handles graphs with cycles by treating edges like springs
        pos = nx.kamada_kawai_layout(di_graph)

    cmap = plt.cm.tab10
    colors = cmap.colors
    node_colors = []
    for node, data in di_graph.nodes(data=True):
        if data.get("is_flag"):
            node_colors.append(colors[1])    # Implicit Flags (edge_diff)
        elif data.get("is_mark"):
            node_colors.append(colors[2])       # Explicit Marks
        else:
            node_colors.append(colors[0]) # Original Forest/Graph Nodes

    nx.draw_networkx_nodes(
        di_graph, pos,
        node_color=node_colors,
        node_size=400,
        edgecolors="black" # Gives nodes a clean border
    )
    nx.draw_networkx_labels(di_graph, pos, font_size=10, font_weight='bold')

    # Filter edges by type
    tree_edges = [(u, v) for u, v, d in di_graph.edges(data=True)]

    # Draw tree edges (Solid Black)
    nx.draw_networkx_edges(di_graph, pos, edgelist=tree_edges, edge_color='black', arrows=True, arrowsize=15)

    plt.title("Spanning Tree Traversal with Directed Cycle Closures")
    plt.axis('off')
    plt.show()


def visualize_circuit_dag(di_graph: nx.DiGraph, figsize=(12, 8)):
    """
    Visualizes the circuit DAG using a multipartite layout based on
    topological generations.
    """
    for layer, nodes in enumerate(nx.topological_generations(di_graph)):
        for node in nodes:
            di_graph.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(di_graph, subset_key="layer", align="vertical")

    labels = {}
    for node, data in di_graph.nodes(data=True):
        op = data.get("op_name")
        targs = "-".join([str(t) for t in data.get("targets", ())])
        labels[node] = f"{op}\n{targs}"

        meas_id = data.get("measurement_id")
        if meas_id is not None:
            labels[node] += f"\nm_{meas_id}"

    plt.figure(figsize=figsize)
    NODE_SIZE = 1500

    nx.draw_networkx_nodes(
        di_graph, pos,
        node_color="#ADD8E6",
        node_size=NODE_SIZE,
        edgecolors="black"
    )

    nx.draw_networkx_edges(
        di_graph, pos,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="gray",
        width=1.5,
        node_size=NODE_SIZE
    )

    nx.draw_networkx_labels(
        di_graph, pos,
        labels=labels,
        font_size=8,
        font_weight="bold"
    )

    plt.title("Circuit Dependency DAG (ASAP Generations)", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

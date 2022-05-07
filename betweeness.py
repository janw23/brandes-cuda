import sys
import networkx as nx

def _read_edges(stream):
    lines = stream.readlines()
    return [tuple(map(int, line.split())) for line in lines]

def _num_nodes(edges):
    n = 0
    for u, v in edges:
        n = max(n, u, v)
    return n

def _make_graph(edges):
    G = nx.Graph()
    num_nodes = _num_nodes(edges)
    G.add_nodes_from(list(range(num_nodes)))
    G.add_edges_from(edges)
    return G


if __name__ == '__main__':
    edges = _read_edges(sys.stdin)
    graph = _make_graph(edges)
    centrality = nx.betweenness_centrality(graph, normalized=False)
    for i in range(len(centrality)):
        print(centrality[i] * 2)
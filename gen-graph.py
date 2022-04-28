import random
import argparse

from cv2 import remap


def _num_all_edges(num_nodes):
    return (1 + num_nodes) * num_nodes // 2

def _index_to_edge(num_nodes, index):
    assert index < _num_all_edges(num_nodes)
    u = index // num_nodes
    v = index % num_nodes
    return (u, v)

def generate_graph(num_nodes, num_edges):
    assert num_edges <= _num_all_edges(num_edges)

    edges = map(lambda idx: _index_to_edge(num_nodes, idx), range(_num_all_edges(num_nodes)))
    edges = filter(lambda e: e[0] != e[1], edges)
    edges = list(edges)
    edges = random.sample(edges, num_edges)
    assert len(edges) == num_edges
    return edges


def morph_graph(edges):
    occuring_nodes = set()
    for u, v in edges:
        occuring_nodes.add(u)
        occuring_nodes.add(v)

    remapping = {k:v for v, k in enumerate(occuring_nodes)}
    return [(remapping[u], remapping[v]) for u, v in edges]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate undirected graph.')
    parser.add_argument('--edges', type=int, required=True,
                            help='the number of edges of the graph')
    parser.add_argument('--seed', type=int, required=False,
                            help='seed for the random number generator')

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        
    
    num_edges = args.edges

    edges = generate_graph(num_edges, num_edges)
    edges = morph_graph(edges)
    edges = sorted(edges, key=lambda p: p[0])

    for edge in edges:
        print(*edge)

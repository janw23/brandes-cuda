import random
import argparse

from cv2 import remap


def _num_all_edges(num_nodes):
    return (num_nodes - 1) * num_nodes // 2

def _index_to_edge(num_nodes, index):
    assert index < _num_all_edges(num_nodes)
    u = index // num_nodes
    v = index % num_nodes
    return (u, v)

def _edge_generator(num_nodes):
    for x in range(num_nodes - 1):
        for y in range(x + 1, num_nodes):
            yield (x, y)

def generate_graph(num_nodes, num_edges):
    assert num_edges <= _num_all_edges(num_nodes)

    egen = _edge_generator(num_nodes)
    edges = list(egen)
    edges = random.sample(edges, num_edges)
    assert len(edges) == num_edges
    return list(edges)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate undirected graph.')
    parser.add_argument('--nodes', type=int, required=True,
                            help='the number of nodes of the graph')
    parser.add_argument('--edges', type=int, required=True,
                            help='the number of edges of the graph')
    parser.add_argument('--seed', type=int, required=False,
                            help='seed for the random number generator')

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        
    
    num_nodes = args.nodes
    num_edges = args.edges

    edges = generate_graph(num_nodes, num_edges)
    edges = sorted(edges)

    for edge in edges:
        print(*edge)

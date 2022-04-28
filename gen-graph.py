import random
import argparse


def _num_all_edges(num_nodes):
    return (1 + num_nodes) * num_nodes // 2

def _index_to_edge(num_nodes, index):
    assert index < _num_all_edges(num_nodes)
    u = index // num_nodes
    v = index % num_nodes
    return (u, v)

def generate_graph(num_nodes, num_edges):
    assert num_edges <= _num_all_edges(num_edges)

    indices = random.sample(range(_num_all_edges(num_nodes)), num_edges)
    edges = map(lambda idx: _index_to_edge(num_nodes, idx), indices)
    return edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate undirected graph.')
    parser.add_argument('--nodes', type=int, required=True,
                            help='the number of nodes of the graph')
    parser.add_argument('--edges', type=int, required=False,
                            help='the number of edges of the graph, leave empty to choose randomly')
    parser.add_argument('--seed', type=int, required=False,
                            help='seed for the random number generator')

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        
    num_nodes = args.nodes
    num_edges = args.edges or random.randint(1, _num_all_edges(num_nodes))

    edges = generate_graph(num_nodes, num_edges)
    for edge in edges:
        print(*edge)

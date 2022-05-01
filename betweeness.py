import sys

import numpy as np
from pandas import Int32Dtype
from torch import int32

NO_EDGE = np.iinfo(np.uint).max

def _read_edges(stream):
    lines = stream.readlines()
    return [tuple(map(int, line.split())) for line in lines]

def make_graph(edges):
    m = max([v for e in edges for v in e]) + 1
    graph = np.ones((m, m), dtype=np.uint) * NO_EDGE
    for u, v in edges:
        graph[u, v] = graph[v, u] = 1

    np.fill_diagonal(graph, 0)
    return graph

def betweeness(graph):
    m = graph.shape[0] # num of nodes
    
    numPaths = np.copy(graph)
    numPaths[numPaths != 1] = 0
    np.fill_diagonal(numPaths, 1)

    for k in range(m):
        for i in range(m):
            for j in range(m):    
                pathSum = NO_EDGE if NO_EDGE in [graph[i,k], graph[k,j]] else graph[i, k] + graph[k, j]
                if pathSum < graph[i, j]:
                    graph[i, j] = pathSum
                    numPaths[i, j] = numPaths[i, k] * numPaths[k, j]

    ranks = [0] * m
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if i != k and k != j and graph[i,j] == graph[i,k] + graph[k,j] and graph[i,j] != NO_EDGE and graph[i,k] != NO_EDGE and graph[k,j] != NO_EDGE:
                    ranks[k] += int(numPaths[i, k] * numPaths[k, j])
    
    return ranks

if __name__ == '__main__':
    edges = _read_edges(sys.stdin)
    graph = make_graph(edges)
    centrality = betweeness(graph)
    for val in centrality:
        print(val)
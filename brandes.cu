#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <stack>
#include <cassert>

using namespace std;

// TODO correct int data types

static void print_usage() {
    cout << "Program usage: ./brandes input-file output-file" << endl;
}

static void check_args(int argc, char *argv[]) {
    if (argc != 3 || !strcmp(argv[1], "") || !strcmp(argv[2], "")) {
        print_usage();
        exit(1);
    }
}

static vector<pair<int, int>> load_edges(string path) {
    ifstream ifs;
    ifs.open(path);
    if(!ifs.is_open()) {
        cerr << "Cannot open file \"" << path << "\"." << endl;
        exit(1);
    }

    vector<pair<int, int>> edges;

    int u, v;
    while(ifs >> u >> v) edges.emplace_back(u, v);

    ifs.close();
    return edges;
} 

static int num_verts(const vector<pair<int, int>> &edges) {
    // Compute number of vertices based on maximum vertex label.
    int n = 0;
    for (const auto &edge : edges) n = max({n, edge.first, edge.second});
    n++;
    return n;
}

static vector<float> compute_betweeness(const vector<pair<int, int>> &edges) {
    int n = num_verts(edges);

    // Create graph as adjacency list, based on edges.
    vector<vector<int>> graph(n);
    for (const auto &edge : edges) {
        graph[edge.first].push_back(edge.second);
        graph[edge.second].push_back(edge.first);
    }

    // Based on https://kops.uni-konstanz.de/bitstream/handle/123456789/5739/algorithm.pdf
    // Compute centralities of each vertex.
    vector<float> centrality(n);
    for (int s = 0; s < n; s++) {
        queue<int> que; // BFS queue
        stack<int> stk; // verts ordered by distance from source
        vector<vector<int>> preds(n); // list of predecessors of each vertex
        vector<int> num_paths(n); // number of paths from source to each vertex
        vector<int> dist(n); // distance from source to each vertex
        num_paths[s] = 1; // there's 1 path to oneself
        fill(dist.begin(), dist.end(), -1); // allows dist to act as 'visited' marker
        dist[s] = 0; // dist to oneself is 0

        que.push(s);
        while(!que.empty()) {
            auto v = que.front();
            que.pop();
            stk.push(v);

            for (auto w : graph[v]) {
                if (dist[w] < 0) { // w visited for the first time
                    que.push(w);
                    dist[w] = dist[v] + 1;
                }
                if (dist[w] == dist[v] + 1) { // shortest path from source to w
                    num_paths[w] += num_paths[v];
                    preds[w].push_back(v);
                }
            }
        }

        vector<float> dependency(n);
        while (!stk.empty()) {
            auto w = stk.top();
            stk.pop();
            for (auto v : preds[w]) {
                dependency[v] += static_cast<float>(num_paths[v]) / num_paths[w] * (1.0f + dependency[w]);
            }
            if (w != s) {
                centrality[w] += dependency[w];
            }
        }
    }

    return centrality;
}

static void save_to_file(string path, const vector<float> &centrality) {
    ofstream ofs;
    ofs.open(path);

    if (!ofs.is_open()) {
        cerr << "Cannot open file \"" << path << "\"." << endl;
        exit(1);
    }

    for (auto val : centrality) {
        ofs << val << endl;
    }

    ofs.close();
}

// Helper function which converts graph into adjacenty lists representation.
static vector<vector<int>> adjacency_lists(const vector<pair<int, int>> &edges) {
    int n = num_verts(edges);
    vector<vector<int>> adjs(n);

    for (auto edge : edges) {
        adjs[edge.first].push_back(edge.second);
        adjs[edge.second].push_back(edge.first);
    }

    return adjs;
}

struct VirtualCSR {
    vector<int> vmap;
    vector<int> vptrs;
    vector<int> adjs;
    vector<int> ptrs; // TODO remove

    VirtualCSR(const vector<pair<int, int>> &edges, int mdeg) {
        auto graph = adjacency_lists(edges);
        
        for (int v = 0; v < graph.size(); v++) { // iterate over real verts
            int u = 0; // index of adjacent vert in v's adjacency list
            while (u < graph[v].size()) {
                vmap.push_back(v); // map new virtual vert to real vert v
                vptrs.push_back(adjs.size()); // mark the beginning of virtual vert's adjacency list
                for (int deg = 0; deg < mdeg && u < graph[v].size(); deg++, u++) {
                    adjs.push_back(graph[v][u]);
                }
            }
        }
    }
};

int main(int argc, char *argv[]) {
    check_args(argc, argv);
    auto edges = load_edges(argv[1]);
    VirtualCSR vcsr(edges, 4);

    auto betweeness = compute_betweeness(edges);
    save_to_file(argv[2], betweeness);

    return 0;
}
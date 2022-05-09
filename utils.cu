#include "utils.cuh"
#include "errors.h"
#include <algorithm>

using namespace std;

DeviceBool::DeviceBool(bool initial) {
    HANDLE_ERROR(cudaMalloc(&device_data, sizeof(bool)));
    set_value(initial); // default value
}

DeviceBool::DeviceBool() : DeviceBool(false) {}

DeviceBool::~DeviceBool() {
    HANDLE_ERROR(cudaFree(device_data));
    device_data = NULL;
}

void DeviceBool::set_value(bool val) {
    HANDLE_ERROR(cudaMemset(device_data, val, sizeof(bool)));
}

bool DeviceBool::get_value() {
    bool val;
    HANDLE_ERROR(cudaMemcpy(&val, device_data, sizeof(bool), cudaMemcpyDeviceToHost));
    return val;
}

int num_verts(const vector<pair<int, int>> &edges) {
    // Compute number of vertices based on maximum vertex label.
    int n = 0;
    for (const auto &edge : edges) n = max({n, edge.first, edge.second});
    n++;
    return n;
}

// Helper function which converts graph into adjacency lists representation.
static vector<vector<int>> adjacency_lists(const vector<pair<int, int>> &edges) {
    int n = num_verts(edges);
    vector<vector<int>> adjs(n);

    for (auto edge : edges) {
        adjs[edge.first].push_back(edge.second);
        adjs[edge.second].push_back(edge.first);
    }

    return adjs;
}

VirtualCSR::VirtualCSR(const vector<pair<int, int>> &edges, int mdeg) {
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

    vptrs.push_back(adjs.size()); // add guard at the end of vptrs
}

// Returns grid_size based on the overall required number of threads and block size.
int grid_size(int min_threads_count, int block_size) {
    return (min_threads_count + block_size - 1) / block_size;
}


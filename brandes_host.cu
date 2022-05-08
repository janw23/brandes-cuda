#include "brandes_host.cuh"

#include <algorithm>

using namespace std;

CUDATimer::Timer::Timer(cudaEvent_t start, cudaEvent_t stop) : stop{stop} {
    cudaEventRecord(start);
}

CUDATimer::Timer::~Timer() {
    cudaEventRecord(stop);
}

CUDATimer::Timer CUDATimer::kernel_timer() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel_evts.emplace_back(start, stop);
    return Timer(start, stop);
}


float CUDATimer::elapsed_time_kernels() {
    float time = 0;
    for (auto evt : kernel_evts) {
        cudaEventSynchronize(evt.second);
        float ms;
        cudaEventElapsedTime(&ms, evt.first, evt.second);
        time += ms;
    }
    return time;
}

float CUDATimer::elapsed_time_memcpy() {
    float time = 0;
    for (auto evt : memcpy_evts) {
        cudaEventSynchronize(evt.second);
        float ms;
        cudaEventElapsedTime(&ms, evt.first, evt.second);
        time += ms;
    }
    return time;
}

CUDATimer::~CUDATimer() {
    for (auto evt : kernel_evts) {
        cudaEventDestroy(evt.first);
        cudaEventDestroy(evt.second);
    }
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

host::VirtualCSR::VirtualCSR(const vector<pair<int, int>> &edges, int mdeg) {
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

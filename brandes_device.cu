#include "brandes_device.cuh"
#include "brandes_host.cuh"
#include "errors.h"

#include <iostream> // TODO remove
#include <cstdio> // TODO remove

using namespace std;

device::GraphDataVirtual::GraphDataVirtual(const vector<pair<int, int>> &edges, int mdeg) {
    num_real_verts = num_verts(edges);
    host::VirtualCSR vcsr(move(edges), mdeg);
    num_virtual_verts = vcsr.vmap.size();

    HANDLE_ERROR(cudaMalloc(&vmap, sizeof(*vmap) * vcsr.vmap.size()));
    HANDLE_ERROR(cudaMalloc(&vptrs, sizeof(*vptrs) * vcsr.vptrs.size()));
    HANDLE_ERROR(cudaMalloc(&adjs, sizeof(*adjs) * vcsr.adjs.size()));

    HANDLE_ERROR(cudaMemcpy(vmap, vcsr.vmap.data(), sizeof(*vmap) * vcsr.vmap.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(vptrs, vcsr.vptrs.data(), sizeof(*vptrs) * vcsr.vptrs.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(adjs, vcsr.adjs.data(), sizeof(*adjs) * vcsr.adjs.size(), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&dist, sizeof(*dist) * num_real_verts));
    HANDLE_ERROR(cudaMalloc(&num_paths, sizeof(*num_paths) * num_real_verts));
    HANDLE_ERROR(cudaMalloc(&delta, sizeof(*delta) * num_real_verts));
}

void device::GraphDataVirtual::free() {
    HANDLE_ERROR(cudaFree(vmap));
    vmap = NULL;
    HANDLE_ERROR(cudaFree(vptrs));
    vptrs = NULL;
    HANDLE_ERROR(cudaFree(adjs));
    adjs = NULL;
    HANDLE_ERROR(cudaFree(dist));
    dist = NULL;
    HANDLE_ERROR(cudaFree(num_paths));
    num_paths = NULL;
    HANDLE_ERROR(cudaFree(delta));
    delta = NULL;
}

__global__
void bc_virtual_initialize(device::GraphDataVirtual gdata, int source) {
    // There's one thread per real vertex;
    int real = blockIdx.x * blockDim.x + threadIdx.x;
    if (real == 0) printf("initialize(source=%d)\n", source); // TODO remove
    if (real >= gdata.num_real_verts) return;

    int is_source = real == source;
    gdata.num_paths[real] = is_source;
    gdata.dist[real] = is_source - 1;
    gdata.delta[real] = 0;
}

__global__
void bc_virtual_forward(device::GraphDataVirtual gdata, int layer, bool *cont) {
    // There's one thread per virtual vertex;
    int virt = blockIdx.x * blockDim.x + threadIdx.x;
    if (virt == 0) printf("forward(layer=%d)\n", layer); // TODO remove
    if (virt >= gdata.num_virtual_verts) return;

    int u = gdata.vmap[virt]; // u is the real vertex associated with the current virtual vertex
    if (gdata.dist[u] == layer) {
        for (int idx = gdata.vptrs[virt]; idx < gdata.vptrs[virt+1]; idx++) { // iterate over adjacent vertices
            int v = gdata.adjs[idx]; // v is the neighbour of current virtual vertex
            if (gdata.dist[v] == -1) {
                gdata.dist[v] = layer + 1;
                *cont = true;
            }
            if (gdata.dist[v] == layer + 1) {
                atomicAdd(&gdata.num_paths[v], gdata.num_paths[u]);
            }
        }
    }
}

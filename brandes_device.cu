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

    HANDLE_ERROR(cudaMalloc(&bc, sizeof(*bc) * num_real_verts));
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
    HANDLE_ERROR(cudaFree(bc));
    bc = NULL;
}

__global__
void fill(double *data, int size, double value) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < size; i += gridDim.x) {
        data[i] = value;
    }
}

__global__
void bc_virtual_prep_fwd(device::GraphDataVirtual gdata, int source) {
    // There's one thread per real vertex;
    int real = blockIdx.x * blockDim.x + threadIdx.x;
    // if (real == 0) printf("prep_forward(source=%d)\n", source); // TODO remove
    if (real >= gdata.num_real_verts) return;

    int is_source = real == source;
    gdata.num_paths[real] = is_source;
    gdata.dist[real] = is_source - 1;
    gdata.delta[real] = 0;
}

__global__
void bc_virtual_fwd(device::GraphDataVirtual gdata, int layer, bool *cont) {
    // There's one thread per virtual vertex;
    int virt = blockIdx.x * blockDim.x + threadIdx.x;
    // if (virt == 0) printf("forward(layer=%d)\n", layer); // TODO remove
    // TODO można trochę zoptymalizować gdata.vptrs wczytując zakres od pamięci dzielonej

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

__global__
void bc_virtual_prep_bwd(device::GraphDataVirtual gdata) {
    int real = blockIdx.x * blockDim.x + threadIdx.x;
    // if (real == 0) printf("prep_backward\n"); // TODO remove
    if (real >= gdata.num_real_verts) return;
    gdata.delta[real] = 1.0 / gdata.num_paths[real];
}

__global__
void bc_virtual_bwd(device::GraphDataVirtual gdata, int layer) {
    // There's one thread per virtual vertex;
    int virt = blockIdx.x * blockDim.x + threadIdx.x;
    // if (virt == 0) printf("backward(layer=%d)\n", layer); // TODO remove
    if (virt >= gdata.num_virtual_verts) return;

    int u = gdata.vmap[virt];
    if (gdata.dist[u] == layer) {
        double sum = 0;
        for (int idx = gdata.vptrs[virt]; idx < gdata.vptrs[virt+1]; idx++) { // iterate over adjacent vertices
            int v = gdata.adjs[idx]; // v is the neighbour of current virtual vertex
            if (gdata.dist[v] == layer + 1) {
                sum += gdata.delta[v];
            }
        }
        atomicAdd(&gdata.delta[u], sum);
    }
}

__global__
void bc_virtual_update(device::GraphDataVirtual gdata, int source) {
    // There's one thread per real vertex;
    int real = blockIdx.x * blockDim.x + threadIdx.x;
    // if (real == 0) printf("update()\n"); // TODO remove
    if (real >= gdata.num_real_verts) return;

    // TODO gdata.dist[real] != -1 does not update verts
    // TODO which were not reachable from the current source
    // TODO but maybe this should be done differently?

    if (real != source && gdata.dist[real] != -1)
        gdata.bc[real] += gdata.num_paths[real] * gdata.delta[real] - 1.0;
}
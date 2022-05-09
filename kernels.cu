#include "kernels.cuh"
#include "utils.cuh"
#include "errors.h"

using namespace std;

GraphDataVirtual::GraphDataVirtual(const vector<pair<uint32_t, uint32_t>> &edges, uint32_t mdeg) {
    num_real_verts = num_verts(edges);
    VirtualCSR vcsr(move(edges), mdeg);
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

void GraphDataVirtual::free() {
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
void fill(double *data, uint32_t size, double value) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (uint32_t i = tid; i < size; i += gridDim.x) {
        data[i] = value;
    }
}

__global__
void bc_virtual_prep_fwd(GraphDataVirtual gdata, uint32_t source) {
    // There's one thread per real vertex;
    uint32_t real = blockIdx.x * blockDim.x + threadIdx.x;
    if (real >= gdata.num_real_verts) return;

    bool is_source = real == source;
    gdata.num_paths[real] = is_source;
    gdata.dist[real] = is_source - 1;
    gdata.delta[real] = 0;
}

__global__
void bc_virtual_fwd(GraphDataVirtual gdata, int32_t layer, bool *cont) {
    // There's one thread per virtual vertex;
    uint32_t virt = blockIdx.x * blockDim.x + threadIdx.x;
    if (virt >= gdata.num_virtual_verts) return;

    uint32_t u = gdata.vmap[virt]; // u is the real vertex associated with the current virtual vertex
    if (gdata.dist[u] == layer) {
        for (uint32_t idx = gdata.vptrs[virt]; idx < gdata.vptrs[virt+1]; idx++) { // iterate over adjacent vertices
            uint32_t v = gdata.adjs[idx]; // v is the neighbour of current virtual vertex
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
void bc_virtual_prep_bwd(GraphDataVirtual gdata) {
    uint32_t real = blockIdx.x * blockDim.x + threadIdx.x;
    if (real >= gdata.num_real_verts) return;
    gdata.delta[real] = 1.0 / gdata.num_paths[real];
}

__global__
void bc_virtual_bwd(GraphDataVirtual gdata, int32_t layer) {
    // There's one thread per virtual vertex;
    uint32_t virt = blockIdx.x * blockDim.x + threadIdx.x;
    if (virt >= gdata.num_virtual_verts) return;

    uint32_t u = gdata.vmap[virt];
    if (gdata.dist[u] == layer) {
        double sum = 0;
        for (uint32_t idx = gdata.vptrs[virt]; idx < gdata.vptrs[virt+1]; idx++) { // iterate over adjacent vertices
            uint32_t v = gdata.adjs[idx]; // v is the neighbour of current virtual vertex
            if (gdata.dist[v] == layer + 1) {
                sum += gdata.delta[v];
            }
        }
        atomicAdd(&gdata.delta[u], sum);
    }
}

__global__
void bc_virtual_update(GraphDataVirtual gdata, uint32_t source) {
    // There's one thread per real vertex;
    uint32_t real = blockIdx.x * blockDim.x + threadIdx.x;
    if (real >= gdata.num_real_verts) return;

    if (real != source && gdata.dist[real] != -1)
        gdata.bc[real] += gdata.num_paths[real] * gdata.delta[real] - 1.0;
}
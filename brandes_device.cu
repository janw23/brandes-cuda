#include "brandes_device.cuh"
#include "errors.h"

#include <iostream> // TODO remove
#include <cstdio> // TODO remove

device::VirtualCSR::VirtualCSR(const host::VirtualCSR &vcsr) {
    HANDLE_ERROR(cudaMalloc(&vmap, sizeof(*vmap) * vcsr.vmap.size()));
    HANDLE_ERROR(cudaMalloc(&vptrs, sizeof(*vptrs) * vcsr.vptrs.size()));
    HANDLE_ERROR(cudaMalloc(&adjs, sizeof(*adjs) * vcsr.adjs.size()));

    HANDLE_ERROR(cudaMemcpy(vmap, vcsr.vmap.data(), sizeof(*vmap) * vcsr.vmap.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(vptrs, vcsr.vptrs.data(), sizeof(*vptrs) * vcsr.vptrs.size(), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(adjs, vcsr.adjs.data(), sizeof(*adjs) * vcsr.adjs.size(), cudaMemcpyHostToDevice));
}

void device::VirtualCSR::free() {
    HANDLE_ERROR(cudaFree(vmap));
    vmap = NULL;
    HANDLE_ERROR(cudaFree(vptrs));
    vptrs = NULL;
    HANDLE_ERROR(cudaFree(adjs));
    adjs = NULL;
}

__global__
void bc_virtual_forward(device::VirtualCSR vcsr, int layer, bool *cont) {
    printf("Hai CUDA\n");
    for (int i = 0; i < 3; i++) {
        printf("%d\n", vcsr.vmap[i]);
    }
}
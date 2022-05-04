#ifndef __BRANDES_DEVICE_CUH__
#define __BRANDES_DEVICE_CUH__

#include "brandes_host.cuh"

namespace device {

    struct GraphDataVirtual {
        // Virtual-CSR representation data.
        int *vmap;
        int *vptrs;
        int *adjs;

        int *dist;
        int *num_paths;
        float *delta;

        int num_virtual_verts;
        int num_real_verts;

        GraphDataVirtual() = delete;
        GraphDataVirtual(const std::vector<std::pair<int, int>> &edges, int mdeg);

        void free();
    };
}

__global__
void bc_virtual_initialize(device::GraphDataVirtual gdata, int source);

__global__
void bc_virtual_forward(device::GraphDataVirtual gdata, int layer, bool *cont);

__global__
void bc_virtual_backward(device::GraphDataVirtual gdata, int layer);

#endif // __BRANDES_DEVICE_CUH__
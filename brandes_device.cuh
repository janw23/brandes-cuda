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
        double *delta;

        double *bc;

        int num_virtual_verts;
        int num_real_verts;

        GraphDataVirtual() = delete;
        GraphDataVirtual(const std::vector<std::pair<int, int>> &edges, int mdeg);

        void free();
    };
}

__global__
void fill(double *data, int size, double value);

__global__
void bc_virtual_prep_fwd(device::GraphDataVirtual gdata, int source);

__global__
void bc_virtual_fwd(device::GraphDataVirtual gdata, int layer, bool *cont);

__global__
void bc_virtual_prep_bwd(device::GraphDataVirtual gdata);

__global__
void bc_virtual_bwd(device::GraphDataVirtual gdata, int layer);

__global__
void bc_virtual_update(device::GraphDataVirtual gdata, int source);

#endif // __BRANDES_DEVICE_CUH__
#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <vector>
#include <cstdint>

struct GraphDataVirtual {
    // Virtual-CSR representation data.
    uint32_t *vmap;
    uint32_t *vptrs;
    uint32_t *adjs;

    int32_t *dist;
    uint32_t *num_paths;
    double *delta;

    double *bc;

    uint32_t num_virtual_verts;
    uint32_t num_real_verts;

    GraphDataVirtual() = delete;
    GraphDataVirtual(const std::vector<std::pair<uint32_t, uint32_t>> &edges, uint32_t mdeg);

    void free();
};  

__global__
void fill(double *data, uint32_t size, double value);

__global__
void bc_virtual_prep_fwd(GraphDataVirtual gdata, uint32_t source);

__global__
void bc_virtual_fwd(GraphDataVirtual gdata, int32_t layer, bool *cont);

__global__
void bc_virtual_prep_bwd(GraphDataVirtual gdata);

__global__
void bc_virtual_bwd(GraphDataVirtual gdata, int32_t layer);

__global__
void bc_virtual_update(GraphDataVirtual gdata, uint32_t source);

#endif
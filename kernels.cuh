#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <vector>

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


__global__
void fill(double *data, int size, double value);

__global__
void bc_virtual_prep_fwd(GraphDataVirtual gdata, int source);

__global__
void bc_virtual_fwd(GraphDataVirtual gdata, int layer, bool *cont);

__global__
void bc_virtual_prep_bwd(GraphDataVirtual gdata);

__global__
void bc_virtual_bwd(GraphDataVirtual gdata, int layer);

__global__
void bc_virtual_update(GraphDataVirtual gdata, int source);

#endif
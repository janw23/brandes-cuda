#ifndef __BRANDES_DEVICE_CUH__
#define __BRANDES_DEVICE_CUH__

#include "brandes_host.cuh"

namespace device {
    struct VirtualCSR {
        int *vmap;
        int *vptrs;
        int *adjs;

        VirtualCSR() = delete;
        VirtualCSR(const host::VirtualCSR &vcsr);
        
        void free();
    };
}

__global__
void bc_virtual_forward(device::VirtualCSR vcsr, int layer, bool *cont);

#endif // __BRANDES_DEVICE_CUH__
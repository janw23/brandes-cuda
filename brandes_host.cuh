#ifndef __BRANDES_HOST_CUH__
#define __BRANDES_HOST_CUH__

#include <vector>

namespace host {
    struct VirtualCSR {
        std::vector<int> vmap;
        std::vector<int> vptrs;
        std::vector<int> adjs;

        VirtualCSR() = delete;
        VirtualCSR(const std::vector<std::pair<int, int>> &edges, int mdeg);
    };
}

int num_verts(const std::vector<std::pair<int, int>> &edges);

#endif // __BRANDES_HOST_CUH__
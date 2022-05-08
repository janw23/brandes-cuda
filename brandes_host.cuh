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


class CUDATimer {
    private:
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> kernel_evts;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> memcpy_evts;

    class Timer {
        cudaEvent_t stop;

        public:
        Timer() = delete;
        Timer(cudaEvent_t start, cudaEvent_t stop);
        ~Timer();
    };

public:
    ~CUDATimer();
    
    Timer kernel_timer();
    Timer memcpy_timer() {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        memcpy_evts.emplace_back(start, stop);
        return CUDATimer::Timer(start, stop);
    }

    float elapsed_time_kernels();
    float elapsed_time_memcpy();
};


int num_verts(const std::vector<std::pair<int, int>> &edges);

#endif // __BRANDES_HOST_CUH__
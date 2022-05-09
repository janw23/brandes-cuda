#ifndef UTILS_CUH
#define UTILS_CUH

#include <vector>
#include <cstdint>

struct DeviceBool {
    bool *device_data;

    DeviceBool(bool initial);
    DeviceBool();

    ~DeviceBool();

    void set_value(bool val);
    bool get_value();
};

struct VirtualCSR {
    std::vector<uint32_t> vmap;
    std::vector<uint32_t> vptrs;
    std::vector<uint32_t> adjs;

    VirtualCSR() = delete;
    VirtualCSR(const std::vector<std::pair<uint32_t, uint32_t>> &edges, uint32_t mdeg);
};

uint32_t num_verts(const std::vector<std::pair<uint32_t, uint32_t>> &edges);

// Helper function which converts graph into adjacency lists representation.
static std::vector<std::vector<uint32_t>> adjacency_lists(const std::vector<std::pair<uint32_t, uint32_t>> &edges);

// Returns grid_size based on the overall required number of threads and block size.
uint32_t grid_size(uint32_t min_threads_count, uint32_t block_size);

#endif
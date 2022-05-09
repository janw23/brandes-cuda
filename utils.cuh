#ifndef UTILS_CUH
#define UTILS_CUH

#include <vector>

struct DeviceBool {
    bool *device_data;

    DeviceBool(bool initial);
    DeviceBool();

    ~DeviceBool();

    void set_value(bool val);
    bool get_value();
};

struct VirtualCSR {
    std::vector<int> vmap;
    std::vector<int> vptrs;
    std::vector<int> adjs;

    VirtualCSR() = delete;
    VirtualCSR(const std::vector<std::pair<int, int>> &edges, int mdeg);
};

int num_verts(const std::vector<std::pair<int, int>> &edges);

// Helper function which converts graph into adjacency lists representation.
static std::vector<std::vector<int>> adjacency_lists(const std::vector<std::pair<int, int>> &edges);

// Returns grid_size based on the overall required number of threads and block size.
int grid_size(int min_threads_count, int block_size);

#endif
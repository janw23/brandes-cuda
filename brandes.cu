#include "brandes_host.cuh"
#include "brandes_device.cuh"
#include "errors.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <stack>
#include <cassert>

using namespace std;

// TODO correct int data types

static void print_usage() {
    cout << "Program usage: ./brandes input-file output-file" << endl;
}

static void check_args(int argc, char *argv[]) {
    if (argc != 3 || !strcmp(argv[1], "") || !strcmp(argv[2], "")) {
        print_usage();
        exit(1);
    }
}

static vector<pair<int, int>> load_edges(string path) {
    ifstream ifs;
    ifs.open(path);
    if(!ifs.is_open()) {
        cout << "Cannot open file \"" << path << "\"." << endl;
        exit(1);
    }

    vector<pair<int, int>> edges;

    int u, v;
    while(ifs >> u >> v) edges.emplace_back(u, v);

    ifs.close();
    return edges;
} 

static void save_to_file(string path, const vector<double> &centrality) {
    ofstream ofs;
    ofs.open(path);

    if (!ofs.is_open()) {
        cout << "Cannot open file \"" << path << "\"." << endl;
        exit(1);
    }

    for (auto val : centrality) {
        ofs << val << endl;
    }

    ofs.close();
}

struct DeviceBool {
    bool *device_data;

    DeviceBool(bool initial) {
        HANDLE_ERROR(cudaMalloc(&device_data, sizeof(bool)));
        set_value(initial); // default value
    }

    DeviceBool() : DeviceBool(false) {}

    ~DeviceBool() {
        HANDLE_ERROR(cudaFree(device_data));
        device_data = NULL;
    }

    void set_value(bool val) {
        HANDLE_ERROR(cudaMemcpy(device_data, &val, sizeof(bool), cudaMemcpyHostToDevice));
    }

    bool get_value() {
        bool val;
        HANDLE_ERROR(cudaMemcpy(&val, device_data, sizeof(bool), cudaMemcpyDeviceToHost));
        return val;
    }
};


// Returns grid_size based on the overall required number of threads and block size.
static int grid_size(int min_threads_count, int block_size) {
    return (min_threads_count + block_size - 1) / block_size;
}

static vector<double> betweeness_on_gpu(const vector<pair<int, int>> &edges) {
    CUDATimer cuda_timer;

    host::VirtualCSR host_vcsr(edges, 4);
    device::GraphDataVirtual gdata(move(edges), 4); // this moves data to device

    const int block_size = 512; // TODO

    // Initialize betweeness centrality array with zeros.
    {
        auto kt = cuda_timer.kernel_timer(); // RAII style timer
        static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
        fill<<<num_blocks, block_size>>>(gdata.bc, gdata.num_real_verts, 0.0);
        HANDLE_ERROR(cudaPeekAtLastError());
    }

    // ALGORITHM BEGIN
    DeviceBool cont;

    for (int s = 0; s < gdata.num_real_verts; s++) { // for each vertex as a source
        cout << (s+1) << " / " << gdata.num_real_verts << "\n";
        // Reset values, because they are source-specific.
        {
            auto kt = cuda_timer.kernel_timer(); // RAII style timer
            static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_prep_fwd<<<num_blocks, block_size>>>(gdata, s);
            HANDLE_ERROR(cudaPeekAtLastError());
        }

        // Run forward phase.
        int layer = 0;
        do {
            cont.set_value(false);
            {
                auto kt = cuda_timer.kernel_timer(); // RAII style timer
                static const int num_blocks = grid_size(gdata.num_virtual_verts, block_size);
                bc_virtual_fwd<<<num_blocks, block_size>>>(gdata, layer, cont.device_data);
                HANDLE_ERROR(cudaPeekAtLastError());
            }
            layer++;
        } while(cont.get_value());

        // Initialize delta values.
        {
            auto kt = cuda_timer.kernel_timer(); // RAII style timer
            static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_prep_bwd<<<num_blocks, block_size>>>(gdata);
            HANDLE_ERROR(cudaPeekAtLastError());
        }

        // Run backward phase
        while (layer > 1) {
            layer--;
            {
                auto kt = cuda_timer.kernel_timer(); // RAII style timer
                static const int num_blocks = grid_size(gdata.num_virtual_verts, block_size);
                bc_virtual_bwd<<<num_blocks, block_size>>>(gdata, layer);
                HANDLE_ERROR(cudaPeekAtLastError());
            }
        }

        // Update bc values.
        {
            auto kt = cuda_timer.kernel_timer(); // RAII style timer
            static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_update<<<num_blocks, block_size>>>(gdata, s);
            HANDLE_ERROR(cudaPeekAtLastError());
        }
    }
    
    // ALGORITHM END

    vector<double> betweeness(gdata.num_real_verts);
    HANDLE_ERROR(cudaMemcpy(betweeness.data(), gdata.bc, sizeof(double) * betweeness.size(), cudaMemcpyDeviceToHost));
    
    gdata.free();
    cerr << "Kernels execution time: " << (int) cuda_timer.elapsed_time_kernels() << "ms\n";
    return betweeness;
}

int main(int argc, char *argv[]) {
    check_args(argc, argv);
    auto edges = load_edges(argv[1]);
    auto betweeness = betweeness_on_gpu(move(edges));
    save_to_file(argv[2], betweeness);

    return 0;
}
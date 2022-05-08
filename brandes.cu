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
        HANDLE_ERROR(cudaMemset(device_data, val, sizeof(bool)));
    }

    bool get_value() {
        bool val;
        HANDLE_ERROR(cudaMemcpy(&val, device_data, sizeof(bool), cudaMemcpyDeviceToHost));
        return val;
    }
};

static void print_progress(int done, int all) {
    static int prev = -1;
    int percent = 100.0f * done / all;
    if (prev != percent) {
        cout << percent << "%\n";
        prev = percent;
    }
}

class Timer {
    cudaEvent_t _start, _stop;

public:
    Timer() {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
        cudaEventRecord(_start);
    }

    void stop() {
        cudaEventRecord(_stop);
    }

    ~Timer() {
        cudaEventSynchronize(_stop);
        float ms;
        cudaEventElapsedTime(&ms, _start, _stop);
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
        cout << "Elapsed kernels time: " << ms << "\n";
    }
};

// Returns grid_size based on the overall required number of threads and block size.
static int grid_size(int min_threads_count, int block_size) {
    return (min_threads_count + block_size - 1) / block_size;
}

static vector<double> betweeness_on_gpu(const vector<pair<int, int>> &edges) {
    host::VirtualCSR host_vcsr(edges, 4);
    device::GraphDataVirtual gdata(move(edges), 4); // this moves data to device

    const int block_size = 512; // TODO
    Timer timer;

    // Initialize betweeness centrality array with zeros.
    {
        static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
        fill<<<num_blocks, block_size>>>(gdata.bc, gdata.num_real_verts, 0.0);
        HANDLE_ERROR(cudaPeekAtLastError());
    }

    // ALGORITHM BEGIN
    DeviceBool cont;

    for (int s = 0; s < gdata.num_real_verts; s++) { // for each vertex as a source
        print_progress(s+1, gdata.num_real_verts); // TODO RM
        // Reset values, because they are source-specific.
        {
            static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_prep_fwd<<<num_blocks, block_size>>>(gdata, s);
            HANDLE_ERROR(cudaPeekAtLastError());
        }

        // Run forward phase.
        int layer = 0;
        do {
            cont.set_value(false);
            for (int tries = 0; tries < 5; tries++) { // This optimizes device -> host memory transfers
                {
                    static const int num_blocks = grid_size(gdata.num_virtual_verts, block_size);
                    bc_virtual_fwd<<<num_blocks, block_size>>>(gdata, layer, cont.device_data);
                    HANDLE_ERROR(cudaPeekAtLastError());
                }
                layer++;
            }
        } while(cont.get_value());

        // Initialize delta values.
        {
            static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_prep_bwd<<<num_blocks, block_size>>>(gdata);
            HANDLE_ERROR(cudaPeekAtLastError());
        }

        // Run backward phase
        while (layer > 1) {
            layer--;
            {
                static const int num_blocks = grid_size(gdata.num_virtual_verts, block_size);
                bc_virtual_bwd<<<num_blocks, block_size>>>(gdata, layer);
                HANDLE_ERROR(cudaPeekAtLastError());
            }
        }

        // Update bc values.
        {
            static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_update<<<num_blocks, block_size>>>(gdata, s);
            HANDLE_ERROR(cudaPeekAtLastError());
        }
    }
    timer.stop();
    // ALGORITHM END

    vector<double> betweeness(gdata.num_real_verts);
    HANDLE_ERROR(cudaMemcpy(betweeness.data(), gdata.bc, sizeof(double) * betweeness.size(), cudaMemcpyDeviceToHost));
    
    gdata.free();
    return betweeness;
}

int main(int argc, char *argv[]) {
    check_args(argc, argv);
    auto edges = load_edges(argv[1]);
    auto betweeness = betweeness_on_gpu(move(edges));
    save_to_file(argv[2], betweeness);

    return 0;
}
#include "kernels.cuh"
#include "utils.cuh"
#include "errors.h"

#include <vector>
#include <fstream>
#include <iostream>

using namespace std;

static void check_args(int argc, char *argv[]) {
    if (argc != 3 || !strcmp(argv[1], "") || !strcmp(argv[2], "")) {
        cout << "Program usage: ./brandes input-file output-file" << endl;
        exit(1);
    }
}

static vector<pair<uint32_t, uint32_t>> load_edges(string path) {
    ifstream ifs;
    ifs.open(path);
    if(!ifs.is_open()) {
        cout << "Cannot open file \"" << path << "\"." << endl;
        exit(1);
    }

    vector<pair<uint32_t, uint32_t>> edges;

    uint32_t u, v;
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

static void print_progress(int done, int all) {
    static int prev = -1;
    int percent = 100.0f * done / all;
    if (prev != percent) {
        cout << percent << "%\n";
        prev = percent;
    }
}

static vector<double> betweeness_on_gpu(const vector<pair<uint32_t, uint32_t>> &edges) {
    cudaEvent_t kernels_start, kernels_stop, mem_start, mem_stop;
    HANDLE_ERROR(cudaEventCreate(&kernels_start));
    HANDLE_ERROR(cudaEventCreate(&kernels_stop));
    HANDLE_ERROR(cudaEventCreate(&mem_start));
    HANDLE_ERROR(cudaEventCreate(&mem_stop));

    HANDLE_ERROR(cudaEventRecord(mem_start)); // TODO currently this also measures generating vcsr on host
    GraphDataVirtual gdata(move(edges), 4); // this moves data to device

    HANDLE_ERROR(cudaEventRecord(kernels_start));

    const int block_size = 512; // TODO
    // Initialize betweeness centrality array with zeros.
    {
        static const int num_blocks = grid_size(gdata.num_real_verts, block_size);
        fill<<<num_blocks, block_size>>>(gdata.bc, gdata.num_real_verts, 0.0);
        HANDLE_ERROR(cudaPeekAtLastError());
    }

    // ALGORITHM BEGIN
    DeviceBool cont;

    for (uint32_t s = 0; s < gdata.num_real_verts; s++) { // for each vertex as a source
        print_progress(s+1, gdata.num_real_verts); // TODO RM
        // Reset values, because they are source-specific.
        {
            static const uint32_t num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_prep_fwd<<<num_blocks, block_size>>>(gdata, s);
            HANDLE_ERROR(cudaPeekAtLastError());
        }

        // Run forward phase.
        int32_t layer = 0;
        do {
            cont.set_value(false);
            for (int tries = 0; tries < 2; tries++) { // This optimizes device -> host memory transfers
                {
                    static const uint32_t num_blocks = grid_size(gdata.num_virtual_verts, block_size);
                    bc_virtual_fwd<<<num_blocks, block_size>>>(gdata, layer, cont.device_data);
                    HANDLE_ERROR(cudaPeekAtLastError());
                }
                layer++;
            }
        } while(cont.get_value());

        // Initialize delta values.
        {
            static const uint32_t num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_prep_bwd<<<num_blocks, block_size>>>(gdata);
            HANDLE_ERROR(cudaPeekAtLastError());
        }

        // Run backward phase
        while (layer > 1) {
            layer--;
            {
                static const uint32_t num_blocks = grid_size(gdata.num_virtual_verts, block_size);
                bc_virtual_bwd<<<num_blocks, block_size>>>(gdata, layer);
                HANDLE_ERROR(cudaPeekAtLastError());
            }
        }

        // Update bc values.
        {
            static const uint32_t num_blocks = grid_size(gdata.num_real_verts, block_size);
            bc_virtual_update<<<num_blocks, block_size>>>(gdata, s);
            HANDLE_ERROR(cudaPeekAtLastError());
        }
    }

    HANDLE_ERROR(cudaEventRecord(kernels_stop));

    // ALGORITHM END

    vector<double> betweeness(gdata.num_real_verts);
    HANDLE_ERROR(cudaMemcpy(betweeness.data(), gdata.bc, sizeof(double) * betweeness.size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(mem_stop));

    gdata.free();

    HANDLE_ERROR(cudaEventSynchronize(kernels_stop));
    HANDLE_ERROR(cudaEventSynchronize(mem_stop));

    float kernels_time, mem_time;
    HANDLE_ERROR(cudaEventElapsedTime(&kernels_time, kernels_start, kernels_stop));
    HANDLE_ERROR(cudaEventElapsedTime(&mem_time, mem_start, mem_stop));

    cerr << round(kernels_time) << "\n";
    cerr << round(mem_time) << "\n";

    HANDLE_ERROR(cudaEventDestroy(kernels_start));
    HANDLE_ERROR(cudaEventDestroy(kernels_stop));
    HANDLE_ERROR(cudaEventDestroy(mem_start));
    HANDLE_ERROR(cudaEventDestroy(mem_stop));

    return betweeness;
}


int main(int argc, char *argv[]) {
    check_args(argc, argv);
    auto edges = load_edges(argv[1]);
    auto betweeness = betweeness_on_gpu(move(edges));
    save_to_file(argv[2], betweeness);

    return 0;
}
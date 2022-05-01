#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

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
        cerr << "Cannot open file \"" << path << "\"." << endl;
        exit(1);
    }

    vector<pair<int, int>> edges;

    while(ifs.good()) {
        int u, v;
        ifs >> u >> v;
        edges.emplace_back(u, v);
    }

    ifs.close();
    return edges;
} 

int main(int argc, char *argv[]) {
    check_args(argc, argv);
    auto edges = load_edges(argv[1]);

    for (auto edge : edges) {
        cout << edge.first << "-" << edge.second << "\n";
    }

    return 0;
}
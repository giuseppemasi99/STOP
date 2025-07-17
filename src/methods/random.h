#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <algorithm>
#include <iostream>


using namespace std;

tuple<vector<vector<vector<int>>>, vector<double>, double> getRandomSolution(int n_machines, int n_demands, int n_models, mt19937& rng)
{
    auto start = chrono::high_resolution_clock::now();

    vector<vector<vector<int>>> x(n_machines, vector<vector<int>>(n_demands, vector<int>(n_models, 0)));

    vector<int> all_indices_machines(n_demands);
    vector<int> all_indices_models(n_demands);

    uniform_int_distribution<> dist_machine(0, n_machines); // inclusive upper bound
    uniform_int_distribution<> dist_model(0, n_models - 1);

    for (int i = 0; i < n_demands; ++i) {
        all_indices_machines[i] = dist_machine(rng);
        all_indices_models[i] = dist_model(rng);
    }

    // Find indices where machine index != n_machines
    vector<int> ind;
    for (int i = 0; i < n_demands; ++i) {
        if (all_indices_machines[i] != n_machines)
            ind.push_back(i);
    }

    for (int idx : ind) {
        int m = all_indices_machines[idx];
        int d = idx;
        int mod = all_indices_models[idx];
        x[m][d][mod] = 1;
    }

    vector<double> gamma_k(n_machines, 0.0);
    if (n_demands != 0) {
        // For each machine, check if any assignment exists (max over demands and models)
        for (int m = 0; m < n_machines; ++m) {
            int found = 0;
            for (int d = 0; d < n_demands; ++d)
                for (int mod = 0; mod < n_models; ++mod)
                    found = max(found, x[m][d][mod]);
            gamma_k[m] = found;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end - start).count();

    return make_tuple(x, gamma_k, elapsed_time);
}
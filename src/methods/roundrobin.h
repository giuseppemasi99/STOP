#include <vector>
#include <chrono>
#include <cmath>
#include <tuple>
#include <algorithm>

using namespace std;

tuple<vector<vector<vector<int>>>, vector<double>, double> getRRSolution(int n_machines, int n_demands, int n_models) {
    auto start = chrono::high_resolution_clock::now();

    // 3D assignment: [n_machines][n_demands][n_models]
    vector<vector<vector<int>>> x(n_machines, vector<vector<int>>(n_demands, vector<int>(n_models, 0)));

    vector<double> gamma_k(n_machines, 0.0);

    if (n_demands == 0) {
        // gamma_k already zero
    } else {
        int m1 = n_demands % (n_machines + 1);
        int n1 = static_cast<int>(ceil(static_cast<double>(n_demands) / (n_machines + 1)));
        int n2 = n_demands / (n_machines + 1);

        for (int k = 0; k < n_machines + 1; ++k) {
            if (k < m1) {
                int start_idx = k * n1;
                int end_idx = (k + 1) * n1;
                vector<int> ind_demands;
                for (int i = start_idx; i < end_idx && i < n_demands; ++i)
                    ind_demands.push_back(i);

                vector<int> ind_models;
                for (int i = 0; i < n_models; ++i)
                    ind_models.push_back(i);

                int rep = n1 / n_models;
                vector<int> ind_models_full;
                for (int r = 0; r < rep; ++r)
                    ind_models_full.insert(ind_models_full.end(), ind_models.begin(), ind_models.end());

                int to_append = n1 % n_models;
                if (to_append > 0) {
                    for (int i = 0; i < to_append; ++i)
                        ind_models_full.push_back(i);
                }

                for (size_t idx = 0; idx < ind_demands.size() && idx < ind_models_full.size(); ++idx) {
                    if (k < n_machines && ind_demands[idx] < n_demands && ind_models_full[idx] < n_models)
                        x[k][ind_demands[idx]][ind_models_full[idx]] = 1;
                }
            } else if (k < n_machines) {
                int start_idx = m1 * n1 + n2 * (k - m1);
                int end_idx = m1 * n1 + n2 * (k + 1 - m1);
                vector<int> ind_demands;
                for (int i = start_idx; i < end_idx && i < n_demands; ++i)
                    ind_demands.push_back(i);

                vector<int> ind_models;
                for (int i = 0; i < n_models; ++i)
                    ind_models.push_back(i);

                int rep = n2 / n_models;
                vector<int> ind_models_full;
                for (int r = 0; r < rep; ++r)
                    ind_models_full.insert(ind_models_full.end(), ind_models.begin(), ind_models.end());

                int to_append = n2 % n_models;
                if (to_append > 0) {
                    for (int i = 0; i < to_append; ++i)
                        ind_models_full.push_back(i);
                }

                for (size_t idx = 0; idx < ind_demands.size() && idx < ind_models_full.size(); ++idx) {
                    if (k < n_machines && ind_demands[idx] < n_demands && ind_models_full[idx] < n_models)
                        x[k][ind_demands[idx]][ind_models_full[idx]] = 1;
                }
            } else {
                continue;
            }
        }

        // Compute gamma_k: for each machine, max over demands and models
        for (int m = 0; m < n_machines; ++m) {
            int found = 0;
            for (int d = 0; d < n_demands; ++d)
                for (int mod = 0; mod < n_models; ++mod)
                    found = std::max(found, x[m][d][mod]);
            gamma_k[m] = found;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end - start).count();

    return make_tuple(x, gamma_k, elapsed_time);
}
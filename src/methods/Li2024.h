#include <vector>
#include <chrono>
#include <cmath>
#include <tuple>
#include <algorithm>
#include "Highs.h"


using namespace std;

tuple<vector<vector<vector<int>>>, vector<double>, double> getLi2024Solution(
    const string& method, double V, const vector<double>& k_t, int n_machines, int n_models, int n_demands,
    const vector<double>& FLOPS_k, const vector<double>& Z_k_const_, const vector<vector<double>>& omega, 
    const vector<double>& sigma, const vector<double>& B_k, int P_avg, const vector<double>& delta_j,
    const vector<double>& Q_k, const vector<double>& epsilon_k, const vector<int>& D_k, const vector<double>& const_k,
    double C_avg, double R_avg) 
{

    if (method == "Li2024-AllDemandsExecuted") {
        assert(V == 0 && "Trying to execute Li2024-AllDemandsExecuted with V > 0");
    } else {
        assert(V > 0 && "Trying to execute Li2024-WithoutCapacityConstrained with V = 0");
    }

    auto start = chrono::high_resolution_clock::now();

    // 3D assignment: [n_machines][n_demands][n_models]
    vector<vector<vector<int>>> x(n_machines, vector<vector<int>>(n_demands, vector<int>(n_models, 0)));
    vector<double> gamma_k(n_machines, 0.0);

    if (n_demands > 0){

        vector<double> c;
        for (int k = 0; k < n_machines; ++k) {
            for (int i = 0; i < n_demands; ++i) {
                for (int j = 0; j < n_models; ++j) {
                    double val = -V 
                        + k_t[k] * Z_k_const_[k] * omega[i][j]
                        + k_t[n_machines + k] * (sigma[i] / B_k[k] + omega[i][j] / FLOPS_k[k])
                        + k_t[2 * n_machines] * (P_avg - delta_j[j]);
                    c.push_back(val);
                }
            }
        }
        for (int k = 0; k < n_machines; ++k) {
            double val = k_t[n_machines + k] * (Q_k[k] / FLOPS_k[k] + epsilon_k[k] * D_k[k] + const_k[k]);
            c.push_back(val);
        }

        int rows, cols;
        if (method == "Li2024-AllDemandsExecuted") {
            rows = n_demands + n_machines * n_demands * n_models + n_machines;
            cols = n_machines + n_machines * n_demands * n_models;
        } else {
            rows = n_demands + n_machines * n_demands * n_models;
            cols = n_machines + n_machines * n_demands * n_models;
        }
        cout << endl;
        cout << endl;
        

        vector<double> A_ub_values;
        vector<int> A_ub_row_indices;
        vector<int> A_ub_col_pointers;
        A_ub_col_pointers.push_back(0);

        for (int col = 0; col < cols; ++col) {
            map<int, double> col_entries; // row -> value
            // Demand constraints
            for (int k = 0; k < n_machines; ++k) {
                int ind_start = k * n_models * n_demands;
                for (int i = 0; i < n_demands; ++i) {
                    for (int j = 0; j < n_models; ++j) {
                        int col_idx = ind_start + i * n_models + j;
                        if (col == col_idx) {
                            col_entries[i] += 1.0;
                        }
                    }
                }
            }
            // Gamma constraints
            for (int k = 0; k < n_machines; ++k) {
                int row_start = n_demands + k * (n_demands * n_models);
                int col_start = k * (n_demands * n_models);
                int col_end = (k + 1) * (n_demands * n_models);
                if (col >= col_start && col < col_end) {
                    int r = row_start + (col - col_start);
                    col_entries[r] += 1.0;
                    int gamma_col = cols - n_machines + k;
                    if (col == gamma_col) {
                        col_entries[r] += -1.0;
                    }
                }
            }
            // Capacity constraints
            if (method == "Li2024-AllDemandsExecuted") {
                vector<double> omega_flat;
                for (int i = 0; i < n_demands; ++i)
                    for (int j = 0; j < n_models; ++j)
                        omega_flat.push_back(omega[i][j]);
                for (int k = 1; k <= n_machines; ++k) {
                    int row_idx = rows - (n_machines - k) - 1;
                    int offset = (k - 1) * n_demands * n_models;
                    if (col >= offset && col < offset + omega_flat.size()) {
                        col_entries[row_idx] += omega_flat[col - offset];
                    }
                }
            }
            int nnz_in_col = 0;
            for (const auto& entry : col_entries) {
                if (entry.second != 0.0) {
                    A_ub_values.push_back(entry.second);
                    A_ub_row_indices.push_back(entry.first);
                    nnz_in_col++;
                }
            }
            A_ub_col_pointers.push_back(A_ub_col_pointers.back() + nnz_in_col);
        }
        
        if (method == "Li2024-AllDemandsExecuted") {
            int extra_row = rows;
            for (int col = 0; col < cols; ++col) {
                if (col < cols - n_machines) {
                    int start = A_ub_col_pointers[col];
                    int end = A_ub_col_pointers[col + 1];
                    A_ub_values.insert(A_ub_values.begin() + end, 1.0);
                    A_ub_row_indices.insert(A_ub_row_indices.begin() + end, extra_row);
                    for (int k = col + 1; k < A_ub_col_pointers.size(); ++k) {
                        A_ub_col_pointers[k]++;
                    }
                }
            }
            rows += 1;
        }

        vector<double> b_ub;
        b_ub.insert(b_ub.end(), n_demands, 1.0);
        b_ub.insert(b_ub.end(), n_machines * n_demands * n_models, 0.0);
        if (method == "Li2024-AllDemandsExecuted") {
            b_ub.insert(b_ub.end(), FLOPS_k.begin(), FLOPS_k.end());
            b_ub.push_back(static_cast<double>(n_demands));
        }

        vector<double> b_lb;
        b_lb.insert(b_lb.end(), n_demands + n_machines * n_demands * n_models, -1.0);
        if (method == "Li2024-AllDemandsExecuted") {
            b_lb.insert(b_lb.end(), n_machines, -1.0);
            b_lb.push_back(static_cast<double>(n_demands));
        }

        Highs highs;
        HighsModelStatus model_status;
        HighsLp lp;
        lp.num_col_ = cols;
        lp.num_row_ = rows;
        lp.col_cost_ = c;
        lp.col_lower_.assign(cols, 0.0);
        lp.col_upper_.assign(cols, 1.0);
        lp.row_lower_ = b_lb;
        lp.row_upper_ = b_ub;
        lp.a_matrix_.format_ = MatrixFormat::kColwise;
        lp.a_matrix_.start_ = A_ub_col_pointers;
        lp.a_matrix_.index_ = A_ub_row_indices;
        lp.a_matrix_.value_ = A_ub_values;
        lp.integrality_.assign(cols, HighsVarType::kInteger);
        highs.passModel(lp);
        HighsStatus status = highs.run();
        const vector<double>& sol_relaxed = highs.getSolution().col_value;
        for (int k = 0; k < n_machines; ++k) {
            for (int i = 0; i < n_demands; ++i) {
                for (int j = 0; j < n_models; ++j) {
                    int idx = k * n_demands * n_models + i * n_models + j;
                    x[k][i][j] = static_cast<int>(round(sol_relaxed[idx]));
                }
            }
        }

        



    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end - start).count();

    return make_tuple(x, gamma_k, elapsed_time);
}
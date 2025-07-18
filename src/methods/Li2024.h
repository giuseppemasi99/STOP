#include <vector>
#include <chrono>
#include <cmath>
#include <tuple>
#include <algorithm>
#include "Highs.h"


using namespace std;

tuple<vector<vector<vector<int>>>, vector<double>, double> getLi2024Solution(
    const string& method, double V, vector<double>& k_t, int n_machines, int n_models, int n_demands,
    const vector<double>& FLOPS_k, const vector<double>& Z_k_const_, const vector<vector<double>>& omega, 
    const vector<double>& sigma, const vector<double>& B_k, double P_avg, const vector<double>& delta_j,
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
        int n_x = n_machines * n_demands * n_models;
        int n_vars = n_x + n_machines;
        int n_rows = n_demands + n_machines * n_demands * n_models;
        bool add_extra_rows = (method == "Li2024-AllDemandsExecuted");
        int extra_rows_start = n_rows;
        int n_extra_rows = add_extra_rows ? (n_machines + 1) : 0;
        if (add_extra_rows) n_rows += n_extra_rows;
        vector<int> Astart(n_vars + 1, 0);
        vector<int> Aindex;
        vector<double> Avalue;
        vector<double> omega_flat;
        if (add_extra_rows) {
            for (int i = 0; i < n_demands; ++i) {
                for (int j = 0; j < n_models; ++j) {
                    omega_flat.push_back(omega[i][j]);
                }
            }
        }
        for (int col = 0; col < n_x; ++col) {
            int k = col / (n_demands * n_models);
            int rem = col % (n_demands * n_models);
            int i = rem / n_models;
            int j = rem % n_models;
            Aindex.push_back(i);
            Avalue.push_back(1.0);
            int row_gamma = n_demands + k * (n_demands * n_models) + i * n_models + j;
            Aindex.push_back(row_gamma);
            Avalue.push_back(1.0);
            if (add_extra_rows) {
                int all_ones_row = extra_rows_start + n_machines;
                Aindex.push_back(all_ones_row);
                Avalue.push_back(1.0);
                for (int k_idx = 0; k_idx < n_machines; ++k_idx) {
                    int block_start = k_idx * n_demands * n_models;
                    int block_end = (k_idx + 1) * n_demands * n_models;
                    if (col >= block_start && col < block_end) {
                        int omega_pos = col - block_start;
                        int omega_row = extra_rows_start + k_idx;
                        Aindex.push_back(omega_row);
                        Avalue.push_back(omega_flat[omega_pos]);
                    }
                }
            }
            Astart[col + 1] = (int)Aindex.size();
        }
        for (int k = 0; k < n_machines; ++k) {
            for (int i = 0; i < n_demands * n_models; ++i) {
                int row_gamma = n_demands + k * (n_demands * n_models) + i;
                Aindex.push_back(row_gamma);
                Avalue.push_back(-1.0);
            }
            Astart[n_x + k + 1] = (int)Aindex.size();
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
        highs.setOptionValue("output_flag", false);
        HighsModelStatus model_status;
        HighsLp lp;
        lp.num_col_ = n_vars;
        lp.num_row_ = n_rows;
        lp.col_cost_ = c;
        lp.col_lower_.assign(n_vars, 0.0);
        lp.col_upper_.assign(n_vars, 1.0);
        lp.row_lower_ = b_lb;
        lp.row_upper_ = b_ub;
        lp.a_matrix_.format_ = MatrixFormat::kColwise;
        lp.a_matrix_.start_ = Astart;
        lp.a_matrix_.index_ = Aindex;
        lp.a_matrix_.value_ = Avalue;
        lp.integrality_.assign(n_vars, HighsVarType::kInteger);
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
        
        vector<double> c2;
        for (size_t i = 0; i < k_t.size() - 1; ++i)
            c2.push_back(-k_t[i]);
        c2.push_back(1 - k_t.back());
        c2.push_back(1.0);
        c2.push_back(1.0);
        int n_cols2 = c2.size();
        int n_rows2 = 2 * n_machines;
        vector<int> Astart2(n_cols2 + 1, 0);
        vector<int> Aindex2;
        vector<double> Avalue2;
        for (int col = 0; col < n_cols2; ++col) {
            int nnz = 0;
            if (col < n_machines) {
                Aindex2.push_back(col);
                Avalue2.push_back(1.0);
                nnz++;
            }
            if (col == n_cols2 - 2) {
                for (int k = 0; k < n_machines; ++k) {
                    Aindex2.push_back(k);
                    Avalue2.push_back(-1.0);
                    nnz++;
                }
            }
            if (col >= n_machines && col < 2 * n_machines) {
                int k = col - n_machines;
                Aindex2.push_back(n_machines + k);
                Avalue2.push_back(1.0);
                nnz++;
            }
            if (col == n_cols2 - 1) {
                for (int k = 0; k < n_machines; ++k) {
                    Aindex2.push_back(n_machines + k);
                    Avalue2.push_back(-1.0);
                    nnz++;
                }
            }
            Astart2[col + 1] = Astart2[col] + nnz;
        }
        vector<double> b_ub2(n_rows2, 0.0);
        vector<double> b_lb2(n_rows2, -kHighsInf);
        vector<double> col_lower2;
        col_lower2.insert(col_lower2.end(), n_machines, 0.0);
        col_lower2.insert(col_lower2.end(), n_machines, 0.0);
        col_lower2.push_back(-100.0);
        col_lower2.push_back(0.0);
        col_lower2.push_back(0.0);
        vector<double> col_upper2;
        col_upper2.insert(col_upper2.end(), n_machines, C_avg);
        col_upper2.insert(col_upper2.end(), n_machines, R_avg);
        col_upper2.push_back(-P_avg);
        col_upper2.push_back(C_avg);
        col_upper2.push_back(R_avg);
        HighsModelStatus model_status2;
        highs.setOptionValue("presolve", false);
        HighsLp lp2;
        lp2.num_col_ = n_cols2;
        lp2.num_row_ = n_rows2;
        lp2.col_cost_ = c2;
        lp2.col_lower_ = col_lower2;
        lp2.col_upper_ = col_upper2;
        lp2.row_lower_ = b_lb2;
        lp2.row_upper_ = b_ub2;
        lp2.a_matrix_.format_ = MatrixFormat::kColwise;
        lp2.a_matrix_.start_ = Astart2;
        lp2.a_matrix_.index_ = Aindex2;
        lp2.a_matrix_.value_ = Avalue2;
        // lp2.integrality_.assign(n_cols2, HighsVarType::kInteger);
        highs.passModel(lp2);
        HighsStatus status2 = highs.run();
        const vector<double>& sol = highs.getSolution().col_value;
        vector<double> z_c(sol.begin(), sol.begin() + n_machines);
        vector<double> z_r(sol.begin() + n_machines, sol.begin() + 2 * n_machines);
        double z_p = sol[2 * n_machines];

        vector<double> x_at_omega(n_machines, 0.0);
        for (int m = 0; m < n_machines; ++m) {
            double sum = 0.0;
            for (int d = 0; d < n_demands; ++d) {
                for (int mod = 0; mod < n_models; ++mod) {
                    sum += x[m][d][mod] * omega[d][mod];
                }
            }
            x_at_omega[m] = sum;
        }
        vector<double> x_at_sigma(n_machines, 0.0);
        for (int m = 0; m < n_machines; ++m) {
            double sum = 0.0;
            for (int d = 0; d < n_demands; ++d) {
                int assigned = 0;
                for (int mod = 0; mod < n_models; ++mod) {
                    assigned = std::max(assigned, x[m][d][mod]);
                }
                sum += assigned * sigma[d];
            }
            x_at_sigma[m] = sum;
        }
        double x_at_delta = 0.0;
        for (int m = 0; m < n_machines; ++m) {
            for (int d = 0; d < n_demands; ++d) {
                for (int mod = 0; mod < n_models; ++mod) {
                    x_at_delta += x[m][d][mod] * (P_avg - delta_j[mod]);
                }
            }
        }
        vector<double> gamma_k(n_machines, 0.0);
        for (int k_idx = 0; k_idx < n_machines; ++k_idx) {
            int max_val = 0;
            for (int i = 0; i < n_demands; ++i)
                for (int j = 0; j < n_models; ++j)
                    if (x[k_idx][i][j] > max_val) max_val = x[k_idx][i][j];
            gamma_k[k_idx] = max_val;
        }
        vector<double> C(n_machines, 0.0);
        for (int k = 0; k < n_machines; ++k)
            C[k] = Z_k_const_[k] * x_at_omega[k];
        vector<double> R(n_machines, 0.0);
        for (int k = 0; k < n_machines; ++k) {
            R[k] = x_at_sigma[k] / B_k[k]
                + x_at_omega[k] / FLOPS_k[k]
                + (Q_k[k] / FLOPS_k[k] + epsilon_k[k] * D_k[k] + const_k[k]) * gamma_k[k];
        }
        double P = x_at_delta;
        vector<double> d_t;
        d_t.insert(d_t.end(), z_c.begin(), z_c.end());
        d_t.insert(d_t.end(), z_r.begin(), z_r.end());
        d_t.push_back(z_p);
        vector<double> CRP;
        CRP.insert(CRP.end(), C.begin(), C.end());
        CRP.insert(CRP.end(), R.begin(), R.end());
        CRP.push_back(P);
        for (size_t i = 0; i < d_t.size(); ++i) {
            d_t[i] -= CRP[i];
        }
        double eta = 1.7*1e-4;
        for (size_t i = 0; i < k_t.size(); ++i) {
            k_t[i] = std::max(k_t[i] - eta * d_t[i], 0.0);
        }

    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end - start).count();
    return make_tuple(x, gamma_k, elapsed_time);
}
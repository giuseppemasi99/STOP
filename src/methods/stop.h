#include "Highs.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <set>
#include "solution.h"

using namespace std;
using namespace chrono;


Solution getSTOPSolution(
    int n_machines, int n_demands, int n_models, double V,
    const vector<vector<double>>& omega,
    const vector<double>& Z_k, const vector<int>& CI_k, const vector<double>& PUE_k, const vector<double>& e_k, double C_avg,
    const vector<double>& W_k, const vector<double>& FLOPS_k, double R_avg, const vector<double>& Q_k,
    const vector<double>& sigma, const vector<double>& B_k,
    double Y, const vector<double>& delta_j, double P_avg,
    const vector<double>& eta_k, const vector<double>& alpha_k, double normalization_constant
) {
    auto start = high_resolution_clock::now();
    
    vector<vector<vector<int>>> x(n_machines, vector<vector<int>>(n_demands, vector<int>(n_models, 0)));
    vector<double> gamma_k(n_machines, 0.0);
    
    double elapsed_time_demands_0 = 0.0;
    double elapsed_time_c = 0.0;
    double elapsed_time_alpha_k = 0.0;
    double elapsed_time_build = 0.0;
    double elapsed_time_relaxed = 0.0;
    double elapsed_time_b_ub_sub = 0.0;
    double elapsed_time_NQ = 0.0;
    double elapsed_time_SQ = 0.0;
    double elapsed_time_integral = 0.0;

    if (n_demands > 0){
        auto start_c = high_resolution_clock::now();
        vector<double> c;
        // x_{k,i,j} terms
        for (int k = 0; k < n_machines; ++k) {
            for (int i = 0; i < n_demands; ++i) {
                for (int j = 0; j < n_models; ++j) {
                    double val = -V
                    + omega[i][j] * (Z_k[k] * 1e-3 * CI_k[k] * PUE_k[k] * e_k[k] * 100.0 / C_avg
                        + 100.0 * W_k[k] / (FLOPS_k[k] * R_avg)
                        + (Q_k[k] / normalization_constant))
                        + W_k[k] * 100.0 * sigma[i] / (B_k[k] * R_avg)
                        + 100.0 * Y * (P_avg - delta_j[j]) / (n_demands * P_avg);
                        c.push_back(val);
                    }
                }
            }
            // gamma_k terms
            for (int k = 0; k < n_machines; ++k) {
                c.push_back(eta_k[k] * 100.0 / R_avg);
            }
        auto end_c = high_resolution_clock::now();
        elapsed_time_c = duration<double>(end_c - start_c).count();
        
        auto start_alpha_k = high_resolution_clock::now();
        auto it = find(alpha_k.begin(), alpha_k.end(), 0.0);

        if (it != alpha_k.end()) {
            vector<int> k_empty;
            for (int idx = 0; idx < (int)alpha_k.size(); ++idx) {
                if (alpha_k[idx] == 0.0) k_empty.push_back(idx);
            }
            int k = k_empty[0];
            double max_flops = FLOPS_k[k];
            for (int idx : k_empty) {
                if (FLOPS_k[idx] > max_flops) {
                    max_flops = FLOPS_k[idx];
                    k = idx;
                }
            }
            for (int i = 0; i < n_demands; ++i) {
                x[k][i][n_models-1] = 1;
            }
            auto end_alpha_k = high_resolution_clock::now();
            elapsed_time_alpha_k = duration<double>(end_alpha_k - start_alpha_k).count();

        } else {

            Highs highs;
            highs.setOptionValue("output_flag", false);
            
            auto start_build = high_resolution_clock::now();
            int n_x = n_machines * n_demands * n_models;
            int n_vars = n_x + n_machines;
            int n_rows = n_demands + n_machines * n_demands * n_models;
            vector<int> Astart(n_vars + 1, 0);
            vector<int> Aindex;
            vector<double> Avalue;
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
            auto end_build = high_resolution_clock::now();
            elapsed_time_build = duration<double>(end_build - start_build).count();
            
            auto start_relaxed = high_resolution_clock::now();
            HighsModel model_relaxed;
            model_relaxed.lp_.num_col_ = n_vars;
            model_relaxed.lp_.num_row_ = n_rows;
            model_relaxed.lp_.a_matrix_.start_ = Astart;
            model_relaxed.lp_.a_matrix_.index_ = Aindex;
            model_relaxed.lp_.a_matrix_.value_ = Avalue;
            model_relaxed.lp_.col_cost_ = c;
            model_relaxed.lp_.col_lower_ = vector<double>(n_vars, 0.0);
            model_relaxed.lp_.col_upper_ = vector<double>(n_vars, 1.0);
            model_relaxed.lp_.row_lower_ = vector<double>(n_rows, -kHighsInf);
            model_relaxed.lp_.row_upper_ = b_ub;
            highs.passModel(model_relaxed);
            HighsStatus status_relaxed = highs.run();
            const vector<double>& sol_relaxed = highs.getSolution().col_value;
            for (int k = 0; k < n_machines; ++k) {
                for (int i = 0; i < n_demands; ++i) {
                    for (int j = 0; j < n_models; ++j) {
                        int idx = k * n_demands * n_models + i * n_models + j;
                        x[k][i][j] = static_cast<int>(round(sol_relaxed[idx]));
                    }
                }
            }
            auto end_relaxed = high_resolution_clock::now();
            elapsed_time_relaxed = duration<double>(end_relaxed - start_relaxed).count();

            bool has_fractional = false;
            for (int k = 0; k < n_machines && !has_fractional; ++k) {
                for (int i = 0; i < n_demands && !has_fractional; ++i) {
                    for (int j = 0; j < n_models; ++j) {
                        if (x[k][i][j] > 0 && x[k][i][j] < 1) {
                            has_fractional = true;
                            break;
                        }
                    }
                }
            }
            if (has_fractional) {
                auto start_b_ub_sub = high_resolution_clock::now();
                vector<double> b_ub_sub(b_ub.begin(), b_ub.begin() + n_demands);
                auto end_b_ub_sub = high_resolution_clock::now();
                elapsed_time_b_ub_sub = duration<double>(end_b_ub_sub - start_b_ub_sub).count();

                auto start_NQ = high_resolution_clock::now();
                vector<double> c_NQ(c.begin(), c.begin() + n_machines * n_demands * n_models);
                vector<int> Astart_NQ(n_x + 1, 0);
                vector<int> Aindex_NQ;
                vector<double> Avalue_NQ;
                for (int col = 0; col < n_x; ++col) {
                    for (int idx = Astart[col]; idx < Astart[col + 1]; ++idx) {
                        int row = Aindex[idx];
                        if (row < n_demands) {
                            Aindex_NQ.push_back(row);
                            Avalue_NQ.push_back(Avalue[idx]);
                        }
                    }
                    Astart_NQ[col + 1] = (int)Aindex_NQ.size();
                }
                HighsModel model_NQ;
                model_NQ.lp_.num_col_ = n_vars - n_machines;
                model_NQ.lp_.num_row_ = n_machines;
                model_NQ.lp_.a_matrix_.start_ = Astart_NQ;
                model_NQ.lp_.a_matrix_.index_ = Aindex_NQ;
                model_NQ.lp_.a_matrix_.value_ = Avalue_NQ;
                model_NQ.lp_.col_cost_ = c_NQ;
                model_NQ.lp_.col_lower_ = vector<double>(n_vars - n_machines, 0.0);
                model_NQ.lp_.col_upper_ = vector<double>(n_vars - n_machines, 1.0);
                model_NQ.lp_.row_lower_ = vector<double>(n_rows, -kHighsInf);
                model_NQ.lp_.row_upper_ = b_ub_sub;
                highs.passModel(model_NQ);
                HighsStatus status_NQ = highs.run();
                const vector<double>& sol_NQ = highs.getSolution().col_value;
                vector<vector<vector<int>>> x_NQ(n_machines, vector<vector<int>>(n_demands, vector<int>(n_models, 0)));
                for (int k = 0; k < n_machines; ++k) {
                    for (int i = 0; i < n_demands; ++i) {
                        for (int j = 0; j < n_models; ++j) {
                            int idx = k * n_demands * n_models + i * n_models + j;
                            x_NQ[k][i][j] = static_cast<int>(round(sol_NQ[idx]));
                        }
                    }
                }
                auto end_NQ = high_resolution_clock::now();
                elapsed_time_NQ = duration<double>(end_NQ - start_NQ).count();

                auto start_SQ = high_resolution_clock::now();
                vector<double> c_SQ = c_NQ;
                for (int k = 0; k < n_machines; ++k)
                    c_SQ.push_back(eta_k[k] * (n_demands * n_models));
                vector<int> Astart_SQ(n_vars + 1, 0);
                vector<int> Aindex_SQ;
                vector<double> Avalue_SQ;
                for (int col = 0; col < n_vars; ++col) {
                    for (int idx = Astart[col]; idx < Astart[col + 1]; ++idx) {
                        int row = Aindex[idx];
                        if (row < n_demands) {
                            Aindex_SQ.push_back(row);
                            Avalue_SQ.push_back(Avalue[idx]);
                        }
                    }
                    Astart_SQ[col + 1] = (int)Aindex_SQ.size();
                }
                HighsModel model_SQ;
                model_SQ.lp_.num_col_ = n_vars;
                model_SQ.lp_.num_row_ = n_machines;
                model_SQ.lp_.a_matrix_.start_ = Astart_SQ;
                model_SQ.lp_.a_matrix_.index_ = Aindex_SQ;
                model_SQ.lp_.a_matrix_.value_ = Avalue_SQ;
                model_SQ.lp_.col_cost_ = c_SQ;
                model_SQ.lp_.col_lower_ = vector<double>(n_vars, 0.0);
                model_SQ.lp_.col_upper_ = vector<double>(n_vars, 1.0);
                model_SQ.lp_.row_lower_ = vector<double>(n_rows, -kHighsInf);
                model_SQ.lp_.row_upper_ = b_ub_sub;
                highs.passModel(model_SQ);
                HighsStatus status_SQ = highs.run();
                const vector<double>& sol_SQ = highs.getSolution().col_value;
                vector<vector<vector<int>>> x_SQ(n_machines, vector<vector<int>>(n_demands, vector<int>(n_models, 0)));
                for (int k = 0; k < n_machines; ++k) {
                    for (int i = 0; i < n_demands; ++i) {
                        for (int j = 0; j < n_models; ++j) {
                            int idx = k * n_demands * n_models + i * n_models + j;
                            x_SQ[k][i][j] = static_cast<int>(round(sol_SQ[idx]));
                        }
                    }
                }
                auto end_SQ = high_resolution_clock::now();
                elapsed_time_SQ = duration<double>(end_SQ - start_SQ).count();

                bool any_diff = false;
                for (int k = 0; k < n_machines && !any_diff; ++k) {
                    for (int i = 0; i < n_demands && !any_diff; ++i) {
                        for (int j = 0; j < n_models; ++j) {
                            if (x_NQ[k][i][j] != x_SQ[k][i][j]) {
                                any_diff = true;
                                break;
                            }
                        }
                    }
                }
                if (any_diff) {
                    auto start_integral = high_resolution_clock::now();
                    vector<int> I0;
                    for (int i = 0; i < n_demands; ++i) {
                        int sum = 0;
                        for (int k = 0; k < n_machines; ++k)
                            for (int j = 0; j < n_models; ++j)
                                sum += x_NQ[k][i][j];
                        if (sum == 0) I0.push_back(i);
                    }
                    vector<int> I1;
                    for (int i = 0; i < n_demands; ++i) {
                        int max_val = 0;
                        for (int k = 0; k < n_machines; ++k)
                            for (int j = 0; j < n_models; ++j)
                                if (x_SQ[k][i][j] > max_val) max_val = x_SQ[k][i][j];
                        if (max_val == 1) I1.push_back(i);
                    }
                    set<int> set_I0(I0.begin(), I0.end());
                    set<int> set_I1(I1.begin(), I1.end());
                    vector<int> intersection;
                    set_intersection(set_I0.begin(), set_I0.end(), set_I1.begin(), set_I1.end(), back_inserter(intersection));
                    if (!intersection.empty()) {
                        throw runtime_error("I0 and I1 have elements in common, which is not allowed.");
                    }
                    vector<double> b_lb(n_rows, -1.0);
                    for (auto i : I0) {
                        b_ub[i] = 0.0;
                    }
                    for (auto i : I1) {
                        b_lb[i] = 1.0;
                    }
                    HighsModel model_integral;
                    model_integral.lp_.num_col_ = n_vars;
                    model_integral.lp_.num_row_ = n_rows;
                    model_integral.lp_.a_matrix_.start_ = Astart;
                    model_integral.lp_.a_matrix_.index_ = Aindex;
                    model_integral.lp_.a_matrix_.value_ = Avalue;
                    model_integral.lp_.col_cost_ = c;
                    model_integral.lp_.col_lower_ = vector<double>(n_vars, 0.0);
                    model_integral.lp_.col_upper_ = vector<double>(n_vars, 1.0);
                    model_integral.lp_.row_lower_ = b_lb;
                    model_integral.lp_.row_upper_ = b_ub;
                    model_integral.lp_.integrality_.assign(n_vars, HighsVarType::kContinuous);
                    for (int idx = 0; idx < n_vars; ++idx) {
                        model_integral.lp_.integrality_[idx] = HighsVarType::kInteger;
                    }
                    highs.passModel(model_integral);
                    HighsStatus status_integral = highs.run();
                    const vector<double>& sol_integral = highs.getSolution().col_value;
                    vector<vector<vector<double>>> x(n_machines, vector<vector<double>>(n_demands, vector<double>(n_models, 0.0)));
                    for (int k = 0; k < n_machines; ++k) {
                        for (int i = 0; i < n_demands; ++i) {
                            for (int j = 0; j < n_models; ++j) {
                                int idx = k * n_demands * n_models + i * n_models + j;
                                x[k][i][j] = static_cast<double>(round(sol_integral[idx]));
                            }
                        }
                    }
                    auto end_integral = high_resolution_clock::now();
                    elapsed_time_integral = duration<double>(end_integral - start_integral).count();

                } else {
                    x = x_SQ;
                }
            }
        }
    } else {
        auto end_demands_0 = high_resolution_clock::now();
        elapsed_time_demands_0 = duration<double>(end_demands_0 - start).count();
    }

    for (int k_idx = 0; k_idx < n_machines; ++k_idx) {
        int max_val = 0;
        for (int i = 0; i < n_demands; ++i)
            for (int j = 0; j < n_models; ++j)
                if (x[k_idx][i][j] > max_val) max_val = x[k_idx][i][j];
        gamma_k[k_idx] = max_val;
    }

    double elapsed_time = elapsed_time_demands_0 + elapsed_time_c + elapsed_time_alpha_k + elapsed_time_build + max(elapsed_time_relaxed, elapsed_time_b_ub_sub + max(elapsed_time_NQ, elapsed_time_SQ) + elapsed_time_integral);
    return Solution(x, gamma_k, elapsed_time);
}
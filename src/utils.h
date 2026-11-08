#include <vector>
#include <random>

using namespace std;

// Carbon emission model

inline vector<double> Z_k_const(const vector<int>& CI_k, const vector<double>& PUE_k, const vector<double>& e_k) {
    vector<double> result(CI_k.size());
    for (size_t k = 0; k < CI_k.size(); ++k) {
        result[k] = 1e-3 * CI_k[k] * PUE_k[k] * e_k[k];
    }
    return result;
}

inline vector<double> c_k(
    const vector<int>& CI_k,
    const vector<double>& PUE_k,
    const vector<double>& e_k,
    const vector<double>& x_at_omega) {
    vector<double> z = Z_k_const(CI_k, PUE_k, e_k);
    vector<double> result(CI_k.size());
    for (size_t k = 0; k < CI_k.size(); ++k) {
        result[k] = z[k] * x_at_omega[k];
    }
    return result;
}


// Response time model

inline vector<double> d_k_comm(
    const vector<double>& x_at_sigma,
    const vector<double>& B_k,
    const vector<double>& epsilon_k,
    const vector<int>& D_k,
    const vector<double>& const_k,
    const vector<double>& gamma_k) {
    vector<double> result(x_at_sigma.size());
    for (size_t k = 0; k < x_at_sigma.size(); ++k) {
        result[k] = x_at_sigma[k] / B_k[k] + (epsilon_k[k] * D_k[k] + const_k[k]) * gamma_k[k];
    }
    return result;
}

inline vector<double> d_k_proc(
    const vector<double>& x_at_omega,
    const vector<double>& FLOPS_k) {
    vector<double> result(x_at_omega.size());
    for (size_t k = 0; k < x_at_omega.size(); ++k) {
        result[k] = x_at_omega[k] / FLOPS_k[k];
    }
    return result;
}

inline vector<double> d_k_queue(
    const vector<double>& Q_k,
    const vector<double>& FLOPS_k,
    const vector<double>& gamma_k) {
    vector<double> result(Q_k.size());
    for (size_t k = 0; k < Q_k.size(); ++k) {
        result[k] = (Q_k[k] / FLOPS_k[k]) * gamma_k[k];
    }
    return result;
}

inline vector<double> R_k(
    const vector<double>& x_at_sigma,
    const vector<double>& B_k,
    const vector<double>& epsilon_k,
    const vector<int>& D_k,
    const vector<double>& const_k,
    const vector<double>& gamma_k,
    const vector<double>& x_at_omega,
    const vector<double>& FLOPS_k,
    const vector<double>& Q_k
) {
    auto comm = d_k_comm(x_at_sigma, B_k, epsilon_k, D_k, const_k, gamma_k);
    auto proc = d_k_proc(x_at_omega, FLOPS_k);
    auto queue = d_k_queue(Q_k, FLOPS_k, gamma_k);
    vector<double> result(comm.size());
    for (size_t k = 0; k < comm.size(); ++k) {
        result[k] = comm[k] + proc[k] + queue[k];
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    double random_err = dist(gen)
    return result + random_err;
}

vector<double> update_W_Perc(
    const vector<double>& W_k,
    const vector<double>& x_at_sigma,
    const vector<double>& B_k,
    const vector<double>& epsilon_k,
    const vector<int>& D_k,
    const vector<double>& const_k,
    const vector<double>& gamma_k,
    const vector<double>& x_at_omega,
    const vector<double>& FLOPS_k,
    const vector<double>& Q_k,
    double R_avg
) {
    vector<double> r = R_k(x_at_sigma, B_k, epsilon_k, D_k, const_k, gamma_k, x_at_omega, FLOPS_k, Q_k);
    vector<double> result(W_k.size(), 0.0);
    for (size_t i = 0; i < W_k.size(); ++i) {
        result[i] = max(r[i] * 100.0 / R_avg - 100.0 + W_k[i], 0.0);
    }
    return result;
}


// Real queue update
vector<double> update_Q(const vector<double>& Q_k, const vector<double>& FLOPS_k, const vector<double>& x_at_omega) {
    vector<double> result(Q_k.size(), 0.0);
    for (size_t i = 0; i < Q_k.size(); ++i) {
        result[i] = max(Q_k[i] + x_at_omega[i] - FLOPS_k[i], 0.0);
    }
    return result;
}

// Emission virtual queue update
vector<double> update_Z_Perc(
    const vector<double>& Z_k,
    const vector<int>& CI_k,
    const vector<double>& PUE_k,
    const vector<double>& e_k,
    const vector<double>& x_at_omega,
    double C_avg
) {
    vector<double> c = c_k(CI_k, PUE_k, e_k, x_at_omega); // c_k does not use t
    vector<double> result(Z_k.size(), 0.0);
    for (size_t i = 0; i < Z_k.size(); ++i) {
        result[i] = max(c[i] * 100.0 / C_avg - 100.0 + Z_k[i], 0.0);
    }
    return result;
}

// Performance virtual queue update
double update_Y_Perc(double Y, double x_at_delta, int n_demands, double P_avg) {
    if (n_demands == 0) {
        return Y;
    }
    return max(Y + (x_at_delta * 100.0) / (n_demands * P_avg), 0.0);
}

vector<double> alpha(
    const vector<double>& Q_k,
    const vector<double>& Z_k,
    const vector<double>& W_k,
    const vector<double>& FLOPS_k,
    const vector<int>& CI_k,
    const vector<double>& PUE_k,
    const vector<double>& e_k
) {
    vector<double> Z_k_const_ = Z_k_const(CI_k, PUE_k, e_k);
    vector<double> result(Q_k.size(), 0.0);
    for (size_t i = 0; i < Q_k.size(); ++i) {
        result[i] = Q_k[i] + Z_k[i] * Z_k_const_[i] + W_k[i] / FLOPS_k[i];
    }
    return result;
}

vector<double> eta(
    const vector<double>& W_k,
    const vector<double>& epsilon_k,
    const vector<int>& D_k,
    const vector<double>& const_k,
    const vector<double>& Q_k,
    const vector<double>& FLOPS_k
) {
    vector<double> result(W_k.size(), 0.0);
    for (size_t i = 0; i < W_k.size(); ++i) {
        result[i] = W_k[i] * (epsilon_k[i] * D_k[i] + const_k[i] + Q_k[i] / FLOPS_k[i]);
    }
    return result;
}
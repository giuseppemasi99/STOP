#include "argparse.h"
#include "utils.h"
#include "model/model.h"
#include "methods/random.h"
#include "methods/roundrobin.h"
#include "methods/stop.h"
#include "methods/Li2024.h"
#include "logger.h"
#include <iostream>
#include <string>
#include <memory>
#include <random>
#include <unordered_map>
#include <sstream>
#include <iomanip>

using namespace std;

int main(int argc, char* argv[]) {
    ArgParse parser(argc, argv);

    string method = parser.get("method");
    string model_name = parser.get("model_name");
    double V = parser.get_double("V");
    int n_machines = parser.get_int("n_machines");
    int max_demands_per_round = parser.get_int("max_demands_per_round");
    double C_avg = parser.get_double("C_avg");
    double R_avg = parser.get_double("R_avg");
    int seed = parser.get_int("seed");
    int min_demands_per_round = 0;

    bool logger_enabled = true;

    int T = 300;

    unique_ptr<Model> model;
    if (model_name == "Yolo") model = make_unique<Yolo>();
    else model = make_unique<Whisper>();

    int n_models = model->get_n_models();
    double normalization_constant = model->normalization_constant(max_demands_per_round, n_machines);

    // Real queue
    vector<double> Q_k(n_machines, 0.0);

    // Carbon emission model
    // grams of CO2 equivalent per KWh  [Italy 331, France 56, Germany 381, Spain 174]
    vector<int> CI_options = {56, 331, 381, 174};
    vector<int> CI_k(n_machines);
    mt19937 rng_CI(seed);
    uniform_int_distribution<> dist(0, CI_options.size() - 1);
    for (int i = 0; i < n_machines; ++i)
        CI_k[i] = CI_options[dist(rng_CI)];
    // grams of CO2 equivalent per KWh  [Italy 331, France 56, Germany 381, Spain 174]
    vector<double> PUE_options = {1.01, 1.1, 1.05, 1.2};
    vector<double> PUE_k(n_machines);
    mt19937 rng_PUE(seed);
    uniform_int_distribution<> dist_pue(0, PUE_options.size() - 1);
    for (int i = 0; i < n_machines; ++i)
        PUE_k[i] = PUE_options[dist_pue(rng_PUE)];
    // Nvidia 5060, 5070, 5080, 5090, A100, H100
    vector<int> TDPs_options = {145, 250, 360, 575, 250, 350};
    vector<double> FLOPSs_options = {19.18, 30.87, 56.28, 104.8, 19.49, 51.22};  // TFLOPS
    vector<int> TDP_k(n_machines);
    vector<double> FLOPS_k(n_machines);
    mt19937 rng_TDP(seed);
    uniform_int_distribution<> dist_TDP(0, TDPs_options.size() - 1);
    mt19937 rng_FLOPs(seed);
    uniform_int_distribution<> dist_FLOPs(0, FLOPSs_options.size() - 1);
    for (int i = 0; i < n_machines; ++i) {
        TDP_k[i] = TDPs_options[dist_TDP(rng_TDP)];
        FLOPS_k[i] = FLOPSs_options[dist_FLOPs(rng_FLOPs)] * 1e3;  // GigaFLOP/s
    }
    vector<double> e_k(n_machines);
    for (int i = 0; i < n_machines; ++i)
        e_k[i] = TDP_k[i] / (3600.0 * FLOPS_k[i]);
    vector<double> Z_k(n_machines, 0.0);

    // Response time model
    vector<double> B_k(n_machines, 1000.0 * 1e6); // byte/s
    vector<double> epsilon_k(n_machines);
    mt19937 rng_epsilon(seed);
    uniform_int_distribution<> dist_epsilon(1, 2);
    for (int i = 0; i < n_machines; ++i)
        epsilon_k[i] = dist_epsilon(rng_epsilon) * 1e-6;
    vector<int> D_k(n_machines);
    mt19937 rng_D(seed);
    uniform_int_distribution<> dist_D(100, 4999);
    for (int i = 0; i < n_machines; ++i)
        D_k[i] = dist_D(rng_D);
    vector<double> const_k(n_machines);
    mt19937 rng_const(seed);
    uniform_int_distribution<> dist_const(10, 99);
    for (int i = 0; i < n_machines; ++i)
        const_k[i] = dist_const(rng_const) * 1e-3; // seconds
    vector<double> W_k(n_machines, 0.0);

    double Y = 0.0;
    vector<double> delta_j_original = model->get_delta_j_original();
    double P_avg = (method.find("STOP") != string::npos) ? 50.0 : model->get_P_avg();

    unordered_map<string, nlohmann::json> params;
    params["seed"] = seed;
    params["method"] = method;
    params["model_name"] = model_name;
    params["V"] = V;
    params["n_machines"] = n_machines;
    params["min_demands_per_round"] = min_demands_per_round;
    params["max_demands_per_round"] = max_demands_per_round;
    params["CI_k"] = CI_k;
    params["PUE_k"] = PUE_k;
    params["TDP_k"] = TDP_k;
    params["FLOPS_k"] = FLOPS_k;
    params["C_avg"] = C_avg;
    params["B_k"] = B_k;
    params["epsilon_k"] = epsilon_k;
    params["D_k"] = D_k;
    params["const_k"] = const_k;
    params["R_avg"] = R_avg;

    ostringstream oss;
    oss << "results_new/" << model_name << "/" << method;
    if (method != "Random" && method != "RoundRobin" && method != "Li2024-AllDemandsExecuted")
        oss << "-V=" << V;
    oss << "/n_m=" << n_machines << "-max_d=" << max_demands_per_round;
    if (method != "Random" && method != "RoundRobin")
        oss << "/C_avg=" << setprecision(8) << C_avg << "-R_avg=" << setprecision(8) << R_avg;
    string folder_name = oss.str();

    Logger logger(folder_name, logger_enabled);
    logger.save_dict(params, "config.json");

    vector<double> delta_j = model->get_delta_j(method);

    // Li2024
    vector<double> k_t(2 * n_machines + 1, 0.0);

    mt19937 rng_demands(seed);
    mt19937 rng_sizes(seed);

    mt19937 rng_random(seed);

    vector<vector<vector<int>>> x;
    vector<double> gamma_k;
    double elapsed_time = 0.0;

    for (int t = 0; t < T; ++t) {
        uniform_int_distribution<> dist(min_demands_per_round, max_demands_per_round - 1);
        int n_demands = dist(rng_demands);
        cout << "Round " << t + 1 << ": " << n_demands << " demands" << endl;
        
        vector<int> sizes = model->get_sizes(n_demands, rng_sizes);
        vector<double> sigma = model->get_sigma(sizes);
        vector<vector<double>> omega = model->get_omega(sizes);

        if (method == "Random")
            tie(x, gamma_k, elapsed_time) = getRandomSolution(n_machines, n_demands, n_models, rng_random);
        else if (method == "RoundRobin")
            tie(x, gamma_k, elapsed_time) = getRRSolution(n_machines, n_demands, n_models);
        else if (method.find("STOP") != string::npos) {
            vector<double> alpha_k = alpha(Q_k, Z_k, W_k, FLOPS_k, CI_k, PUE_k, e_k);
            vector<double> eta_k = eta(W_k, epsilon_k, D_k, const_k, Q_k, FLOPS_k);
            tie(x, gamma_k, elapsed_time) = getSTOPSolution(
                n_machines, n_demands, n_models, V, omega, Z_k, CI_k, PUE_k, e_k, C_avg, W_k, FLOPS_k, 
                R_avg, Q_k, sigma, B_k, Y, delta_j, P_avg, eta_k, alpha_k, normalization_constant    
            );
        }
        else if (method.find("Li2024") != string::npos) {
            vector<double> Z_k_const_ = Z_k_const(CI_k, PUE_k, e_k);
            tie(x, gamma_k, elapsed_time) = getLi2024Solution(
                method, V, k_t, n_machines, n_models, n_demands, FLOPS_k, Z_k_const_, 
                omega, sigma, B_k, P_avg, delta_j, Q_k, epsilon_k, D_k, const_k, C_avg, R_avg
            );
        }
        else
            throw runtime_error("Unknown method: " + method);

        vector<double> x_at_omega(n_machines, 0.0);
        for (int m = 0; m < n_machines; ++m) {
            double sum = 0.0;
            for (int d = 0; d < n_demands; ++d) {
                for (int mod = 0; mod < n_models; ++mod) {
                    sum += x[m][d][mod] * omega[d][mod];
                }
            }
            x_at_omega[m] = round(sum);
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

        vector<double> emission_k = c_k(CI_k, PUE_k, e_k, x_at_omega);  // gCO2
        vector<double> response_time_k = R_k(x_at_sigma, B_k, epsilon_k, D_k, const_k, gamma_k, x_at_omega, FLOPS_k, Q_k);  // s
        double performance;
        int total_assignments = 0;
        double weighted_sum = 0.0;
        int non_zero_count = 0;
        for (int m = 0; m < n_machines; ++m) {
            for (int d = 0; d < n_demands; ++d) {
                for (int mod = 0; mod < n_models; ++mod) {
                    total_assignments += x[m][d][mod];
                    if (x[m][d][mod] != 0) {
                        weighted_sum += x[m][d][mod] * delta_j_original[mod];
                        non_zero_count += x[m][d][mod];
                    }
                }
            }
        }
        if (total_assignments == 0) {
            performance = P_avg;
        } else {
            performance = weighted_sum / non_zero_count;
        }

        vector<int> throughput(n_models, 0);
        if (total_assignments == 0) {
            throughput = vector<int>(n_models, 0);
        } else {
            for (int m = 0; m < n_machines; ++m) {
                for (int d = 0; d < n_demands; ++d) {
                    for (int mod = 0; mod < n_models; ++mod) {
                        throughput[mod] += x[m][d][mod];
                    }
                }
            }
        }

        if (method.find("STOP") != string::npos) {
            Z_k = update_Z_Perc(Z_k, CI_k, PUE_k, e_k, x_at_omega, C_avg);
            W_k = update_W_Perc(W_k, x_at_sigma, B_k, epsilon_k, D_k, const_k, gamma_k, x_at_omega, FLOPS_k, Q_k, R_avg);
            Y = update_Y_Perc(Y, x_at_delta, n_demands, P_avg);
        }
        Q_k = update_Q(Q_k, FLOPS_k, x_at_omega);


        // Store round results in an unordered_map (using nlohmann::json for heterogeneous types)
        unordered_map<string, nlohmann::json> round_result = {
            {"t", t},
            {"arrivals", n_demands},
            {"throughput", throughput},
            {"elapsed_time", elapsed_time},
            {"emission_k", emission_k},
            {"response_time_k", response_time_k},
            {"performance", performance},
            {"Q_k", Q_k},
            {"Z_k", Z_k},
            {"W_k", W_k},
            {"Y", Y}
        };
        logger.save_dict(round_result, "t=" + to_string(t) + ".json");

    }

    return 0;

}
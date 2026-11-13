#include "argparse.h"
#include "utils.h"
#include "model/model.h"
#include "model/performance.h"
#include "methods/random.h"
#include "methods/roundrobin.h"
#include "methods/stop.h"
#include "methods/Li2024.h"
#include "logger.h"
#include "gt_utils.h"
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
    string demands_json = parser.get("demands_json");
    double V = parser.get_double("V");
    int n_machines = parser.get_int("n_machines");
    int max_demands_per_round = parser.get_int("max_demands_per_round");
    double C_avg = parser.get_double("C_avg");
    double R_avg = parser.get_double("R_avg");
    int seed = parser.get_int("seed");
    int min_demands_per_round = parser.get_int("min_demands_per_round", 10);

    bool logger_enabled = true;

    int T = 300;

    unique_ptr<Model> model;
    if (model_name == "Yolo") model = make_unique<Yolo>();
    else if (model_name == "Whisper") {
        model = make_unique<Whisper>();
    } else if (model_name == "Whisper3") {
        model = make_unique<WhisperThreeModels>();
    } else {
        throw runtime_error("Unknown model: " + model_name);
    }

    unique_ptr<DemandDataset> demand_dataset;
    if (!demands_json.empty()) {
        demand_dataset = make_unique<DemandDataset>(demands_json, model_name);
    }
    if (n_machines <= 0 || min_demands_per_round < 0 || max_demands_per_round <= min_demands_per_round)
        throw invalid_argument("Require n_machines > 0 and 0 <= min_demands_per_round < max_demands_per_round");

    int n_models = model->get_n_models();
    double normalization_constant = model->normalization_constant(max_demands_per_round, n_machines);

    normalization_constant *= (min_demands_per_round + max_demands_per_round - 1.0) / max_demands_per_round;
    if (demand_dataset) {
        // Use the same model weighting as each synthetic workload's normalization.
        double mean_cost = 0.0;
        if (model_name == "Yolo") {
            mean_cost = demand_dataset->mean_gflops(1);
        } else {
            for (int j = 0; j < n_models; ++j)
                mean_cost += demand_dataset->mean_gflops(j) / n_models;
        }
        normalization_constant = mean_cost * ((min_demands_per_round + max_demands_per_round - 1.0) / 2.0) / n_machines;
    }

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
    double P_avg_original = model->get_P_avg();
    const bool scaled_performance = method.find("STOP") != string::npos;
    if (demand_dataset) {
        delta_j_original = demand_dataset->mean_performance();
        P_avg_original = accumulate(delta_j_original.begin(), delta_j_original.end(), 0.0) / n_models;
    }
    double P_avg = scaled_performance ? model->scale_performance(P_avg_original) : P_avg_original;
    // The extrapolated target may be zero or negative. Keep a fixed positive
    // normalization so its sign never reverses the accuracy feedback.
    const double accuracy_normalization = 50.0;
    vector<double> delta_j = delta_j_original;
    if (scaled_performance)
        for (auto& score : delta_j) score = model->scale_performance(score);

    unordered_map<string, nlohmann::json> params;
    params["seed"] = seed;
    params["method"] = method;
    params["model_name"] = model_name;
    if (demand_dataset) params["demands_json"] = demands_json;
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
    params["delta_j_original"] = delta_j_original;
    params["delta_j"] = delta_j;
    params["P_avg_original"] = P_avg_original;
    params["P_avg"] = P_avg;
    params["accuracy_normalization"] = accuracy_normalization;

    ostringstream oss;
    oss << "results/" << model_name << "/" << method;
    if (method != "Random" && method != "RoundRobin" && method != "Li2024-AllDemandsExecuted")
        oss << "-V=" << V;
    oss << "/n_m=" << n_machines << "-max_d=" << max_demands_per_round;
    if (method != "Random" && method != "RoundRobin")
        oss << "/C_avg=" << setprecision(8) << C_avg << "-R_avg=" << setprecision(8) << R_avg;
    oss << "/id=" << seed;
    string folder_name = oss.str();

    Logger logger(folder_name, logger_enabled);
    logger.save_dict(params, "config.json");


    // Li2024
    vector<double> k_t(2 * n_machines + 1, 0.0);

    mt19937 rng_demands(seed);
    mt19937 rng_sizes(seed);

    mt19937 rng_random(seed);

    vector<vector<vector<int>>> x;
    vector<double> gamma_k;
    double elapsed_time = 0.0;

    string model_status_string = "None";

    for (int t = 0; t < T; ++t) {

        uniform_int_distribution<> dist(min_demands_per_round, max_demands_per_round - 1);
        int n_demands = dist(rng_demands);
        cout << "Round " << t + 1 << ": " << n_demands << " demands" << endl;
        
        vector<Demand> demands;
        vector<int> sizes;
        vector<double> sigma;
        vector<vector<double>> omega;
        if (demand_dataset) {
            demands = demand_dataset->sample(n_demands, rng_sizes);
            sizes = model->get_sizes(demands);
            sigma = model->get_sigma(demands);
            omega = model->get_omega(demands);
        } else {
            sizes = model->get_sizes(n_demands, rng_sizes);
            sigma = model->get_sigma(sizes);
            omega = model->get_omega(sizes);
        }

        if (method == "Random")
            tie(x, gamma_k, elapsed_time) = getRandomSolution(n_machines, n_demands, n_models, rng_random);
        else if (method == "RoundRobin")
            tie(x, gamma_k, elapsed_time) = getRRSolution(n_machines, n_demands, n_models);
        else if (method.find("STOP") != string::npos) {
            vector<double> alpha_k = alpha(Q_k, Z_k, W_k, FLOPS_k, CI_k, PUE_k, e_k);
            vector<double> eta_k = eta(W_k, epsilon_k, D_k, const_k, Q_k, FLOPS_k);
            tie(x, gamma_k, elapsed_time) = getSTOPSolution(
                n_machines, n_demands, n_models, V, omega, Z_k, CI_k, PUE_k, e_k, C_avg, W_k, FLOPS_k, 
                R_avg, Q_k, sigma, B_k, Y, delta_j, P_avg, eta_k, alpha_k, normalization_constant, accuracy_normalization
            );
        }
        else if (method.find("Li2024") != string::npos) {
            vector<double> Z_k_const_ = Z_k_const(CI_k, PUE_k, e_k);
            tie(x, gamma_k, elapsed_time, model_status_string) = getLi2024Solution(
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
        vector<int> throughput(n_models, 0);
        vector<double> score_sums(n_models, 0.0);
        for (int m = 0; m < n_machines; ++m) {
            for (int d = 0; d < n_demands; ++d) {
                for (int mod = 0; mod < n_models; ++mod) {
                    int assigned = x[m][d][mod];
                    if (assigned == 0) continue;
                    double score = demand_dataset ? demands[d].performance[mod] : delta_j_original[mod];
                    throughput[mod] += assigned;
                    score_sums[mod] += assigned * score;
                }
            }
        }
        const auto feedback = performance_feedback(*model, throughput, score_sums, P_avg, scaled_performance);
        double x_at_delta = feedback.deficit;
        double performance = feedback.overall;

        vector<double> emission_k = c_k(CI_k, PUE_k, e_k, x_at_omega);  // gCO2
        vector<double> response_time_k = R_k(x_at_sigma, B_k, epsilon_k, D_k, const_k, gamma_k, x_at_omega, FLOPS_k, Q_k);  // s

        if (method.find("STOP") != string::npos) {
            // read B_k's 
            B_k = query_residuals("../residuals.json", t, n_machines);
            Z_k = update_Z_Perc(Z_k, CI_k, PUE_k, e_k, x_at_omega, C_avg);
            W_k = update_W_Perc(W_k, x_at_sigma, B_k, epsilon_k, D_k, const_k, gamma_k, x_at_omega, FLOPS_k, Q_k, R_avg);
            Y = update_Y_Perc(Y, x_at_delta, n_demands, accuracy_normalization);
        }
        Q_k = update_Q(Q_k, FLOPS_k, x_at_omega);

        unordered_map<string, nlohmann::json> round_result = {
            {"t", t},
            {"arrivals", n_demands},
            {"throughput", throughput},
            {"elapsed_time", elapsed_time},
            {"emission_k", emission_k},
            {"response_time_k", response_time_k},
            {"performance", performance},
            {"performance_j", feedback.per_model},
            {"x_at_delta", x_at_delta},
            {"Q_k", Q_k},
            {"Z_k", Z_k},
            {"W_k", W_k},
            {"Y", Y},
            {"status", model_status_string}

        };
        logger.save_dict(round_result, "t=" + to_string(t) + ".json");

    }

    return 0;

}
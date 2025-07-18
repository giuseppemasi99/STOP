#pragma once
#include <string>
#include <array>
#include <random>


using namespace std;

class Model {
protected:
    double polyval(const array<double, 3>& coeffs, double x) const {
            return coeffs[0]*x*x + coeffs[1]*x + coeffs[2];
        }
public:
    virtual ~Model() = default;
    virtual string name() const { return "Model"; }
    virtual int get_n_models() const { return 0; }
    virtual double normalization_constant(int max_demands_per_round, int n_machines) const {return 0.0;}
    virtual vector<double> get_delta_j_original() const { return {}; }
    virtual vector<double> get_delta_j(string& method) const { return {}; }
    virtual double get_P_avg() const { return 0.0; }
    virtual vector<int> get_sizes(int n_demands, mt19937& rng) const { return {}; }
    virtual vector<double> get_sigma(const vector<int>& sizes) const { return {}; }
    virtual vector<vector<double>> get_omega(const vector<int>& sizes) const { return {}; }

};

class Yolo : public Model {
private:
    int n_pixels_min = 279552;
    int n_pixels_max = 15925248;
    const array<double, 3> p_nano   = {1.8501e-13, 8.25e-09, 1e-09};
    const array<double, 3> p_medium = {3.7003e-13, 1.4e-08, 1e-09};
    const array<double, 3> p_large  = {7.4005e-13, 2.2e-08, 1e-09};
    vector<double> delta_j = {85, 90, 95};
    double P_avg = 90.0;

public:
    string name() const override { return "Yolo"; }
    int get_n_models() const override { return 3; }
    vector<double> get_delta_j_original() const override {return delta_j;}
    double get_P_avg() const override {return P_avg;}
    double normalization_constant(int max_demands_per_round, int n_machines) const override{
        double mean_pixels = (n_pixels_min + n_pixels_max) / 2.0;
        double mean_operations_per_demand = polyval(p_medium, mean_pixels);
        double mean_arrivals = ((0.0 + max_demands_per_round) / 2.0) / n_machines;
        return mean_operations_per_demand * mean_arrivals;
    }
    vector<double> get_delta_j(string& method) const override {
        if (method.find("STOP") != string::npos) {
            vector<double> new_delta_j;
            for (double j : delta_j) {
                new_delta_j.push_back(0.02 * (j * j) + 6.3 * j - 679);
            }
            return new_delta_j;
        }
        return delta_j;
    }
    vector<int> get_sizes(int n_demands, mt19937& rng) const override {
        vector<int> sizes(n_demands);
        uniform_int_distribution<> dist(n_pixels_min, n_pixels_max);
        for (int i = 0; i < n_demands; ++i) {
            sizes[i] = dist(rng);
        }
        return sizes;
    }
    vector<double> get_sigma(const vector<int>& sizes) const override {
        // File Size in bytes = (pixel dimensions x bit depth) / 8
        vector<double> sigma(sizes.size());
        for (size_t i = 0; i < sizes.size(); ++i) {
            sigma[i] = (sizes[i] * 24) / 8;
        }
        return sigma;
    }
    vector<vector<double>> get_omega(const vector<int>& sizes) const override {
        vector<vector<double>> omega(sizes.size(), vector<double>(3));
        for (size_t i = 0; i < sizes.size(); ++i) {
            omega[i][0] = polyval(p_nano, sizes[i]);
            omega[i][1] = polyval(p_medium, sizes[i]);
            omega[i][2] = polyval(p_large, sizes[i]);
        }
        return omega;
    }

};

class Whisper : public Model {
private:
    double mean_seconds = 60.0;
    vector<double> delta_j = {93, 94, 95, 96};
    double P_avg = 94.5;
    double flops_tiny = 54.888938496;
    double flops_small = 141.163216896;
    double flops_medium = 732.594597888;
    double flops_large = 2500.971347968;

public:
    string name() const override { return "Whisper"; }
    int get_n_models() const override { return 4; }
    vector<double> get_delta_j_original() const override {return delta_j;}
    double get_P_avg() const override {return P_avg;}
    double normalization_constant(int max_demands_per_round, int n_machines) const override{
        double mean_operations_per_demand = (flops_tiny + flops_small + flops_medium + flops_large) / 4.0;
        double mean_arrivals = ((0.0 + max_demands_per_round) / 2.0) / n_machines;
        return mean_operations_per_demand * mean_arrivals;
    }
    vector<double> get_delta_j(string& method) const override {
        if (method.find("STOP") != string::npos) {
            vector<double> new_delta_j;
            new_delta_j.push_back(1);
            new_delta_j.push_back(33);
            new_delta_j.push_back(66);
            new_delta_j.push_back(100);
            return new_delta_j;
        }
        return delta_j;
    }
    vector<int> get_sizes(int n_demands, mt19937& rng) const override {
        vector<int> sizes(n_demands);
        cauchy_distribution<double> dist(mean_seconds, 30.0);
        for (int i = 0; i < n_demands; ++i) {
            int sample;
            do {
                sample = static_cast<int>(round(dist(rng)));
            } while (sample < 1 || sample > 300);
            sizes[i] = sample;
        }
        return sizes;
    }
    vector<double> get_sigma(const vector<int>& sizes) const override {
        // File Size in bytes = (audio length in seconds x sample rate x bit depth x channels) / 8
        vector<double> sigma(sizes.size());
        for (size_t i = 0; i < sizes.size(); ++i) {
            sigma[i] = (sizes[i] * 16000 * 16 * 1) / 8 + sizes[i]*2.5*6;
        }
        return sigma;
    }
    vector<vector<double>> get_omega(const vector<int>& sizes) const override {
        vector<vector<double>> omega(sizes.size(), vector<double>(4));
        for (size_t i = 0; i < sizes.size(); ++i) {
            omega[i][0] = (sizes[i] / 30 + 1) * flops_tiny;
            omega[i][1] = (sizes[i] / 30 + 1) * flops_small;
            omega[i][2] = (sizes[i] / 30 + 1) * flops_medium;
            omega[i][3] = (sizes[i] / 30 + 1) * flops_large;
        }
        return omega;
    }


};
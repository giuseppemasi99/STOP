#pragma once
#include <vector>

using namespace std;

struct Solution {
    vector<vector<vector<int>>> x;
    vector<double> gamma_k;
    double elapsed_time;

    Solution(
        const vector<vector<vector<int>>>& x_,
        const vector<double>& gamma_k_,
        double elapsed_time_
    ) : x(x_), gamma_k(gamma_k_), elapsed_time(elapsed_time_) {}
};
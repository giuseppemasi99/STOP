#pragma once
#include "model.h"
#include <numeric>

struct PerformanceFeedback {
    vector<int> counts;
    vector<double> per_model;
    double overall;
    double deficit;
};

// Aggregate assigned outcomes before scaling, skipping models with no assignments.
inline PerformanceFeedback performance_feedback(
    const Model& model, const vector<int>& counts, const vector<double>& score_sums,
    double target, bool scaled) {
    if (counts.size() != static_cast<size_t>(model.get_n_models()) || counts.size() != score_sums.size())
        throw invalid_argument("Performance totals do not match the selected model");
    PerformanceFeedback result{counts, vector<double>(counts.size(), numeric_limits<double>::quiet_NaN()),
                               numeric_limits<double>::quiet_NaN(), 0.0};
    int served = 0;
    double total = 0.0;
    for (size_t j = 0; j < counts.size(); ++j) {
        if (counts[j] == 0) continue;
        result.per_model[j] = score_sums[j] / counts[j];
        double score = scaled ? model.scale_performance(result.per_model[j]) : result.per_model[j];
        result.deficit += counts[j] * (target - score);
        served += counts[j];
        total += score_sums[j];
    }
    if (served > 0) result.overall = total / served;
    return result;
}

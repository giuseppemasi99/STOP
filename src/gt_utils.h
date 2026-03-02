#pragma once

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "third_party/json.hpp" 
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct CountryResidual {
    std::string country;
    double      residual_gbps;
};

// ---------------------------------------------------------------------------
// Core function
// ---------------------------------------------------------------------------

/**
 * Reads residuals.json and returns the first n_machines entries for time step t.
 *
 * @param json_path     Path to residuals.json produced by the Python script.
 * @param t             Time-step index (0-based integer, stored as a string key in JSON).
 * @param n_machines    How many entries to return.
 * @return              Vector of CountryResidual, length min(n_machines, available_entries).
 *                      Entries are in the natural iteration order of the JSON object
 *                      (alphabetical, as produced by json.dump(..., sort_keys=True)).
 */
inline std::vector<CountryResidual> query_residuals(
    const std::string& json_path,
    int                t,
    int                n_machines)
{
    // --- 1. Open file ---
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + json_path);
    }

    // --- 2. Parse ---
    json root;
    try {
        ifs >> root;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("JSON parse error: ") + e.what());
    }

    // --- 3. Look up time step key (integer t -> string key "t") ---
    const std::string step_key = std::to_string(t);
    if (!root.contains(step_key)) {
        throw std::out_of_range(
            "Time step " + step_key + " not found in " + json_path +
            ". Available steps: 0 - " + std::to_string(root.size() - 1));
    }

    const json& step_obj = root.at(step_key);
    if (!step_obj.is_object()) {
        throw std::runtime_error(
            "Expected a JSON object at step key \"" + step_key + "\".");
    }

    // --- 4. Collect entries ---
    std::vector<CountryResidual> result;
    result.reserve(std::min(n_machines, static_cast<int>(step_obj.size())));

    int count = 0;
    for (auto it = step_obj.begin(); it != step_obj.end() && count < n_machines; ++it, ++count) {
        if (!it.value().is_number()) {
            throw std::runtime_error(
                "Expected a numeric value for key \"" + it.key() + "\" at step " + step_key);
        }
        result.push_back({ it.key(), it.value().get<double>() });
    }

    return result;
}
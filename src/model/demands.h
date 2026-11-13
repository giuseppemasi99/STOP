#pragma once

#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Size is pixels for YOLO and rounded-up seconds for Whisper.
// Costs follow the selected model family's get_omega ordering.
struct Demand {
    int size;
    double input_bytes;
    std::vector<double> gflops;
    std::vector<double> performance; // Realized scores on a 0..100 scale.
};

class DemandDataset {
    std::vector<Demand> demands_;

    static double positive_number(const nlohmann::json& value) {
        if (!value.is_number())
            throw std::runtime_error("expected a positive number");
        double number = value.get<double>();
        if (!std::isfinite(number) || number <= 0.0)
            throw std::runtime_error("expected a finite positive number");
        return number;
    }

    static double realized_performance(const nlohmann::json& record,
                                       const nlohmann::json& result) {
        if (result.contains("performance")) {
            const auto& value = result.at("performance");
            if (!value.is_number()) throw std::runtime_error("performance must be numeric");
            double score = value.get<double>();
            if (!std::isfinite(score) || score < 0.0 || score > 100.0)
                throw std::runtime_error("performance must be finite and between 0 and 100");
            return score;
        }
        if (record.contains("ground_truth_label") && result.contains("predicted_label")) {
            if (!record.at("ground_truth_label").is_string() || !result.at("predicted_label").is_string())
                throw std::runtime_error("ground_truth_label and predicted_label must be strings");
            return record.at("ground_truth_label") == result.at("predicted_label") ? 100.0 : 0.0;
        }
        if (result.contains("correct") && result.at("correct").is_boolean())
            return result.at("correct").get<bool>() ? 100.0 : 0.0;
        throw std::runtime_error("each model needs performance, labels, or boolean correct");
    }

public:
    explicit DemandDataset(const std::string& path, const std::string& model_name = "Yolo") {
        std::vector<std::string> model_names;
        if (model_name == "Yolo") model_names = {"nano", "medium", "large"};
        else if (model_name == "Whisper") model_names = {"tiny", "small", "medium", "large"};
        else if (model_name == "Whisper3") model_names = {"tiny", "small", "medium"};
        else throw std::invalid_argument("Unsupported demand model: " + model_name);
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("Cannot open demands JSON: " + path);
        nlohmann::json records;
        try {
            input >> records;
        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error("Invalid demands JSON " + path + ": " + e.what());
        }
        if (!records.is_array() || records.empty())
            throw std::runtime_error("Demands JSON must be a non-empty array: " + path);

        for (std::size_t i = 0; i < records.size(); ++i) {
            try {
                const auto& record = records.at(i);
                double size;
                if (model_name == "Yolo") {
                    double width = positive_number(record.at("width"));
                    double height = positive_number(record.at("height"));
                    size = width * height;
                    if (std::floor(width) != width || std::floor(height) != height ||
                        size > std::numeric_limits<int>::max())
                        throw std::runtime_error("width and height must be integers with a pixel count fitting in int");
                } else {
                    // Preserve the existing integer-seconds get_sizes interface.
                    // Byte counts and GFLOPs still use the exact supplied measurements.
                    size = std::ceil(positive_number(record.at("duration_seconds")));
                    if (size > std::numeric_limits<int>::max())
                        throw std::runtime_error("duration_seconds exceeds the supported size range");
                }
                Demand demand{static_cast<int>(size),
                              positive_number(record.at("input_bytes")), {}, {}};
                for (const auto& name : model_names) {
                    const auto& result = record.at("models").at(name);
                    demand.gflops.push_back(positive_number(result.at("gflops")));
                    demand.performance.push_back(realized_performance(record, result));
                }
                demands_.push_back(std::move(demand));
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid demand at index " + std::to_string(i) +
                                         " in " + path + ": " + e.what());
            }
        }
    }

    // Each draw is independent: records can recur within and across timesteps.
    std::vector<Demand> sample(int count, std::mt19937& rng) const {
        if (count < 0)
            throw std::invalid_argument("Demand count must be non-negative");
        std::uniform_int_distribution<std::size_t> draw(0, demands_.size() - 1);
        std::vector<Demand> sampled;
        sampled.reserve(count);
        for (int i = 0; i < count; ++i)
            sampled.push_back(demands_[draw(rng)]);
        return sampled;
    }

    std::vector<double> mean_performance() const {
        std::vector<double> means(demands_.front().performance.size(), 0.0);
        for (std::size_t i = 0; i < demands_.size(); ++i)
            for (std::size_t j = 0; j < means.size(); ++j)
                means[j] += (demands_[i].performance[j] - means[j]) / (i + 1);
        return means;
    }

    double mean_gflops(std::size_t model_index) const {
        double mean = 0.0;
        for (std::size_t i = 0; i < demands_.size(); ++i)
            mean += (demands_[i].gflops.at(model_index) - mean) / (i + 1);
        return mean;
    }
};

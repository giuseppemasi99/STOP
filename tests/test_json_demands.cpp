#include "model/model.h"
#include "model/performance.h"
#include <chrono>
#include <filesystem>
#include <iostream>

using nlohmann::json;

static void require(bool condition, const string& message) {
    if (!condition) throw runtime_error(message);
}

struct Fixture {
    filesystem::path path = filesystem::temp_directory_path() /
        ("stop-demands-test-" + to_string(chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    void write(const json& value) const { ofstream(path) << value.dump(); }
    ~Fixture() { error_code error; filesystem::remove(path, error); }
};

int main() {
    try {
        Fixture fixture;
        json first = {
            {"width", 10}, {"height", 20}, {"input_bytes", 123},
            {"models", {{"nano", {{"gflops", 1}}},
                        {"medium", {{"gflops", 2}}},
                        {"large", {{"gflops", 3}}}}},
            {"correct", true}
        };
        for (auto& result : first["models"]) result["correct"] = true;
        json second = first;
        second["width"] = 30;
        second["input_bytes"] = 456;
        second["models"]["medium"]["gflops"] = 8;
        json records = json::array({first, second});
        fixture.write(records);
        DemandDataset dataset(fixture.path.string());
        require(dataset.mean_gflops(1) == 5.0, "empirical normalization cost");
        mt19937 rng(17), repeat_rng(17);
        auto batch = dataset.sample(100, rng);
        auto repeated = dataset.sample(100, repeat_rng);
        Yolo model;
        auto sizes = model.get_sizes(batch);
        auto sigma = model.get_sigma(batch);
        auto omega = model.get_omega(batch);
        bool saw_first = false, saw_second = false;
        for (size_t i = 0; i < batch.size(); ++i) {
            require(batch[i].size == repeated[i].size, "seed reproducibility");
            bool is_first = sizes[i] == 200;
            saw_first |= is_first;
            saw_second |= !is_first;
            require(sizes[i] == (is_first ? 200 : 600), "sampled pixels");
            require(sigma[i] == (is_first ? 123 : 456), "bytes stay aligned with pixels");
            require(omega[i] == vector<double>({1, is_first ? 2.0 : 8.0, 3}), "model costs stay aligned");
        }
        require(saw_first && saw_second, "both records sampled");
        require(batch.size() == 100, "sampling with replacement exceeds dataset size");
        auto empty = dataset.sample(0, rng);
        require(model.get_sizes(empty).empty() && model.get_sigma(empty).empty() &&
                model.get_omega(empty).empty(), "zero-demand round");
        fixture.write(json::array({first}));
        DemandDataset singleton(fixture.path.string());
        for (const auto& demand : singleton.sample(10, rng))
            require(demand.size == 200, "singleton replacement");

        auto rejects = [&](const json& invalid) {
            fixture.write(invalid);
            try { DemandDataset bad(fixture.path.string()); }
            catch (const exception&) { return; }
            throw runtime_error("invalid input accepted");
        };
        rejects(json::array());
        rejects(json::object());
        for (const string key : {"width", "height", "input_bytes", "models"}) {
            json bad = first;
            bad.erase(key);
            rejects(json::array({bad}));
        }
        for (json value : {json(-1), json(0), json("10"), json(nullptr)}) {
            json bad = first;
            bad["width"] = value;
            rejects(json::array({bad}));
        }
        json bad = first;
        bad["width"] = 1.5;
        rejects(json::array({bad}));
        bad = first;
        bad["width"] = 2147483647;
        rejects(json::array({bad}));
        bad = first;
        bad["models"].erase("large");
        rejects(json::array({bad}));
        bad = first;
        bad["models"]["nano"]["gflops"] = -1;
        rejects(json::array({bad}));
        bad = first;
        bad["input_bytes"] = 0;
        rejects(json::array({bad}));
        bool negative_rejected = false;
        try { dataset.sample(-1, rng); }
        catch (const invalid_argument&) { negative_rejected = true; }
        require(negative_rejected, "negative demand count");

        // Synthetic accessors remain usable through the original interface.
        Model& base = model;
        auto synthetic_sizes = base.get_sizes(3, rng);
        require(base.get_sigma(synthetic_sizes).size() == 3 &&
                base.get_omega(synthetic_sizes).size() == 3, "synthetic compatibility");
        json audio = {
            {"duration_seconds", 12.25}, {"input_bytes", 392000},
            {"models", {{"tiny", {{"gflops", 10}}},
                        {"small", {{"gflops", 20}}},
                        {"medium", {{"gflops", 30}}},
                        {"large", {{"gflops", 40}}}}}
        };
        for (auto& result : audio["models"]) result["correct"] = true;
        json audio_second = audio;
        audio_second["duration_seconds"] = 65;
        audio_second["input_bytes"] = 2080000;
        for (const string name : {"tiny", "small", "medium", "large"})
            audio_second["models"][name]["gflops"] =
                audio["models"][name]["gflops"].get<double>() * 3;
        fixture.write(json::array({audio, audio_second}));
        DemandDataset whisper_dataset(fixture.path.string(), "Whisper");
        Whisper whisper;
        mt19937 audio_rng(17), audio_repeat_rng(17);
        auto audio_batch = whisper_dataset.sample(100, audio_rng);
        auto audio_repeat = whisper_dataset.sample(100, audio_repeat_rng);
        auto durations = whisper.get_sizes(audio_batch);
        auto audio_sigma = whisper.get_sigma(audio_batch);
        auto audio_omega = whisper.get_omega(audio_batch);
        bool saw_short = false, saw_long = false;
        for (size_t i = 0; i < audio_batch.size(); ++i) {
            bool short_audio = durations[i] == 13;
            saw_short |= short_audio;
            saw_long |= !short_audio;
            require(durations[i] == (short_audio ? 13 : 65), "audio duration rounds up");
            require(audio_batch[i].size == audio_repeat[i].size, "audio seed reproducibility");
            require(audio_sigma[i] == (short_audio ? 392000 : 2080000), "audio bytes alignment");
            require(audio_omega[i] == (short_audio ? vector<double>{10, 20, 30, 40} :
                                                    vector<double>{30, 60, 90, 120}),
                    "four Whisper costs stay ordered and aligned");
        }
        require(saw_short && saw_long, "both audio records sampled with replacement");
        require(whisper_dataset.mean_gflops(3) == 80, "Whisper empirical model mean");
        auto audio_empty = whisper_dataset.sample(0, audio_rng);
        require(whisper.get_sizes(audio_empty).empty() && whisper.get_sigma(audio_empty).empty() &&
                whisper.get_omega(audio_empty).empty(), "Whisper zero-demand round");
        DemandDataset whisper3_dataset(fixture.path.string(), "Whisper3");
        WhisperThreeModels whisper3;
        auto audio3 = whisper3_dataset.sample(10, audio_rng);
        for (const auto& costs : whisper3.get_omega(audio3))
            require(costs.size() == 3 && costs[2] == 3 * costs[0], "Whisper3 cost ordering");

        auto rejects_audio = [&](const json& invalid) {
            fixture.write(json::array({invalid}));
            try { DemandDataset invalid_dataset(fixture.path.string(), "Whisper"); }
            catch (const exception&) { return; }
            throw runtime_error("invalid audio input accepted");
        };
        for (json value : {json(0), json(-1), json("12"), json(nullptr), json(2147483648.0)}) {
            json invalid = audio;
            invalid["duration_seconds"] = value;
            rejects_audio(invalid);
        }
        json invalid = audio;
        invalid.erase("duration_seconds");
        rejects_audio(invalid);
        invalid = audio;
        invalid["models"].erase("large");
        rejects_audio(invalid);
        invalid = audio;
        invalid["models"]["tiny"]["gflops"] = 0;
        rejects_audio(invalid);
        Model& whisper_base = whisper;
        auto synthetic_audio = whisper_base.get_sizes(3, audio_rng);
        require(whisper_base.get_sigma(synthetic_audio).size() == 3 &&
                whisper_base.get_omega(synthetic_audio)[0].size() == 4,
                "synthetic Whisper compatibility");
        auto close = [&](double actual, double expected, const string& context) {
            require(abs(actual - expected) < 1e-8, context);
        };
        for (const Model* family : vector<const Model*>{&model, &whisper, &whisper3}) {
            const auto raw = family->get_delta_j_original();
            const auto scaled = family->scaled_accuracy_anchors();
            for (size_t j = 0; j < raw.size(); ++j) {
                close(family->scale_performance(raw[j]), scaled[j], "exact scaling anchor");
                if (j > 0)
                    close(family->scale_performance((raw[j-1] + raw[j]) / 2),
                          (scaled[j-1] + scaled[j]) / 2, "linear interpolation");
            }
            close(family->scale_performance(raw.front() - 1),
                  scaled[0] - (scaled[1]-scaled[0])/(raw[1]-raw[0]), "lower extrapolation");
            size_t last = raw.size() - 1;
            close(family->scale_performance(raw.back() + 1),
                  scaled[last] + (scaled[last]-scaled[last-1])/(raw[last]-raw[last-1]),
                  "upper extrapolation");
        }
        close(whisper.scale_performance(94.5), 49.5, "target uses the same interpolation");
        close(model.scale_performance(0), -832, "YOLO incorrect singleton score");
        close(model.scale_performance(100), 150, "YOLO correct singleton score");
        auto singleton_feedback = performance_feedback(model, {1, 1, 1}, {100, 100, 0}, 50, true);
        close(singleton_feedback.overall, 200.0/3, "raw accuracy remains unscaled");
        close(singleton_feedback.deficit, 682, "singleton outcomes");
        auto balanced = performance_feedback(model, {100, 100, 100}, {9000, 9000, 9000}, 50, true);
        close(balanced.deficit, 0, "balanced target accuracy");
        auto unbalanced = performance_feedback(model, {1000, 0, 1}, {90000, 0, 0}, 50, true);
        close(unbalanced.deficit, 882, "unbalanced assignments are weighted by counts");
        require(isnan(unbalanced.per_model[1]), "unused model has no observed accuracy");
        auto aggregate_first = performance_feedback(model, {100, 0, 0}, {9000, 0, 0}, 50, true);
        close(aggregate_first.deficit, 0, "scale the model average rather than each outcome");
        auto no_assignments = performance_feedback(model, {0, 0, 0}, {0, 0, 0}, 50, true);
        close(no_assignments.deficit, 0, "no assignments do not change accuracy queue");
        require(isnan(no_assignments.overall), "no assignments have undefined accuracy");
        auto raw_feedback = performance_feedback(model, {1, 0, 1}, {100, 0, 0}, 90, false);
        close(raw_feedback.deficit, 80, "unscaled baseline deficit");

        // Per-model predictions, not an image-level correctness flag, define outcomes.
        json labeled = first;
        labeled["ground_truth_label"] = "Bear";
        for (auto& result : labeled["models"]) {
            result.erase("correct");
            result["predicted_label"] = "Bear";
        }
        labeled["models"]["large"]["predicted_label"] = "Dog";
        fixture.write(json::array({labeled}));
        DemandDataset labeled_dataset(fixture.path.string());
        require(labeled_dataset.mean_performance() == vector<double>({100, 100, 0}),
                "label-based model means");
        auto labeled_batch = labeled_dataset.sample(2, rng);
        require(labeled_batch[0].performance == vector<double>({100, 100, 0}),
                "sampled outcomes retained in model order");
        auto means = labeled_dataset.mean_performance();
        close(accumulate(means.begin(), means.end(), 0.0)/means.size(), 200.0/3,
              "dataset target is the mean of model means");
        json failed_prediction = labeled;
        failed_prediction["models"]["nano"].erase("predicted_label");
        rejects(json::array({failed_prediction}));
        for (json value : {json(-1), json(101), json("90"), json(nullptr)}) {
            json invalid_score = first;
            invalid_score["models"]["nano"]["performance"] = value;
            rejects(json::array({invalid_score}));
        }
        json measured_audio = audio;
        for (auto& result : measured_audio["models"]) {
            result.erase("correct");
            result["performance"] = 94.25;
        }
        fixture.write(json::array({measured_audio}));
        DemandDataset measured_dataset(fixture.path.string(), "Whisper");
        close(measured_dataset.mean_performance()[0], 94.25, "numeric audio performance");
        cout << "JSON demand tests passed\n";
        return 0;
    } catch (const exception& e) {
        cerr << e.what() << '\n';
        return 1;
    }
}

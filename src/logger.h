#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace std;

class Logger {
    string log_dir;
    bool enabled;
public:
    Logger(const string& path, bool enable) : log_dir(path), enabled(enable) {}

    void save_dict(const nlohmann::json& dict, const string& filename) const {
        if (enabled){
            filesystem::create_directories(log_dir);
            ofstream ofs(log_dir + "/" + filename);
            ofs << dict.dump(4);
        }
    }
};
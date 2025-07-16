
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdexcept>


class ArgParse {
public:
    std::unordered_map<std::string, std::string> args;

    ArgParse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) == 0 && i + 1 < argc) {
                args[arg.substr(2)] = argv[++i];
            }
        }
    }

    std::string get(const std::string& name, const std::string& default_val = "") const {
        auto it = args.find(name);
        return it != args.end() ? it->second : default_val;
    }

    int get_int(const std::string& name, int default_val = 0) const {
        auto it = args.find(name);
        if (it != args.end()) return std::stoi(it->second);
        return default_val;
    }

    double get_double(const std::string& name, double default_val = 0.0) const {
        auto it = args.find(name);
        if (it != args.end()) return std::stod(it->second);
        return default_val;
    }
};
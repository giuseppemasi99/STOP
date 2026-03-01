cmake_minimum_required(VERSION 3.10)
project(STOP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(HIGHS REQUIRED IMPORTED_TARGET highs)

find_package(nlohmann_json CONFIG REQUIRED)

add_executable(STOP src/main.cpp)
target_link_libraries(STOP PRIVATE PkgConfig::HIGHS nlohmann_json::nlohmann_json)
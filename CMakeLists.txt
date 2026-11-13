cmake_minimum_required(VERSION 3.10)
project(STOP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(highs CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(STOP src/main.cpp)

target_link_libraries(STOP PRIVATE highs::highs nlohmann_json::nlohmann_json)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_json_demands tests/test_json_demands.cpp)
    target_include_directories(test_json_demands PRIVATE src)
    target_link_libraries(test_json_demands PRIVATE nlohmann_json::nlohmann_json)
    add_test(NAME json_demands COMMAND test_json_demands)
endif()

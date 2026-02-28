#pragma once
#include <chrono>

class Timer {
public:
    void start() {
        begin_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(
            end - begin_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point begin_;
};
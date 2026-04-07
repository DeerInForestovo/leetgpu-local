#pragma once
#include <chrono>
#include <cuda_runtime.h>

class CpuTimer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start() { cudaEventRecord(start_); }
    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    float elapsed_ms() const {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

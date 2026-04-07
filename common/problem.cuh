#pragma once
#include <cstdio>
#include "cuda_check.cuh"
#include "timer.cuh"

class Problem {
public:
    virtual ~Problem() = default;

    virtual const char* name() const = 0;

    // Allocate memory, generate input data, copy inputs to device
    virtual void setup() = 0;

    // CPU reference implementation (results stored in host memory)
    virtual void cpu_solution() = 0;

    // GPU kernel launch (inputs already on device, results stored in device memory)
    virtual void gpu_solution() = 0;

    // Copy GPU results back to host, compare with CPU results
    virtual bool verify() = 0;

    // Free all host and device memory
    virtual void teardown() = 0;

    void run() {
        printf("  \033[1m▸ %s\033[0m\n", name());
        printf("  \033[90m────────────────────────────────────────\033[0m\n");

        setup();

        CpuTimer cpu_timer;
        cpu_timer.start();
        cpu_solution();
        cpu_timer.stop();
        double cpu_ms = cpu_timer.elapsed_ms();

        // warmup: first kernel launch incurs JIT / context init overhead
        gpu_solution();
        CUDA_CHECK(cudaDeviceSynchronize());

        GpuTimer gpu_timer;
        gpu_timer.start();
        gpu_solution();
        gpu_timer.stop();
        float gpu_ms = gpu_timer.elapsed_ms();

        printf("  \033[90mCPU\033[0m  %10.3f ms\n", cpu_ms);
        printf("  \033[33mGPU\033[0m  %10.3f ms\n", gpu_ms);
        printf("\n");

        bool ok = verify();
        if (ok)
            printf("  \033[32m✓ PASSED\033[0m");
        else
            printf("  \033[31m✗ FAILED\033[0m");

        if (gpu_ms > 0) {
            double speedup = cpu_ms / gpu_ms;
            printf("            \033[1mSpeedup %.1fx\033[0m", speedup);

            // visual bar (max 30 chars)
            printf("\n  \033[90m");
            int bar_len = (int)(speedup * 0.3);
            if (bar_len > 30) bar_len = 30;
            if (bar_len < 1) bar_len = 1;
            for (int i = 0; i < bar_len; i++) printf("█");
            printf("\033[0m");
        }

        printf("\n  \033[90m────────────────────────────────────────\033[0m\n\n");

        teardown();
    }
};

// Place at the bottom of each problem file to register the class
#define REGISTER_PROBLEM(ProblemClass)                                         \
    Problem* create_problem() { return new ProblemClass(); }

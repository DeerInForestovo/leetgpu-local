#include <cstdio>
#include <cuda_runtime.h>
#include "common/cuda_check.cuh"
#include "common/problem.cuh"

extern Problem* create_problem();

static void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\n");
    printf("  \033[1mLeetGPU Local\033[0m\n");
    printf("  \033[90m────────────────────────────────────────\033[0m\n");
    printf("  \033[36m%s\033[0m\n", prop.name);
    printf("  \033[90msm_%d%d  ·  %d SMs  ·  %d MB\033[0m\n",
           prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.totalGlobalMem / (1024 * 1024)));
    printf("\n");
}

int main() {
    print_device_info();

    Problem* p = create_problem();
    p->run();
    delete p;

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

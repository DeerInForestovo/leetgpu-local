// Matrix Multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
// Uses shared memory tiling to reduce global memory traffic.

#include "common/problem.cuh"
#include "common/util.cuh"

static constexpr int TILE = 32;

        // TODO: implement your GPU solution here

class MatrixMul : public Problem {
    static constexpr int M = 1024;
    static constexpr int K = 1024;
    static constexpr int N = 1024;

    float *h_A{}, *h_B{}, *h_cpu_out{}, *h_gpu_out{};
    float *d_A{}, *d_B{}, *d_C{};

public:
    const char* name() const override {
        return "Matrix Multiplication (1024x1024)";
    }

    void setup() override {
        h_A       = new float[M * K];
        h_B       = new float[K * N];
        h_cpu_out = new float[M * N];
        h_gpu_out = new float[M * N];

        fill_rand(h_A, M * K, -1.0f, 1.0f, 42);
        fill_rand(h_B, K * N, -1.0f, 1.0f, 123);

        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void cpu_solution() override {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += h_A[i * K + k] * h_B[k * N + j];
                h_cpu_out[i * N + j] = sum;
            }
    }

    void gpu_solution() override {
        // TODO: implement your GPU solution here
    }

    bool verify() override {
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_C, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return verify_equals(h_cpu_out, h_gpu_out, M * N, 1e-2f);
    }

    void teardown() override {
        delete[] h_A;
        delete[] h_B;
        delete[] h_cpu_out;
        delete[] h_gpu_out;
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
};

REGISTER_PROBLEM(MatrixMul);

// Vector Addition: C[i] = A[i] + B[i], N = 16M

#include "common/problem.cuh"
#include "common/util.cuh"

        // TODO: implement your GPU solution here

class VectorAdd : public Problem {
    static constexpr int N = 1 << 24;
    static constexpr size_t BYTES = N * sizeof(float);

    float *h_a{}, *h_b{}, *h_cpu_out{}, *h_gpu_out{};
    float *d_a{}, *d_b{}, *d_c{};

public:
    const char* name() const override { return "Vector Addition (N=16M)"; }

    void setup() override {
        h_a       = new float[N];
        h_b       = new float[N];
        h_cpu_out = new float[N];
        h_gpu_out = new float[N];

        fill_rand(h_a, N, 0.0f, 100.0f, 42);
        fill_rand(h_b, N, 0.0f, 100.0f, 123);

        CUDA_CHECK(cudaMalloc(&d_a, BYTES));
        CUDA_CHECK(cudaMalloc(&d_b, BYTES));
        CUDA_CHECK(cudaMalloc(&d_c, BYTES));
        CUDA_CHECK(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice));
    }

    void cpu_solution() override {
        for (int i = 0; i < N; i++)
            h_cpu_out[i] = h_a[i] + h_b[i];
    }

    void gpu_solution() override {
        // TODO: implement your GPU solution here
    }

    bool verify() override {
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_c, BYTES, cudaMemcpyDeviceToHost));
        return verify_equals(h_cpu_out, h_gpu_out, N);
    }

    void teardown() override {
        delete[] h_a;
        delete[] h_b;
        delete[] h_cpu_out;
        delete[] h_gpu_out;
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }
};

REGISTER_PROBLEM(VectorAdd);

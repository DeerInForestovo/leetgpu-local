// Softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// Uses Online Softmax: fuse max and sum into one pass with running (max, sum).
// Pass 1: reduce to get global (max, sum_exp).  Pass 2: apply exp(x-m)/s.

#include "common/problem.cuh"
#include "common/util.cuh"

static constexpr int BLOCK_SIZE = 256;

struct SoftmaxPair {
    float m; // running max
    float s; // running sum of exp(x_i - m)
};

        // TODO: implement your GPU solution here

class Softmax : public Problem {
    static constexpr int N = 1 << 20; // 1M elements
    static constexpr int STATS_BLOCKS = 256;
    static constexpr size_t BYTES = N * sizeof(float);

    float *h_input{}, *h_cpu_out{}, *h_gpu_out{};
    float *d_input{}, *d_output{};
    SoftmaxPair *d_block_stats{}, *d_global_stats{};

public:
    const char* name() const override { return "Softmax (N=1M, Online)"; }

    void setup() override {
        h_input   = new float[N];
        h_cpu_out = new float[N];
        h_gpu_out = new float[N];

        fill_rand(h_input, N, -10.0f, 10.0f, 42);

        CUDA_CHECK(cudaMalloc(&d_input, BYTES));
        CUDA_CHECK(cudaMalloc(&d_output, BYTES));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, BYTES,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_block_stats,
                              STATS_BLOCKS * sizeof(SoftmaxPair)));
        CUDA_CHECK(cudaMalloc(&d_global_stats, sizeof(SoftmaxPair)));
    }

    void cpu_solution() override {
        float max_val = h_input[0];
        for (int i = 1; i < N; i++)
            if (h_input[i] > max_val) max_val = h_input[i];

        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            h_cpu_out[i] = expf(h_input[i] - max_val);
            sum += h_cpu_out[i];
        }
        for (int i = 0; i < N; i++)
            h_cpu_out[i] /= sum;
    }

    void gpu_solution() override {
        // TODO: implement your GPU solution here
    }

    bool verify() override {
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output, BYTES,
                              cudaMemcpyDeviceToHost));
        return verify_equals(h_cpu_out, h_gpu_out, N, 1e-4f);
    }

    void teardown() override {
        delete[] h_input;
        delete[] h_cpu_out;
        delete[] h_gpu_out;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_block_stats));
        CUDA_CHECK(cudaFree(d_global_stats));
    }
};

REGISTER_PROBLEM(Softmax);

// Reduction: compute the sum of an array of floats using parallel reduction.
// Input: array of N floats. Output: single float sum.

#include "common/problem.cuh"
#include "common/util.cuh"

static constexpr int BLOCK_SIZE = 256;

        // TODO: implement your GPU solution here

class Reduction : public Problem {
    static constexpr int N = 1 << 24; // 16M elements
    static constexpr size_t BYTES = N * sizeof(float);

    float *h_input{};
    float cpu_sum{};
    float gpu_sum{};

    float *d_input{}, *d_partial{}, *d_out{};
    int num_blocks{};

public:
    const char* name() const override { return "Reduction Sum (N=16M)"; }

    void setup() override {
        h_input = new float[N];
        fill_rand(h_input, N, -1.0f, 1.0f, 42);

        CUDA_CHECK(cudaMalloc(&d_input, BYTES));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, BYTES,
                              cudaMemcpyHostToDevice));

        num_blocks = ceil_div(N, BLOCK_SIZE * 2);
        CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    }

    void cpu_solution() override {
        // Kahan summation for better accuracy as reference
        double sum = 0.0, c = 0.0;
        for (int i = 0; i < N; i++) {
            double y = static_cast<double>(h_input[i]) - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        cpu_sum = static_cast<float>(sum);
    }

    void gpu_solution() override {
        // TODO: implement your GPU solution here
    }

    bool verify() override {
        CUDA_CHECK(cudaMemcpy(&gpu_sum, d_out, sizeof(float),
                              cudaMemcpyDeviceToHost));
        float diff = fabsf(cpu_sum - gpu_sum);
        float rel = (cpu_sum != 0.0f) ? diff / fabsf(cpu_sum) : diff;
        printf("  CPU sum = %.6f\n", cpu_sum);
        printf("  GPU sum = %.6f\n", gpu_sum);
        printf("  Relative error = %.2e\n", rel);
        // Floating-point reduction order differs; allow relative tolerance
        return rel < 1e-4f;
    }

    void teardown() override {
        delete[] h_input;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(d_out));
    }
};

REGISTER_PROBLEM(Reduction);

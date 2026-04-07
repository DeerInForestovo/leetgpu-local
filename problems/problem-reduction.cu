// Reduction: compute the sum of an array of floats using parallel reduction.
// Input: array of N floats. Output: single float sum.

#include "common/problem.cuh"
#include "common/util.cuh"

static constexpr int BLOCK_SIZE = 256;

// === BEGIN SOLUTION ===
__global__ void reduce_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * (BLOCK_SIZE * 2) + tid;

    // Each thread loads two elements and adds them
    float val = 0.0f;
    if (i < n)     val += input[i];
    if (i + BLOCK_SIZE < n) val += input[i + BLOCK_SIZE];
    sdata[tid] = val;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction (no __syncthreads needed within a warp)
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}
// === END SOLUTION ===

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
        // === BEGIN SOLUTION ===
        // First pass: reduce N elements to num_blocks partial sums
        reduce_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, N);

        // Iteratively reduce until single value remains
        int remaining = num_blocks;
        float* src = d_partial;
        float* dst = d_out;

        while (remaining > 1) {
            int blocks = ceil_div(remaining, BLOCK_SIZE * 2);
            reduce_kernel<<<blocks, BLOCK_SIZE>>>(src, dst, remaining);
            remaining = blocks;
            // Ping-pong between buffers
            float* tmp = src;
            src = dst;
            dst = tmp;
        }
        // Final result is in src[0]; copy to d_out if not already there
        if (src != d_out) {
            CUDA_CHECK(cudaMemcpy(d_out, src, sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
        // === END SOLUTION ===
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

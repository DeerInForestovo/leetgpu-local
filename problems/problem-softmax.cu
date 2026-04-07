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

// === BEGIN SOLUTION ===
__device__ SoftmaxPair merge_pair(SoftmaxPair a, SoftmaxPair b) {
    float m = fmaxf(a.m, b.m);
    float s = a.s * expf(a.m - m) + b.s * expf(b.m - m);
    return {m, s};
}

// Pass 1a: each block reduces a chunk of floats into one SoftmaxPair.
// Grid-stride loop so we can use fewer blocks than elements.
__global__ void softmax_stats_kernel(const float* x, SoftmaxPair* block_out,
                                     int n) {
    __shared__ SoftmaxPair smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    SoftmaxPair local = {-INFINITY, 0.0f};
    for (int i = gid; i < n; i += gridDim.x * BLOCK_SIZE) {
        SoftmaxPair elem = {x[i], 1.0f};
        local = merge_pair(local, elem);
    }

    smem[tid] = local;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s >= 1; s >>= 1) {
        if (tid < s)
            smem[tid] = merge_pair(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        block_out[blockIdx.x] = smem[0];
}

// Pass 1b: reduce the per-block pairs into a single global pair.
__global__ void reduce_pairs_kernel(const SoftmaxPair* data, SoftmaxPair* out,
                                    int n) {
    __shared__ SoftmaxPair smem[BLOCK_SIZE];
    int tid = threadIdx.x;

    SoftmaxPair local = {-INFINITY, 0.0f};
    for (int i = tid; i < n; i += BLOCK_SIZE)
        local = merge_pair(local, data[i]);

    smem[tid] = local;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s >= 1; s >>= 1) {
        if (tid < s)
            smem[tid] = merge_pair(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        out[0] = smem[0];
}

// Pass 2: apply softmax using the global (max, sum).
__global__ void softmax_apply_kernel(const float* x, float* out,
                                     const SoftmaxPair* stats, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = expf(x[i] - stats[0].m) / stats[0].s;
}
// === END SOLUTION ===

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
        // === BEGIN SOLUTION ===
        // Pass 1a: per-block online reduction → (max, sum_exp) per block
        softmax_stats_kernel<<<STATS_BLOCKS, BLOCK_SIZE>>>(
            d_input, d_block_stats, N);
        // Pass 1b: reduce 256 block pairs → 1 global pair
        reduce_pairs_kernel<<<1, BLOCK_SIZE>>>(
            d_block_stats, d_global_stats, STATS_BLOCKS);
        // Pass 2: apply exp(x - max) / sum
        softmax_apply_kernel<<<ceil_div(N, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_input, d_output, d_global_stats, N);
        // === END SOLUTION ===
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

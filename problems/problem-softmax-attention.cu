// Softmax Attention: O = softmax(Q @ K^T) @ V
// Q: M x d, K: N x d, V: N x d → O: M x d
// softmax is applied row-wise on the M x N score matrix S = Q @ K^T.
//
// Fused implementation: each block processes one query row, computing
// S[row], softmax(S[row]), and O[row] without materializing the full M x N matrix.

#include "common/problem.cuh"
#include "common/util.cuh"

static constexpr int BLOCK_DIM = 256;

// === BEGIN SOLUTION ===

// Shared memory layout (all dynamic, one contiguous allocation):
//   [q_shared: d floats] [scores: N floats] [reduce_buf: BLOCK_DIM floats]
//
// Grid:  M blocks  (one block per query row)
// Block: BLOCK_DIM threads
// smem:  (d + N + BLOCK_DIM) * sizeof(float)
//
// Requires N <= BLOCK_DIM (each thread handles at most one score).
// Requires d <= BLOCK_DIM (each thread loads at most one query element).

__global__ void attention_kernel(const float* Q, const float* K,
                                 const float* V, float* O,
                                 int M, int N, int d) {
    int row = blockIdx.x;   // which query row this block handles
    if (row >= M) return;

    int tid = threadIdx.x;  // thread index within block [0, BLOCK_DIM)

    // Partition shared memory into three non-overlapping regions
    extern __shared__ float smem[];
    float* q_shared   = smem;            // [0     .. d)         — query vector cache
    float* scores     = smem + d;        // [d     .. d+N)       — attention scores
    float* reduce_buf = smem + d + N;    // [d+N   .. d+N+BLOCK_DIM) — reduction scratch

    // Pointer to this row's query vector in global memory
    const float* q = Q + row * d;

    //=========================================================================
    // Step 1: Compute S[row, j] = dot(Q[row], K[j]) for j = 0..N-1
    //=========================================================================
    // Load the d-dimensional query vector into shared memory.
    // d <= BLOCK_DIM, so threads [0..d-1] each load one element.
    // Threads [d..BLOCK_DIM-1] do nothing here.
    for (int i = tid; i < d; i += BLOCK_DIM)
        q_shared[i] = q[i];
    __syncthreads();  // all threads must see q_shared before computing dots

    // Each thread computes one dot product (since N <= BLOCK_DIM).
    // Thread tid computes scores[tid] = dot(q_shared, K[tid]).
    // Threads [N..BLOCK_DIM-1] do nothing if N < BLOCK_DIM.
    for (int j = tid; j < N; j += BLOCK_DIM) {
        float dot = 0.0f;
        const float* kj = K + j * d;   // pointer to K[j] row
        for (int k = 0; k < d; k++)
            dot += q_shared[k] * kj[k]; // accumulate dot product
        scores[j] = dot;
    }
    __syncthreads();  // all scores written; needed before cross-thread reads

    //=========================================================================
    // Step 2a: Find row-wise max of scores (for numerical stability)
    //=========================================================================
    // Each thread reads its own scores[tid] (one element since N <= BLOCK_DIM).
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += BLOCK_DIM)
        local_max = fmaxf(local_max, scores[j]);

    // Block-wide max reduction:
    // Each thread writes its local_max to reduce_buf, then tree-reduce.
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_DIM / 2; s >= 1; s >>= 1) {
        if (tid < s)
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float row_max = reduce_buf[0];  // broadcast: all threads read the global max
    __syncthreads();

    //=========================================================================
    // Step 2b: Compute exp(score - max) and their sum
    //=========================================================================
    // Each thread modifies its own scores[tid] in-place.
    // No cross-thread hazard (one score per thread), no sync needed before this.
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += BLOCK_DIM) {
        scores[j] = expf(scores[j] - row_max);
        local_sum += scores[j];
    }

    // Block-wide sum reduction (same pattern as max)
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_DIM / 2; s >= 1; s >>= 1) {
        if (tid < s)
            reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float row_sum = reduce_buf[0];  // broadcast: all threads read the global sum

    //=========================================================================
    // Step 2c: Normalize — scores[j] /= sum
    //=========================================================================
    // Each thread normalizes its own scores[tid].
    for (int j = tid; j < N; j += BLOCK_DIM)
        scores[j] /= row_sum;
    __syncthreads();  // ALL scores must be finalized before step 3 reads them

    //=========================================================================
    // Step 3: Output — O[row, k] = sum_j( scores[j] * V[j][k] )
    //=========================================================================
    // d <= BLOCK_DIM, so threads [0..d-1] each compute one output element.
    // Thread tid computes O[row][tid] by reading ALL N scores (shared mem)
    // and the corresponding column of V (global mem, strided access).
    for (int k = tid; k < d; k += BLOCK_DIM) {
        float val = 0.0f;
        for (int j = 0; j < N; j++)
            val += scores[j] * V[j * d + k];  // V[j][k], stride-d access
        O[row * d + k] = val;
    }
}

// === END SOLUTION ===

class SoftmaxAttention : public Problem {
    static constexpr int M_DIM = 512;  // num queries
    static constexpr int N_DIM = 256;  // num keys/values
    static constexpr int D_DIM = 128;  // head dimension

    float *h_Q{}, *h_K{}, *h_V{};
    float *h_cpu_out{}, *h_gpu_out{};
    float *d_Q{}, *d_K{}, *d_V{}, *d_O{};

public:
    const char* name() const override {
        return "Softmax Attention (M=512, N=256, d=128)";
    }

    void setup() override {
        h_Q       = new float[M_DIM * D_DIM];
        h_K       = new float[N_DIM * D_DIM];
        h_V       = new float[N_DIM * D_DIM];
        h_cpu_out = new float[M_DIM * D_DIM];
        h_gpu_out = new float[M_DIM * D_DIM];

        fill_rand(h_Q, M_DIM * D_DIM, -1.0f, 1.0f, 42);
        fill_rand(h_K, N_DIM * D_DIM, -1.0f, 1.0f, 123);
        fill_rand(h_V, N_DIM * D_DIM, -1.0f, 1.0f, 456);

        CUDA_CHECK(cudaMalloc(&d_Q, M_DIM * D_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, N_DIM * D_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, N_DIM * D_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, M_DIM * D_DIM * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, M_DIM * D_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, N_DIM * D_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, N_DIM * D_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void cpu_solution() override {
        for (int i = 0; i < M_DIM; i++) {
            float row_max = -INFINITY;
            float* scores = new float[N_DIM];

            // S[i][j] = dot(Q[i], K[j])
            for (int j = 0; j < N_DIM; j++) {
                float dot = 0.0f;
                for (int k = 0; k < D_DIM; k++)
                    dot += h_Q[i * D_DIM + k] * h_K[j * D_DIM + k];
                scores[j] = dot;
                row_max = fmaxf(row_max, dot);
            }

            // softmax: exp(s - max) / sum
            float row_sum = 0.0f;
            for (int j = 0; j < N_DIM; j++) {
                scores[j] = expf(scores[j] - row_max);
                row_sum += scores[j];
            }
            for (int j = 0; j < N_DIM; j++)
                scores[j] /= row_sum;

            // O[i][k] = sum_j scores[j] * V[j][k]
            for (int k = 0; k < D_DIM; k++) {
                float val = 0.0f;
                for (int j = 0; j < N_DIM; j++)
                    val += scores[j] * h_V[j * D_DIM + k];
                h_cpu_out[i * D_DIM + k] = val;
            }

            delete[] scores;
        }
    }

    void gpu_solution() override {
        // === BEGIN SOLUTION ===
        size_t smem_bytes = (D_DIM + N_DIM + BLOCK_DIM) * sizeof(float);
        attention_kernel<<<M_DIM, BLOCK_DIM, smem_bytes>>>(
            d_Q, d_K, d_V, d_O, M_DIM, N_DIM, D_DIM);
        // === END SOLUTION ===
    }

    bool verify() override {
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_O, M_DIM * D_DIM * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return verify_equals(h_cpu_out, h_gpu_out, M_DIM * D_DIM, 1e-3f);
    }

    void teardown() override {
        delete[] h_Q;
        delete[] h_K;
        delete[] h_V;
        delete[] h_cpu_out;
        delete[] h_gpu_out;
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
};

REGISTER_PROBLEM(SoftmaxAttention);

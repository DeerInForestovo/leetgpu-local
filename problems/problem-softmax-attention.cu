// Softmax Attention: O = softmax(Q @ K^T) @ V
// Q: M x d, K: N x d, V: N x d → O: M x d
// softmax is applied row-wise on the M x N score matrix S = Q @ K^T.
//
// Fused implementation: each block processes one query row, computing
// S[row], softmax(S[row]), and O[row] without materializing the full M x N matrix.

#include "common/problem.cuh"
#include "common/util.cuh"

static constexpr int BLOCK_DIM = 256;

        // TODO: implement your GPU solution here

class SoftmaxAttention : public Problem {
    static constexpr int M_DIM = 512;  // num queries
    static constexpr int N_DIM = 256;  // num keys/values
    static constexpr int D_DIM = 128;  // head dimension

    float *h_Q{}, *h_K{}, *h_V{};
    float *h_cpu_out{}, *h_gpu_out{};
    float *d_Q{}, *d_K_T{}, *d_V{}, *d_O{};

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

        // Precompute K^T on host: K is (N x d), K_T is (d x N)
        // K_T[k][j] = K[j][k], stored row-major as K_T[k * N + j]
        float* h_K_T = new float[D_DIM * N_DIM];
        for (int j = 0; j < N_DIM; j++)
            for (int k = 0; k < D_DIM; k++)
                h_K_T[k * N_DIM + j] = h_K[j * D_DIM + k];

        CUDA_CHECK(cudaMalloc(&d_Q, M_DIM * D_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K_T, D_DIM * N_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, N_DIM * D_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, M_DIM * D_DIM * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, M_DIM * D_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K_T, h_K_T, D_DIM * N_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, N_DIM * D_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        delete[] h_K_T;
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
        // TODO: implement your GPU solution here
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
        CUDA_CHECK(cudaFree(d_K_T));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
};

REGISTER_PROBLEM(SoftmaxAttention);

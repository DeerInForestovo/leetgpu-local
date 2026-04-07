// Matrix Transpose: B = A^T, where A is M x N and B is N x M

#include "common/problem.cuh"
#include "common/util.cuh"

static constexpr int TILE = 32;

// === BEGIN SOLUTION ===
__global__ void transpose_kernel(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE][TILE + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < M && y < N)
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
}
// === END SOLUTION ===

class MatrixTranspose : public Problem {
    static constexpr int M = 4096;
    static constexpr int N = 4096;
    static constexpr size_t BYTES = M * N * sizeof(float);

    float *h_A{}, *h_cpu_out{}, *h_gpu_out{};
    float *d_A{}, *d_B{};

public:
    const char* name() const override { return "Matrix Transpose (4096x4096)"; }

    void setup() override {
        h_A       = new float[M * N];
        h_cpu_out = new float[M * N];
        h_gpu_out = new float[M * N];

        fill_rand(h_A, M * N, -1.0f, 1.0f, 42);

        CUDA_CHECK(cudaMalloc(&d_A, BYTES));
        CUDA_CHECK(cudaMalloc(&d_B, BYTES));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice));
    }

    void cpu_solution() override {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                h_cpu_out[j * M + i] = h_A[i * N + j];
    }

    void gpu_solution() override {
        // === BEGIN SOLUTION ===
        dim3 block(TILE, TILE);
        dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
        transpose_kernel<<<grid, block>>>(d_A, d_B, M, N);
        // === END SOLUTION ===
    }

    bool verify() override {
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_B, BYTES, cudaMemcpyDeviceToHost));
        return verify_equals(h_cpu_out, h_gpu_out, M * N);
    }

    void teardown() override {
        delete[] h_A;
        delete[] h_cpu_out;
        delete[] h_gpu_out;
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
    }
};

REGISTER_PROBLEM(MatrixTranspose);

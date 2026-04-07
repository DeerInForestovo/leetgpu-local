# LeetGPU Local

A minimal framework for practicing CUDA kernel programming. Each problem includes data generation, a CPU reference solution, a GPU implementation, correctness verification, and performance comparison.

Tested on NVIDIA RTX 4060 (Ada Lovelace, sm_89).

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc)
- GNU Make

## Project Structure

```
.
├── Makefile
├── main.cu                         # Entry point: prints device info, runs problem
├── common/
│   ├── cuda_check.cuh              # CUDA error checking macros
│   ├── timer.cuh                   # CPU and GPU timer utilities
│   ├── util.cuh                    # Shared utilities (random gen, verification, math)
│   └── problem.cuh                 # Problem base class and REGISTER_PROBLEM macro
└── problems/
    └── problem-vector-add.cu       # Example: vector addition
```

## Usage

```bash
make problem-vector-add          # Build and run
make build-problem-vector-add    # Build only
make debug-problem-vector-add    # Build with debug symbols (-G -g)
make profile-problem-vector-add  # Build and profile with nsys
make list                        # List all available problems
make clean                       # Remove build artifacts
```

## Adding a New Problem

Create `problems/problem-<name>.cu` and implement the `Problem` interface:

```cpp
#include "common/problem.cuh"
#include "common/util.cuh"

__global__ void my_kernel(/* ... */) {
    // GPU kernel
}

class MyProblem : public Problem {
    float *h_in{}, *h_cpu_out{}, *h_gpu_out{};
    float *d_in{}, *d_out{};
    int N{};

public:
    const char* name() const override { return "My Problem"; }
    void setup() override       { /* fill_rand(h_in, N, 0.f, 1.f); cudaMalloc... */ }
    void cpu_solution() override { /* CPU reference */ }
    void gpu_solution() override { /* my_kernel<<<ceil_div(N,256), 256>>>(...) */ }
    bool verify() override      { /* cudaMemcpy back; return verify_equals(...) */ }
    void teardown() override    { /* delete[], cudaFree */ }
};

REGISTER_PROBLEM(MyProblem);
```

Then run it with `make problem-<name>`.

The base class `Problem::run()` handles the execution flow automatically: setup, CPU timing, GPU warmup, GPU timing, verification, and teardown.

## Utilities (`common/util.cuh`)

Common helpers available to all problems:

| Function | Description |
|---|---|
| `fill_rand(float* arr, int n, float lo, float hi, unsigned seed)` | Fill array with random floats |
| `fill_rand(int* arr, int n, int lo, int hi, unsigned seed)` | Fill array with random ints |
| `rand_float(float lo, float hi)` | Single random float |
| `rand_int(int lo, int hi)` | Single random int |
| `verify_equals(const float* a, const float* b, int n, float eps)` | Compare float arrays with tolerance |
| `verify_equals(const int* a, const int* b, int n)` | Compare int arrays |
| `ceil_div(int a, int b)` | Ceiling division (useful for grid size) |

## Branches

| Branch | Description |
|---|---|
| `main` | Framework only, no problems |
| `solution` | All problems with complete GPU implementations |
| `challenge` | All problems with GPU kernel left empty for you to implement |

Development workflow: write complete solutions on `solution`, then generate `challenge` by stripping GPU implementations.

## Changing GPU Architecture

Edit the `ARCH` variable in `Makefile` to match your GPU:

```makefile
ARCH := -arch=sm_89    # RTX 4060
```

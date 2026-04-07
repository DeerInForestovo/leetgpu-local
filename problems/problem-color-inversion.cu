// Color Inversion: invert R, G, B channels (255 - x), keep Alpha unchanged.
// Image is a 1D array of RGBA unsigned chars, size = width * height * 4.

#include "common/problem.cuh"
#include "common/util.cuh"

        // TODO: implement your GPU solution here

class ColorInversion : public Problem {
    static constexpr int WIDTH  = 3840;
    static constexpr int HEIGHT = 2160;
    static constexpr int NUM_PIXELS = WIDTH * HEIGHT;
    static constexpr size_t BYTES = NUM_PIXELS * 4 * sizeof(unsigned char);

    unsigned char *h_image{}, *h_cpu_out{}, *h_gpu_out{};
    unsigned char *d_image{};

public:
    const char* name() const override {
        return "Color Inversion (3840x2160 RGBA)";
    }

    void setup() override {
        h_image   = new unsigned char[NUM_PIXELS * 4];
        h_cpu_out = new unsigned char[NUM_PIXELS * 4];
        h_gpu_out = new unsigned char[NUM_PIXELS * 4];

        srand(42);
        for (int i = 0; i < NUM_PIXELS * 4; i++)
            h_image[i] = static_cast<unsigned char>(rand() % 256);

        CUDA_CHECK(cudaMalloc(&d_image, BYTES));
    }

    void cpu_solution() override {
        for (int i = 0; i < NUM_PIXELS; i++) {
            int base = i * 4;
            h_cpu_out[base + 0] = 255 - h_image[base + 0];
            h_cpu_out[base + 1] = 255 - h_image[base + 1];
            h_cpu_out[base + 2] = 255 - h_image[base + 2];
            h_cpu_out[base + 3] = h_image[base + 3];
        }
    }

    void gpu_solution() override {
        CUDA_CHECK(cudaMemcpy(d_image, h_image, BYTES, cudaMemcpyHostToDevice));
        // TODO: implement your GPU solution here
    }

    bool verify() override {
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_image, BYTES,
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < NUM_PIXELS * 4; i++) {
            if (h_cpu_out[i] != h_gpu_out[i]) {
                int pixel = i / 4;
                const char* ch = "RGBA";
                printf("  MISMATCH at pixel %d (%c): cpu=%u gpu=%u\n",
                       pixel, ch[i % 4], h_cpu_out[i], h_gpu_out[i]);
                return false;
            }
        }
        return true;
    }

    void teardown() override {
        delete[] h_image;
        delete[] h_cpu_out;
        delete[] h_gpu_out;
        CUDA_CHECK(cudaFree(d_image));
    }
};

REGISTER_PROBLEM(ColorInversion);

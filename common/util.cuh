#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---- Random data generation ----

inline float rand_float(float lo, float hi) {
    return lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo);
}

inline int rand_int(int lo, int hi) {
    return lo + rand() % (hi - lo + 1);
}

inline void fill_rand(float* arr, int n, float lo = 0.0f, float hi = 1.0f,
                       unsigned seed = 42) {
    srand(seed);
    for (int i = 0; i < n; i++)
        arr[i] = rand_float(lo, hi);
}

inline void fill_rand(int* arr, int n, int lo = 0, int hi = 100,
                       unsigned seed = 42) {
    srand(seed);
    for (int i = 0; i < n; i++)
        arr[i] = rand_int(lo, hi);
}

// ---- Verification ----

inline bool verify_equals(const float* a, const float* b, int n,
                           float eps = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > eps) {
            printf("  MISMATCH at [%d]: expected=%.6f got=%.6f (diff=%.2e)\n",
                   i, a[i], b[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

inline bool verify_equals(const int* a, const int* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("  MISMATCH at [%d]: expected=%d got=%d\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// ---- Math helpers ----

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

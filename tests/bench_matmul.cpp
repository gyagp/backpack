#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include "../src/reference_matmul.h"

static void bench_tiled_f16(size_t M, size_t N, size_t K, int warmup, int iters) {
    std::vector<float16_t> A(M * K);
    std::vector<float16_t> B(K * N);
    std::vector<float16_t> C(M * N);

    for (size_t i = 0; i < A.size(); i++)
        A[i] = f32_to_f16((float)((i * 7 + 3) % 17) * 0.1f - 0.8f);
    for (size_t i = 0; i < B.size(); i++)
        B[i] = f32_to_f16((float)((i * 11 + 5) % 19) * 0.1f - 0.9f);

    for (int i = 0; i < warmup; i++)
        cpu_matmul_f16(A.data(), B.data(), C.data(), M, N, K);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        cpu_matmul_f16(A.data(), B.data(), C.data(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double avg_s = elapsed_s / iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / avg_s) / 1e12;

    printf("  %zux%zux%zu: %.4f ms/iter, %.6f TFLOPS (CPU reference)\n",
           M, N, K, avg_s * 1e3, tflops);
}

int main() {
    // Theoretical GPU peak (adjust for your hardware)
    // Example: RTX 4090 f16 tensor = 330 TFLOPS, shader f16 = 165 TFLOPS
    // Example: RTX 3090 f16 = 71 TFLOPS
    // Set via environment variable GPU_PEAK_TFLOPS, default 165.0 (RTX 4090 shader f16)
    const char* peak_env = std::getenv("GPU_PEAK_TFLOPS");
    double gpu_peak_tflops = peak_env ? std::atof(peak_env) : 165.0;

    printf("=== Tiled f16 Matmul Benchmark (CPU reference) ===\n");
    printf("GPU theoretical peak: %.1f TFLOPS (set GPU_PEAK_TFLOPS to override)\n\n", gpu_peak_tflops);

    struct { size_t M, N, K; int warmup, iters; } configs[] = {
        {1024, 1024, 1024, 2, 5},
        {2048, 2048, 2048, 1, 2},
        {4096, 4096, 4096, 1, 1},
    };

    double tflops_results[3];

    for (int c = 0; c < 3; c++) {
        auto& cfg = configs[c];
        size_t M = cfg.M, N = cfg.N, K = cfg.K;

        std::vector<float16_t> A(M * K);
        std::vector<float16_t> B(K * N);
        std::vector<float16_t> C(M * N);

        for (size_t i = 0; i < A.size(); i++)
            A[i] = f32_to_f16((float)((i * 7 + 3) % 17) * 0.1f - 0.8f);
        for (size_t i = 0; i < B.size(); i++)
            B[i] = f32_to_f16((float)((i * 11 + 5) % 19) * 0.1f - 0.9f);

        for (int i = 0; i < cfg.warmup; i++)
            cpu_matmul_f16(A.data(), B.data(), C.data(), M, N, K);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < cfg.iters; i++)
            cpu_matmul_f16(A.data(), B.data(), C.data(), M, N, K);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_s = std::chrono::duration<double>(end - start).count();
        double avg_s = elapsed_s / cfg.iters;
        double flops = 2.0 * M * N * K;
        double tflops = (flops / avg_s) / 1e12;
        double pct = (tflops / gpu_peak_tflops) * 100.0;

        tflops_results[c] = tflops;

        printf("%zux%zux%zu: %.2f ms/iter, %.6f TFLOPS, %.2f%% of GPU peak\n",
               M, N, K, avg_s * 1e3, tflops, pct);
    }

    printf("\n=== Summary ===\n");
    double max_tflops = 0;
    for (int i = 0; i < 3; i++)
        if (tflops_results[i] > max_tflops) max_tflops = tflops_results[i];

    double pct_peak = (max_tflops / gpu_peak_tflops) * 100.0;
    printf("Best: %.6f TFLOPS (%.2f%% of %.1f TFLOPS GPU peak)\n",
           max_tflops, pct_peak, gpu_peak_tflops);

    if (pct_peak < 50.0) {
        printf("\n=== Bottleneck Analysis ===\n");
        printf("Performance is <50%% of GPU theoretical peak.\n");
        printf("NOTE: This benchmark runs the CPU reference implementation.\n");
        printf("Expected bottlenecks for the GPU tiled f16 shader:\n");
        printf("  1. Memory bandwidth: f16 data requires global memory loads per tile\n");
        printf("  2. Tile size tuning: BM=64, BN=64, BK=16 may not optimally utilize\n");
        printf("     the GPU's register file and shared memory capacity\n");
        printf("  3. f16 packing overhead: unpack2x16float adds ALU cost per element\n");
        printf("  4. Lack of tensor core utilization: WGSL does not expose hardware\n");
        printf("     matrix multiply instructions (e.g., WMMA/MMA)\n");
        printf("Future optimization: profile with GPU timestamps to isolate\n");
        printf("memory vs compute bound, then tune tile dimensions accordingly.\n");
    }

    return 0;
}

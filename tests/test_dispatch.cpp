#include "../src/dispatch.h"
#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_dispatch_add_aligned(const GpuContext& ctx) {
    const int N = 128;
    std::vector<float> a(N), b(N), expected(N);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(N - i);
        expected[i] = a[i] + b[i];
    }

    auto buf_a = create_storage_buffer(ctx.device, N * sizeof(float), a.data());
    auto buf_b = create_storage_buffer(ctx.device, N * sizeof(float), b.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/add.wgsl");

    dispatch_elementwise(ctx.device, ctx.queue, pipeline, {buf_a, buf_b, buf_out}, N);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "dispatch add aligned FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  dispatch_elementwise add (N=128): PASS\n");
}

static void test_dispatch_add_nonaligned(const GpuContext& ctx) {
    const int N = 100;
    std::vector<float> a(N), b(N), expected(N);
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        expected[i] = 3.0f;
    }

    auto buf_a = create_storage_buffer(ctx.device, N * sizeof(float), a.data());
    auto buf_b = create_storage_buffer(ctx.device, N * sizeof(float), b.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/add.wgsl");

    dispatch_elementwise(ctx.device, ctx.queue, pipeline, {buf_a, buf_b, buf_out}, N);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "dispatch add nonaligned FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  dispatch_elementwise add (N=100, non-aligned): PASS\n");
}

static void test_dispatch_large(const GpuContext& ctx) {
    const int N = 1000;
    std::vector<float> a(N), b(N), expected(N);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i * 0.01f;
        b[i] = (float)i * 0.02f;
        expected[i] = a[i] + b[i];
    }

    auto buf_a = create_storage_buffer(ctx.device, N * sizeof(float), a.data());
    auto buf_b = create_storage_buffer(ctx.device, N * sizeof(float), b.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/add.wgsl");

    dispatch_elementwise(ctx.device, ctx.queue, pipeline, {buf_a, buf_b, buf_out}, N);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-4f) {
            std::fprintf(stderr, "dispatch large FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  dispatch_elementwise add (N=1000, multi-workgroup): PASS\n");
}

static void test_dispatch_mul(const GpuContext& ctx) {
    const int N = 64;
    std::vector<float> a(N), b(N), expected(N);
    for (int i = 0; i < N; i++) {
        a[i] = 2.0f;
        b[i] = 3.0f;
        expected[i] = 6.0f;
    }

    auto buf_a = create_storage_buffer(ctx.device, N * sizeof(float), a.data());
    auto buf_b = create_storage_buffer(ctx.device, N * sizeof(float), b.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/mul.wgsl");

    dispatch_elementwise(ctx.device, ctx.queue, pipeline, {buf_a, buf_b, buf_out}, N);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "dispatch mul FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  dispatch_elementwise mul (N=64): PASS\n");
}

static void test_dispatch_custom_workgroup_size(const GpuContext& ctx) {
    const int N = 512;
    std::vector<float> a(N), b(N), expected(N);
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 1.0f;
        expected[i] = 2.0f;
    }

    auto buf_a = create_storage_buffer(ctx.device, N * sizeof(float), a.data());
    auto buf_b = create_storage_buffer(ctx.device, N * sizeof(float), b.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/add.wgsl");

    dispatch_elementwise(ctx.device, ctx.queue, pipeline, {buf_a, buf_b, buf_out}, N, 64);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "dispatch custom wg FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  dispatch_elementwise with explicit workgroup_size=64: PASS\n");
}

int main() {
    std::printf("Running dispatch helper tests...\n");
    auto ctx = create_gpu_context();

    test_dispatch_add_aligned(ctx);
    test_dispatch_add_nonaligned(ctx);
    test_dispatch_large(ctx);
    test_dispatch_mul(ctx);
    test_dispatch_custom_workgroup_size(ctx);

    std::printf("All dispatch helper tests PASSED.\n");
    return 0;
}

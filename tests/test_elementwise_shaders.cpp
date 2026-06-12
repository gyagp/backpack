#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void test_add(const GpuContext& ctx) {
    const int N = 128;
    std::vector<float> a(N), b(N), expected(N);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(N - i) * 0.3f;
        expected[i] = a[i] + b[i];
    }

    auto buf_a = create_storage_buffer(ctx.device, N * sizeof(float), a.data());
    auto buf_b = create_storage_buffer(ctx.device, N * sizeof(float), b.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);

    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/add.wgsl");

    wgpu::BindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = buf_a; entries[0].size = N * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_b; entries[1].size = N * sizeof(float);
    entries[2].binding = 2; entries[2].buffer = buf_out; entries[2].size = N * sizeof(float);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    auto bind_group = ctx.device.CreateBindGroup(&bg_desc);

    auto encoder = ctx.device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups((N + 63) / 64);
    pass.End();
    auto commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "add FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  add.wgsl: PASS\n");
}

static void test_mul(const GpuContext& ctx) {
    const int N = 128;
    std::vector<float> a(N), b(N), expected(N);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i * 0.1f;
        b[i] = (float)(i + 1) * 0.2f;
        expected[i] = a[i] * b[i];
    }

    auto buf_a = create_storage_buffer(ctx.device, N * sizeof(float), a.data());
    auto buf_b = create_storage_buffer(ctx.device, N * sizeof(float), b.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);

    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/mul.wgsl");

    wgpu::BindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = buf_a; entries[0].size = N * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_b; entries[1].size = N * sizeof(float);
    entries[2].binding = 2; entries[2].buffer = buf_out; entries[2].size = N * sizeof(float);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    auto bind_group = ctx.device.CreateBindGroup(&bg_desc);

    auto encoder = ctx.device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups((N + 63) / 64);
    pass.End();
    auto commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "mul FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  mul.wgsl: PASS\n");
}

static void test_scale(const GpuContext& ctx) {
    const int N = 128;
    float scale_val = 2.5f;
    std::vector<float> input(N), expected(N);
    for (int i = 0; i < N; i++) {
        input[i] = (float)i * 0.7f;
        expected[i] = input[i] * scale_val;
    }

    auto buf_in = create_storage_buffer(ctx.device, N * sizeof(float), input.data());
    auto buf_out = create_storage_buffer(ctx.device, N * sizeof(float), nullptr);
    auto buf_scale = create_buffer(ctx.device, sizeof(float),
                                   wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopySrc,
                                   &scale_val);

    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/scale.wgsl");

    wgpu::BindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = buf_in; entries[0].size = N * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_out; entries[1].size = N * sizeof(float);
    entries[2].binding = 2; entries[2].buffer = buf_scale; entries[2].size = sizeof(float);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    auto bind_group = ctx.device.CreateBindGroup(&bg_desc);

    auto encoder = ctx.device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups((N + 63) / 64);
    pass.End();
    auto commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "scale FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  scale.wgsl: PASS\n");
}

static void test_add_nonaligned(const GpuContext& ctx) {
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

    wgpu::BindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = buf_a; entries[0].size = N * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_b; entries[1].size = N * sizeof(float);
    entries[2].binding = 2; entries[2].buffer = buf_out; entries[2].size = N * sizeof(float);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    auto bind_group = ctx.device.CreateBindGroup(&bg_desc);

    auto encoder = ctx.device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups((N + 63) / 64);
    pass.End();
    auto commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, N);
    for (int i = 0; i < N; i++) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > 1e-5f) {
            std::fprintf(stderr, "add_nonaligned FAIL [%d]: got=%.6f expected=%.6f\n", i, result[i], expected[i]);
            std::exit(1);
        }
    }
    std::printf("  add.wgsl (non-aligned N=100): PASS\n");
}

int main() {
    std::printf("Running elementwise shader tests...\n");
    auto ctx = create_gpu_context();

    test_add(ctx);
    test_mul(ctx);
    test_scale(ctx);
    test_add_nonaligned(ctx);

    std::printf("All elementwise shader tests PASSED.\n");
    return 0;
}

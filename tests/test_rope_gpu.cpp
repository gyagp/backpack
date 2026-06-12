#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void cpu_rope(const float* input, float* output, uint32_t seq_len,
                     uint32_t head_dim, float theta_base) {
    uint32_t half_dim = head_dim / 2;
    for (uint32_t pos = 0; pos < seq_len; pos++) {
        for (uint32_t d = 0; d < half_dim; d++) {
            float freq = 1.0f / std::pow(theta_base, (float)d / (float)half_dim);
            float angle = (float)pos * freq;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            float x0 = input[pos * head_dim + d];
            float x1 = input[pos * head_dim + d + half_dim];

            output[pos * head_dim + d] = x0 * cos_a - x1 * sin_a;
            output[pos * head_dim + d + half_dim] = x0 * sin_a + x1 * cos_a;
        }
    }
}

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t seq_len, uint32_t head_dim, float theta_base,
                    const char* label) {
    uint32_t count = seq_len * head_dim;
    std::vector<float> input(count);
    for (uint32_t i = 0; i < count; i++)
        input[i] = ((float)((i * 7 + 3) % 19) - 9.0f) * 0.5f;

    std::vector<float> ref(count);
    cpu_rope(input.data(), ref.data(), seq_len, head_dim, theta_base);

    auto buf_input = create_storage_buffer(ctx.device, count * sizeof(float), input.data());
    auto buf_output = create_storage_buffer(ctx.device, count * sizeof(float), nullptr);

    uint32_t theta_bits;
    std::memcpy(&theta_bits, &theta_base, sizeof(float));
    uint32_t params[4] = {head_dim, seq_len, theta_bits, 0};
    auto buf_params = create_buffer(ctx.device, sizeof(params),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                    params);

    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = buf_input;  entries[0].size = count * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_output; entries[1].size = count * sizeof(float);
    entries[2].binding = 2; entries[2].buffer = buf_params; entries[2].size = sizeof(params);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    uint32_t wg_count = (count + 63) / 64;

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_count, 1, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_output, count);

    int fail = 0;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float err = std::fabs(result[i] - ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
            if (fail < 5)
                std::printf("  MISMATCH %s[%u]: got=%f expected=%f err=%f\n", label, i, result[i], ref[i], err);
            fail++;
        }
    }

    if (fail) {
        std::printf("FAIL %s: %d/%u mismatches (max_err=%.6f)\n", label, fail, count, max_err);
        return 1;
    }
    std::printf("PASS %s (seq=%u, dim=%u, theta=%.1f, max_err=%.6f)\n", label, seq_len, head_dim, theta_base, max_err);
    return 0;
}

int main() {
    std::printf("Running RoPE GPU tests...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/rope.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 1, 64, 10000.0f, "1x64");
    fail += run_test(ctx, pipeline, 4, 64, 10000.0f, "4x64");
    fail += run_test(ctx, pipeline, 8, 128, 10000.0f, "8x128");
    fail += run_test(ctx, pipeline, 16, 128, 500000.0f, "16x128_theta500k");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll RoPE GPU tests PASSED.\n");
    return 0;
}

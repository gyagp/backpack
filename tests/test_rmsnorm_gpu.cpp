#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct RmsnormParams {
    uint32_t row_length;
    float epsilon;
};

static void cpu_rmsnorm(const float* input, float* output, uint32_t rows, uint32_t row_length, float epsilon) {
    for (uint32_t r = 0; r < rows; r++) {
        const float* in_row = input + r * row_length;
        float* out_row = output + r * row_length;
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < row_length; i++) {
            sum_sq += in_row[i] * in_row[i];
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / (float)row_length + epsilon);
        for (uint32_t i = 0; i < row_length; i++) {
            out_row[i] = in_row[i] * inv_rms;
        }
    }
}

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t rows, uint32_t row_length, float epsilon, const char* label) {
    size_t total = (size_t)rows * row_length;
    std::vector<float> input(total);
    for (size_t i = 0; i < total; i++)
        input[i] = ((float)((i * 7 + 3) % 19) - 9.0f) * 0.1f;

    std::vector<float> ref(total);
    cpu_rmsnorm(input.data(), ref.data(), rows, row_length, epsilon);

    auto buf_input = create_storage_buffer(ctx.device, total * sizeof(float), input.data());
    auto buf_output = create_storage_buffer(ctx.device, total * sizeof(float), nullptr);

    RmsnormParams params{row_length, epsilon};
    auto buf_params = create_buffer(ctx.device, sizeof(RmsnormParams),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                    &params);

    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = buf_input;  entries[0].size = total * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_output; entries[1].size = total * sizeof(float);
    entries[2].binding = 2; entries[2].buffer = buf_params; entries[2].size = sizeof(RmsnormParams);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    uint32_t wg_count = (rows + 63) / 64;

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_count, 1, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_output, total);

    int fail = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float err = std::fabs(result[i] - ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
            if (fail < 5)
                std::printf("  MISMATCH %s[%zu]: got=%f expected=%f err=%f\n", label, i, result[i], ref[i], err);
            fail++;
        }
    }

    if (fail) {
        std::printf("FAIL %s: %d/%zu mismatches (max_err=%.6f)\n", label, fail, total, max_err);
        return 1;
    }
    std::printf("PASS %s (rows=%u row_length=%u, max_err=%.6f)\n", label, rows, row_length, max_err);
    return 0;
}

int main() {
    std::printf("Running rmsnorm GPU tests...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/rmsnorm.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 1, 64, 1e-5f, "1x64");
    fail += run_test(ctx, pipeline, 4, 128, 1e-5f, "4x128");
    fail += run_test(ctx, pipeline, 16, 256, 1e-6f, "16x256");
    fail += run_test(ctx, pipeline, 64, 512, 1e-5f, "64x512");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll rmsnorm GPU tests PASSED.\n");
    return 0;
}

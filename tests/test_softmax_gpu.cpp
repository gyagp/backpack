#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void cpu_softmax(const float* input, float* output, uint32_t row_len,
                        uint32_t num_rows) {
    for (uint32_t r = 0; r < num_rows; r++) {
        const float* row_in = input + r * row_len;
        float* row_out = output + r * row_len;

        float row_max = row_in[0];
        for (uint32_t i = 1; i < row_len; i++)
            row_max = std::fmax(row_max, row_in[i]);

        float sum = 0.0f;
        for (uint32_t i = 0; i < row_len; i++) {
            row_out[i] = std::exp(row_in[i] - row_max);
            sum += row_out[i];
        }

        float inv_sum = 1.0f / sum;
        for (uint32_t i = 0; i < row_len; i++)
            row_out[i] *= inv_sum;
    }
}

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t row_len, uint32_t num_rows, const char* label) {
    uint32_t count = row_len * num_rows;
    std::vector<float> input(count);
    for (uint32_t i = 0; i < count; i++)
        input[i] = ((float)((i * 7 + 3) % 19) - 9.0f) * 0.5f;

    std::vector<float> ref(count);
    cpu_softmax(input.data(), ref.data(), row_len, num_rows);

    auto buf_input = create_storage_buffer(ctx.device, count * sizeof(float), input.data());
    auto buf_output = create_storage_buffer(ctx.device, count * sizeof(float), nullptr);

    uint32_t params[2] = {row_len, num_rows};
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

    // One workgroup per row
    uint32_t wg_count = num_rows;

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
    std::printf("PASS %s (rows=%u, cols=%u, max_err=%.6f)\n", label, num_rows, row_len, max_err);
    return 0;
}

int main() {
    std::printf("Running Softmax GPU tests...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/softmax.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 64, 1, "1x64");
    fail += run_test(ctx, pipeline, 128, 4, "4x128");
    fail += run_test(ctx, pipeline, 256, 8, "8x256");
    fail += run_test(ctx, pipeline, 512, 16, "16x512");
    fail += run_test(ctx, pipeline, 1024, 2, "2x1024");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll Softmax GPU tests PASSED.\n");
    return 0;
}

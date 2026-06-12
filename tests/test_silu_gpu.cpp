#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float cpu_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t count, const char* label) {
    std::vector<float> input(count);
    for (uint32_t i = 0; i < count; i++)
        input[i] = ((float)((i * 7 + 3) % 19) - 9.0f) * 0.5f;

    std::vector<float> ref(count);
    for (uint32_t i = 0; i < count; i++)
        ref[i] = cpu_silu(input[i]);

    auto buf_input = create_storage_buffer(ctx.device, count * sizeof(float), input.data());
    auto buf_output = create_storage_buffer(ctx.device, count * sizeof(float), nullptr);

    wgpu::BindGroupEntry entries[2] = {};
    entries[0].binding = 0; entries[0].buffer = buf_input;  entries[0].size = count * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_output; entries[1].size = count * sizeof(float);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 2;
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
    std::printf("PASS %s (count=%u, max_err=%.6f)\n", label, count, max_err);
    return 0;
}

int main() {
    std::printf("Running SiLU GPU tests...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/silu.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 64, "64");
    fail += run_test(ctx, pipeline, 256, "256");
    fail += run_test(ctx, pipeline, 1024, "1024");
    fail += run_test(ctx, pipeline, 4096, "4096");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll SiLU GPU tests PASSED.\n");
    return 0;
}

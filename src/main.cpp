#include "gpu_context.h"
#include "buffer_utils.h"
#include "shader_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
    auto ctx = create_gpu_context();

    wgpu::AdapterInfo info{};
    ctx.adapter.GetInfo(&info);
    std::printf("Adapter: %s\n", std::string(info.device).c_str());
    std::printf("Backend: %d\n", static_cast<int>(info.backendType));

    // Buffer round-trip test
    float test_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto buf = create_storage_buffer(ctx.device, sizeof(test_data), test_data);
    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf, 4);
    std::printf("Roundtrip: %.1f %.1f %.1f %.1f\n",
                result[0], result[1], result[2], result[3]);

    // Compute shader dispatch test: doubles each element
    constexpr size_t N = 64;
    float input[N];
    for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

    auto input_buf = create_storage_buffer(ctx.device, sizeof(input), input);
    auto output_buf = create_storage_buffer(ctx.device, sizeof(input), nullptr);

    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/double.wgsl");

    wgpu::BindGroupEntry entries[2]{};
    entries[0].binding = 0;
    entries[0].buffer = input_buf;
    entries[0].size = sizeof(input);
    entries[1].binding = 1;
    entries[1].buffer = output_buf;
    entries[1].size = sizeof(input);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 2;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(1);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto gpu_result = read_buffer_as<float>(ctx.device, ctx.queue, output_buf, N);

    bool pass_test = true;
    for (size_t i = 0; i < N; i++) {
        float expected = static_cast<float>(i) * 2.0f;
        if (std::fabs(gpu_result[i] - expected) > 1e-6f) {
            std::fprintf(stderr, "FAIL at [%zu]: got %.6f, expected %.6f\n",
                         i, gpu_result[i], expected);
            pass_test = false;
        }
    }

    if (pass_test) {
        std::printf("Compute shader test PASSED: all %zu values doubled correctly\n", N);
    } else {
        std::fprintf(stderr, "Compute shader test FAILED\n");
        return 1;
    }

    return 0;
}

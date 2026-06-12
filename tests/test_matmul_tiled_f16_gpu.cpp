#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"
#include "../src/reference_matmul.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct MatmulParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batch_size;
    uint32_t stride_A;
    uint32_t stride_C;
};

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t M, uint32_t N, uint32_t K, const char* label) {
    std::vector<float16_t> A(M * K), B(K * N);
    for (size_t i = 0; i < M * K; i++)
        A[i] = f32_to_f16((float)((i * 3 + 5) % 17) * 0.1f - 0.8f);
    for (size_t i = 0; i < K * N; i++)
        B[i] = f32_to_f16((float)((i * 7 + 13) % 11) * 0.15f - 0.7f);

    std::vector<float16_t> C_ref(M * N);
    cpu_matmul_f16(A.data(), B.data(), C_ref.data(), M, N, K);

    // A and B are packed as array<u32> (two f16 per u32)
    size_t a_u32 = (M * K + 1) / 2;
    size_t b_u32 = (K * N + 1) / 2;

    auto buf_a = create_storage_buffer(ctx.device, a_u32 * sizeof(uint32_t), A.data());
    auto buf_b = create_storage_buffer(ctx.device, b_u32 * sizeof(uint32_t), B.data());
    auto buf_c = create_storage_buffer(ctx.device, M * N * sizeof(float), nullptr);

    MatmulParams params{M, N, K, 1, M * K, M * N};
    auto buf_params = create_buffer(ctx.device, sizeof(MatmulParams),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                    &params);

    wgpu::BindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = buf_a; entries[0].size = a_u32 * sizeof(uint32_t);
    entries[1].binding = 1; entries[1].buffer = buf_b; entries[1].size = b_u32 * sizeof(uint32_t);
    entries[2].binding = 2; entries[2].buffer = buf_c; entries[2].size = M * N * sizeof(float);
    entries[3].binding = 3; entries[3].buffer = buf_params; entries[3].size = sizeof(MatmulParams);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 4;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    uint32_t wg_x = (M + 127) / 128;
    uint32_t wg_y = (N + 127) / 128;

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_x, wg_y, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_c, M * N);

    int fail = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < M * N; i++) {
        float got = result[i];
        float exp = f16_to_f32(C_ref[i]);
        float err = std::fabs(got - exp);
        if (err > max_err) max_err = err;
        float tol = 1e-2f + std::fabs(exp) * 0.05f;
        if (err > tol) {
            if (fail < 5)
                std::printf("  MISMATCH %s[%zu]: got=%f expected=%f err=%f\n", label, i, got, exp, err);
            fail++;
        }
    }

    if (fail) {
        std::printf("FAIL %s: %d/%zu mismatches (max_err=%.6f)\n", label, fail, (size_t)(M * N), max_err);
        return 1;
    }
    std::printf("PASS %s (M=%u N=%u K=%u, max_err=%.6f)\n", label, M, N, K, max_err);
    return 0;
}

int main() {
    std::printf("Running matmul_tiled_f16 GPU tests...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/matmul_tiled_f16.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 128, 128, 128, "128x128x128");
    fail += run_test(ctx, pipeline, 256, 256, 256, "256x256x256");
    fail += run_test(ctx, pipeline, 128, 256, 64, "128x256x64");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll matmul_tiled_f16 GPU tests PASSED.\n");
    return 0;
}

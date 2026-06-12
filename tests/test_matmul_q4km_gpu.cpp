#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"
#include "../src/reference_matmul.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct MatmulParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batch_size;
    uint32_t stride_A;
    uint32_t stride_C;
};

static void pack_q4_k_m_block(const float* vals, uint8_t* block) {
    std::memset(block, 0, Q4_K_BYTES_PER_BLOCK);

    float vmin = vals[0], vmax = vals[0];
    for (int i = 1; i < Q4_K_BLOCK_SIZE; i++) {
        if (vals[i] < vmin) vmin = vals[i];
        if (vals[i] > vmax) vmax = vals[i];
    }

    float range = vmax - vmin;
    float d_val = range > 0 ? range / 15.0f : 1.0f;
    float dmin_val = vmin < 0 ? -vmin : 0.0f;

    uint16_t d_f16 = f32_to_f16(d_val);
    uint16_t dmin_f16 = f32_to_f16(dmin_val);
    std::memcpy(block, &d_f16, 2);
    std::memcpy(block + 2, &dmin_f16, 2);

    uint8_t* scales = block + 4;
    for (int i = 0; i < 4; i++) {
        scales[i] = 1;
        scales[i + 4] = 1;
    }
    for (int i = 0; i < 4; i++) {
        scales[8 + i] = 1 | (1 << 4);
    }

    float d_f = f16_to_f32(d_f16);
    float dmin_f = f16_to_f32(dmin_f16);
    uint8_t* quants = block + 16;

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 16; i++) {
            float v_lo = vals[j * 32 + i];
            float v_hi = vals[j * 32 + i + 16];
            int q_lo = (int)roundf((v_lo + dmin_f) / d_f);
            int q_hi = (int)roundf((v_hi + dmin_f) / d_f);
            if (q_lo < 0) q_lo = 0; if (q_lo > 15) q_lo = 15;
            if (q_hi < 0) q_hi = 0; if (q_hi > 15) q_hi = 15;
            quants[j * 16 + i] = (uint8_t)(q_lo | (q_hi << 4));
        }
    }
}

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t M, uint32_t N, uint32_t K, const char* label) {
    size_t blocks_per_col = K / Q4_K_BLOCK_SIZE;

    std::vector<float> B_float(K * N);
    for (size_t i = 0; i < K * N; i++)
        B_float[i] = (float)((i * 7 + 13) % 11) * 0.3f - 1.5f;

    std::vector<uint8_t> B_q4(N * blocks_per_col * Q4_K_BYTES_PER_BLOCK);
    for (size_t n = 0; n < N; n++) {
        std::vector<float> col(K);
        for (size_t k = 0; k < K; k++) col[k] = B_float[k * N + n];
        for (size_t blk = 0; blk < blocks_per_col; blk++) {
            pack_q4_k_m_block(col.data() + blk * Q4_K_BLOCK_SIZE,
                              B_q4.data() + (n * blocks_per_col + blk) * Q4_K_BYTES_PER_BLOCK);
        }
    }

    std::vector<float16_t> A(M * K);
    for (size_t i = 0; i < M * K; i++)
        A[i] = f32_to_f16((float)((i * 3 + 5) % 9) * 0.2f - 0.8f);

    std::vector<float16_t> C_ref(M * N);
    cpu_matmul_q4km_f16(A.data(), B_q4.data(), C_ref.data(), M, N, K);

    size_t a_u32 = (M * K + 1) / 2;
    size_t b_u32 = (N * blocks_per_col * Q4_K_BYTES_PER_BLOCK + 3) / 4;

    auto buf_a = create_storage_buffer(ctx.device, a_u32 * sizeof(uint32_t), A.data());
    auto buf_b = create_storage_buffer(ctx.device, b_u32 * sizeof(uint32_t), B_q4.data());
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

    uint32_t wg_x = (M + 15) / 16;
    uint32_t wg_y = (N + 15) / 16;

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
        float tol = 0.5f + std::fabs(exp) * 0.05f;
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
    std::printf("Running matmul_q4_k_m GPU tests...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/matmul_q4_k_m.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 2, 2, 256, "2x2x256");
    fail += run_test(ctx, pipeline, 8, 16, 256, "8x16x256");
    fail += run_test(ctx, pipeline, 4, 8, 512, "4x8x512");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll matmul_q4_k_m GPU tests PASSED.\n");
    return 0;
}

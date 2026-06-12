#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/reference_matmul.h"
#include "../src/ffn.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void cpu_silu(float* x, size_t n) {
    for (size_t i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

static void cpu_ffn(const float16_t* hidden, const float16_t* gate_w,
                    const float16_t* up_w, const float16_t* down_w,
                    float* output,
                    uint32_t M, uint32_t K, uint32_t N_ff) {
    std::vector<float16_t> gate_out_f16(M * N_ff);
    cpu_matmul_f16(hidden, gate_w, gate_out_f16.data(), M, N_ff, K);

    std::vector<float> gate_out(M * N_ff);
    for (size_t i = 0; i < M * N_ff; i++)
        gate_out[i] = f16_to_f32(gate_out_f16[i]);
    cpu_silu(gate_out.data(), M * N_ff);

    std::vector<float16_t> up_out_f16(M * N_ff);
    cpu_matmul_f16(hidden, up_w, up_out_f16.data(), M, N_ff, K);

    std::vector<float> fused(M * N_ff);
    for (size_t i = 0; i < M * N_ff; i++)
        fused[i] = gate_out[i] * f16_to_f32(up_out_f16[i]);

    std::vector<float16_t> fused_f16(M * N_ff);
    for (size_t i = 0; i < M * N_ff; i++)
        fused_f16[i] = f32_to_f16(fused[i]);

    std::vector<float16_t> out_f16(M * K);
    cpu_matmul_f16(fused_f16.data(), down_w, out_f16.data(), M, K, N_ff);

    for (size_t i = 0; i < M * K; i++)
        output[i] = f16_to_f32(out_f16[i]);
}

static int run_test(const GpuContext& ctx,
                    uint32_t M, uint32_t K, uint32_t N_ff, const char* label) {
    std::vector<float16_t> hidden(M * K);
    std::vector<float16_t> gate_w(K * N_ff);
    std::vector<float16_t> up_w(K * N_ff);
    std::vector<float16_t> down_w(N_ff * K);

    for (size_t i = 0; i < M * K; i++)
        hidden[i] = f32_to_f16((float)((i * 3 + 5) % 17) * 0.05f - 0.4f);
    for (size_t i = 0; i < K * N_ff; i++)
        gate_w[i] = f32_to_f16((float)((i * 7 + 13) % 11) * 0.04f - 0.2f);
    for (size_t i = 0; i < K * N_ff; i++)
        up_w[i] = f32_to_f16((float)((i * 11 + 3) % 13) * 0.03f - 0.15f);
    for (size_t i = 0; i < N_ff * K; i++)
        down_w[i] = f32_to_f16((float)((i * 5 + 7) % 19) * 0.03f - 0.25f);

    std::vector<float> cpu_out(M * K);
    cpu_ffn(hidden.data(), gate_w.data(), up_w.data(), down_w.data(),
            cpu_out.data(), M, K, N_ff);

    size_t h_u32 = (M * K + 1) / 2;
    size_t g_u32 = (K * N_ff + 1) / 2;
    size_t u_u32 = (K * N_ff + 1) / 2;
    size_t d_u32 = (N_ff * K + 1) / 2;

    auto buf_hidden = create_storage_buffer(ctx.device, h_u32 * 4, hidden.data());
    auto buf_gate = create_storage_buffer(ctx.device, g_u32 * 4, gate_w.data());
    auto buf_up = create_storage_buffer(ctx.device, u_u32 * 4, up_w.data());
    auto buf_down = create_storage_buffer(ctx.device, d_u32 * 4, down_w.data());

    PipelineCache pipeline_cache(ctx.device);
    auto buf_out = ffn_forward(ctx.device, ctx.queue, pipeline_cache,
                               buf_hidden, buf_gate, buf_up, buf_down,
                               M, K, N_ff);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, M * K);

    int fail = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < M * K; i++) {
        float got = result[i];
        float exp = cpu_out[i];
        float err = std::fabs(got - exp);
        if (err > max_err) max_err = err;
        float tol = 1e-2f + std::fabs(exp) * 0.05f;
        if (err > tol) {
            if (fail < 5)
                std::printf("  MISMATCH %s[%zu]: got=%f expected=%f err=%f\n",
                            label, i, got, exp, err);
            fail++;
        }
    }

    if (fail) {
        std::printf("FAIL %s: %d/%zu mismatches (max_err=%.6f)\n",
                    label, fail, (size_t)(M * K), max_err);
        return 1;
    }
    std::printf("PASS %s (M=%u K=%u N_ff=%u, max_err=%.6f)\n",
                label, M, K, N_ff, max_err);
    return 0;
}

int main() {
    std::printf("Running FFN GPU tests...\n");
    auto ctx = create_gpu_context();

    int fail = 0;
    fail += run_test(ctx, 1, 64, 128, "1x64x128");
    fail += run_test(ctx, 2, 64, 128, "2x64x128");
    fail += run_test(ctx, 1, 128, 256, "1x128x256");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll FFN GPU tests PASSED.\n");
    return 0;
}

#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static uint32_t pack2x16float(float a, float b) {
    uint16_t ha = f32_to_f16(a);
    uint16_t hb = f32_to_f16(b);
    return static_cast<uint32_t>(ha) | (static_cast<uint32_t>(hb) << 16);
}

static std::vector<uint32_t> pack_f32_to_f16_pairs(const float* data, uint32_t count) {
    uint32_t pair_count = count / 2;
    std::vector<uint32_t> packed(pair_count);
    for (uint32_t i = 0; i < pair_count; i++) {
        packed[i] = pack2x16float(data[i * 2], data[i * 2 + 1]);
    }
    return packed;
}

static void cpu_attention(const float* Q, const float* K, const float* V,
                          float* output, uint32_t seq_len, uint32_t head_dim,
                          uint32_t num_heads, uint32_t num_kv_heads) {
    float scale = 1.0f / std::sqrt((float)head_dim);

    for (uint32_t h = 0; h < num_heads; h++) {
        uint32_t kv_h = h % num_kv_heads;
        for (uint32_t q_row = 0; q_row < seq_len; q_row++) {
            const float* q_ptr = Q + (h * seq_len + q_row) * head_dim;

            std::vector<float> scores(seq_len);
            for (uint32_t k_col = 0; k_col < seq_len; k_col++) {
                const float* k_ptr = K + (kv_h * seq_len + k_col) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++)
                    dot += q_ptr[d] * k_ptr[d];
                scores[k_col] = dot * scale;
            }

            float row_max = scores[0];
            for (uint32_t i = 1; i < seq_len; i++)
                row_max = std::fmax(row_max, scores[i]);

            float sum = 0.0f;
            for (uint32_t i = 0; i < seq_len; i++) {
                scores[i] = std::exp(scores[i] - row_max);
                sum += scores[i];
            }
            float inv_sum = 1.0f / sum;
            for (uint32_t i = 0; i < seq_len; i++)
                scores[i] *= inv_sum;

            float* out_ptr = output + (h * seq_len + q_row) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < seq_len; s++)
                    acc += scores[s] * V[(kv_h * seq_len + s) * head_dim + d];
                out_ptr[d] = acc;
            }
        }
    }
}

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t seq_len, uint32_t head_dim, uint32_t num_heads,
                    uint32_t num_kv_heads, const char* label) {
    uint32_t qo_count = num_heads * seq_len * head_dim;
    uint32_t kv_count = num_kv_heads * seq_len * head_dim;
    uint32_t kv_u32_count = kv_count / 2;

    std::vector<float> q_data(qo_count), k_data(kv_count), v_data(kv_count);
    for (uint32_t i = 0; i < qo_count; i++)
        q_data[i] = ((float)((i * 7 + 3) % 19) - 9.0f) * 0.1f;
    for (uint32_t i = 0; i < kv_count; i++) {
        k_data[i] = ((float)((i * 11 + 5) % 23) - 11.0f) * 0.1f;
        v_data[i] = ((float)((i * 13 + 7) % 29) - 14.0f) * 0.1f;
    }

    std::vector<float> ref(qo_count);
    cpu_attention(q_data.data(), k_data.data(), v_data.data(), ref.data(),
                  seq_len, head_dim, num_heads, num_kv_heads);

    auto k_packed = pack_f32_to_f16_pairs(k_data.data(), kv_count);
    auto v_packed = pack_f32_to_f16_pairs(v_data.data(), kv_count);

    auto buf_q = create_storage_buffer(ctx.device, qo_count * sizeof(float), q_data.data());
    auto buf_k = create_storage_buffer(ctx.device, kv_u32_count * sizeof(uint32_t), k_packed.data());
    auto buf_v = create_storage_buffer(ctx.device, kv_u32_count * sizeof(uint32_t), v_packed.data());
    auto buf_out = create_storage_buffer(ctx.device, qo_count * sizeof(float), nullptr);

    float scale = 1.0f / std::sqrt((float)head_dim);
    uint32_t params[8] = {};
    params[0] = seq_len;
    params[1] = head_dim;
    params[2] = num_heads;
    params[3] = num_kv_heads;
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, sizeof(float));
    params[4] = scale_bits;

    auto buf_params = create_buffer(ctx.device, sizeof(params),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                    params);

    wgpu::BindGroupEntry entries[5] = {};
    entries[0].binding = 0; entries[0].buffer = buf_q;      entries[0].size = qo_count * sizeof(float);
    entries[1].binding = 1; entries[1].buffer = buf_k;      entries[1].size = kv_u32_count * sizeof(uint32_t);
    entries[2].binding = 2; entries[2].buffer = buf_v;      entries[2].size = kv_u32_count * sizeof(uint32_t);
    entries[3].binding = 3; entries[3].buffer = buf_out;    entries[3].size = qo_count * sizeof(float);
    entries[4].binding = 4; entries[4].buffer = buf_params; entries[4].size = sizeof(params);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 5;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(num_heads, seq_len, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result = read_buffer_as<float>(ctx.device, ctx.queue, buf_out, qo_count);

    int fail = 0;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < qo_count; i++) {
        float err = std::fabs(result[i] - ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) {
            if (fail < 5)
                std::printf("  MISMATCH %s[%u]: got=%f expected=%f err=%f\n", label, i, result[i], ref[i], err);
            fail++;
        }
    }

    if (fail) {
        std::printf("FAIL %s: %d/%u mismatches (max_err=%.6f)\n", label, fail, qo_count, max_err);
        return 1;
    }
    std::printf("PASS %s (heads=%u, kv_heads=%u, seq=%u, dim=%u, max_err=%.6f)\n",
                label, num_heads, num_kv_heads, seq_len, head_dim, max_err);
    return 0;
}

int main() {
    std::printf("Running Attention GPU tests (f16 KV)...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/attention.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 8, 64, 4, 4, "MHA_8x64_4h");
    fail += run_test(ctx, pipeline, 16, 32, 2, 2, "MHA_16x32_2h");
    fail += run_test(ctx, pipeline, 8, 64, 4, 2, "GQA_8x64_4h2kv");
    fail += run_test(ctx, pipeline, 16, 64, 8, 2, "GQA_16x64_8h2kv");
    fail += run_test(ctx, pipeline, 4, 32, 1, 1, "single_4x32");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll Attention GPU tests PASSED.\n");
    return 0;
}

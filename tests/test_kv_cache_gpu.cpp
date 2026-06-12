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

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        float result = 0.0f;
        std::memcpy(&result, &sign, 4);
        return result;
    }
    if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }
    uint32_t bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

static uint32_t pack2x16float(float a, float b) {
    uint16_t ha = f32_to_f16(a);
    uint16_t hb = f32_to_f16(b);
    return static_cast<uint32_t>(ha) | (static_cast<uint32_t>(hb) << 16);
}

static void unpack2x16float(uint32_t packed, float& a, float& b) {
    a = f16_to_f32(static_cast<uint16_t>(packed & 0xFFFF));
    b = f16_to_f32(static_cast<uint16_t>(packed >> 16));
}

static void cpu_kv_cache_update_f16(std::vector<uint32_t>& K_cache_packed,
                                     std::vector<uint32_t>& V_cache_packed,
                                     const float* new_K, const float* new_V,
                                     uint32_t num_kv_heads, uint32_t max_seq_len,
                                     uint32_t head_dim, uint32_t seq_pos) {
    uint32_t half_hd = head_dim / 2;
    for (uint32_t head = 0; head < num_kv_heads; head++) {
        for (uint32_t dp = 0; dp < half_hd; dp++) {
            uint32_t d = dp * 2;
            uint32_t cache_offset = (head * max_seq_len + seq_pos) * half_hd + dp;
            uint32_t src_offset = head * head_dim + d;
            K_cache_packed[cache_offset] = pack2x16float(new_K[src_offset], new_K[src_offset + 1]);
            V_cache_packed[cache_offset] = pack2x16float(new_V[src_offset], new_V[src_offset + 1]);
        }
    }
}

static int run_test(const GpuContext& ctx, const wgpu::ComputePipeline& pipeline,
                    uint32_t num_kv_heads, uint32_t max_seq_len,
                    uint32_t head_dim, uint32_t seq_pos, const char* label) {
    uint32_t half_hd = head_dim / 2;
    uint32_t cache_u32_count = num_kv_heads * max_seq_len * half_hd;
    uint32_t new_count = num_kv_heads * head_dim;

    std::vector<uint32_t> k_cache_packed(cache_u32_count, 0u);
    std::vector<uint32_t> v_cache_packed(cache_u32_count, 0u);
    std::vector<float> new_k(new_count), new_v(new_count);

    for (uint32_t i = 0; i < new_count; i++) {
        new_k[i] = ((float)((i * 7 + 3) % 23) - 11.0f) * 0.1f;
        new_v[i] = ((float)((i * 11 + 5) % 29) - 14.0f) * 0.1f;
    }

    std::vector<uint32_t> ref_k = k_cache_packed;
    std::vector<uint32_t> ref_v = v_cache_packed;
    cpu_kv_cache_update_f16(ref_k, ref_v, new_k.data(), new_v.data(),
                            num_kv_heads, max_seq_len, head_dim, seq_pos);

    auto buf_k_cache = create_storage_buffer(ctx.device, cache_u32_count * sizeof(uint32_t), k_cache_packed.data());
    auto buf_v_cache = create_storage_buffer(ctx.device, cache_u32_count * sizeof(uint32_t), v_cache_packed.data());
    auto buf_new_k = create_storage_buffer(ctx.device, new_count * sizeof(float), new_k.data());
    auto buf_new_v = create_storage_buffer(ctx.device, new_count * sizeof(float), new_v.data());

    uint32_t params[8] = { num_kv_heads, max_seq_len, head_dim, seq_pos, 1, 0, 0, 0 };
    auto buf_params = create_buffer(ctx.device, sizeof(params),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                    params);

    wgpu::BindGroupEntry entries[5] = {};
    entries[0].binding = 0; entries[0].buffer = buf_k_cache; entries[0].size = cache_u32_count * sizeof(uint32_t);
    entries[1].binding = 1; entries[1].buffer = buf_v_cache; entries[1].size = cache_u32_count * sizeof(uint32_t);
    entries[2].binding = 2; entries[2].buffer = buf_new_k;   entries[2].size = new_count * sizeof(float);
    entries[3].binding = 3; entries[3].buffer = buf_new_v;   entries[3].size = new_count * sizeof(float);
    entries[4].binding = 4; entries[4].buffer = buf_params;  entries[4].size = sizeof(params);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 5;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    uint32_t total_pairs = num_kv_heads * half_hd;
    uint32_t workgroups = (total_pairs + 255) / 256;

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(workgroups, 1, 1);
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto result_k = read_buffer_as<uint32_t>(ctx.device, ctx.queue, buf_k_cache, cache_u32_count);
    auto result_v = read_buffer_as<uint32_t>(ctx.device, ctx.queue, buf_v_cache, cache_u32_count);

    int fail = 0;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < cache_u32_count; i++) {
        float rk0, rk1, ek0, ek1, rv0, rv1, ev0, ev1;
        unpack2x16float(result_k[i], rk0, rk1);
        unpack2x16float(ref_k[i], ek0, ek1);
        unpack2x16float(result_v[i], rv0, rv1);
        unpack2x16float(ref_v[i], ev0, ev1);
        float err = std::fmax(std::fmax(std::fabs(rk0 - ek0), std::fabs(rk1 - ek1)),
                              std::fmax(std::fabs(rv0 - ev0), std::fabs(rv1 - ev1)));
        if (err > max_err) max_err = err;
        if (err > 1e-2f) {
            if (fail < 5)
                std::printf("  MISMATCH %s[%u]: K got=(%f,%f) exp=(%f,%f)\n",
                            label, i, rk0, rk1, ek0, ek1);
            fail++;
        }
    }

    if (fail) {
        std::printf("FAIL %s: %d/%u mismatches (max_err=%.6f)\n", label, fail, cache_u32_count, max_err);
        return 1;
    }
    std::printf("PASS %s (kv_heads=%u, max_seq=%u, dim=%u, pos=%u, max_err=%.6f)\n",
                label, num_kv_heads, max_seq_len, head_dim, seq_pos, max_err);
    return 0;
}

int main() {
    std::printf("Running KV-cache update GPU tests (f16 packed)...\n");
    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/kv_cache_update.wgsl");

    int fail = 0;
    fail += run_test(ctx, pipeline, 4, 32, 64, 0, "pos0_4h_32s_64d");
    fail += run_test(ctx, pipeline, 4, 32, 64, 15, "mid_4h_32s_64d");
    fail += run_test(ctx, pipeline, 4, 32, 64, 31, "last_4h_32s_64d");
    fail += run_test(ctx, pipeline, 2, 16, 128, 7, "2h_16s_128d_pos7");
    fail += run_test(ctx, pipeline, 1, 8, 32, 3, "single_8s_32d_pos3");
    fail += run_test(ctx, pipeline, 8, 64, 64, 0, "8h_64s_64d_pos0");

    if (fail) {
        std::printf("\n%d test(s) FAILED\n", fail);
        return 1;
    }
    std::printf("\nAll KV-cache update GPU tests PASSED.\n");
    return 0;
}

#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"
#include "../src/reference_dequant.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f32_to_f16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static void build_random_q4_k_block(uint8_t* block, std::mt19937& rng) {
    std::memset(block, 0, Q4_K_BYTES_PER_BLOCK);

    std::uniform_real_distribution<float> scale_dist(0.01f, 2.0f);
    float d = scale_dist(rng);
    float dmin = scale_dist(rng);

    uint16_t d_bits = f32_to_f16(d);
    uint16_t dmin_bits = f32_to_f16(dmin);
    std::memcpy(block, &d_bits, 2);
    std::memcpy(block + 2, &dmin_bits, 2);

    uint8_t scales[8], mins[8];
    std::uniform_int_distribution<int> sc_dist(0, 63);
    for (int i = 0; i < 8; i++) {
        scales[i] = (uint8_t)sc_dist(rng);
        mins[i] = (uint8_t)sc_dist(rng);
    }

    uint8_t* s = block + 4;
    for (int j = 0; j < 4; j++) {
        s[j]     = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6);
        s[j + 4] = (mins[j] & 63)   | ((mins[j + 4] >> 4) << 6);
    }
    for (int j = 0; j < 4; j++) {
        s[8 + j] = (scales[j + 4] & 0xF) | ((mins[j + 4] & 0xF) << 4);
    }

    std::uniform_int_distribution<int> q_dist(0, 255);
    for (int i = 0; i < 128; i++) {
        block[16 + i] = (uint8_t)q_dist(rng);
    }
}

int main() {
    constexpr int N_BLOCKS = 16;
    constexpr int TOTAL_FLOATS = N_BLOCKS * Q4_K_BLOCK_SIZE;

    std::mt19937 rng(42);
    std::vector<uint8_t> quant_data(N_BLOCKS * Q4_K_BYTES_PER_BLOCK);
    for (int i = 0; i < N_BLOCKS; i++) {
        build_random_q4_k_block(quant_data.data() + i * Q4_K_BYTES_PER_BLOCK, rng);
    }

    std::vector<float> cpu_out = cpu_dequant_q4_k_m_n(quant_data.data(), N_BLOCKS);

    auto ctx = create_gpu_context();
    std::printf("GPU test: Q4_K_M dequant, %d blocks (%d values)\n", N_BLOCKS, TOTAL_FLOATS);

    uint64_t quant_buf_size = quant_data.size();
    if (quant_buf_size % 4 != 0) {
        quant_buf_size += 4 - (quant_buf_size % 4);
    }
    auto quant_buf = create_storage_buffer(ctx.device, quant_buf_size, quant_data.data());

    uint64_t out_buf_size = TOTAL_FLOATS * sizeof(float);
    auto output_buf = create_storage_buffer(ctx.device, out_buf_size, nullptr);

    uint32_t n_blocks = N_BLOCKS;
    auto params_buf = create_buffer(ctx.device, sizeof(uint32_t),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopySrc,
                                    &n_blocks);

    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/dequant_q4_k_m.wgsl");

    wgpu::BindGroupEntry entries[3]{};
    entries[0].binding = 0;
    entries[0].buffer = quant_buf;
    entries[0].size = quant_buf_size;
    entries[1].binding = 1;
    entries[1].buffer = output_buf;
    entries[1].size = out_buf_size;
    entries[2].binding = 2;
    entries[2].buffer = params_buf;
    entries[2].size = sizeof(uint32_t);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups((N_BLOCKS + 63) / 64);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    ctx.queue.Submit(1, &commands);

    auto gpu_out = read_buffer_as<float>(ctx.device, ctx.queue, output_buf, TOTAL_FLOATS);

    float max_err = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < TOTAL_FLOATS; i++) {
        float err = std::fabs(gpu_out[i] - cpu_out[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
            if (fail_count < 10) {
                std::fprintf(stderr, "MISMATCH [%d]: gpu=%.6f cpu=%.6f err=%.6f\n",
                             i, gpu_out[i], cpu_out[i], err);
            }
            fail_count++;
        }
    }

    std::printf("Max abs error: %.6e\n", max_err);
    if (fail_count > 0) {
        std::fprintf(stderr, "FAIL: %d/%d values exceed 1e-3 tolerance\n", fail_count, TOTAL_FLOATS);
        return 1;
    }

    std::printf("PASS: all %d values match within 1e-3 tolerance\n", TOTAL_FLOATS);
    return 0;
}

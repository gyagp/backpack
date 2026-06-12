#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"
#include "../src/reference_dequant.h"

#include <chrono>
#include <cstdio>
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

struct BenchResult {
    const char* name;
    int n_blocks;
    int total_elements;
    size_t input_bytes;
    size_t output_bytes;
    int iterations;
    double avg_ms;
    double throughput_gbps;
};

static void build_random_q8_0_blocks(uint8_t* data, int n_blocks, std::mt19937& rng) {
    std::uniform_real_distribution<float> scale_dist(0.01f, 2.0f);
    std::uniform_int_distribution<int> q_dist(-128, 127);
    for (int b = 0; b < n_blocks; b++) {
        uint8_t* block = data + b * Q8_0_BYTES_PER_BLOCK;
        std::memset(block, 0, Q8_0_BYTES_PER_BLOCK);
        float d = scale_dist(rng);
        uint16_t d_bits = f32_to_f16(d);
        std::memcpy(block, &d_bits, 2);
        int8_t* quants = reinterpret_cast<int8_t*>(block + 2);
        for (int i = 0; i < Q8_0_BLOCK_SIZE; i++)
            quants[i] = (int8_t)q_dist(rng);
    }
}

static void build_random_q4_k_m_blocks(uint8_t* data, int n_blocks, std::mt19937& rng) {
    std::uniform_real_distribution<float> scale_dist(0.01f, 2.0f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (int b = 0; b < n_blocks; b++) {
        uint8_t* block = data + b * Q4_K_BYTES_PER_BLOCK;
        std::memset(block, 0, Q4_K_BYTES_PER_BLOCK);
        float d = scale_dist(rng);
        float dmin = scale_dist(rng);
        uint16_t d_bits = f32_to_f16(d);
        uint16_t dmin_bits = f32_to_f16(dmin);
        std::memcpy(block, &d_bits, 2);
        std::memcpy(block + 2, &dmin_bits, 2);
        for (int i = 4; i < Q4_K_BYTES_PER_BLOCK; i++)
            block[i] = (uint8_t)byte_dist(rng);
    }
}

static BenchResult bench_q8_0(const GpuContext& ctx, int n_blocks, int warmup, int iters) {
    int total_floats = n_blocks * Q8_0_BLOCK_SIZE;

    std::mt19937 rng(42);
    std::vector<uint8_t> quant_data(n_blocks * Q8_0_BYTES_PER_BLOCK);
    build_random_q8_0_blocks(quant_data.data(), n_blocks, rng);

    uint64_t quant_buf_size = quant_data.size();
    if (quant_buf_size % 4 != 0)
        quant_buf_size += 4 - (quant_buf_size % 4);
    auto quant_buf = create_storage_buffer(ctx.device, quant_buf_size, quant_data.data());

    uint64_t out_buf_size = total_floats * sizeof(float);
    auto output_buf = create_storage_buffer(ctx.device, out_buf_size, nullptr);

    uint32_t nb = n_blocks;
    auto params_buf = create_buffer(ctx.device, sizeof(uint32_t),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopySrc, &nb);

    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/dequant_q8_0.wgsl");

    wgpu::BindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = quant_buf; entries[0].size = quant_buf_size;
    entries[1].binding = 1; entries[1].buffer = output_buf; entries[1].size = out_buf_size;
    entries[2].binding = 2; entries[2].buffer = params_buf; entries[2].size = sizeof(uint32_t);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    uint32_t workgroups = (n_blocks + 63) / 64;

    for (int i = 0; i < warmup; i++) {
        wgpu::CommandEncoder enc = ctx.device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(workgroups);
        pass.End();
        wgpu::CommandBuffer cmd = enc.Finish();
        ctx.queue.Submit(1, &cmd);
        ctx.device.Tick();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        wgpu::CommandEncoder enc = ctx.device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(workgroups);
        pass.End();
        wgpu::CommandBuffer cmd = enc.Finish();
        ctx.queue.Submit(1, &cmd);
    }
    ctx.device.Tick();
    // Force GPU completion by reading back a single value
    read_buffer_as<float>(ctx.device, ctx.queue, output_buf, 1);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double avg_ms = (elapsed_s / iters) * 1e3;
    size_t input_bytes = (size_t)n_blocks * Q8_0_BYTES_PER_BLOCK;
    size_t output_bytes = (size_t)total_floats * sizeof(float);
    double total_bytes = (double)(input_bytes + output_bytes) * iters;
    double gbps = total_bytes / elapsed_s / 1e9;

    return {"Q8_0", n_blocks, total_floats, input_bytes, output_bytes, iters, avg_ms, gbps};
}

static BenchResult bench_q4_k_m(const GpuContext& ctx, int n_blocks, int warmup, int iters) {
    int total_floats = n_blocks * Q4_K_BLOCK_SIZE;

    std::mt19937 rng(42);
    std::vector<uint8_t> quant_data(n_blocks * Q4_K_BYTES_PER_BLOCK);
    build_random_q4_k_m_blocks(quant_data.data(), n_blocks, rng);

    uint64_t quant_buf_size = quant_data.size();
    if (quant_buf_size % 4 != 0)
        quant_buf_size += 4 - (quant_buf_size % 4);
    auto quant_buf = create_storage_buffer(ctx.device, quant_buf_size, quant_data.data());

    uint64_t out_buf_size = total_floats * sizeof(float);
    auto output_buf = create_storage_buffer(ctx.device, out_buf_size, nullptr);

    uint32_t nb = n_blocks;
    auto params_buf = create_buffer(ctx.device, sizeof(uint32_t),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopySrc, &nb);

    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/dequant_q4_k_m.wgsl");

    wgpu::BindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = quant_buf; entries[0].size = quant_buf_size;
    entries[1].binding = 1; entries[1].buffer = output_buf; entries[1].size = out_buf_size;
    entries[2].binding = 2; entries[2].buffer = params_buf; entries[2].size = sizeof(uint32_t);

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bind_group = ctx.device.CreateBindGroup(&bg_desc);

    uint32_t workgroups = (n_blocks + 63) / 64;

    for (int i = 0; i < warmup; i++) {
        wgpu::CommandEncoder enc = ctx.device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(workgroups);
        pass.End();
        wgpu::CommandBuffer cmd = enc.Finish();
        ctx.queue.Submit(1, &cmd);
        ctx.device.Tick();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        wgpu::CommandEncoder enc = ctx.device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(workgroups);
        pass.End();
        wgpu::CommandBuffer cmd = enc.Finish();
        ctx.queue.Submit(1, &cmd);
    }
    ctx.device.Tick();
    read_buffer_as<float>(ctx.device, ctx.queue, output_buf, 1);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double avg_ms = (elapsed_s / iters) * 1e3;
    size_t input_bytes = (size_t)n_blocks * Q4_K_BYTES_PER_BLOCK;
    size_t output_bytes = (size_t)total_floats * sizeof(float);
    double total_bytes = (double)(input_bytes + output_bytes) * iters;
    double gbps = total_bytes / elapsed_s / 1e9;

    return {"Q4_K_M", n_blocks, total_floats, input_bytes, output_bytes, iters, avg_ms, gbps};
}

static void print_result(const BenchResult& r) {
    std::printf("  %s: %d blocks (%d elements), input=%.2f MB, output=%.2f MB\n",
                r.name, r.n_blocks, r.total_elements,
                r.input_bytes / 1e6, r.output_bytes / 1e6);
    std::printf("    %d iterations, avg %.3f ms/iter, throughput: %.2f GB/s\n",
                r.iterations, r.avg_ms, r.throughput_gbps);
}

int main() {
    auto ctx = create_gpu_context();
    std::printf("=== Dequantization Throughput Benchmark ===\n\n");

    struct Config { int n_blocks; int warmup; int iters; };
    Config configs[] = {
        {4096,   3, 20},   // ~131K elements (Q8_0) / ~1M elements (Q4_K_M)
        {32768,  2, 10},   // ~1M elements (Q8_0)  / ~8M elements (Q4_K_M)
        {131072, 1, 5},    // ~4M elements (Q8_0)  / ~33M elements (Q4_K_M)
    };

    std::printf("--- Q8_0 (%d values/block, %d bytes/block) ---\n",
                Q8_0_BLOCK_SIZE, Q8_0_BYTES_PER_BLOCK);
    std::vector<BenchResult> q8_results;
    for (auto& cfg : configs) {
        auto r = bench_q8_0(ctx, cfg.n_blocks, cfg.warmup, cfg.iters);
        print_result(r);
        q8_results.push_back(r);
    }

    std::printf("\n--- Q4_K_M (%d values/block, %d bytes/block) ---\n",
                Q4_K_BLOCK_SIZE, Q4_K_BYTES_PER_BLOCK);
    std::vector<BenchResult> q4_results;
    for (auto& cfg : configs) {
        auto r = bench_q4_k_m(ctx, cfg.n_blocks, cfg.warmup, cfg.iters);
        print_result(r);
        q4_results.push_back(r);
    }

    std::printf("\n=== Summary ===\n");
    double best_q8 = 0, best_q4 = 0;
    for (auto& r : q8_results) if (r.throughput_gbps > best_q8) best_q8 = r.throughput_gbps;
    for (auto& r : q4_results) if (r.throughput_gbps > best_q4) best_q4 = r.throughput_gbps;
    std::printf("  Q8_0   best throughput: %.2f GB/s\n", best_q8);
    std::printf("  Q4_K_M best throughput: %.2f GB/s\n", best_q4);

    return 0;
}

#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/shader_utils.h"
#include "../src/reference_matmul.h"

#include <chrono>
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

struct BenchResult {
    uint32_t M, N, K;
    int iterations;
    double avg_ms;
    double tflops;
    double pct_peak;
};

static BenchResult bench_matmul(const GpuContext& ctx,
                                const wgpu::ComputePipeline& pipeline,
                                uint32_t M, uint32_t N, uint32_t K,
                                int warmup, int iters,
                                double gpu_peak_tflops) {
    std::vector<float16_t> A(M * K), B(K * N);
    for (size_t i = 0; i < A.size(); i++)
        A[i] = f32_to_f16((float)((i * 7 + 3) % 17) * 0.1f - 0.8f);
    for (size_t i = 0; i < B.size(); i++)
        B[i] = f32_to_f16((float)((i * 11 + 5) % 19) * 0.1f - 0.9f);

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

    for (int i = 0; i < warmup; i++) {
        wgpu::CommandEncoder enc = ctx.device.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(wg_x, wg_y, 1);
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
        pass.DispatchWorkgroups(wg_x, wg_y, 1);
        pass.End();
        wgpu::CommandBuffer cmd = enc.Finish();
        ctx.queue.Submit(1, &cmd);
    }
    ctx.device.Tick();
    read_buffer_as<float>(ctx.device, ctx.queue, buf_c, 1);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double avg_ms = (elapsed_s / iters) * 1e3;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (elapsed_s / iters)) / 1e12;
    double pct = (tflops / gpu_peak_tflops) * 100.0;

    return {M, N, K, iters, avg_ms, tflops, pct};
}

int main() {
    const char* peak_env = std::getenv("GPU_PEAK_TFLOPS");
    double gpu_peak_tflops = peak_env ? std::atof(peak_env) : 165.0;

    std::printf("=== GPU Matmul Tiled f16 Benchmark ===\n");
    std::printf("Shader: matmul_tiled_f16.wgsl\n");
    std::printf("GPU theoretical peak: %.1f TFLOPS (set GPU_PEAK_TFLOPS to override)\n\n",
                gpu_peak_tflops);

    auto ctx = create_gpu_context();
    auto pipeline = load_compute_pipeline(ctx.device, "src/shaders/matmul_tiled_f16.wgsl");

    struct Config {
        uint32_t M, N, K;
        int warmup, iters;
    } configs[] = {
        {1024, 1024, 1024, 5, 20},
        {2048, 2048, 2048, 3, 10},
        {4096, 4096, 4096, 2, 5},
    };

    std::vector<BenchResult> results;

    for (auto& cfg : configs) {
        std::printf("Benchmarking %ux%ux%u ...\n", cfg.M, cfg.N, cfg.K);
        auto r = bench_matmul(ctx, pipeline, cfg.M, cfg.N, cfg.K,
                              cfg.warmup, cfg.iters, gpu_peak_tflops);
        std::printf("  %ux%ux%u: %.2f ms/iter, %.4f TFLOPS, %.2f%% of GPU peak\n",
                    r.M, r.N, r.K, r.avg_ms, r.tflops, r.pct_peak);
        results.push_back(r);
    }

    std::printf("\n=== Summary ===\n");
    double best_tflops = 0;
    for (auto& r : results)
        if (r.tflops > best_tflops) best_tflops = r.tflops;

    std::printf("Best: %.4f TFLOPS (%.2f%% of %.1f TFLOPS GPU peak)\n",
                best_tflops, (best_tflops / gpu_peak_tflops) * 100.0, gpu_peak_tflops);

    return 0;
}

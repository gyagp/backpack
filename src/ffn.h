#pragma once

#include <dawn/webgpu_cpp.h>
#include <cstdint>
#include <vector>

#include "buffer_utils.h"
#include "dispatch.h"
#include "shader_utils.h"

struct MatmulTiledParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batch_size;
    uint32_t stride_A;
    uint32_t stride_C;
};

struct FusedGateUpParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batch_size;
    uint32_t stride_input;
    uint32_t stride_out;
};

inline void dispatch_fused_gate_up_silu_enc(const wgpu::Device& device,
                                            wgpu::CommandEncoder& encoder,
                                            const wgpu::ComputePipeline& pipeline,
                                            wgpu::Buffer input_f16,
                                            wgpu::Buffer gate_w_f16,
                                            wgpu::Buffer up_w_f16,
                                            wgpu::Buffer out_f32,
                                            uint32_t M, uint32_t N, uint32_t K) {
    FusedGateUpParams p{M, N, K, 1, (M * K + 1) / 2, M * N};
    auto params_buf = create_buffer(device, sizeof(FusedGateUpParams),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, &p);

    wgpu::BindGroupEntry entries[5] = {};
    entries[0].binding = 0; entries[0].buffer = input_f16;  entries[0].size = input_f16.GetSize();
    entries[1].binding = 1; entries[1].buffer = gate_w_f16; entries[1].size = gate_w_f16.GetSize();
    entries[2].binding = 2; entries[2].buffer = up_w_f16;   entries[2].size = up_w_f16.GetSize();
    entries[3].binding = 3; entries[3].buffer = out_f32;    entries[3].size = out_f32.GetSize();
    entries[4].binding = 4; entries[4].buffer = params_buf; entries[4].size = params_buf.GetSize();

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 5;
    bg_desc.entries = entries;
    wgpu::BindGroup bg = device.CreateBindGroup(&bg_desc);

    uint32_t wg_x = (M + 127) / 128;
    uint32_t wg_y = (N + 127) / 128;

    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups(wg_x, wg_y, 1);
    pass.End();
}

inline void dispatch_matmul_tiled_f16_enc(const wgpu::Device& device,
                                          wgpu::CommandEncoder& encoder,
                                          const wgpu::ComputePipeline& pipeline,
                                          wgpu::Buffer A_f16, wgpu::Buffer B_f16,
                                          wgpu::Buffer C_f32,
                                          uint32_t M, uint32_t N, uint32_t K) {
    MatmulTiledParams p{M, N, K, 1, (M * K + 1) / 2, M * N};
    auto params_buf = create_buffer(device, sizeof(MatmulTiledParams),
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, &p);

    wgpu::BindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = A_f16; entries[0].size = A_f16.GetSize();
    entries[1].binding = 1; entries[1].buffer = B_f16; entries[1].size = B_f16.GetSize();
    entries[2].binding = 2; entries[2].buffer = C_f32; entries[2].size = C_f32.GetSize();
    entries[3].binding = 3; entries[3].buffer = params_buf; entries[3].size = params_buf.GetSize();

    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = 4;
    bg_desc.entries = entries;
    wgpu::BindGroup bg = device.CreateBindGroup(&bg_desc);

    uint32_t wg_x = (M + 127) / 128;
    uint32_t wg_y = (N + 127) / 128;

    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups(wg_x, wg_y, 1);
    pass.End();
}

inline void dispatch_matmul_tiled_f16(const wgpu::Device& device,
                                      const wgpu::Queue& queue,
                                      const wgpu::ComputePipeline& pipeline,
                                      wgpu::Buffer A_f16, wgpu::Buffer B_f16,
                                      wgpu::Buffer C_f32,
                                      uint32_t M, uint32_t N, uint32_t K) {
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    dispatch_matmul_tiled_f16_enc(device, encoder, pipeline, A_f16, B_f16, C_f32, M, N, K);
    wgpu::CommandBuffer cmd = encoder.Finish();
    queue.Submit(1, &cmd);
}

// FFN forward pass for LLaMA-style models:
//   output = down_w @ (silu(gate_w @ hidden) * (up_w @ hidden))
//
// hidden:  [M x K] f16 packed as array<u32>
// gate_w:  [K x N_ff] f16 packed
// up_w:    [K x N_ff] f16 packed
// down_w:  [N_ff x K] f16 packed  (note: output dim = K)
// output:  [M x K] f32
//
// M = sequence length (typically 1 for token-at-a-time)
// K = hidden_dim
// N_ff = intermediate_dim
inline wgpu::Buffer ffn_forward(const wgpu::Device& device,
                                const wgpu::Queue& queue,
                                PipelineCache& pipeline_cache,
                                wgpu::Buffer hidden_f16,
                                wgpu::Buffer gate_w_f16,
                                wgpu::Buffer up_w_f16,
                                wgpu::Buffer down_w_f16,
                                uint32_t M, uint32_t hidden_dim, uint32_t intermediate_dim,
                                wgpu::CommandEncoder* ext_encoder = nullptr) {
    auto fused_gate_up_pipeline = pipeline_cache.get("src/shaders/fused_gate_up_silu.wgsl");
    auto matmul_pipeline = pipeline_cache.get("src/shaders/matmul_tiled_f16.wgsl");
    auto pack_pipeline = pipeline_cache.get("src/shaders/pack_f32_to_f16.wgsl");

    wgpu::CommandEncoder local_encoder;
    if (!ext_encoder) {
        local_encoder = device.CreateCommandEncoder();
        ext_encoder = &local_encoder;
    }
    wgpu::CommandEncoder& enc = *ext_encoder;

    uint32_t K = hidden_dim;
    uint32_t N_ff = intermediate_dim;

    // Step 1: fused = silu(hidden @ gate_w) * (hidden @ up_w) -> [M x N_ff] f32
    auto fused = create_storage_buffer(device, M * N_ff * sizeof(float), nullptr);
    dispatch_fused_gate_up_silu_enc(device, enc, fused_gate_up_pipeline,
                                   hidden_f16, gate_w_f16, up_w_f16, fused,
                                   M, N_ff, K);

    // Step 2: pack fused f32 -> f16 for down projection
    uint32_t fused_elems = M * N_ff;
    uint32_t fused_u32 = (fused_elems + 1) / 2;
    auto fused_f16 = create_storage_buffer(device, fused_u32 * sizeof(uint32_t), nullptr);
    dispatch_elementwise_enc(device, enc, pack_pipeline, {fused, fused_f16},
                         fused_u32, 64);

    // Step 3: output = matmul(fused_f16 [M x N_ff], down_w [N_ff x K]) -> [M x K] f32
    auto output = create_storage_buffer(device, M * K * sizeof(float), nullptr);
    dispatch_matmul_tiled_f16_enc(device, enc, matmul_pipeline,
                              fused_f16, down_w_f16, output, M, K, N_ff);

    if (local_encoder != nullptr) {
        wgpu::CommandBuffer cmd = enc.Finish();
        queue.Submit(1, &cmd);
    }

    return output;
}

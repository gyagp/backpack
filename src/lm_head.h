#pragma once

#include <dawn/webgpu_cpp.h>
#include <cstdint>

#include "buffer_utils.h"
#include "dispatch.h"
#include "shader_utils.h"
#include "transformer_layer.h"

// lm_head_forward: final RMSNorm + linear projection to vocab logits
//
// hidden:        [seq_len × hidden_dim] f32
// norm_weight:   [hidden_dim] f32
// output_weight: [hidden_dim × vocab_size] f16 packed as u32
//
// Returns: [seq_len × vocab_size] f32 logits
inline wgpu::Buffer lm_head_forward(const wgpu::Device& device,
                                    const wgpu::Queue& queue,
                                    PipelineCache& pipeline_cache,
                                    wgpu::Buffer hidden,
                                    wgpu::Buffer norm_weight,
                                    wgpu::Buffer output_weight,
                                    uint32_t seq_len,
                                    uint32_t hidden_dim,
                                    uint32_t vocab_size,
                                    float norm_epsilon) {
    auto rmsnorm_pipeline = pipeline_cache.get("src/shaders/rmsnorm.wgsl");
    auto mul_pipeline = pipeline_cache.get("src/shaders/mul.wgsl");
    auto pack_pipeline = pipeline_cache.get("src/shaders/pack_f32_to_f16.wgsl");
    auto matmul_pipeline = pipeline_cache.get("src/shaders/matmul_tiled_f16.wgsl");

    uint32_t total = seq_len * hidden_dim;

    // 1. RMSNorm
    auto normed = create_storage_buffer(device, total * sizeof(float), nullptr);
    detail::dispatch_rmsnorm(device, queue, rmsnorm_pipeline,
                             hidden, normed, seq_len, hidden_dim, norm_epsilon);

    // 2. Multiply by norm weights (elementwise, seq_len must be 1)
    auto normed_w = create_storage_buffer(device, total * sizeof(float), nullptr);
    dispatch_elementwise(device, queue, mul_pipeline,
                         {normed, norm_weight, normed_w}, hidden_dim);

    // 3. Pack to f16
    uint32_t u32_count = (total + 1) / 2;
    auto normed_f16 = create_storage_buffer(device, u32_count * sizeof(uint32_t), nullptr);
    dispatch_elementwise(device, queue, pack_pipeline, {normed_w, normed_f16}, u32_count, 64);

    // 4. Matmul: [seq_len × hidden_dim] @ [hidden_dim × vocab_size] -> [seq_len × vocab_size]
    auto logits = create_storage_buffer(device, seq_len * vocab_size * sizeof(float), nullptr);
    dispatch_matmul_tiled_f16(device, queue, matmul_pipeline,
                              normed_f16, output_weight, logits, seq_len, vocab_size, hidden_dim);

    return logits;
}

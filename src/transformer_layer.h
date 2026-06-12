#pragma once

#include <dawn/webgpu_cpp.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "buffer_utils.h"
#include "dispatch.h"
#include "ffn.h"
#include "shader_utils.h"

struct TransformerLayer {
    wgpu::Buffer attn_norm;   // [hidden_dim] f32
    wgpu::Buffer ffn_norm;    // [hidden_dim] f32
    wgpu::Buffer w_q;         // [hidden_dim × n_heads*head_dim] f16 packed as u32
    wgpu::Buffer w_k;         // [hidden_dim × n_kv_heads*head_dim] f16 packed
    wgpu::Buffer w_v;         // [hidden_dim × n_kv_heads*head_dim] f16 packed
    wgpu::Buffer w_o;         // [n_heads*head_dim × hidden_dim] f16 packed
    wgpu::Buffer w_gate;      // [hidden_dim × intermediate_dim] f16 packed
    wgpu::Buffer w_up;        // [hidden_dim × intermediate_dim] f16 packed
    wgpu::Buffer w_down;      // [intermediate_dim × hidden_dim] f16 packed
};

struct TransformerLayerConfig {
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq_len;
    float rope_theta;
    float norm_epsilon;
};

struct TransformerPipelines {
    wgpu::ComputePipeline rmsnorm_scaled;
    wgpu::ComputePipeline add;
    wgpu::ComputePipeline matmul;
    wgpu::ComputePipeline pack;
    wgpu::ComputePipeline rope;
    wgpu::ComputePipeline kv_cache;
    wgpu::ComputePipeline attention;
    wgpu::ComputePipeline fused_qkv;
};

inline TransformerPipelines load_transformer_pipelines(PipelineCache& cache) {
    return {
        cache.get("src/shaders/rmsnorm_scaled.wgsl"),
        cache.get("src/shaders/add.wgsl"),
        cache.get("src/shaders/matmul_tiled_f16.wgsl"),
        cache.get("src/shaders/pack_f32_to_f16.wgsl"),
        cache.get("src/shaders/rope.wgsl"),
        cache.get("src/shaders/kv_cache_update.wgsl"),
        cache.get("src/shaders/flash_attention.wgsl"),
        cache.get("src/shaders/fused_qkv.wgsl"),
    };
}

namespace detail {

inline void encode_compute(const wgpu::Device& device, wgpu::CommandEncoder& enc,
                           const wgpu::ComputePipeline& pipeline,
                           const wgpu::BindGroupEntry* entries, uint32_t entry_count,
                           uint32_t wg_x, uint32_t wg_y = 1, uint32_t wg_z = 1) {
    wgpu::BindGroupDescriptor bg_desc{};
    bg_desc.layout = pipeline.GetBindGroupLayout(0);
    bg_desc.entryCount = entry_count;
    bg_desc.entries = entries;
    wgpu::BindGroup bg = device.CreateBindGroup(&bg_desc);

    wgpu::ComputePassEncoder pass = enc.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups(wg_x, wg_y, wg_z);
    pass.End();
}

inline void submit_compute(const wgpu::Device& device, const wgpu::Queue& queue,
                           const wgpu::ComputePipeline& pipeline,
                           const wgpu::BindGroupEntry* entries, uint32_t entry_count,
                           uint32_t wg_x, uint32_t wg_y = 1, uint32_t wg_z = 1) {
    wgpu::CommandEncoder enc = device.CreateCommandEncoder();
    encode_compute(device, enc, pipeline, entries, entry_count, wg_x, wg_y, wg_z);
    wgpu::CommandBuffer cmd = enc.Finish();
    queue.Submit(1, &cmd);
}

inline wgpu::BindGroupEntry make_entry(uint32_t binding, wgpu::Buffer buf) {
    wgpu::BindGroupEntry e{};
    e.binding = binding;
    e.buffer = buf;
    e.offset = 0;
    e.size = buf.GetSize();
    return e;
}

inline void dispatch_rmsnorm(const wgpu::Device& device, const wgpu::Queue& queue,
                             const wgpu::ComputePipeline& pipeline,
                             wgpu::Buffer input, wgpu::Buffer output,
                             uint32_t rows, uint32_t row_length, float epsilon) {
    struct { uint32_t row_length; float epsilon; } params{row_length, epsilon};
    auto pb = create_buffer(device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, &params);
    wgpu::BindGroupEntry entries[3] = {
        make_entry(0, input), make_entry(1, output), make_entry(2, pb)
    };
    submit_compute(device, queue, pipeline, entries, 3, (rows + 63) / 64);
}

inline void encode_rmsnorm_scaled(const wgpu::Device& device, wgpu::CommandEncoder& enc,
                           const wgpu::ComputePipeline& pipeline,
                           wgpu::Buffer input, wgpu::Buffer weights, wgpu::Buffer output,
                           uint32_t rows, uint32_t row_length, float epsilon) {
    struct { uint32_t row_length; float epsilon; } params{row_length, epsilon};
    auto pb = create_buffer(device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, &params);
    wgpu::BindGroupEntry entries[4] = {
        make_entry(0, input), make_entry(1, weights),
        make_entry(2, output), make_entry(3, pb)
    };
    encode_compute(device, enc, pipeline, entries, 4, rows);
}

inline void encode_rope(const wgpu::Device& device, wgpu::CommandEncoder& enc,
                        const wgpu::ComputePipeline& pipeline,
                        wgpu::Buffer input, wgpu::Buffer output,
                        uint32_t num_heads, uint32_t head_dim, float theta,
                        uint32_t seq_pos_offset = 0, uint32_t M = 1) {
    uint32_t theta_bits;
    std::memcpy(&theta_bits, &theta, 4);
    uint32_t params[8] = {head_dim, num_heads, theta_bits, 0, seq_pos_offset, M, 0, 0};
    auto pb = create_buffer(device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, params);
    wgpu::BindGroupEntry entries[3] = {
        make_entry(0, input), make_entry(1, output), make_entry(2, pb)
    };
    uint32_t total = M * num_heads * head_dim;
    encode_compute(device, enc, pipeline, entries, 3, (total + 63) / 64);
}

inline void encode_kv_cache_update(const wgpu::Device& device, wgpu::CommandEncoder& enc,
                                   const wgpu::ComputePipeline& pipeline,
                                   wgpu::Buffer k_cache, wgpu::Buffer v_cache,
                                   wgpu::Buffer new_k, wgpu::Buffer new_v,
                                   uint32_t n_kv_heads, uint32_t max_seq_len,
                                   uint32_t head_dim, uint32_t seq_pos,
                                   uint32_t M = 1) {
    uint32_t params[8] = {n_kv_heads, max_seq_len, head_dim, seq_pos, M, 0, 0, 0};
    auto pb = create_buffer(device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, params);
    wgpu::BindGroupEntry entries[5] = {
        make_entry(0, k_cache), make_entry(1, v_cache),
        make_entry(2, new_k), make_entry(3, new_v), make_entry(4, pb)
    };
    uint32_t total = M * n_kv_heads * head_dim;
    encode_compute(device, enc, pipeline, entries, 5, (total + 255) / 256);
}

inline void encode_attention(const wgpu::Device& device, wgpu::CommandEncoder& enc,
                             const wgpu::ComputePipeline& pipeline,
                             wgpu::Buffer Q, wgpu::Buffer K, wgpu::Buffer V,
                             wgpu::Buffer output,
                             uint32_t kv_seq_len, uint32_t head_dim,
                             uint32_t num_heads, uint32_t num_kv_heads,
                             uint32_t q_seq_len = 1, uint32_t q_seq_offset = 0) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    uint32_t params[8] = {kv_seq_len, head_dim, num_heads, num_kv_heads, scale_bits, q_seq_offset, 0, 0};
    auto pb = create_buffer(device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, params);
    wgpu::BindGroupEntry entries[5] = {
        make_entry(0, Q), make_entry(1, K), make_entry(2, V),
        make_entry(3, output), make_entry(4, pb)
    };
    encode_compute(device, enc, pipeline, entries, 5, num_heads, q_seq_len);
}

inline wgpu::Buffer pack_f32_to_f16(const wgpu::Device& device, const wgpu::Queue& queue,
                                     const wgpu::ComputePipeline& pipeline,
                                     wgpu::Buffer input_f32, uint32_t element_count) {
    uint32_t u32_count = (element_count + 1) / 2;
    auto output = create_storage_buffer(device, u32_count * sizeof(uint32_t), nullptr);
    dispatch_elementwise(device, queue, pipeline, {input_f32, output}, u32_count, 64);
    return output;
}

inline wgpu::Buffer encode_pack_f32_to_f16(const wgpu::Device& device, wgpu::CommandEncoder& enc,
                                            const wgpu::ComputePipeline& pipeline,
                                            wgpu::Buffer input_f32, uint32_t element_count) {
    uint32_t u32_count = (element_count + 1) / 2;
    auto output = create_storage_buffer(device, u32_count * sizeof(uint32_t), nullptr);
    dispatch_elementwise_enc(device, enc, pipeline, {input_f32, output}, u32_count, 64);
    return output;
}

inline void encode_fused_qkv(const wgpu::Device& device, wgpu::CommandEncoder& enc,
                             const wgpu::ComputePipeline& pipeline,
                             wgpu::Buffer input_f16,
                             wgpu::Buffer Wq, wgpu::Buffer Wk, wgpu::Buffer Wv,
                             wgpu::Buffer outQ, wgpu::Buffer outK, wgpu::Buffer outV,
                             uint32_t M, uint32_t N, uint32_t K) {
    struct { uint32_t M, N, K, batch_size, stride_input, stride_out; }
        params{M, N, K, 1, (M * K + 1) / 2, M * N};
    auto pb = create_buffer(device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, &params);
    wgpu::BindGroupEntry entries[8] = {
        make_entry(0, input_f16), make_entry(1, Wq), make_entry(2, Wk),
        make_entry(3, Wv), make_entry(4, outQ), make_entry(5, outK),
        make_entry(6, outV), make_entry(7, pb)
    };
    uint32_t wg_x = (M + 127) / 128;
    uint32_t wg_y = (N + 127) / 128;
    encode_compute(device, enc, pipeline, entries, 8, wg_x, wg_y, 1);
}

} // namespace detail

// Transformer layer forward pass supporting batched prefill (M>1) and decode (M=1).
// hidden: [M × hidden_dim] f32
// k_cache, v_cache: [n_kv_heads × max_seq_len × head_dim] f32 (pre-allocated)
// seq_pos: starting sequence position for this batch
// M: number of tokens in this batch
// Returns: [M × hidden_dim] f32
inline wgpu::Buffer transformer_layer_forward(
    const wgpu::Device& device,
    const wgpu::Queue& queue,
    const TransformerPipelines& pipelines,
    PipelineCache& pipeline_cache,
    const TransformerLayer& layer,
    const TransformerLayerConfig& cfg,
    wgpu::Buffer hidden,
    wgpu::Buffer k_cache,
    wgpu::Buffer v_cache,
    uint32_t seq_pos,
    uint32_t M = 1)
{
    uint32_t D = cfg.hidden_dim;
    uint32_t n_h = cfg.n_heads;
    uint32_t n_kv = cfg.n_kv_heads;
    uint32_t hd = cfg.head_dim;
    uint32_t kv_seq_len = seq_pos + M;

    wgpu::CommandEncoder enc = device.CreateCommandEncoder();

    // --- Attention block ---

    // 1. Fused RMSNorm + scale by attention norm weights
    auto normed_w = create_storage_buffer(device, M * D * sizeof(float), nullptr);
    detail::encode_rmsnorm_scaled(device, enc, pipelines.rmsnorm_scaled,
                           hidden, layer.attn_norm, normed_w, M, D, cfg.norm_epsilon);

    // 2. Pack to f16 for matmul
    auto normed_f16 = detail::encode_pack_f32_to_f16(device, enc, pipelines.pack, normed_w, M * D);

    // 3. QKV projections
    uint32_t q_dim = n_h * hd;
    uint32_t kv_dim = n_kv * hd;

    auto q_f32 = create_storage_buffer(device, M * q_dim * sizeof(float), nullptr);
    auto k_f32 = create_storage_buffer(device, M * kv_dim * sizeof(float), nullptr);
    auto v_f32 = create_storage_buffer(device, M * kv_dim * sizeof(float), nullptr);
    {
        if (q_dim == kv_dim) {
            detail::encode_fused_qkv(device, enc, pipelines.fused_qkv,
                                     normed_f16, layer.w_q, layer.w_k, layer.w_v,
                                     q_f32, k_f32, v_f32, M, q_dim, D);
        } else {
            dispatch_matmul_tiled_f16_enc(device, enc, pipelines.matmul,
                                          normed_f16, layer.w_q, q_f32, M, q_dim, D);

            dispatch_matmul_tiled_f16_enc(device, enc, pipelines.matmul,
                                          normed_f16, layer.w_k, k_f32, M, kv_dim, D);

            dispatch_matmul_tiled_f16_enc(device, enc, pipelines.matmul,
                                          normed_f16, layer.w_v, v_f32, M, kv_dim, D);
        }
    }

    // 4. RoPE on Q and K
    auto q_roped = create_storage_buffer(device, M * q_dim * sizeof(float), nullptr);
    auto k_roped = create_storage_buffer(device, M * kv_dim * sizeof(float), nullptr);
    detail::encode_rope(device, enc, pipelines.rope,
                        q_f32, q_roped, n_h, hd, cfg.rope_theta, seq_pos, M);
    detail::encode_rope(device, enc, pipelines.rope,
                        k_f32, k_roped, n_kv, hd, cfg.rope_theta, seq_pos, M);

    // 5. KV cache update
    detail::encode_kv_cache_update(device, enc, pipelines.kv_cache,
                                   k_cache, v_cache, k_roped, v_f32,
                                   n_kv, cfg.max_seq_len, hd, seq_pos, M);

    // 6. Attention
    auto attn_out = create_storage_buffer(device, M * q_dim * sizeof(float), nullptr);
    detail::encode_attention(device, enc, pipelines.attention,
                             q_roped, k_cache, v_cache, attn_out,
                             kv_seq_len, hd, n_h, n_kv, M, seq_pos);

    // 7. Output projection
    auto attn_f16 = detail::encode_pack_f32_to_f16(device, enc, pipelines.pack, attn_out, M * q_dim);
    auto projected = create_storage_buffer(device, M * D * sizeof(float), nullptr);
    dispatch_matmul_tiled_f16_enc(device, enc, pipelines.matmul,
                              attn_f16, layer.w_o, projected, M, D, q_dim);

    // 8. Residual add: hidden + projected
    auto post_attn = create_storage_buffer(device, M * D * sizeof(float), nullptr);
    dispatch_elementwise_enc(device, enc, pipelines.add,
                         {hidden, projected, post_attn}, M * D);

    // --- FFN block ---

    // 3. Fused RMSNorm + scale by FFN norm weights
    auto normed2_w = create_storage_buffer(device, M * D * sizeof(float), nullptr);
    detail::encode_rmsnorm_scaled(device, enc, pipelines.rmsnorm_scaled,
                           post_attn, layer.ffn_norm, normed2_w, M, D, cfg.norm_epsilon);

    // 4. Pack to f16 for FFN
    auto normed2_f16 = detail::encode_pack_f32_to_f16(device, enc, pipelines.pack, normed2_w, M * D);

    // 5. FFN forward
    auto ffn_out = ffn_forward(device, queue, pipeline_cache, normed2_f16,
                               layer.w_gate, layer.w_up, layer.w_down,
                               M, D, cfg.intermediate_dim, &enc);

    // 6. Residual add: post_attn + ffn_out
    auto output = create_storage_buffer(device, M * D * sizeof(float), nullptr);
    dispatch_elementwise_enc(device, enc, pipelines.add,
                         {post_attn, ffn_out, output}, M * D);

    // Single submit for the entire layer
    wgpu::CommandBuffer cmd = enc.Finish();
    queue.Submit(1, &cmd);

    return output;
}

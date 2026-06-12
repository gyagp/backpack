#pragma once

#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "argmax.h"
#include "buffer_utils.h"
#include "chat_template.h"
#include "gguf_parser.h"
#include "gpu_context.h"
#include "lm_head.h"
#include "mmap_file.h"
#include "model_arch.h"
#include "model_config.h"
#include "sampling.h"
#include "shader_utils.h"
#include "tensor.h"
#include "tokenizer.h"
#include "transformer_layer.h"
#include "weight_loader.h"

struct GenerateResult {
    std::vector<uint32_t> tokens;
    double prefill_tok_per_sec;
    double decode_tok_per_sec;
};

struct GenerateParams {
    uint32_t max_tokens = 128;
    float temperature = 0.0f;
    uint32_t top_k = 40;
    float top_p = 0.9f;
    float norm_epsilon = 1e-5f;
    GpuBackend backend = GpuBackend::Default;
};

inline wgpu::Buffer embedding_lookup(const wgpu::Device& device,
                                     const wgpu::Queue& queue,
                                     const Tensor& embed_table,
                                     uint32_t token_id,
                                     uint32_t hidden_dim) {
    auto row = create_storage_buffer(device, hidden_dim * sizeof(float), nullptr);

    if (embed_table.dtype == DType::f32) {
        uint64_t offset = static_cast<uint64_t>(token_id) * hidden_dim * sizeof(float);
        wgpu::CommandEncoder enc = device.CreateCommandEncoder();
        enc.CopyBufferToBuffer(embed_table.buffer, offset, row, 0,
                               hidden_dim * sizeof(float));
        wgpu::CommandBuffer cmd = enc.Finish();
        queue.Submit(1, &cmd);
    } else if (embed_table.dtype == DType::f16) {
        struct { uint32_t offset; uint32_t count; } params;
        params.offset = token_id * hidden_dim;
        params.count = hidden_dim;
        auto params_buf = create_buffer(device, sizeof(params),
                                        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                        &params);

        static wgpu::ComputePipeline pipeline = nullptr;
        if (!pipeline) {
            pipeline = load_compute_pipeline(device, "src/shaders/embed_f16_to_f32.wgsl");
        }

        wgpu::BindGroupEntry entries[3] = {};
        entries[0].binding = 0; entries[0].buffer = embed_table.buffer; entries[0].size = embed_table.buffer.GetSize();
        entries[1].binding = 1; entries[1].buffer = row;                entries[1].size = row.GetSize();
        entries[2].binding = 2; entries[2].buffer = params_buf;         entries[2].size = params_buf.GetSize();

        wgpu::BindGroupDescriptor bg_desc{};
        bg_desc.layout = pipeline.GetBindGroupLayout(0);
        bg_desc.entryCount = 3;
        bg_desc.entries = entries;
        auto bind_group = device.CreateBindGroup(&bg_desc);

        wgpu::CommandEncoder enc = device.CreateCommandEncoder();
        auto pass = enc.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups((hidden_dim + 63) / 64);
        pass.End();
        queue.Submit(1, &enc.Finish());
    } else {
        throw std::runtime_error("embedding_lookup: unsupported dtype");
    }

    return row;
}

inline wgpu::Buffer embedding_lookup_batch(const wgpu::Device& device,
                                           const wgpu::Queue& queue,
                                           const Tensor& embed_table,
                                           const std::vector<uint32_t>& token_ids,
                                           uint32_t hidden_dim) {
    uint32_t M = static_cast<uint32_t>(token_ids.size());
    auto output = create_storage_buffer(device, M * hidden_dim * sizeof(float), nullptr);

    if (embed_table.dtype == DType::f32) {
        wgpu::CommandEncoder enc = device.CreateCommandEncoder();
        for (uint32_t i = 0; i < M; i++) {
            uint64_t src_offset = static_cast<uint64_t>(token_ids[i]) * hidden_dim * sizeof(float);
            uint64_t dst_offset = static_cast<uint64_t>(i) * hidden_dim * sizeof(float);
            enc.CopyBufferToBuffer(embed_table.buffer, src_offset, output, dst_offset,
                                   hidden_dim * sizeof(float));
        }
        wgpu::CommandBuffer cmd = enc.Finish();
        queue.Submit(1, &cmd);
    } else if (embed_table.dtype == DType::f16) {
        static wgpu::ComputePipeline pipeline = nullptr;
        if (!pipeline) {
            pipeline = load_compute_pipeline(device, "src/shaders/embed_f16_to_f32.wgsl");
        }

        wgpu::CommandEncoder enc = device.CreateCommandEncoder();
        for (uint32_t i = 0; i < M; i++) {
            struct { uint32_t offset; uint32_t count; } params;
            params.offset = token_ids[i] * hidden_dim;
            params.count = hidden_dim;
            auto params_buf = create_buffer(device, sizeof(params),
                                            wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                            &params);

            wgpu::BindGroupEntry entries[3] = {};
            entries[0].binding = 0; entries[0].buffer = embed_table.buffer; entries[0].size = embed_table.buffer.GetSize();
            entries[1].binding = 1; entries[1].buffer = output;              entries[1].size = output.GetSize();
            entries[2].binding = 2; entries[2].buffer = params_buf;          entries[2].size = params_buf.GetSize();

            // Need a per-token output view — use offset in the bind group
            // Actually embed_f16_to_f32.wgsl writes to output[i] where i = gid.x (0..count-1)
            // So we need separate dispatches with separate output buffers, or modify the shader.
            // Simpler: create per-token output slices and copy. But buffers can't be sub-sliced.
            // Instead, do individual lookups into temp buffers and copy.
            auto row = create_storage_buffer(device, hidden_dim * sizeof(float), nullptr);
            entries[1].buffer = row; entries[1].size = row.GetSize();

            wgpu::BindGroupDescriptor bg_desc{};
            bg_desc.layout = pipeline.GetBindGroupLayout(0);
            bg_desc.entryCount = 3;
            bg_desc.entries = entries;
            auto bind_group = device.CreateBindGroup(&bg_desc);

            auto pass = enc.BeginComputePass();
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, bind_group);
            pass.DispatchWorkgroups((hidden_dim + 63) / 64);
            pass.End();

            enc.CopyBufferToBuffer(row, 0, output,
                                   static_cast<uint64_t>(i) * hidden_dim * sizeof(float),
                                   hidden_dim * sizeof(float));
        }
        wgpu::CommandBuffer cmd = enc.Finish();
        queue.Submit(1, &cmd);
    } else {
        throw std::runtime_error("embedding_lookup_batch: unsupported dtype");
    }

    return output;
}

inline GenerateResult generate(const std::string& gguf_path,
                               const std::string& prompt,
                               const GenerateParams& params = {}) {
    auto ctx = create_gpu_context(params.backend);
    MmapFile mmap(gguf_path);
    auto gguf = GGUFFile::parse(mmap.data(), mmap.size());
    auto config = parse_model_config(gguf);
    auto weights = load_weights(gguf, mmap, ctx);

    auto vocab = extract_vocab(gguf);
    BPETokenizer tokenizer(vocab);

    auto formatted = format_chat({{  "user", prompt }}, true);
    auto input_ids = tokenizer.encode(formatted);

    PipelineCache pipeline_cache(ctx.device);
    auto pipelines = load_transformer_pipelines(pipeline_cache);

    Tensor& embed_table = weights.at("token_embd.weight");
    wgpu::Buffer output_norm = weights.at("output_norm.weight").buffer;
    wgpu::Buffer output_weight = weights.at("output.weight").buffer;

    uint32_t n_layers = config.n_layers;
    uint32_t hidden_dim = config.hidden_dim;
    uint32_t head_dim = hidden_dim / config.n_heads;
    uint32_t max_seq_len = (std::min)(config.context_length, params.max_tokens + static_cast<uint32_t>(input_ids.size()));

    TransformerLayerConfig layer_cfg{};
    layer_cfg.hidden_dim = hidden_dim;
    layer_cfg.intermediate_dim = config.intermediate_dim;
    layer_cfg.n_heads = config.n_heads;
    layer_cfg.n_kv_heads = config.n_kv_heads;
    layer_cfg.head_dim = head_dim;
    layer_cfg.max_seq_len = max_seq_len;
    layer_cfg.rope_theta = config.rope_theta;
    layer_cfg.norm_epsilon = params.norm_epsilon;

    std::vector<TransformerLayer> layers(n_layers);
    for (uint32_t i = 0; i < n_layers; i++) {
        std::string pfx = "blk." + std::to_string(i) + ".";
        layers[i].attn_norm = weights.at(pfx + "attn_norm.weight").buffer;
        layers[i].ffn_norm  = weights.at(pfx + "ffn_norm.weight").buffer;
        layers[i].w_q       = weights.at(pfx + "attn_q.weight").buffer;
        layers[i].w_k       = weights.at(pfx + "attn_k.weight").buffer;
        layers[i].w_v       = weights.at(pfx + "attn_v.weight").buffer;
        layers[i].w_o       = weights.at(pfx + "attn_output.weight").buffer;
        layers[i].w_gate    = weights.at(pfx + "ffn_gate.weight").buffer;
        layers[i].w_up      = weights.at(pfx + "ffn_up.weight").buffer;
        layers[i].w_down    = weights.at(pfx + "ffn_down.weight").buffer;
    }

    uint64_t kv_cache_size = static_cast<uint64_t>(config.n_kv_heads) * max_seq_len * head_dim * sizeof(float);
    std::vector<wgpu::Buffer> k_caches(n_layers), v_caches(n_layers);
    for (uint32_t i = 0; i < n_layers; i++) {
        k_caches[i] = create_storage_buffer(ctx.device, kv_cache_size, nullptr);
        v_caches[i] = create_storage_buffer(ctx.device, kv_cache_size, nullptr);
    }

    std::mt19937 rng(42);
    GenerateResult result{};

    // === Prefill phase: process all input tokens in one batched forward pass ===
    auto prefill_start = std::chrono::high_resolution_clock::now();

    uint32_t prefill_len = static_cast<uint32_t>(input_ids.size());
    wgpu::Buffer hidden = embedding_lookup_batch(ctx.device, ctx.queue, embed_table,
                                                  input_ids, hidden_dim);

    for (uint32_t l = 0; l < n_layers; l++) {
        hidden = transformer_layer_forward(ctx.device, ctx.queue, pipelines,
                                           pipeline_cache, layers[l], layer_cfg, hidden,
                                           k_caches[l], v_caches[l], 0, prefill_len);
    }
    uint32_t seq_pos = prefill_len;

    // Extract last token's hidden state for lm_head
    wgpu::Buffer last_hidden;
    if (prefill_len > 1) {
        last_hidden = create_storage_buffer(ctx.device, hidden_dim * sizeof(float), nullptr);
        wgpu::CommandEncoder enc = ctx.device.CreateCommandEncoder();
        uint64_t src_offset = static_cast<uint64_t>(prefill_len - 1) * hidden_dim * sizeof(float);
        enc.CopyBufferToBuffer(hidden, src_offset, last_hidden, 0,
                               hidden_dim * sizeof(float));
        wgpu::CommandBuffer cmd = enc.Finish();
        ctx.queue.Submit(1, &cmd);
    } else {
        last_hidden = hidden;
    }

    auto logits_buf = lm_head_forward(ctx.device, ctx.queue, pipeline_cache, last_hidden,
                                      output_norm, output_weight,
                                      1, hidden_dim, config.vocab_size,
                                      params.norm_epsilon);
    uint32_t first_token;
    if (params.temperature <= 0.0f) {
        first_token = gpu_argmax(ctx.device, ctx.queue, pipeline_cache,
                                 logits_buf, config.vocab_size);
    } else {
        auto logits = read_buffer_as<float>(ctx.device, ctx.queue, logits_buf,
                                            config.vocab_size);
        first_token = sample_topk(logits.data(), config.vocab_size,
                                  params.top_k, params.temperature, rng);
    }
    result.tokens.push_back(first_token);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_secs = std::chrono::duration<double>(prefill_end - prefill_start).count();
    result.prefill_tok_per_sec = static_cast<double>(input_ids.size()) / prefill_secs;

    // === Decode phase: generate subsequent tokens ===
    auto decode_start = std::chrono::high_resolution_clock::now();
    uint32_t decode_count = 0;

    for (uint32_t step = 1; step < params.max_tokens; step++) {
        uint32_t cur_token = result.tokens.back();

        hidden = embedding_lookup(ctx.device, ctx.queue, embed_table,
                                 cur_token, hidden_dim);

        for (uint32_t l = 0; l < n_layers; l++) {
            hidden = transformer_layer_forward(ctx.device, ctx.queue, pipelines,
                                               pipeline_cache, layers[l], layer_cfg, hidden,
                                               k_caches[l], v_caches[l], seq_pos);
        }
        seq_pos++;

        auto logits_buf = lm_head_forward(ctx.device, ctx.queue, pipeline_cache, hidden,
                                          output_norm, output_weight,
                                          1, hidden_dim, config.vocab_size,
                                          params.norm_epsilon);
        uint32_t next_token;
        if (params.temperature <= 0.0f) {
            next_token = gpu_argmax(ctx.device, ctx.queue, pipeline_cache,
                                    logits_buf, config.vocab_size);
        } else {
            auto logits = read_buffer_as<float>(ctx.device, ctx.queue, logits_buf,
                                                config.vocab_size);
            next_token = sample_topk(logits.data(), config.vocab_size,
                                     params.top_k, params.temperature, rng);
        }

        result.tokens.push_back(next_token);
        decode_count++;

        // EOS check (token id 0 or common EOS ids)
        if (next_token == 0) break;
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    double decode_secs = std::chrono::duration<double>(decode_end - decode_start).count();
    result.decode_tok_per_sec = decode_count > 0
        ? static_cast<double>(decode_count) / decode_secs
        : 0.0;

    return result;
}

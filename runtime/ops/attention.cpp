/**
 * ops/attention.cpp — Attention ops using embedded WGSL kernels.
 * GroupQueryAttention, MultiHeadAttention, RotaryEmbedding.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include "../wgsl_template.h"
#include <cstdio>
#include <cstring>
#include <cmath>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

static TensorDtype computeOutDtype(TensorDtype dtype) {
    if (dtype == TensorDtype::Float16) return TensorDtype::Float16;
    if (dtype == TensorDtype::Float32) return TensorDtype::Float32;
    return dtype;
}

static float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = (sign << 31) | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

static void ensureFloat32(OpContext& ex, GpuTensor& t) {
    if (t.dtype != TensorDtype::Float16) { ex.EnsureGpu(t); return; }
    int64_t count = tensorNel(&t);
    if (count <= 0) return;
    size_t fp16Bytes = (size_t)count * 2;
    const uint8_t* src = nullptr;
    std::vector<uint8_t> gpuReadback;
    if (!t.cpuData.empty() && t.cpuData.size() >= fp16Bytes) {
        src = t.cpuData.data();
    } else if (t.buffer.handle) {
        ex.EnsureGpu(t);
        ex.FlushPendingWork();
        gpuReadback = ex.getGpu()->readBuffer(t.buffer, fp16Bytes);
        if (gpuReadback.size() >= fp16Bytes) src = gpuReadback.data();
    }
    if (!src) return;
    std::vector<float> f32((size_t)count);
    auto* fp16 = reinterpret_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < count; i++) f32[(size_t)i] = fp16ToFloat(fp16[i]);
    t.shape = t.shape; // preserve shape
    t.dtype = TensorDtype::Float32;
    t.cpuData.clear();
    t.buffer = ex.getGpu()->createBuffer("attn_f32", f32.size() * 4);
    ex.getGpu()->writeBuffer(t.buffer, f32.data(), f32.size() * 4);
    t.isCpuOnly = false;
}

static bool readTensorInt64Values(OpContext& ex, GpuTensor* t,
                                  int64_t maxElements,
                                  std::vector<int64_t>& out) {
    out.clear();
    if (!t || t->dtype != TensorDtype::Int64) return false;
    int64_t nel = tensorNel(t);
    if (nel < 0 || nel > maxElements) return false;

    if (t->cpuData.size() >= (size_t)nel * 8) {
        out.resize((size_t)nel);
        memcpy(out.data(), t->cpuData.data(), (size_t)nel * 8);
        return true;
    }

    if (!t->buffer.handle || t->buffer.size < (size_t)nel * 8) return false;
    ex.FlushPendingWork();
    auto raw = ex.getGpu()->readBuffer(t->buffer, (size_t)nel * 8);
    if (raw.size() < (size_t)nel * 8) return false;
    t->cpuData.resize((size_t)nel * 8);
    memcpy(t->cpuData.data(), raw.data(), raw.size());
    out.resize((size_t)nel);
    memcpy(out.data(), raw.data(), raw.size());
    return true;
}

// ─── GroupQueryAttention (ONNX-native with KV cache + RoPE) ──────────────────
// Handles the full ONNX GQA op:
//   in[0]=Q, in[1]=K, in[2]=V, in[3]=past_key, in[4]=past_value,
//   in[5]=seqlen_k (past_length per batch), in[6]=total_seqlen,
//   in[7]=cos_cache, in[8]=sin_cache, in[9]=(empty), in[10]=attention_bias
//
//   out[0]=output, out[1]=present_key, out[2]=present_value

static void opGQA(OpContext& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Q = in[0];
    auto* K = in.size() > 1 ? in[1] : nullptr;
    auto* V = in.size() > 2 ? in[2] : nullptr;
    if (!Q || !K || !V || !Q->IsValid() || !K->IsValid() || !V->IsValid()) return;

    auto* pastKey = (in.size() > 3 && in[3] && in[3]->IsValid()) ? in[3] : nullptr;
    auto* pastVal = (in.size() > 4 && in[4] && in[4]->IsValid()) ? in[4] : nullptr;
    auto* seqLenK = (in.size() > 5 && in[5] && in[5]->IsValid()) ? in[5] : nullptr;
    auto* totalSeqLen = (in.size() > 6 && in[6] && in[6]->IsValid()) ? in[6] : nullptr;
    auto* cosCache = (in.size() > 7 && in[7] && in[7]->IsValid()) ? in[7] : nullptr;
    auto* sinCache = (in.size() > 8 && in[8] && in[8]->IsValid()) ? in[8] : nullptr;

    // Check if this is a KV-cache-style GQA (has past_key input)
    bool hasKVCache = (pastKey != nullptr || pastVal != nullptr);

    // If no KV cache inputs, fall back to simple bidirectional attention
    if (!hasKVCache && !cosCache) {
        ensureFloat32(ex, *Q);
        ensureFloat32(ex, *K);
        ensureFloat32(ex, *V);
        ex.EnsureGpu(*Q);
        ex.EnsureGpu(*K);
        ex.EnsureGpu(*V);

        int64_t num_heads = n.GetInt("num_heads", 32);
        int64_t kv_heads = n.GetInt("kv_num_heads", num_heads);
        float scale = n.GetFloat("scale", 1.0f / sqrtf(128.0f));

        int64_t T = 1, head_dim = 1;
        if (Q->shape.size() >= 3) {
            num_heads = Q->shape[Q->shape.size() - 2];
            head_dim = Q->shape.back();
            for (size_t i = 0; i + 2 < Q->shape.size(); i++) T *= Q->shape[i];
        } else {
            T = (Q->shape.size() >= 2) ? Q->shape[Q->shape.size() - 2] : 1;
            head_dim = Q->shape.back() / num_heads;
        }
        if (K->shape.size() >= 3) kv_heads = K->shape[K->shape.size() - 2];

        *out[0] = ex.AllocTensor(Q->shape, computeOutDtype(Q->dtype));
        for (size_t i = 1; i < out.size(); i++)
            if (out[i]) *out[i] = ex.AllocTensor({1}, computeOutDtype(Q->dtype));

        uint32_t scale_u32; memcpy(&scale_u32, &scale, 4);
        uint32_t params[8] = {(uint32_t)T, (uint32_t)num_heads, (uint32_t)head_dim,
                               (uint32_t)T, scale_u32, (uint32_t)kv_heads, 0, 0};
        auto paramBuf = ex.getParamBuffer(32);
        ex.getGpu()->writeBuffer(paramBuf, params, 32);

        auto& pl = ex.GetPipelineT("bidirectional_attn", 5, []() { return instantiateTemplate(WGSL_BIDIRECTIONAL_ATTN_T, TensorDtype::Float32); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)((head_dim + 127) / 128), (uint32_t)num_heads, (uint32_t)T, "gqa");
        return;
    }

    // ─── Full GQA with KV cache and RoPE ─────────────────────────────────

    // Record original dtype for output casting (attention computes in f32 internally)
    TensorDtype origDtype = Q->dtype;

    int64_t num_heads = n.GetInt("num_heads", 32);
    int64_t kv_heads = n.GetInt("kv_num_heads", num_heads);
    float scale = n.GetFloat("scale", 0.0f);
    int64_t doRotary = n.GetInt("do_rotary", 0);

    // Q: [batch, seq_q, num_heads * head_dim]
    int64_t batch = Q->shape[0];
    int64_t seqQ = (Q->shape.size() >= 2) ? Q->shape[1] : 1;
    int64_t qDim = Q->shape.back();
    int64_t head_dim = qDim / num_heads;

    if (scale <= 0.0f) scale = 1.0f / sqrtf((float)head_dim);

    // past KV: [batch, kv_heads, past_seq, head_dim]
    int64_t pastSeq = 0;
    if (pastKey && pastKey->shape.size() >= 4) pastSeq = pastKey->shape[2];
    int64_t totalSeq = pastSeq + seqQ;

    // Position offset from past_key shape — no GPU readback needed.
    // ORT also derives position from the KV cache length.
    int64_t posOffset = pastSeq;

    // GPU fp16→f32 cast helper: dispatches a cast kernel without CPU sync.
    auto gpuCastF16ToF32 = [&](GpuTensor& t) {
        if (t.dtype != TensorDtype::Float16) return;
        ex.EnsureGpu(t);
        int64_t nel = tensorNel(&t);
        if (nel <= 0) return;
        GpuTensor f32t = ex.AllocTensor(t.shape, TensorDtype::Float32);
        uint32_t p[4] = {(uint32_t)nel, 0, 0, 0};
        auto pb = ex.getParamBuffer(16);
        ex.getGpu()->writeBuffer(pb, p, 16);
        auto& pl = ex.GetPipelineT("cast_f16_to_f32", 3, []() { return std::string(WGSL_CAST_F16_TO_F32); });
        auto bg = ex.MakeBindGroup(pl, {{0, t.buffer}, {1, f32t.buffer}, {2, pb}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)((nel + 255) / 256), 1, 1, "attn_cast");
        t = f32t;
    };

    // Convert Q, K, V to f32 on GPU — no CPU readback
    gpuCastF16ToF32(*Q);
    gpuCastF16ToF32(*K);
    gpuCastF16ToF32(*V);
    ex.EnsureGpu(*Q);
    ex.EnsureGpu(*K);
    ex.EnsureGpu(*V);

    // Apply RoPE on GPU if needed
    int64_t rotaryDim = head_dim;
    if (doRotary && cosCache && sinCache) {
        gpuCastF16ToF32(*cosCache);
        gpuCastF16ToF32(*sinCache);
        ex.EnsureGpu(*cosCache);
        ex.EnsureGpu(*sinCache);

        if (cosCache->shape.size() >= 2) rotaryDim = cosCache->shape.back() * 2;

        auto& roPl = ex.GetPipelineT("rope_inplace", 4, []() { return instantiateTemplate(WGSL_ROPE_INPLACE_T, TensorDtype::Float32); });

        if (seqQ > 1) {
            // Batched RoPE: all T tokens at once
            auto& roBatchPl = ex.GetPipelineT("rope_inplace_batched", 4, []() {
                return std::string(WGSL_ROPE_INPLACE_BATCHED);
            });

            // RoPE on Q for all tokens
            uint32_t ropeParams[4] = {(uint32_t)num_heads, (uint32_t)head_dim,
                                       (uint32_t)rotaryDim, (uint32_t)posOffset};
            auto rpBuf = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(rpBuf, ropeParams, 16);
            auto roBg = ex.MakeBindGroup(roBatchPl, {
                {0, Q->buffer}, {1, cosCache->buffer}, {2, sinCache->buffer}, {3, rpBuf}});
            ex.QueueDispatch(roBatchPl.pipeline, roBg,
                (uint32_t)((num_heads + 63) / 64), (uint32_t)seqQ, 1, "rope_q_b");

            // RoPE on K for all tokens
            ropeParams[0] = (uint32_t)kv_heads;
            auto rpBufK = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(rpBufK, ropeParams, 16);
            auto roBgK = ex.MakeBindGroup(roBatchPl, {
                {0, K->buffer}, {1, cosCache->buffer}, {2, sinCache->buffer}, {3, rpBufK}});
            ex.QueueDispatch(roBatchPl.pipeline, roBgK,
                (uint32_t)((kv_heads + 63) / 64), (uint32_t)seqQ, 1, "rope_k_b");
        } else {
            // Single-token RoPE (decode path)
            uint32_t ropeParams[4] = {(uint32_t)num_heads, (uint32_t)head_dim,
                                       (uint32_t)rotaryDim, (uint32_t)posOffset};
            auto rpBuf = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(rpBuf, ropeParams, 16);
            ex.RegisterReplayParam(rpBuf, 12, ExecutionContext::ReplayParamUpdate::PosOffset);
            auto roBg = ex.MakeBindGroup(roPl, {
                {0, Q->buffer}, {1, cosCache->buffer}, {2, sinCache->buffer}, {3, rpBuf}});
            ex.QueueDispatch(roPl.pipeline, roBg,
                (uint32_t)((num_heads + 63) / 64), 1, 1, "rope_q");

            ropeParams[0] = (uint32_t)kv_heads;
            auto rpBufK = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(rpBufK, ropeParams, 16);
            ex.RegisterReplayParam(rpBufK, 12, ExecutionContext::ReplayParamUpdate::PosOffset);
            auto roBgK = ex.MakeBindGroup(roPl, {
                {0, K->buffer}, {1, cosCache->buffer}, {2, sinCache->buffer}, {3, rpBufK}});
            ex.QueueDispatch(roPl.pipeline, roBgK,
                (uint32_t)((kv_heads + 63) / 64), 1, 1, "rope_k");
        }
    }

    // Build present KV on GPU: append new K/V to past cache
    // Static KV cache mode: if the past_key buffer is already f32 and large enough
    // to hold totalSeq tokens, write the new token in-place (no full copy).
    // The caller pre-allocates a max-sized f32 buffer and reuses it across steps.
    size_t kvSlotBytes = (size_t)(batch * kv_heads * totalSeq * head_dim * 4);
    bool staticKV = (pastKey && pastKey->dtype == TensorDtype::Float32 &&
                     pastKey->buffer.handle && pastKey->buffer.size >= kvSlotBytes &&
                     pastVal && pastVal->dtype == TensorDtype::Float32 &&
                     pastVal->buffer.handle && pastVal->buffer.size >= kvSlotBytes);

    // Determine max_seq for static layout (from buffer size)
    int64_t maxSeq = totalSeq;
    if (staticKV) {
        maxSeq = (int64_t)(pastKey->buffer.size / (batch * kv_heads * head_dim * 4));
    }

    GpuTensor presentKey, presentVal;
    if (staticKV) {
        // Static mode: reuse the same buffer, just write the new token at pastSeq
        ex.EnsureGpu(*pastKey);
        ex.EnsureGpu(*pastVal);
        presentKey.shape = {batch, kv_heads, totalSeq, head_dim};
        presentKey.dtype = TensorDtype::Float32;
        presentKey.buffer = pastKey->buffer;
        presentVal.shape = {batch, kv_heads, totalSeq, head_dim};
        presentVal.dtype = TensorDtype::Float32;
        presentVal.buffer = pastVal->buffer;

        // Write new K/V at positions [pastSeq, pastSeq+seqQ)
        auto& kvPl = ex.GetPipelineT("kv_cache_write", 3, []() { return instantiateTemplate(WGSL_KV_CACHE_WRITE_T, TensorDtype::Float32); });

        if (seqQ > 1) {
            // Batched KV write: all T tokens at once
            auto& kvBatchPl = ex.GetPipelineT("kv_cache_write_batched", 3, []() {
                return std::string(WGSL_KV_CACHE_WRITE_BATCHED);
            });

            uint32_t kvParams[4] = {(uint32_t)kv_heads, (uint32_t)head_dim,
                                     (uint32_t)pastSeq, (uint32_t)maxSeq};
            auto kvpBuf = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(kvpBuf, kvParams, 16);

            auto kvBgK = ex.MakeBindGroup(kvBatchPl, {
                {0, K->buffer}, {1, presentKey.buffer}, {2, kvpBuf}});
            ex.QueueDispatch(kvBatchPl.pipeline, kvBgK,
                (uint32_t)((kv_heads * head_dim + 255) / 256), (uint32_t)seqQ, 1, "kv_write_k_b");

            auto kvBgV = ex.MakeBindGroup(kvBatchPl, {
                {0, V->buffer}, {1, presentVal.buffer}, {2, kvpBuf}});
            ex.QueueDispatch(kvBatchPl.pipeline, kvBgV,
                (uint32_t)((kv_heads * head_dim + 255) / 256), (uint32_t)seqQ, 1, "kv_write_v_b");
        } else {
            // Single-token KV write (decode path)
            uint32_t kvParams[4] = {(uint32_t)kv_heads, (uint32_t)head_dim,
                                     (uint32_t)pastSeq, (uint32_t)maxSeq};
            auto kvpBuf = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(kvpBuf, kvParams, 16);
            ex.RegisterReplayParam(kvpBuf, 8, ExecutionContext::ReplayParamUpdate::PastSeq);

            auto kvBgK = ex.MakeBindGroup(kvPl, {
                {0, K->buffer}, {1, presentKey.buffer}, {2, kvpBuf}});
            ex.QueueDispatch(kvPl.pipeline, kvBgK,
                (uint32_t)((kv_heads * head_dim + 255) / 256), 1, 1, "kv_write_k");

            auto kvBgV = ex.MakeBindGroup(kvPl, {
                {0, V->buffer}, {1, presentVal.buffer}, {2, kvpBuf}});
            ex.QueueDispatch(kvPl.pipeline, kvBgV,
                (uint32_t)((kv_heads * head_dim + 255) / 256), 1, 1, "kv_write_v");
        }
    } else {
        // Dynamic mode: allocate new buffer and copy past + append new
        presentKey = ex.AllocTensor({batch, kv_heads, totalSeq, head_dim}, TensorDtype::Float32);
        presentVal = ex.AllocTensor({batch, kv_heads, totalSeq, head_dim}, TensorDtype::Float32);

        GPUBuffer pastKeyBuf, pastValBuf;
        if (pastKey && pastSeq > 0) {
            gpuCastF16ToF32(*pastKey);
            ex.EnsureGpu(*pastKey);
            pastKeyBuf = pastKey->buffer;
        } else {
            pastKeyBuf = ex.getParamBuffer(4);
        }
        if (pastVal && pastSeq > 0) {
            gpuCastF16ToF32(*pastVal);
            ex.EnsureGpu(*pastVal);
            pastValBuf = pastVal->buffer;
        } else {
            pastValBuf = ex.getParamBuffer(4);
        }

        uint32_t kvParams[4] = {(uint32_t)kv_heads, (uint32_t)head_dim,
                                 (uint32_t)pastSeq, (uint32_t)totalSeq};
        auto kvpBuf = ex.getParamBuffer(16);
        ex.getGpu()->writeBuffer(kvpBuf, kvParams, 16);

        auto& kvPl = ex.GetPipelineT("kv_cache_append", 4, []() { return instantiateTemplate(WGSL_KV_CACHE_APPEND_T, TensorDtype::Float32); });
        auto kvBgK = ex.MakeBindGroup(kvPl, {
            {0, K->buffer}, {1, pastKeyBuf}, {2, presentKey.buffer}, {3, kvpBuf}});
        ex.QueueDispatch(kvPl.pipeline, kvBgK,
            (uint32_t)((kv_heads * head_dim + 255) / 256), 1, 1, "kv_append_k");

        auto kvBgV = ex.MakeBindGroup(kvPl, {
            {0, V->buffer}, {1, pastValBuf}, {2, presentVal.buffer}, {3, kvpBuf}});
        ex.QueueDispatch(kvPl.pipeline, kvBgV,
            (uint32_t)((kv_heads * head_dim + 255) / 256), 1, 1, "kv_append_v");
    }

    // Compute attention on GPU
    *out[0] = ex.AllocTensor({batch, seqQ, num_heads * head_dim}, TensorDtype::Float32);

    {
        uint32_t scale_u32; memcpy(&scale_u32, &scale, 4);

        if (seqQ > 1) {
            // Batched prefill: single dispatch for all T query tokens
            auto& prefillPl = ex.GetPipelineT("gqa_prefill", 5, []() {
                return std::string(WGSL_GQA_PREFILL);
            });

            uint32_t prefillParams[8] = {(uint32_t)num_heads, (uint32_t)head_dim,
                                          (uint32_t)pastSeq, (uint32_t)kv_heads,
                                          scale_u32, (uint32_t)maxSeq, (uint32_t)seqQ, 0};
            auto ppBuf = ex.getParamBuffer(32);
            ex.getGpu()->writeBuffer(ppBuf, prefillParams, 32);

            auto prefillBg = ex.MakeBindGroup(prefillPl, {
                {0, Q->buffer}, {1, presentKey.buffer}, {2, presentVal.buffer},
                {3, out[0]->buffer}, {4, ppBuf}});
            ex.QueueDispatch(prefillPl.pipeline, prefillBg,
                (uint32_t)num_heads, (uint32_t)((seqQ + 3) / 4), 1, "gqa_prefill");
        } else {
            // Single-token decode path
            auto& attnPl = ex.GetPipelineT("gqa_decode", 5, []() { return instantiateTemplate(WGSL_GQA_DECODE_T, TensorDtype::Float32); });

            uint32_t attnParams[8] = {(uint32_t)num_heads, (uint32_t)head_dim,
                                       (uint32_t)totalSeq, (uint32_t)kv_heads,
                                       scale_u32, (uint32_t)maxSeq, 0, 0};
            auto apBuf = ex.getParamBuffer(32);
            ex.getGpu()->writeBuffer(apBuf, attnParams, 32);
            ex.RegisterReplayParam(apBuf, 8, ExecutionContext::ReplayParamUpdate::TotalSeq);

            auto attnBg = ex.MakeBindGroup(attnPl, {
                {0, Q->buffer}, {1, presentKey.buffer}, {2, presentVal.buffer},
                {3, out[0]->buffer}, {4, apBuf}});
            ex.QueueDispatch(attnPl.pipeline, attnBg,
                1, (uint32_t)num_heads, 1, "gqa_decode");
        }
    }

    // Cast f32 attention output back to f16 if input was f16
    if (origDtype == TensorDtype::Float16 && ex.getGpu()->supportsShaderF16) {
        int64_t nel = out[0]->ElementCount();
        GpuTensor f16t = ex.AllocTensor(out[0]->shape, TensorDtype::Float16);
        uint32_t cp[4] = {(uint32_t)nel, 0, 0, 0};
        auto cpBuf = ex.getParamBuffer(16);
        ex.getGpu()->writeBuffer(cpBuf, cp, 16);
        auto& cPl = ex.GetPipelineT("cast_f32_to_f16", 3, []() { return std::string(WGSL_CAST_F32_TO_F16); });
        auto cBg = ex.MakeBindGroup(cPl, {{0, out[0]->buffer}, {1, f16t.buffer}, {2, cpBuf}});
        ex.QueueDispatch(cPl.pipeline, cBg,
            (uint32_t)((nel + 255) / 256), 1, 1, "attn_cast_f16");
        *out[0] = f16t;
    }

    // Output present KV — in static mode, skip f16 cast (stays f32, same buffer)
    if (!staticKV && origDtype == TensorDtype::Float16 && ex.getGpu()->supportsShaderF16) {
        auto castToF16 = [&](GpuTensor& t, const char* label) {
            int64_t nel = t.ElementCount();
            GpuTensor f16t = ex.AllocTensor(t.shape, TensorDtype::Float16);
            uint32_t cp[4] = {(uint32_t)nel, 0, 0, 0};
            auto cpBuf = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(cpBuf, cp, 16);
            auto& cPl = ex.GetPipelineT("cast_f32_to_f16", 3, []() { return std::string(WGSL_CAST_F32_TO_F16); });
            auto cBg = ex.MakeBindGroup(cPl, {{0, t.buffer}, {1, f16t.buffer}, {2, cpBuf}});
            ex.QueueDispatch(cPl.pipeline, cBg,
                (uint32_t)((nel + 255) / 256), 1, 1, label);
            t = f16t;
        };
        castToF16(presentKey, "kv_cast_k_f16");
        castToF16(presentVal, "kv_cast_v_f16");
    }

    // Output present KV
    if (out.size() > 1 && out[1]) *out[1] = presentKey;
    if (out.size() > 2 && out[2]) *out[2] = presentVal;
}

static void opMHA(OpContext& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Q = in[0]; auto* K = in[1]; auto* V = in[2];
    if (!Q || !K || !V || !Q->IsValid() || !K->IsValid() || !V->IsValid()) return;

    // The attention kernel uses array<f32>; convert fp16 inputs to f32.
    ensureFloat32(ex, *Q);
    ensureFloat32(ex, *K);
    ensureFloat32(ex, *V);
    ex.EnsureGpu(*Q);
    ex.EnsureGpu(*K);
    ex.EnsureGpu(*V);

    int64_t num_heads = n.GetInt("num_heads", 30);
    float scale = n.GetFloat("scale", 1.0f / sqrtf(128.0f));

    // MHA input layout: [batch, seq_len, num_heads * head_dim]
    // Derive T_q, T_kv and head_dim from shapes and num_heads attribute.
    int64_t T_q = 1, T_kv = 1, head_dim = 128;
    int64_t q_last = Q->shape.back();
    head_dim = q_last / num_heads;

    if (Q->shape.size() == 3) {
        T_q = Q->shape[1];   // [batch, seq, features]
    } else if (Q->shape.size() == 2) {
        T_q = Q->shape[0];   // [seq, features]
    }

    if (K->shape.size() == 3) {
        T_kv = K->shape[1];
    } else if (K->shape.size() == 2) {
        T_kv = K->shape[0];
    }

    *out[0] = ex.AllocTensor(Q->shape, computeOutDtype(Q->dtype));
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({1}, computeOutDtype(Q->dtype));

    uint32_t scale_u32; memcpy(&scale_u32, &scale, 4);
    uint32_t params[8] = {(uint32_t)T_q, (uint32_t)num_heads, (uint32_t)head_dim,
                           (uint32_t)T_kv, scale_u32, 0, 0, 0};

    auto paramBuf = ex.getParamBuffer(32);
    ex.getGpu()->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipelineT("bidirectional_attn", 5, []() { return instantiateTemplate(WGSL_BIDIRECTIONAL_ATTN_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((head_dim + 127) / 128), (uint32_t)num_heads, (uint32_t)T_q, "mha");
}

// ─── RotaryEmbedding ─────────────────────────────────────────────────────────

static void opRotaryEmbedding(OpContext& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* PosIds = in.size() > 1 ? in[1] : nullptr;
    auto* CosCache = in.size() > 2 ? in[2] : nullptr;
    auto* SinCache = in.size() > 3 ? in[3] : nullptr;
    if (!X || !X->IsValid()) return;

    // RoPE kernel uses array<f32>; convert fp16 input to f32.
    ensureFloat32(ex, *X);

    int64_t interleaved = n.GetInt("interleaved", 0);
    int64_t total = tensorNel(X);
    int64_t head_dim = X->shape.back();

    // Compute seq_len from X shape for position auto-increment
    // X shape: [batch, heads, seq_len, head_dim] for 4D
    //       or [batch, seq_len, features] for 3D
    int64_t seqLen = 1;
    if (X->shape.size() >= 4) seqLen = X->shape[X->shape.size() - 2];
    else if (X->shape.size() == 3) seqLen = X->shape[1];
    else if (X->shape.size() == 2) seqLen = X->shape[0];

    *out[0] = ex.AllocTensor(X->shape, computeOutDtype(X->dtype));

    if (!CosCache || !SinCache || !CosCache->IsValid() || !SinCache->IsValid()) {
        *out[0] = *X;
        return;
    }

    int64_t nPosIds = PosIds ? tensorNel(PosIds) : 0;

    // Convert int64 PosIds to int32 if needed
    GPUBuffer posIdsBuf;
    if (PosIds && PosIds->IsValid()) {
        ex.EnsureGpu(*PosIds);
        if (PosIds->dtype == TensorDtype::Int64 && nPosIds <= 65536) {
            std::vector<int64_t> posIds;
            if (readTensorInt64Values(ex, PosIds, 65536, posIds)) {
                std::vector<int32_t> i32(nPosIds);
                for (int64_t i = 0; i < nPosIds; i++) {
                    i32[i] = (int32_t)posIds[(size_t)i];
                }
                posIdsBuf = ex.getGpu()->createBuffer("rope_posids32", nPosIds * 4);
                ex.getGpu()->writeBuffer(posIdsBuf, i32.data(), nPosIds * 4);
            } else {
                posIdsBuf = PosIds->buffer;
            }
        } else {
            posIdsBuf = PosIds->buffer;
        }
    } else {
        int32_t zero = 0;
        posIdsBuf = ex.getParamBuffer(4);
        ex.getGpu()->writeBuffer(posIdsBuf, &zero, 4);
    }

    if (CosCache && CosCache->IsValid()) ensureFloat32(ex, *CosCache);
    if (SinCache && SinCache->IsValid()) ensureFloat32(ex, *SinCache);
    ex.EnsureGpu(*CosCache);
    ex.EnsureGpu(*SinCache);

    uint32_t params[8] = {(uint32_t)total, (uint32_t)head_dim,
                           (uint32_t)interleaved, (uint32_t)nPosIds,
                           (uint32_t)seqLen, 0, 0, 0};
    auto paramBuf = ex.getParamBuffer(32);
    ex.getGpu()->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipelineT("rotary_embedding", 6, []() { return instantiateTemplate(WGSL_ROTARY_EMBEDDING_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, posIdsBuf}, {2, CosCache->buffer},
        {3, SinCache->buffer}, {4, out[0]->buffer}, {5, paramBuf}});

    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "rope");
}

REGISTER_OP(GroupQueryAttention, opGQA)
REGISTER_OP(MultiHeadAttention, opMHA)
REGISTER_OP(RotaryEmbedding, opRotaryEmbedding)

// ─── GQA with Attention Sink + Sliding Window ──────────────────────────────
// Custom op for models with attention sinks (learnable logits that absorb
// probability mass) and sliding window attention.
//
// Inputs: Q (f32), K_cache (f32), V_cache (f32), Sinks (f32, per-head logits)
// Attrs: num_heads, kv_heads, head_dim, kv_stride, kv_start, T_win
//
// KV cache management (circular buffer, eviction) must be handled by the
// caller before dispatching this op.

static void opGQAAttnSink(OpContext& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Q_in = in[0];
    auto* K_cache = in.size() > 1 ? in[1] : nullptr;
    auto* V_cache = in.size() > 2 ? in[2] : nullptr;
    auto* Sinks = in.size() > 3 ? in[3] : nullptr;

    if (!Q_in || !K_cache || !V_cache || !Sinks ||
        !Q_in->IsValid() || !K_cache->IsValid() || !V_cache->IsValid() || !Sinks->IsValid()) {
        fprintf(stderr, "GQAAttnSink: missing inputs\n");
        return;
    }

    ensureFloat32(ex, *Q_in);
    ensureFloat32(ex, *K_cache);
    ensureFloat32(ex, *V_cache);
    ensureFloat32(ex, *Sinks);

    uint32_t num_heads = (uint32_t)n.GetInt("num_heads", 32);
    uint32_t kv_heads = (uint32_t)n.GetInt("kv_heads", num_heads);
    uint32_t head_dim = (uint32_t)n.GetInt("head_dim", 64);
    uint32_t kv_stride = (uint32_t)n.GetInt("kv_stride", 0);
    uint32_t kv_start = (uint32_t)n.GetInt("kv_start", 0);
    uint32_t T_win = (uint32_t)n.GetInt("window_size", 0);

    float scale = 1.0f / sqrtf((float)head_dim);

    auto outShape = Q_in->shape;
    *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);

    uint32_t params[8] = {num_heads, head_dim, T_win, kv_heads,
                           0, kv_stride, kv_start, 0};
    memcpy(&params[4], &scale, sizeof(float));  // bitcast f32 → u32

    auto paramBuf = ex.getParamBuffer(32);
    ex.getGpu()->writeBuffer(paramBuf, params, 28);

    auto& pl = ex.GetPipelineT("gqa_decode_attn_sink", 6, []() {
        return std::string(WGSL_GQA_DECODE_ATTN_SINK);
    });
    auto bg = ex.MakeBindGroup(pl, {
        {0, Q_in->buffer}, {1, K_cache->buffer}, {2, V_cache->buffer},
        {3, Sinks->buffer}, {4, out[0]->buffer}, {5, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg, 1, num_heads, 1, "gqa_attn_sink");
}

REGISTER_OP(GQAAttnSink, opGQAAttnSink)

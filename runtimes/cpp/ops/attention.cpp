/**
 * ops/attention.cpp — Attention ops using embedded WGSL kernels.
 * GroupQueryAttention, MultiHeadAttention, RotaryEmbedding.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <cstdio>
#include <cstring>
#include <cmath>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

static TensorDtype computeOutDtype(TensorDtype dtype) {
    return (dtype == TensorDtype::Float16 || dtype == TensorDtype::Float32)
             ? TensorDtype::Float32
             : dtype;
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

static void ensureFloat32(GraphExecutor& ex, GpuTensor& t) {
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
        gpuReadback = ex.gpu->readBuffer(t.buffer, fp16Bytes);
        if (gpuReadback.size() >= fp16Bytes) src = gpuReadback.data();
    }
    if (!src) return;
    std::vector<float> f32((size_t)count);
    auto* fp16 = reinterpret_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < count; i++) f32[(size_t)i] = fp16ToFloat(fp16[i]);
    t.shape = t.shape; // preserve shape
    t.dtype = TensorDtype::Float32;
    t.cpuData.clear();
    t.buffer = ex.gpu->createBuffer("attn_f32", f32.size() * 4);
    ex.gpu->writeBuffer(t.buffer, f32.data(), f32.size() * 4);
    t.isCpuOnly = false;
}

static bool readTensorInt64Values(GraphExecutor& ex, GpuTensor* t,
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
    auto raw = ex.gpu->readBuffer(t->buffer, (size_t)nel * 8);
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

static void opGQA(GraphExecutor& ex, const OnnxGraphNode& n,
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
        auto paramBuf = ex.gpu->createBuffer("gqa_p", 32);
        ex.gpu->writeBuffer(paramBuf, params, 32);

        auto& pl = ex.GetPipeline("bidirectional_attn", WGSL_BIDIRECTIONAL_ATTN, 5);
        auto bg = ex.MakeBindGroup(pl, {
            {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)((head_dim + 127) / 128), (uint32_t)num_heads, (uint32_t)T, "gqa"});
        return;
    }

    // ─── Full GQA with KV cache and RoPE (CPU path for correctness) ─────

    int64_t num_heads = n.GetInt("num_heads", 32);
    int64_t kv_heads = n.GetInt("kv_num_heads", num_heads);
    float scale = n.GetFloat("scale", 0.125f);
    int64_t doRotary = n.GetInt("do_rotary", 0);
    int64_t rotaryInterleaved = n.GetInt("rotary_interleaved", 0);

    // Q: [batch, seq_q, num_heads * head_dim]
    // K: [batch, seq_q, kv_heads * head_dim]
    // V: [batch, seq_q, kv_heads * head_dim]
    int64_t batch = Q->shape[0];
    int64_t seqQ = (Q->shape.size() >= 2) ? Q->shape[1] : 1;
    int64_t qDim = Q->shape.back();
    int64_t head_dim = qDim / num_heads;

    // past KV: [batch, kv_heads, past_seq, head_dim]
    int64_t pastSeq = 0;
    if (pastKey && pastKey->shape.size() >= 4) {
        pastSeq = pastKey->shape[2];
    }
    int64_t totalSeq = pastSeq + seqQ;

    // Read seqlen_k to get the position offset for RoPE
    int64_t posOffset = pastSeq;
    if (seqLenK) {
        std::vector<int64_t> slk;
        if (readTensorInt64Values(ex, seqLenK, 16, slk) && !slk.empty()) {
            posOffset = slk[0];
        }
    }

    // Read all inputs to CPU
    std::vector<float> qData, kData, vData;
    std::vector<float> pastKeyData, pastValData;
    std::vector<float> cosData, sinData;

    auto readToFloat = [&](GpuTensor* t, const std::string& name, std::vector<float>& dst) {
        if (!t) return;
        ensureFloat32(ex, *t);
        ex.EnsureGpu(*t);
        ex.FlushPendingWork();
        int64_t nel = tensorNel(t);
        if (nel <= 0) return;
        auto rb = ex.gpu->readBuffer(t->buffer, nel * 4);
        dst.resize(nel);
        memcpy(dst.data(), rb.data(), nel * 4);
    };

    readToFloat(Q, n.inputs[0], qData);
    readToFloat(K, n.inputs[1], kData);
    readToFloat(V, n.inputs[2], vData);
    if (pastKey && pastSeq > 0) readToFloat(pastKey, n.inputs[3], pastKeyData);
    if (pastVal && pastSeq > 0) readToFloat(pastVal, n.inputs[4], pastValData);
    if (cosCache && doRotary) readToFloat(cosCache, n.inputs[7], cosData);
    if (sinCache && doRotary) readToFloat(sinCache, n.inputs[8], sinData);

    int64_t headGroup = num_heads / kv_heads;
    int64_t rotaryDim = head_dim;
    if (!cosData.empty() && cosCache) {
        // cos_cache shape: [max_pos, head_dim/2]
        rotaryDim = cosCache->shape.back() * 2;
    }

    // Reshape Q: [batch, seqQ, num_heads * head_dim] → [batch, seqQ, num_heads, head_dim]
    // Apply RoPE to Q and K
    auto applyRoPE = [&](float* data, int64_t nHeads, int64_t pos) {
        if (cosData.empty() || sinData.empty()) return;
        int64_t cosStride = rotaryDim / 2;
        for (int64_t h = 0; h < nHeads; h++) {
            float* head = data + h * head_dim;
            for (int64_t d = 0; d < rotaryDim / 2; d++) {
                float cosVal = cosData[pos * cosStride + d];
                float sinVal = sinData[pos * cosStride + d];
                float x0 = head[d];
                float x1 = head[d + rotaryDim / 2];
                head[d] = x0 * cosVal - x1 * sinVal;
                head[d + rotaryDim / 2] = x1 * cosVal + x0 * sinVal;
            }
        }
    };

    // Apply RoPE to Q and K for each position in the sequence
    if (doRotary) {
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t s = 0; s < seqQ; s++) {
                int64_t pos = posOffset + s;
                float* qHead = qData.data() + (b * seqQ + s) * num_heads * head_dim;
                float* kHead = kData.data() + (b * seqQ + s) * kv_heads * head_dim;
                applyRoPE(qHead, num_heads, pos);
                applyRoPE(kHead, kv_heads, pos);
            }
        }
    }

    // Build full KV: concat past + current
    // present_key: [batch, kv_heads, totalSeq, head_dim]
    std::vector<float> fullKey(batch * kv_heads * totalSeq * head_dim, 0.0f);
    std::vector<float> fullVal(batch * kv_heads * totalSeq * head_dim, 0.0f);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < kv_heads; h++) {
            // Copy past KV
            for (int64_t s = 0; s < pastSeq; s++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t dstIdx = ((b * kv_heads + h) * totalSeq + s) * head_dim + d;
                    int64_t srcIdx = ((b * kv_heads + h) * pastSeq + s) * head_dim + d;
                    fullKey[dstIdx] = pastKeyData.empty() ? 0.0f : pastKeyData[srcIdx];
                    fullVal[dstIdx] = pastValData.empty() ? 0.0f : pastValData[srcIdx];
                }
            }
            // Copy current KV (K is [batch, seqQ, kv_heads * head_dim], need to reshape)
            for (int64_t s = 0; s < seqQ; s++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t dstIdx = ((b * kv_heads + h) * totalSeq + (pastSeq + s)) * head_dim + d;
                    int64_t srcIdx = (b * seqQ + s) * kv_heads * head_dim + h * head_dim + d;
                    fullKey[dstIdx] = kData[srcIdx];
                    fullVal[dstIdx] = vData[srcIdx];
                }
            }
        }
    }

    // Compute attention: output[batch, seqQ, num_heads, head_dim]
    std::vector<float> output(batch * seqQ * num_heads * head_dim, 0.0f);
    std::vector<float> attnScores(totalSeq);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < num_heads; h++) {
            int64_t kvH = h / headGroup;  // map Q head to KV head
            for (int64_t sq = 0; sq < seqQ; sq++) {
                // Q[b, sq, h, :] dot K[b, kvH, sk, :] for all sk
                float* qVec = qData.data() + (b * seqQ + sq) * num_heads * head_dim + h * head_dim;

                // Compute attention scores
                float maxScore = -1e30f;
                for (int64_t sk = 0; sk < totalSeq; sk++) {
                    float* kVec = fullKey.data() + ((b * kv_heads + kvH) * totalSeq + sk) * head_dim;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; d++) {
                        dot += qVec[d] * kVec[d];
                    }
                    dot *= scale;
                    attnScores[sk] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                // Softmax
                float sumExp = 0.0f;
                for (int64_t sk = 0; sk < totalSeq; sk++) {
                    attnScores[sk] = expf(attnScores[sk] - maxScore);
                    sumExp += attnScores[sk];
                }
                if (sumExp > 0.0f) {
                    for (int64_t sk = 0; sk < totalSeq; sk++) {
                        attnScores[sk] /= sumExp;
                    }
                }

                // Weighted sum of values
                float* outVec = output.data() + (b * seqQ + sq) * num_heads * head_dim + h * head_dim;
                for (int64_t sk = 0; sk < totalSeq; sk++) {
                    float w = attnScores[sk];
                    float* vVec = fullVal.data() + ((b * kv_heads + kvH) * totalSeq + sk) * head_dim;
                    for (int64_t d = 0; d < head_dim; d++) {
                        outVec[d] += w * vVec[d];
                    }
                }
            }
        }
    }

    // Write outputs
    *out[0] = ex.AllocTensor({batch, seqQ, num_heads * head_dim}, TensorDtype::Float32);
    ex.gpu->writeBuffer(out[0]->buffer, output.data(), output.size() * sizeof(float));

    // present_key/value: [batch, kv_heads, totalSeq, head_dim]
    if (out.size() > 1 && out[1]) {
        *out[1] = ex.AllocTensor({batch, kv_heads, totalSeq, head_dim}, TensorDtype::Float32);
        ex.gpu->writeBuffer(out[1]->buffer, fullKey.data(), fullKey.size() * sizeof(float));
    }
    if (out.size() > 2 && out[2]) {
        *out[2] = ex.AllocTensor({batch, kv_heads, totalSeq, head_dim}, TensorDtype::Float32);
        ex.gpu->writeBuffer(out[2]->buffer, fullVal.data(), fullVal.size() * sizeof(float));
    }
}

static void opMHA(GraphExecutor& ex, const OnnxGraphNode& n,
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

    auto paramBuf = ex.gpu->createBuffer("mha_p", 32);
    ex.gpu->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipeline("bidirectional_attn", WGSL_BIDIRECTIONAL_ATTN, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((head_dim + 127) / 128), (uint32_t)num_heads, (uint32_t)T_q, "mha"});
}

// ─── RotaryEmbedding ─────────────────────────────────────────────────────────

static void opRotaryEmbedding(GraphExecutor& ex, const OnnxGraphNode& n,
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
                posIdsBuf = ex.gpu->createBuffer("rope_posids32", nPosIds * 4);
                ex.gpu->writeBuffer(posIdsBuf, i32.data(), nPosIds * 4);
            } else {
                posIdsBuf = PosIds->buffer;
            }
        } else {
            posIdsBuf = PosIds->buffer;
        }
    } else {
        int32_t zero = 0;
        posIdsBuf = ex.gpu->createBuffer("rope_pos0", 4);
        ex.gpu->writeBuffer(posIdsBuf, &zero, 4);
    }

    if (CosCache && CosCache->IsValid()) ensureFloat32(ex, *CosCache);
    if (SinCache && SinCache->IsValid()) ensureFloat32(ex, *SinCache);
    ex.EnsureGpu(*CosCache);
    ex.EnsureGpu(*SinCache);

    uint32_t params[8] = {(uint32_t)total, (uint32_t)head_dim,
                           (uint32_t)interleaved, (uint32_t)nPosIds,
                           (uint32_t)seqLen, 0, 0, 0};
    auto paramBuf = ex.gpu->createBuffer("rope_p", 32);
    ex.gpu->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipeline("rotary_embedding", WGSL_ROTARY_EMBEDDING, 6);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, posIdsBuf}, {2, CosCache->buffer},
        {3, SinCache->buffer}, {4, out[0]->buffer}, {5, paramBuf}});

    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "rope"});
}

REGISTER_OP(GroupQueryAttention, opGQA)
REGISTER_OP(MultiHeadAttention, opMHA)
REGISTER_OP(RotaryEmbedding, opRotaryEmbedding)

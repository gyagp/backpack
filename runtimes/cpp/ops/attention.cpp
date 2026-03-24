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

// ─── GroupQueryAttention / MultiHeadAttention ────────────────────────────────
// Uses bidirectional_attn kernel with online softmax.

static void opGQA(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Q = in[0];
    auto* K = in.size() > 1 ? in[1] : nullptr;
    auto* V = in.size() > 2 ? in[2] : nullptr;
    if (!Q || !K || !V || !Q->IsValid() || !K->IsValid() || !V->IsValid()) return;

    int64_t num_heads = n.GetInt("num_heads", 32);
    int64_t kv_heads = n.GetInt("kv_num_heads", num_heads);
    float scale = n.GetFloat("scale", 1.0f / sqrtf(128.0f));

    int64_t T = (Q->shape.size() >= 2) ? Q->shape[Q->shape.size()-2] : 1;
    int64_t qDim = Q->shape.back();
    int64_t head_dim = qDim / num_heads;

    *out[0] = ex.AllocTensor(Q->shape, Q->dtype);
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({1}, Q->dtype);

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
}

static void opMHA(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Q = in[0]; auto* K = in[1]; auto* V = in[2];
    if (!Q || !K || !V || !Q->IsValid() || !K->IsValid() || !V->IsValid()) return;

    int64_t num_heads = n.GetInt("num_heads", 30);
    float scale = n.GetFloat("scale", 1.0f / sqrtf(128.0f));

    int64_t T_q = (Q->shape.size() >= 2) ? Q->shape[Q->shape.size()-2] : 1;
    int64_t T_kv = (K->shape.size() >= 2) ? K->shape[K->shape.size()-2] : T_q;
    int64_t qDim = Q->shape.back();
    int64_t head_dim = qDim / num_heads;

    *out[0] = ex.AllocTensor(Q->shape, Q->dtype);
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({1}, Q->dtype);

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

    int64_t interleaved = n.GetInt("interleaved", 0);
    int64_t total = tensorNel(X);
    int64_t head_dim = X->shape.back();

    *out[0] = ex.AllocTensor(X->shape, X->dtype);

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
            const uint8_t* p = nullptr;
            if (PosIds->isCpuOnly && !PosIds->cpuData.empty()) p = PosIds->cpuData.data();
            else if (!PosIds->cpuData.empty()) p = PosIds->cpuData.data();
            if (p) {
                std::vector<int32_t> i32(nPosIds);
                for (int64_t i = 0; i < nPosIds; i++) {
                    int64_t v; memcpy(&v, p + i*8, 8); i32[i] = (int32_t)v;
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

    ex.EnsureGpu(*CosCache);
    ex.EnsureGpu(*SinCache);

    uint32_t params[4] = {(uint32_t)total, (uint32_t)head_dim,
                           (uint32_t)interleaved, (uint32_t)nPosIds};
    auto paramBuf = ex.gpu->createBuffer("rope_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

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

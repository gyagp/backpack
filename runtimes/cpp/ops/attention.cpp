/**
 * ops/attention.cpp — Attention ops with GPU kernels.
 * GroupQueryAttention, MultiHeadAttention, RotaryEmbedding.
 */

#include "../graph_executor.h"
#include <cstdio>
#include <cstring>
#include <cmath>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

// ─── GroupQueryAttention ─────────────────────────────────────────────────────
// Fast GQA: each thread handles one output element (token, head, dim).
// Thread computes full score loop over KV tokens using register accumulation.
// No shared memory reduction needed — each thread independently computes
// the same attention weights and selects its own V dimension.

static const char* WGSL_GQA_FAST = R"WGSL(
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let T = _params_[0];
    let num_heads = _params_[1];
    let head_dim = _params_[2];
    let kv_heads = _params_[3];
    let scale = bitcast<f32>(_params_[4]);

    // Each thread handles one (token, head, dim) = one output element
    let d = gid.x;        // dimension within head (0..head_dim-1)
    let head = gid.y;     // head index
    let q_tok = gid.z;    // query token index
    if (d >= head_dim || head >= num_heads || q_tok >= T) { return; }

    let kv_head = head / (num_heads / kv_heads);
    let q_base = q_tok * num_heads * head_dim + head * head_dim;

    // Pass 1: compute max score
    var max_score: f32 = -1e30;
    for (var kv = 0u; kv < T; kv++) {
        var score: f32 = 0.0;
        let k_base = kv * kv_heads * head_dim + kv_head * head_dim;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Pass 2: softmax + weighted V
    var sum_exp: f32 = 0.0;
    var acc: f32 = 0.0;
    for (var kv = 0u; kv < T; kv++) {
        var score: f32 = 0.0;
        let k_base = kv * kv_heads * head_dim + kv_head * head_dim;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score *= scale;
        let w = exp(score - max_score);
        sum_exp += w;
        acc += w * V[kv * kv_heads * head_dim + kv_head * head_dim + d];
    }

    Out[q_base + d] = acc / max(sum_exp, 1e-9);
}
)WGSL";

static void opGQA(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Q = in[0];
    auto* K = in.size() > 1 ? in[1] : nullptr;
    auto* V = in.size() > 2 ? in[2] : nullptr;
    if (!Q || !K || !V || !Q->IsValid() || !K->IsValid() || !V->IsValid()) return;

    int64_t num_heads = n.GetInt("num_heads", 32);
    int64_t kv_heads = n.GetInt("kv_num_heads", 8);
    float scale = n.GetFloat("scale", 1.0f / sqrtf(128.0f));

    int64_t T = (Q->shape.size() >= 2) ? Q->shape[Q->shape.size()-2] : 1;
    int64_t qDim = Q->shape.back();
    int64_t head_dim = qDim / num_heads;

    *out[0] = ex.AllocTensor(Q->shape, Q->dtype);
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({1}, Q->dtype);

    uint32_t scale_u32; memcpy(&scale_u32, &scale, 4);
    uint32_t params[8] = {(uint32_t)T, (uint32_t)num_heads, (uint32_t)head_dim,
                           (uint32_t)kv_heads, scale_u32, 0, 0, 0};
    auto paramBuf = ex.gpu->createBuffer("gqa_p", 32);
    ex.gpu->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipeline("gqa_fast", WGSL_GQA_FAST, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    // Dispatch: (ceil(head_dim/128), num_heads, T)
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((head_dim + 127) / 128), (uint32_t)num_heads, (uint32_t)T, "gqa"});
}

// ─── MultiHeadAttention ──────────────────────────────────────────────────────
// Fast MHA: each thread handles one output element (token, head, dim).
// No shared memory — each thread independently computes attention.

static const char* WGSL_MHA_FAST = R"WGSL(
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let T_q = _params_[0];
    let num_heads = _params_[1];
    let head_dim = _params_[2];
    let T_kv = _params_[3];
    let scale = bitcast<f32>(_params_[4]);

    let d = gid.x;
    let head = gid.y;
    let q_tok = gid.z;
    if (d >= head_dim || head >= num_heads || q_tok >= T_q) { return; }

    let q_base = q_tok * num_heads * head_dim + head * head_dim;

    // Pass 1: max score
    var max_score: f32 = -1e30;
    for (var kv = 0u; kv < T_kv; kv++) {
        var score: f32 = 0.0;
        let k_base = kv * num_heads * head_dim + head * head_dim;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        max_score = max(max_score, score * scale);
    }

    // Pass 2: softmax + weighted V
    var sum_exp: f32 = 0.0;
    var acc: f32 = 0.0;
    for (var kv = 0u; kv < T_kv; kv++) {
        var score: f32 = 0.0;
        let k_base = kv * num_heads * head_dim + head * head_dim;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        let w = exp(score * scale - max_score);
        sum_exp += w;
        acc += w * V[kv * num_heads * head_dim + head * head_dim + d];
    }

    Out[q_base + d] = acc / max(sum_exp, 1e-9);
}
)WGSL";

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

    auto& pl = ex.GetPipeline("mha_fast", WGSL_MHA_FAST, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    // Dispatch: (ceil(head_dim/128), num_heads, T_q)
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((head_dim + 127) / 128), (uint32_t)num_heads, (uint32_t)T_q, "mha"});
}

// ─── RotaryEmbedding ─────────────────────────────────────────────────────────
// Apply rotary position embeddings: x = x * cos + rotate_half(x) * sin

static const char* WGSL_ROPE = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> PosIds: array<i32>;
@group(0) @binding(2) var<storage, read> CosCache: array<f32>;
@group(0) @binding(3) var<storage, read> SinCache: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let head_dim = _params_[1];
    let interleaved = _params_[2];

    let idx = gid.x;
    if (idx >= total) { return; }

    let half = head_dim / 2u;
    let d = idx % head_dim;

    // Position from PosIds
    let pos_idx = idx / head_dim;
    let nPosIds = _params_[3];
    let pos = select(0, PosIds[pos_idx % nPosIds], nPosIds > 0u);

    if (interleaved != 0u) {
        // Interleaved: pairs of (x_2i, x_2i+1) rotated together
        let pair = d / 2u;
        let cos_val = CosCache[u32(pos) * half + pair];
        let sin_val = SinCache[u32(pos) * half + pair];
        let base = idx - d;
        if (d % 2u == 0u) {
            Y[idx] = X[idx] * cos_val - X[base + d + 1u] * sin_val;
        } else {
            Y[idx] = X[base + d - 1u] * sin_val + X[idx] * cos_val;
        }
    } else {
        // Non-interleaved: first half and second half rotated
        if (d < half) {
            let cos_val = CosCache[u32(pos) * half + d];
            let sin_val = SinCache[u32(pos) * half + d];
            let base = idx - d;
            Y[idx] = X[idx] * cos_val - X[base + d + half] * sin_val;
        } else {
            let d2 = d - half;
            let cos_val = CosCache[u32(pos) * half + d2];
            let sin_val = SinCache[u32(pos) * half + d2];
            let base = idx - d;
            Y[idx] = X[base + d2] * sin_val + X[idx] * cos_val;
        }
    }
}
)WGSL";

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
        // No cos/sin cache — just copy
        *out[0] = *X;
        return;
    }

    int64_t nPosIds = PosIds ? tensorNel(PosIds) : 0;

    // Convert int64 PosIds to int32 if needed
    GPUBuffer posIdsBuf;
    if (PosIds && PosIds->IsValid()) {
        ex.EnsureGpu(*PosIds);
        if (PosIds->dtype == TensorDtype::Int64 && nPosIds <= 65536) {
            // Convert int64 to int32 on CPU
            const uint8_t* p = nullptr;
            if (PosIds->isCpuOnly && !PosIds->cpuData.empty()) p = PosIds->cpuData.data();
            else if (!PosIds->cpuData.empty()) p = PosIds->cpuData.data();
            if (p) {
                std::vector<int32_t> i32(nPosIds);
                for (int64_t i = 0; i < nPosIds; i++) {
                    int64_t v; memcpy(&v, p + i*8, 8);
                    i32[i] = (int32_t)v;
                }
                posIdsBuf = ex.gpu->createBuffer("rope_posids32", nPosIds * 4);
                ex.gpu->writeBuffer(posIdsBuf, i32.data(), nPosIds * 4);
            } else {
                posIdsBuf = PosIds->buffer;
            }
        } else if (PosIds->dtype == TensorDtype::Float32) {
            // Float position IDs — cast to int32
            // For now, use buffer directly (kernel reads i32 but gets float bits)
            // TODO: proper float→int conversion
            posIdsBuf = PosIds->buffer;
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

    auto& pl = ex.GetPipeline("rope", WGSL_ROPE, 6);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, posIdsBuf}, {2, CosCache->buffer},
        {3, SinCache->buffer}, {4, out[0]->buffer}, {5, paramBuf}});

    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "rope"});
}

REGISTER_OP(GroupQueryAttention, opGQA)
REGISTER_OP(MultiHeadAttention, opMHA)
REGISTER_OP(RotaryEmbedding, opRotaryEmbedding)

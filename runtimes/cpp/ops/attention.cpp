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
// Tiled GQA: each workgroup handles one (head, query_token) pair.
// Uses shared memory for dot product reduction.

static const char* WGSL_GQA_TILED = R"WGSL(
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const WG: u32 = 128u;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let T = _params_[0];
    let num_heads = _params_[1];
    let head_dim = _params_[2];
    let kv_heads = _params_[3];
    let scale = bitcast<f32>(_params_[4]);

    let q_tok = wid.x;
    let head = wid.y;
    if (q_tok >= T || head >= num_heads) { return; }
    let kv_head = head / (num_heads / kv_heads);

    let d = lid.x;
    if (d >= head_dim) { return; }

    let q_val = Q[q_tok * num_heads * head_dim + head * head_dim + d];

    // Pass 1: max score
    var local_max: f32 = -1e30;
    for (var kv = 0u; kv < T; kv++) {
        let k_val = K[kv * kv_heads * head_dim + kv_head * head_dim + d];
        partial[d] = q_val * k_val * scale;
        workgroupBarrier();
        for (var s = WG/2u; s > 0u; s >>= 1u) {
            if (d < s && d + s < head_dim) { partial[d] += partial[d + s]; }
            workgroupBarrier();
        }
        if (d == 0u) { local_max = max(local_max, partial[0]); }
        workgroupBarrier();
    }
    if (d == 0u) { shared_max = local_max; }
    workgroupBarrier();
    let max_s = shared_max;

    // Pass 2: softmax + weighted V
    var acc: f32 = 0.0;
    var local_sum: f32 = 0.0;
    for (var kv = 0u; kv < T; kv++) {
        let k_val = K[kv * kv_heads * head_dim + kv_head * head_dim + d];
        partial[d] = q_val * k_val * scale;
        workgroupBarrier();
        for (var s = WG/2u; s > 0u; s >>= 1u) {
            if (d < s && d + s < head_dim) { partial[d] += partial[d + s]; }
            workgroupBarrier();
        }
        let w = exp(partial[0] - max_s);
        workgroupBarrier();
        if (d == 0u) { local_sum += w; }
        acc += w * V[kv * kv_heads * head_dim + kv_head * head_dim + d];
    }
    if (d == 0u) { shared_sum = local_sum; }
    workgroupBarrier();

    Out[q_tok * num_heads * head_dim + head * head_dim + d] = acc / max(shared_sum, 1e-9);
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

    auto& pl = ex.GetPipeline("gqa_tiled", WGSL_GQA_TILED, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    // Dispatch: (T, num_heads, 1)
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)T, (uint32_t)num_heads, 1, "gqa"});
}

// ─── MultiHeadAttention ──────────────────────────────────────────────────────
// Proper tiled MHA: each workgroup handles one (batch, head, query_token)
// Uses shared memory for Q row, iterates over KV in tiles.
// Layout: Q/K/V are [batch, seq_len, num_heads * head_dim] (3D)
//   or [batch, num_heads, seq_len, head_dim] after reshape (4D)
// ORT MHA always receives [batch, seq_len, num_heads * head_dim]

static const char* WGSL_MHA_TILED = R"WGSL(
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

// Each workgroup computes attention for one (head, query_token) pair
// Workgroup size = head_dim (capped at 128)
const WG: u32 = 128u;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> partial_sums: array<f32, WG>;
var<workgroup> partial_maxs: array<f32, WG>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let T_q = _params_[0];
    let num_heads = _params_[1];
    let head_dim = _params_[2];
    let T_kv = _params_[3];
    let scale = bitcast<f32>(_params_[4]);

    let q_tok = wid.x;
    let head = wid.y;
    if (q_tok >= T_q || head >= num_heads) { return; }

    let d = lid.x;
    if (d >= head_dim) { return; }

    // Load Q[q_tok, head, d]
    let q_val = Q[q_tok * num_heads * head_dim + head * head_dim + d];

    // Pass 1: Find max score across all KV tokens
    var local_max: f32 = -1e30;
    for (var kv = 0u; kv < T_kv; kv++) {
        let k_val = K[kv * num_heads * head_dim + head * head_dim + d];
        let partial = q_val * k_val * scale;
        // Reduce across head_dim using shared memory
        partial_sums[d] = partial;
        workgroupBarrier();
        // Tree reduction
        for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
            if (d < stride && d + stride < head_dim) {
                partial_sums[d] += partial_sums[d + stride];
            }
            workgroupBarrier();
        }
        if (d == 0u) {
            local_max = max(local_max, partial_sums[0]);
        }
        workgroupBarrier();
    }
    if (d == 0u) { shared_max = local_max; }
    workgroupBarrier();
    let max_score = shared_max;

    // Pass 2: Compute softmax weights and weighted V sum
    var acc: f32 = 0.0;
    var local_sum: f32 = 0.0;

    for (var kv = 0u; kv < T_kv; kv++) {
        let k_val = K[kv * num_heads * head_dim + head * head_dim + d];
        let partial = q_val * k_val * scale;
        partial_sums[d] = partial;
        workgroupBarrier();
        // Reduce
        for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
            if (d < stride && d + stride < head_dim) {
                partial_sums[d] += partial_sums[d + stride];
            }
            workgroupBarrier();
        }
        let score = partial_sums[0];
        workgroupBarrier();

        let w = exp(score - max_score);
        if (d == 0u) { local_sum += w; }
        let v_val = V[kv * num_heads * head_dim + head * head_dim + d];
        acc += w * v_val;
    }

    if (d == 0u) { shared_sum = local_sum; }
    workgroupBarrier();

    // Write output
    Out[q_tok * num_heads * head_dim + head * head_dim + d] = acc / max(shared_sum, 1e-9);
}
)WGSL";

static void opMHA(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Q = in[0]; auto* K = in[1]; auto* V = in[2];
    if (!Q || !K || !V || !Q->IsValid() || !K->IsValid() || !V->IsValid()) return;

    int64_t num_heads = n.GetInt("num_heads", 30);
    float scale = n.GetFloat("scale", 1.0f / sqrtf(128.0f));

    // Q/K/V shape: [batch, seq_len, num_heads * head_dim]
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

    auto& pl = ex.GetPipeline("mha_tiled", WGSL_MHA_TILED, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, Q->buffer}, {1, K->buffer}, {2, V->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    // Dispatch: (T_q, num_heads, 1) — each workgroup = one (token, head)
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)T_q, (uint32_t)num_heads, 1, "mha"});
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

    uint32_t params[4] = {(uint32_t)total, (uint32_t)head_dim,
                           (uint32_t)interleaved, (uint32_t)nPosIds};
    auto paramBuf = ex.gpu->createBuffer("rope_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    GPUBuffer posIdsBuf;
    if (PosIds && PosIds->IsValid()) {
        posIdsBuf = PosIds->buffer;
    } else {
        int32_t zero = 0;
        posIdsBuf = ex.gpu->createBuffer("rope_pos0", 4);
        ex.gpu->writeBuffer(posIdsBuf, &zero, 4);
    }

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

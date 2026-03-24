/**
 * ops/norm.cpp — Normalization ops using embedded WGSL kernels.
 * SimplifiedLayerNormalization (RMSNorm), SkipSimplifiedLayerNormalization,
 * LayerNormalization, InstanceNormalization.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <cstdio>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

// ─── RMSNorm (SimplifiedLayerNormalization) ──────────────────────────────────
// Uses the optimized rms_norm kernel from compiler/kernels/shared/ when available,
// falls back to a simple per-row kernel otherwise.

static const char* WGSL_RMSNORM_SIMPLE = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params { stride: i32, N: i32, eps: f32, };
@group(0) @binding(4) var<storage, read> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let N = u32(params.N);
    if (row >= u32(params.stride) / N) { return; }
    let base = row * N;
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) { let v = X[base + i]; sum_sq += v * v; }
    let rms = sqrt(sum_sq / f32(N) + params.eps);
    let inv_rms = 1.0 / rms;
    for (var i = 0u; i < N; i++) { Y[base + i] = X[base + i] * inv_rms * W[i]; }
    if (gid.x == 0u) { Rstd[row] = inv_rms; }
}
)WGSL";

static void opSimplifiedLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* W = in.size() > 1 ? in[1] : nullptr;
    if (!X || !X->IsValid() || X->shape.empty()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;

    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({nRows}, X->dtype);

    if (!W || !W->IsValid()) return;

    // Allocate Rstd output
    GPUBuffer rstdBuf;
    if (out.size() > 1 && out[1] && out[1]->IsValid()) {
        rstdBuf = out[1]->buffer;
    } else {
        rstdBuf = ex.gpu->createBuffer("rstd", std::max((int64_t)4, nRows * 4));
    }

    // Use simple RMSNorm (the optimized rms_norm requires subgroups which
    // might not match the binding layout for the graph executor)
    struct { int32_t stride; int32_t N; float eps; } p;
    p.stride = (int32_t)(nRows * hiddenDim);
    p.N = (int32_t)hiddenDim;
    p.eps = eps;
    auto paramBuf = ex.gpu->createBuffer("rmsn_p", 16);
    ex.gpu->writeBuffer(paramBuf, &p, 12);

    auto& pl = ex.GetPipeline("rmsnorm_simple", WGSL_RMSNORM_SIMPLE, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, out[0]->buffer}, {2, W->buffer},
        {3, rstdBuf}, {4, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((nRows + 255) / 256), 1, 1, "rmsnorm"});
}

// ─── SkipSimplifiedLayerNorm (residual add + RMSNorm) ────────────────────────

static const char* WGSL_SKIP_RMSNORM = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Skip: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let v = X[base + i] + Skip[base + i];
        SkipOut[base + i] = v;
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / sqrt(sum_sq / f32(N) + eps);
    for (var i = 0u; i < N; i++) {
        Y[base + i] = SkipOut[base + i] * inv_rms * W[i];
    }
}
)WGSL";

static void opSkipSimplifiedLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* Skip = in.size() > 1 ? in[1] : nullptr;
    auto* W = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid() || X->shape.empty()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;

    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor(X->shape, X->dtype);

    if (!Skip || !W || !Skip->IsValid() || !W->IsValid()) return;

    GPUBuffer skipOutBuf;
    if (out.size() > 3 && out[3] && out[3]->IsValid()) {
        skipOutBuf = out[3]->buffer;
    } else {
        auto skipOutTensor = ex.AllocTensor(X->shape, X->dtype);
        skipOutBuf = skipOutTensor.buffer;
        if (out.size() > 3 && out[3]) *out[3] = skipOutTensor;
    }

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)hiddenDim, (uint32_t)nRows, eps_u32, 0};
    auto paramBuf = ex.gpu->createBuffer("srmsn_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("skip_rmsnorm", WGSL_SKIP_RMSNORM, 6);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, Skip->buffer}, {2, W->buffer},
        {3, out[0]->buffer}, {4, skipOutBuf}, {5, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((nRows + 255) / 256), 1, 1, "skip_rmsnorm"});
}

// ─── LayerNormalization ──────────────────────────────────────────────────────

static void opLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* W = in.size() > 1 ? in[1] : nullptr;
    auto* B = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid() || X->shape.empty()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;
    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    if (!W || !W->IsValid()) return;

    GPUBuffer biasBuf;
    if (B && B->IsValid()) { biasBuf = B->buffer; }
    else {
        std::vector<float> zeros((size_t)hiddenDim, 0.0f);
        biasBuf = ex.gpu->createBuffer("ln_b0", hiddenDim * 4);
        ex.gpu->writeBuffer(biasBuf, zeros.data(), hiddenDim * 4);
    }

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)hiddenDim, (uint32_t)nRows, eps_u32, 0};
    auto paramBuf = ex.gpu->createBuffer("ln_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("layer_norm", WGSL_LAYER_NORM, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((nRows + 255) / 256), 1, 1, "layernorm"});
}

// ─── InstanceNormalization ───────────────────────────────────────────────────

static void opInstanceNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* Scale = in.size() > 1 ? in[1] : nullptr;
    auto* Bias = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t N_batch = (X->shape.size() >= 4) ? X->shape[0] : 1;
    int64_t C = (X->shape.size() >= 3) ? X->shape[X->shape.size()-3] : 1;
    int64_t HW = 1;
    for (size_t i = X->shape.size()-2; i < X->shape.size(); i++) HW *= X->shape[i];

    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    if (!Scale || !Bias || !Scale->IsValid() || !Bias->IsValid()) return;

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)C, (uint32_t)HW, (uint32_t)N_batch, eps_u32};
    auto paramBuf = ex.gpu->createBuffer("in_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("instance_norm", WGSL_INSTANCE_NORM, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, Scale->buffer}, {2, Bias->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((N_batch * C + 255) / 256), 1, 1, "instancenorm"});
}

REGISTER_OP(SimplifiedLayerNormalization, opSimplifiedLayerNorm)
REGISTER_OP(SkipSimplifiedLayerNormalization, opSkipSimplifiedLayerNorm)
REGISTER_OP(LayerNormalization, opLayerNorm)
REGISTER_OP(InstanceNormalization, opInstanceNorm)

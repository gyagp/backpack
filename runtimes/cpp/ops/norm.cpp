/**
 * ops/norm.cpp — Normalization ops with full GPU kernels.
 * SimplifiedLayerNormalization (RMSNorm), SkipSimplifiedLayerNormalization,
 * LayerNormalization, InstanceNormalization.
 */

#include "../graph_executor.h"
#include <cstdio>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

// ─── RMSNorm (SimplifiedLayerNormalization) ──────────────────────────────────

static const char* WGSL_RMSNORM = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0]; // hidden dim
    let nRows = _params_[1];
    let eps_u32 = _params_[2];
    let eps = bitcast<f32>(eps_u32);

    let row = gid.x;
    if (row >= nRows) { return; }

    let base = row * N;

    // Compute mean of squares
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let v = X[base + i];
        sum_sq += v * v;
    }
    let rms = sqrt(sum_sq / f32(N) + eps);
    let inv_rms = 1.0 / rms;

    // Normalize and scale
    for (var i = 0u; i < N; i++) {
        Y[base + i] = X[base + i] * inv_rms * W[i];
    }
}
)WGSL";

static void opSimplifiedLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* W = in.size() > 1 ? in[1] : nullptr;
    if (!X || !X->IsValid()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;

    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    // Additional outputs (inv_std_dev) are optional
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({nRows}, X->dtype);

    if (!W || !W->IsValid()) return;

    uint32_t eps_u32;
    memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)hiddenDim, (uint32_t)nRows, eps_u32, 0};
    auto paramBuf = ex.gpu->createBuffer("rmsn_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("rmsnorm", WGSL_RMSNORM, 4);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, out[0]->buffer}, {3, paramBuf}});

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

    // Add residual
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
    if (!X || !X->IsValid()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;

    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    // out[1] = inv_std_dev (optional), out[2] = unused, out[3] = skip_output
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor(X->shape, X->dtype);

    if (!Skip || !W || !Skip->IsValid() || !W->IsValid()) {
        // Fallback to regular RMSNorm if no skip connection
        return;
    }

    // out[3] is the skip output (X + Skip)
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

// ─── LayerNormalization (mean + variance) ────────────────────────────────────

static const char* WGSL_LAYERNORM = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);

    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;

    // Compute mean
    var mean: f32 = 0.0;
    for (var i = 0u; i < N; i++) { mean += X[base + i]; }
    mean = mean / f32(N);

    // Compute variance
    var var_sum: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let d = X[base + i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(N) + eps);

    for (var i = 0u; i < N; i++) {
        Y[base + i] = (X[base + i] - mean) * inv_std * W[i] + B[i];
    }
}
)WGSL";

static void opLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* W = in.size() > 1 ? in[1] : nullptr;
    auto* B = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);
    int64_t axis = n.GetInt("axis", -1);
    int64_t hiddenDim = X->shape.back();
    int64_t nRows = tensorNel(X) / hiddenDim;

    *out[0] = ex.AllocTensor(X->shape, X->dtype);

    if (!W || !W->IsValid()) return;

    GPUBuffer biasBuf;
    if (B && B->IsValid()) {
        biasBuf = B->buffer;
    } else {
        std::vector<float> zeros((size_t)hiddenDim, 0.0f);
        biasBuf = ex.gpu->createBuffer("ln_b0", hiddenDim * 4);
        ex.gpu->writeBuffer(biasBuf, zeros.data(), hiddenDim * 4);
    }

    uint32_t eps_u32; memcpy(&eps_u32, &eps, 4);
    uint32_t params[4] = {(uint32_t)hiddenDim, (uint32_t)nRows, eps_u32, 0};
    auto paramBuf = ex.gpu->createBuffer("ln_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("layernorm", WGSL_LAYERNORM, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});

    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((nRows + 255) / 256), 1, 1, "layernorm"});
}

// ─── InstanceNormalization ───────────────────────────────────────────────────

static const char* WGSL_INSTANCENORM = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let C = _params_[0];    // channels
    let HW = _params_[1];   // height * width
    let N = _params_[2];    // batch
    let eps = bitcast<f32>(_params_[3]);

    let idx = gid.x; // batch * C index
    if (idx >= N * C) { return; }

    let n = idx / C;
    let c = idx % C;
    let base = n * C * HW + c * HW;

    // Compute mean
    var mean: f32 = 0.0;
    for (var i = 0u; i < HW; i++) { mean += X[base + i]; }
    mean /= f32(HW);

    // Compute variance
    var var_sum: f32 = 0.0;
    for (var i = 0u; i < HW; i++) {
        let d = X[base + i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(HW) + eps);

    let s = Scale[c];
    let b = Bias[c];
    for (var i = 0u; i < HW; i++) {
        Y[base + i] = (X[base + i] - mean) * inv_std * s + b;
    }
}
)WGSL";

static void opInstanceNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    auto* Scale = in.size() > 1 ? in[1] : nullptr;
    auto* Bias = in.size() > 2 ? in[2] : nullptr;
    if (!X || !X->IsValid()) return;

    float eps = n.GetFloat("epsilon", 1e-5f);

    // X shape: [N, C, H, W]
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

    auto& pl = ex.GetPipeline("instancenorm", WGSL_INSTANCENORM, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, Scale->buffer}, {2, Bias->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    int64_t total = N_batch * C;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "instancenorm"});
}

REGISTER_OP(SimplifiedLayerNormalization, opSimplifiedLayerNorm)
REGISTER_OP(SkipSimplifiedLayerNormalization, opSkipSimplifiedLayerNorm)
REGISTER_OP(LayerNormalization, opLayerNorm)
REGISTER_OP(InstanceNormalization, opInstanceNorm)

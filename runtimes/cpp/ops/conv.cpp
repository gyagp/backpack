/**
 * ops/conv.cpp — Convolution and resize ops with GPU kernels.
 * Conv (2D convolution), Resize (upsampling).
 */

#include "../graph_executor.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

// ─── Conv2D ──────────────────────────────────────────────────────────────────
// Standard 2D convolution: Y[n,co,oh,ow] = sum_{ci,kh,kw} X[n,ci,oh*sh+kh,ow*sw+kw] * W[co,ci,kh,kw] + bias[co]

static const char* WGSL_CONV2D = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                let ih = i32(oh * stride_h + kh) - i32(pad_h);
                let iw = i32(ow * stride_w + kw) - i32(pad_w);
                if (ih >= 0 && ih < i32(H_in) && iw >= 0 && iw < i32(W_in)) {
                    let x_val = X[n * C_in * H_in * W_in + ci * H_in * W_in + u32(ih) * W_in + u32(iw)];
                    let w_val = W[co * ci_per_group * KH * KW + ci_local * KH * KW + kh * KW + kw];
                    acc += x_val * w_val;
                }
            }
        }
    }

    Y[idx] = acc + Bias[co];
}
)WGSL";

static void opConv(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0]; // [N, C_in, H, W]
    auto* W = in[1]; // [C_out, C_in/group, KH, KW]
    auto* B = in.size() > 2 ? in[2] : nullptr; // [C_out]
    if (!X || !W || !X->IsValid() || !W->IsValid()) return;

    int64_t batch = X->shape[0];
    int64_t C_in = X->shape[1];
    int64_t H_in = X->shape[2];
    int64_t W_in = X->shape[3];
    int64_t C_out = W->shape[0];
    int64_t KH = W->shape[2];
    int64_t KW = W->shape[3];

    // Get attributes
    int64_t group = n.GetInt("group", 1);
    std::vector<int64_t> pads = {0,0,0,0}, strides = {1,1};
    if (n.attrIntLists.count("pads")) pads = n.attrIntLists.at("pads");
    if (n.attrIntLists.count("strides")) strides = n.attrIntLists.at("strides");
    int64_t pad_h = pads.size() >= 1 ? pads[0] : 0;
    int64_t pad_w = pads.size() >= 2 ? pads[1] : 0;
    int64_t stride_h = strides.size() >= 1 ? strides[0] : 1;
    int64_t stride_w = strides.size() >= 2 ? strides[1] : 1;

    int64_t H_out = (H_in + 2 * pad_h - KH) / stride_h + 1;
    int64_t W_out = (W_in + 2 * pad_w - KW) / stride_w + 1;

    *out[0] = ex.AllocTensor({batch, C_out, H_out, W_out}, X->dtype);

    GPUBuffer biasBuf;
    if (B && B->IsValid()) {
        biasBuf = B->buffer;
    } else {
        std::vector<float> zeros((size_t)C_out, 0.0f);
        biasBuf = ex.gpu->createBuffer("conv_b0", C_out * 4);
        ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 4);
    }

    // 14 params
    uint32_t params[16] = {
        (uint32_t)batch, (uint32_t)C_in, (uint32_t)H_in, (uint32_t)W_in,
        (uint32_t)C_out, (uint32_t)KH, (uint32_t)KW,
        (uint32_t)pad_h, (uint32_t)pad_w, (uint32_t)stride_h, (uint32_t)stride_w,
        (uint32_t)H_out, (uint32_t)W_out, (uint32_t)group, 0, 0
    };
    auto paramBuf = ex.gpu->createBuffer("conv_p", 64);
    ex.gpu->writeBuffer(paramBuf, params, 64);

    auto& pl = ex.GetPipeline("conv2d", WGSL_CONV2D, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});

    int64_t total = batch * C_out * H_out * W_out;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "conv2d"});
}

// ─── Resize (nearest-neighbor upsampling) ────────────────────────────────────

static const char* WGSL_RESIZE = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0]; // batch
    let C = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let H_out = _params_[4];
    let W_out = _params_[5];

    let total = N * C * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C * H_out * W_out);
    let c = (idx / (H_out * W_out)) % C;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    // Nearest neighbor
    let ih = min(oh * H_in / H_out, H_in - 1u);
    let iw = min(ow * W_in / W_out, W_in - 1u);

    Y[idx] = X[n * C * H_in * W_in + c * H_in * W_in + ih * W_in + iw];
}
)WGSL";

static void opResize(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    if (!X || !X->IsValid() || X->shape.size() < 4) return;

    // Read scale factors from input[2] (scales tensor)
    // Flush pending before CPU readback
    if (!ex.pendingDispatches_.empty()) {
        ex.gpu->submitOnly(ex.pendingDispatches_, false);
        ex.gpu->waitForQueue();
        ex.pendingDispatches_.clear();
    }

    int64_t N = X->shape[0], C = X->shape[1], H_in = X->shape[2], W_in = X->shape[3];
    int64_t H_out = H_in, W_out = W_in;

    // Try to read scales from input[2]
    if (in.size() > 2 && in[2] && in[2]->IsValid()) {
        int64_t nScales = tensorNel(in[2]);
        if (nScales >= 4) {
            std::vector<float> scales(nScales);
            auto rb = ex.gpu->readBuffer(in[2]->buffer, nScales * sizeof(float));
            memcpy(scales.data(), rb.data(), nScales * sizeof(float));
            // scales: [N_scale, C_scale, H_scale, W_scale]
            H_out = (int64_t)(H_in * scales[nScales-2]);
            W_out = (int64_t)(W_in * scales[nScales-1]);
        }
    }

    *out[0] = ex.AllocTensor({N, C, H_out, W_out}, X->dtype);

    uint32_t params[8] = {(uint32_t)N, (uint32_t)C, (uint32_t)H_in, (uint32_t)W_in,
                           (uint32_t)H_out, (uint32_t)W_out, 0, 0};
    auto paramBuf = ex.gpu->createBuffer("resize_p", 32);
    ex.gpu->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipeline("resize", WGSL_RESIZE, 3);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, out[0]->buffer}, {2, paramBuf}});

    int64_t total = N * C * H_out * W_out;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "resize"});
}

REGISTER_OP(Conv, opConv)
REGISTER_OP(Resize, opResize)

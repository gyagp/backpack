/**
 * ops/conv.cpp — Convolution and resize ops using embedded WGSL kernels.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

// ─── Conv2D ──────────────────────────────────────────────────────────────────

static void opConv(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0]; auto* W = in[1];
    auto* B = in.size() > 2 ? in[2] : nullptr;
    if (!X || !W || !X->IsValid() || !W->IsValid() || X->shape.size() < 4) return;

    int64_t batch = X->shape[0], C_in = X->shape[1];
    int64_t H_in = X->shape[2], W_in = X->shape[3];
    int64_t C_out = W->shape[0], KH = W->shape[2], KW = W->shape[3];
    int64_t group = n.GetInt("group", 1);

    std::vector<int64_t> pads = {0,0,0,0}, strides = {1,1};
    if (n.attrIntLists.count("pads")) pads = n.attrIntLists.at("pads");
    if (n.attrIntLists.count("strides")) strides = n.attrIntLists.at("strides");
    int64_t pad_h = pads.size() >= 1 ? pads[0] : 0;
    int64_t pad_w = pads.size() >= 2 ? pads[1] : 0;
    int64_t stride_h = strides.size() >= 1 ? strides[0] : 1;
    int64_t stride_w = strides.size() >= 2 ? strides[1] : 1;
    int64_t H_out = (H_in + 2*pad_h - KH) / stride_h + 1;
    int64_t W_out = (W_in + 2*pad_w - KW) / stride_w + 1;

    *out[0] = ex.AllocTensor({batch, C_out, H_out, W_out}, X->dtype);

    GPUBuffer biasBuf;
    if (B && B->IsValid()) { biasBuf = B->buffer; }
    else {
        std::vector<float> zeros((size_t)C_out, 0.0f);
        biasBuf = ex.gpu->createBuffer("conv_b0", C_out * 4);
        ex.gpu->writeBuffer(biasBuf, zeros.data(), C_out * 4);
    }

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

// ─── Resize ──────────────────────────────────────────────────────────────────

static void opResize(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    if (!X || !X->IsValid() || X->shape.size() < 4) return;

    int64_t N = X->shape[0], C = X->shape[1], H_in = X->shape[2], W_in = X->shape[3];
    int64_t H_out = H_in, W_out = W_in;

    // Try to read scales from input[2]
    if (in.size() > 2 && in[2] && in[2]->IsValid()) {
        int64_t nScales = tensorNel(in[2]);
        if (nScales >= 4) {
            // Need CPU readback — flush pending first
            if (!ex.pendingDispatches_.empty()) {
                ex.gpu->submitOnly(ex.pendingDispatches_, false);
                ex.gpu->waitForQueue();
                ex.pendingDispatches_.clear();
            }
            auto rb = ex.gpu->readBuffer(in[2]->buffer, nScales * sizeof(float));
            auto* scales = (const float*)rb.data();
            H_out = (int64_t)(H_in * scales[nScales-2]);
            W_out = (int64_t)(W_in * scales[nScales-1]);
        }
    }

    *out[0] = ex.AllocTensor({N, C, H_out, W_out}, X->dtype);

    uint32_t params[8] = {(uint32_t)N, (uint32_t)C, (uint32_t)H_in, (uint32_t)W_in,
                           (uint32_t)H_out, (uint32_t)W_out, 0, 0};
    auto paramBuf = ex.gpu->createBuffer("resize_p", 32);
    ex.gpu->writeBuffer(paramBuf, params, 32);

    auto& pl = ex.GetPipeline("resize_nearest", WGSL_RESIZE_NEAREST, 3);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, out[0]->buffer}, {2, paramBuf}});
    int64_t total = N * C * H_out * W_out;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "resize"});
}

REGISTER_OP(Conv, opConv)
REGISTER_OP(Resize, opResize)

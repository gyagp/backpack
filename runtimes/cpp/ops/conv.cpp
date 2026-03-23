/**
 * ops/conv.cpp — Convolution ops.
 * Conv, Resize.
 */

#include "../graph_executor.h"

static void opConv(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // 2D Convolution — TODO
    auto* X = in[0];
    auto* W = in[1];
    if (!X || !W || !X->IsValid() || !W->IsValid()) return;

    // Output shape: [N, C_out, H_out, W_out]
    auto outShape = X->shape;
    if (outShape.size() >= 2) outShape[1] = W->shape[0]; // C_out
    // TODO: compute H_out, W_out from kernel/padding/stride
    *out[0] = ex.AllocTensor(outShape, X->dtype);
}

static void opResize(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Upsampling — TODO
    auto* X = in[0];
    if (!X || !X->IsValid()) return;
    *out[0] = ex.AllocTensor(X->shape, X->dtype); // TODO: scaled shape
}

REGISTER_OP(Conv, opConv)
REGISTER_OP(Resize, opResize)

/**
 * ops/norm.cpp — Normalization ops.
 * SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization,
 * LayerNormalization, InstanceNormalization.
 */

#include "../graph_executor.h"

static void opSimplifiedLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // RMSNorm — TODO: use existing WGSL kernel
    auto* X = in[0];
    if (!X || !X->IsValid()) return;
    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    // Additional outputs (inv_std_dev) are optional
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({1}, X->dtype);
}

static void opSkipSimplifiedLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Skip (residual add) + RMSNorm — TODO
    auto* X = in[0];
    if (!X || !X->IsValid()) return;
    *out[0] = ex.AllocTensor(X->shape, X->dtype);
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor(X->shape, X->dtype);
}

static void opLayerNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    if (!X || !X->IsValid()) return;
    *out[0] = ex.AllocTensor(X->shape, X->dtype);
}

static void opInstanceNorm(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    if (!X || !X->IsValid()) return;
    *out[0] = ex.AllocTensor(X->shape, X->dtype);
}

REGISTER_OP(SimplifiedLayerNormalization, opSimplifiedLayerNorm)
REGISTER_OP(SkipSimplifiedLayerNormalization, opSkipSimplifiedLayerNorm)
REGISTER_OP(LayerNormalization, opLayerNorm)
REGISTER_OP(InstanceNormalization, opInstanceNorm)

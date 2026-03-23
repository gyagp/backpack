/**
 * ops/attention.cpp — Attention ops.
 * GroupQueryAttention, MultiHeadAttention, RotaryEmbedding.
 */

#include "../graph_executor.h"

static void opGQA(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Fused GroupQueryAttention — TODO
    auto* Q = in[0];
    if (!Q || !Q->IsValid()) return;
    *out[0] = ex.AllocTensor(Q->shape, Q->dtype);
    // GQA can have present_key, present_value outputs
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({1}, Q->dtype);
}

static void opMHA(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Standard MultiHeadAttention — TODO
    auto* Q = in[0];
    if (!Q || !Q->IsValid()) return;
    *out[0] = ex.AllocTensor(Q->shape, Q->dtype);
    for (size_t i = 1; i < out.size(); i++)
        if (out[i]) *out[i] = ex.AllocTensor({1}, Q->dtype);
}

static void opRotaryEmbedding(GraphExecutor& ex, const OnnxGraphNode& n,
    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Standalone RoPE — TODO
    auto* X = in[0];
    if (!X || !X->IsValid()) return;
    *out[0] = ex.AllocTensor(X->shape, X->dtype);
}

REGISTER_OP(GroupQueryAttention, opGQA)
REGISTER_OP(MultiHeadAttention, opMHA)
REGISTER_OP(RotaryEmbedding, opRotaryEmbedding)

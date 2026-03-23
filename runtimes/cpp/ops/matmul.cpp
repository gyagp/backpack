/**
 * ops/matmul.cpp — Matrix multiplication ops.
 * MatMul, MatMulNBits, Gemm, GatherBlockQuantized.
 */

#include "../graph_executor.h"
#include <cstdio>

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0; int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

static void opMatMul(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // TODO: GPU fp32 matmul kernel
    auto* A = in[0]; auto* B = in[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;
    // Output shape: [..., M, N] where A=[...,M,K], B=[...,K,N]
    auto outShape = A->shape;
    if (outShape.size() >= 2 && B->shape.size() >= 1)
        outShape.back() = B->shape.back();
    *out[0] = ex.AllocTensor(outShape, A->dtype);
}

static void opMatMulNBits(GraphExecutor& ex, const OnnxGraphNode& n,
                           const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // TODO: Q4/Q8 quantized matmul kernel
    auto* X = in[0];
    if (!X || !X->IsValid()) return;
    int64_t N = n.GetInt("N");
    int64_t K = n.GetInt("K");
    auto outShape = X->shape;
    if (!outShape.empty()) outShape.back() = N;
    *out[0] = ex.AllocTensor(outShape, X->dtype);
}

static void opGemm(GraphExecutor& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // TODO: Gemm = alpha*A*B + beta*C
    auto* A = in[0]; auto* B = in[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;
    int64_t transB = n.GetInt("transB", 0);
    int64_t M = A->shape.size() >= 2 ? A->shape[A->shape.size()-2] : 1;
    int64_t N = transB ? B->shape[B->shape.size()-2] : B->shape.back();
    *out[0] = ex.AllocTensor({M, N}, A->dtype);
}

static void opGatherBlockQuantized(GraphExecutor& ex, const OnnxGraphNode& n,
                                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // Quantized embedding lookup — TODO
    auto* indices = in.size() > 1 ? in[1] : nullptr;
    if (!indices || !indices->IsValid()) return;
    int64_t nIdx = tensorNel(indices);
    // TODO: proper embedding size
    *out[0] = ex.AllocTensor({1, nIdx, 2560}, TensorDtype::Float32);
}

REGISTER_OP(MatMul, opMatMul)
REGISTER_OP(MatMulNBits, opMatMulNBits)
REGISTER_OP(Gemm, opGemm)
REGISTER_OP(GatherBlockQuantized, opGatherBlockQuantized)

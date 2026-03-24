/**
 * ops/matmul.cpp — Matrix multiplication ops using embedded WGSL kernels.
 *
 * MatMulNBits: Q4 quantized matmul
 * MatMul: fp32 matmul
 * Gemm: matmul + bias
 * GatherBlockQuantized: quantized embedding
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// ─── MatMulNBits ─────────────────────────────────────────────────────────────

static void opMatMulNBits(GraphExecutor& ex, const OnnxGraphNode& n,
                           const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0]; auto* W = in[1]; auto* S = in[2];
    if (!X || !W || !S || !X->IsValid() || !W->IsValid() || !S->IsValid()) return;
    ex.EnsureGpu(*X);

    uint32_t N = (uint32_t)n.GetInt("N");
    uint32_t K = (uint32_t)n.GetInt("K");
    int64_t M = 1;
    for (size_t i = 0; i + 1 < X->shape.size(); i++) M *= X->shape[i];

    auto outShape = X->shape;
    outShape.back() = N;
    *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);

    uint32_t params[4] = {(uint32_t)M, N, K, 0};
    auto paramBuf = ex.gpu->createBuffer("mmnb_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("matmul_q4", WGSL_MATMUL_Q4, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, S->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (N + 7) / 8, (uint32_t)M, 1, "matmul_q4"});
}

// ─── MatMul ──────────────────────────────────────────────────────────────────

static void opMatMul(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;
    ex.EnsureGpu(*A); ex.EnsureGpu(*B);

    int64_t K = A->shape.back();
    int64_t M = (A->shape.size() >= 2) ? A->shape[A->shape.size()-2] : 1;
    int64_t N_out = B->shape.back();
    auto outShape = A->shape;
    outShape.back() = N_out;
    *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);

    uint32_t params[4] = {(uint32_t)M, (uint32_t)N_out, (uint32_t)K, 0};
    auto paramBuf = ex.gpu->createBuffer("mm_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("matmul_f32", WGSL_MATMUL_F32, 4);
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, out[0]->buffer}, {3, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((N_out + 15) / 16), (uint32_t)((M + 15) / 16), 1, "matmul_f32"});
}

// ─── Gemm ────────────────────────────────────────────────────────────────────

static void opGemm(GraphExecutor& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;
    ex.EnsureGpu(*A); ex.EnsureGpu(*B);

    int64_t transB = n.GetInt("transB", 0);
    int64_t M = A->shape.size() >= 2 ? A->shape[0] : 1;
    int64_t K = A->shape.back();
    int64_t N_out = transB ? B->shape[0] : B->shape.back();

    *out[0] = ex.AllocTensor({M, N_out}, TensorDtype::Float32);

    GPUBuffer biasBuf;
    if (in.size() > 2 && in[2] && in[2]->IsValid()) {
        ex.EnsureGpu(*in[2]);
        biasBuf = in[2]->buffer;
    } else {
        std::vector<float> zeros((size_t)N_out, 0.0f);
        biasBuf = ex.gpu->createBuffer("gemm_b0", N_out * 4);
        ex.gpu->writeBuffer(biasBuf, zeros.data(), N_out * 4);
    }

    uint32_t params[4] = {(uint32_t)M, (uint32_t)N_out, (uint32_t)K, (uint32_t)transB};
    auto paramBuf = ex.gpu->createBuffer("gemm_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("gemm", WGSL_GEMM, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((N_out + 15) / 16), (uint32_t)((M + 15) / 16), 1, "gemm"});
}

// ─── GatherBlockQuantized ────────────────────────────────────────────────────

static void opGatherBlockQuantized(GraphExecutor& ex, const OnnxGraphNode& n,
                                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* W = in[0]; auto* Indices = in[1];
    auto* Scales = in.size() > 2 ? in[2] : nullptr;
    if (!W || !Indices || !W->IsValid() || !Indices->IsValid()) return;
    ex.EnsureGpu(*W); ex.EnsureGpu(*Indices);

    int64_t nIdx = 1;
    for (auto d : Indices->shape) nIdx *= d;

    // Detect Q4 vs Q8 from buffer size vs element count
    int64_t bits = n.GetInt("bits", 0);
    if (bits == 0) {
        int64_t totalElements = 1;
        for (auto d : W->shape) totalElements *= d;
        int64_t rawBytes = W->buffer.size;
        bits = (totalElements > 0 && rawBytes > 0 && (double)rawBytes * 8.0 / totalElements < 6) ? 4 : 8;
    }

    int64_t block_size_attr = n.GetInt("block_size", 32);
    uint32_t n_groups, bs, K;
    if (bits == 4) {
        K = (uint32_t)W->shape.back();
        bs = (uint32_t)block_size_attr;
        n_groups = K / bs;
    } else if (W->shape.size() >= 3) {
        n_groups = (uint32_t)W->shape[1];
        bs = (uint32_t)W->shape[2];
        K = n_groups * bs;
    } else if (W->shape.size() == 2) {
        K = (uint32_t)W->shape[1];
        bs = (uint32_t)block_size_attr;
        n_groups = K / bs;
    } else { K = 1; bs = 1; n_groups = 1; }

    auto outShape = Indices->shape;
    outShape.push_back(K);
    *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);

    if (!Scales || !Scales->IsValid()) return;
    ex.EnsureGpu(*Scales);

    // Handle int64 indices → int32 conversion
    GPUBuffer idxBuf = Indices->buffer;
    if (Indices->dtype == TensorDtype::Int64) {
        const uint8_t* idxPtr = nullptr;
        if (Indices->isCpuOnly && !Indices->cpuData.empty())
            idxPtr = Indices->cpuData.data();
        else if (!Indices->cpuData.empty())
            idxPtr = Indices->cpuData.data();
        else if (auto* init = ex.GetInitData(n.inputs[1]); init && init->data)
            idxPtr = init->data;

        if (idxPtr) {
            std::vector<int32_t> i32(nIdx);
            for (int64_t i = 0; i < nIdx; i++) {
                int64_t v; memcpy(&v, idxPtr + i * 8, 8);
                i32[i] = (int32_t)v;
            }
            idxBuf = ex.gpu->createBuffer("gbq_idx32", nIdx * 4);
            ex.gpu->writeBuffer(idxBuf, i32.data(), nIdx * 4);
        }
    }

    uint32_t params[4] = {(uint32_t)nIdx, K, n_groups, bs};
    auto paramBuf = ex.gpu->createBuffer("gbq_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    const char* kernelSrc = (bits == 4) ? WGSL_GATHER_BQ_Q4 : WGSL_GATHER_BQ_Q8;
    const char* plName = (bits == 4) ? "gather_bq_q4" : "gather_bq_q8";
    auto& pl = ex.GetPipeline(plName, kernelSrc, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, W->buffer}, {1, Scales->buffer}, {2, idxBuf},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (K + 255) / 256, (uint32_t)nIdx, 1, "gather_bq"});
}

REGISTER_OP(MatMul, opMatMul)
REGISTER_OP(MatMulNBits, opMatMulNBits)
REGISTER_OP(Gemm, opGemm)
REGISTER_OP(GatherBlockQuantized, opGatherBlockQuantized)

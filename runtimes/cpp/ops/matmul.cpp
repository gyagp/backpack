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
#include <vector>

static float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) f = (sign << 31) | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

static bool ensureTensorFloat32(GraphExecutor& ex, GpuTensor& tensor, const std::string& name) {
    if (tensor.dtype == TensorDtype::Float32) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }
    if (tensor.dtype != TensorDtype::Float16) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }

    size_t count = (size_t)tensor.ElementCount();
    size_t bytes = count * sizeof(uint16_t);
    const uint8_t* src = nullptr;
    std::vector<uint8_t> raw;
    if (!tensor.cpuData.empty() && tensor.cpuData.size() >= bytes) {
        src = tensor.cpuData.data();
    } else if (auto* init = ex.GetInitData(name); init && init->data && init->size >= bytes) {
        src = init->data;
    } else if (tensor.buffer.handle && tensor.buffer.size >= bytes) {
        ex.FlushPendingWork();
        raw = ex.gpu->readBuffer(tensor.buffer, bytes);
        if (raw.size() >= bytes) src = raw.data();
    }
    if (!src) return false;

    std::vector<float> values(count, 0.0f);
    auto* srcFp16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < count; i++) values[i] = fp16ToFloat(srcFp16[i]);

    GpuTensor rebuilt;
    rebuilt.shape = tensor.shape;
    rebuilt.dtype = TensorDtype::Float32;
    rebuilt.cpuData.resize(values.size() * sizeof(float));
    memcpy(rebuilt.cpuData.data(), values.data(), rebuilt.cpuData.size());
    rebuilt.buffer = ex.gpu->createBuffer(name.empty() ? "mmnb_f32_cast" : name,
                                          rebuilt.cpuData.size());
    ex.gpu->writeBuffer(rebuilt.buffer, rebuilt.cpuData.data(), rebuilt.cpuData.size());
    rebuilt.isCpuOnly = false;
    tensor = std::move(rebuilt);
    return tensor.buffer.handle != nullptr;
}

static bool readTensorFloats(GraphExecutor& ex, const GpuTensor& tensor,
                             const std::string& name, std::vector<float>& out) {
    out.clear();
    size_t count = (size_t)tensor.ElementCount();
    out.resize(count);
    if (tensor.dtype == TensorDtype::Float32) {
        size_t bytes = count * sizeof(float);
        if (!tensor.cpuData.empty() && tensor.cpuData.size() >= bytes) {
            memcpy(out.data(), tensor.cpuData.data(), bytes);
            return true;
        }
        if (auto* init = ex.GetInitData(name); init && init->data && init->size >= bytes) {
            memcpy(out.data(), init->data, bytes);
            return true;
        }
        if (tensor.buffer.handle && tensor.buffer.size >= bytes) {
            ex.FlushPendingWork();
            auto raw = ex.gpu->readBuffer(tensor.buffer, bytes);
            if (raw.size() >= bytes) {
                memcpy(out.data(), raw.data(), bytes);
                return true;
            }
        }
        out.clear();
        return false;
    }
    if (tensor.dtype == TensorDtype::Float16) {
        size_t bytes = count * sizeof(uint16_t);
        const uint8_t* src = nullptr;
        std::vector<uint8_t> raw;
        if (!tensor.cpuData.empty() && tensor.cpuData.size() >= bytes) src = tensor.cpuData.data();
        else if (auto* init = ex.GetInitData(name); init && init->data && init->size >= bytes) src = init->data;
        else if (tensor.buffer.handle && tensor.buffer.size >= bytes) {
            ex.FlushPendingWork();
            raw = ex.gpu->readBuffer(tensor.buffer, bytes);
            if (raw.size() >= bytes) src = raw.data();
        }
        if (!src) {
            out.clear();
            return false;
        }
        auto* srcFp16 = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < count; i++) out[i] = fp16ToFloat(srcFp16[i]);
        return true;
    }
    out.clear();
    return false;
}

static std::vector<int64_t> computeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

static bool runBatchedMatMulCpu(GraphExecutor& ex, const OnnxGraphNode& n,
                                GpuTensor& A, GpuTensor& B, GpuTensor& outTensor) {
    std::vector<float> aData, bData;
    if (!readTensorFloats(ex, A, n.inputs.empty() ? std::string() : n.inputs[0], aData) ||
        !readTensorFloats(ex, B, n.inputs.size() > 1 ? n.inputs[1] : std::string(), bData)) {
        return false;
    }

    std::vector<int64_t> aShape = A.shape;
    std::vector<int64_t> bShape = B.shape;
    if (aShape.size() == 1) aShape.insert(aShape.begin(), 1);
    if (bShape.size() == 1) bShape.push_back(1);

    int64_t M = aShape[aShape.size() - 2];
    int64_t K = aShape[aShape.size() - 1];
    int64_t Kb = bShape[bShape.size() - 2];
    int64_t N = bShape[bShape.size() - 1];
    if (K != Kb) return false;

    std::vector<int64_t> aBatch(aShape.begin(), aShape.end() - 2);
    std::vector<int64_t> bBatch(bShape.begin(), bShape.end() - 2);
    size_t batchRank = std::max(aBatch.size(), bBatch.size());
    std::vector<int64_t> aBatchPadded(batchRank, 1), bBatchPadded(batchRank, 1), outBatch(batchRank, 1);
    for (size_t i = 0; i < aBatch.size(); i++) aBatchPadded[batchRank - aBatch.size() + i] = aBatch[i];
    for (size_t i = 0; i < bBatch.size(); i++) bBatchPadded[batchRank - bBatch.size() + i] = bBatch[i];
    for (size_t i = 0; i < batchRank; i++) {
        if (aBatchPadded[i] != bBatchPadded[i] && aBatchPadded[i] != 1 && bBatchPadded[i] != 1) return false;
        outBatch[i] = std::max(aBatchPadded[i], bBatchPadded[i]);
    }

    std::vector<int64_t> outShape = outBatch;
    outShape.push_back(M);
    outShape.push_back(N);
    int64_t batchCount = 1;
    for (auto d : outBatch) batchCount *= d;
    std::vector<float> outData((size_t)std::max<int64_t>(1, batchCount * M * N), 0.0f);

    auto aStrides = computeStrides(aShape);
    auto bStrides = computeStrides(bShape);
    std::vector<int64_t> outBatchStrides = computeStrides(outBatch.empty() ? std::vector<int64_t>{1} : outBatch);
    std::vector<int64_t> coords(batchRank, 0);

    for (int64_t batchIndex = 0; batchIndex < std::max<int64_t>(1, batchCount); batchIndex++) {
        int64_t rem = batchIndex;
        for (size_t i = 0; i < batchRank; i++) {
            int64_t stride = outBatch.empty() ? 1 : outBatchStrides[i];
            coords[i] = outBatch.empty() ? 0 : rem / stride;
            if (!outBatch.empty()) rem %= stride;
        }

        int64_t aBase = 0;
        for (size_t i = 0; i < batchRank; i++) {
            int64_t coord = (aBatchPadded[i] == 1) ? 0 : coords[i];
            size_t ai = i + (aShape.size() - 2 - batchRank);
            if (ai < aStrides.size()) aBase += coord * aStrides[ai];
        }
        int64_t bBase = 0;
        for (size_t i = 0; i < batchRank; i++) {
            int64_t coord = (bBatchPadded[i] == 1) ? 0 : coords[i];
            size_t bi = i + (bShape.size() - 2 - batchRank);
            if (bi < bStrides.size()) bBase += coord * bStrides[bi];
        }

        size_t outBase = (size_t)batchIndex * (size_t)(M * N);
        for (int64_t m = 0; m < M; m++) {
            for (int64_t nCol = 0; nCol < N; nCol++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    float av = aData[(size_t)(aBase + m * aStrides[aShape.size() - 2] + k * aStrides[aShape.size() - 1])];
                    float bv = bData[(size_t)(bBase + k * bStrides[bShape.size() - 2] + nCol * bStrides[bShape.size() - 1])];
                    sum += av * bv;
                }
                outData[outBase + (size_t)m * (size_t)N + (size_t)nCol] = sum;
            }
        }
    }

    outTensor = ex.AllocTensor(outShape, TensorDtype::Float32);
    ex.gpu->writeBuffer(outTensor.buffer, outData.data(), outData.size() * sizeof(float));
    return true;
}

static const WGPULimits& effectiveLimits(const GPUContext& gpu) {
    return gpu.deviceLimits.maxComputeInvocationsPerWorkgroup != 0
        ? gpu.deviceLimits
        : gpu.adapterLimits;
}

// ─── MatMulNBits ─────────────────────────────────────────────────────────────

static void opMatMulNBits(GraphExecutor& ex, const OnnxGraphNode& n,
                           const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0]; auto* W = in[1]; auto* S = in[2];
    if (!X || !W || !S || !X->IsValid() || !W->IsValid() || !S->IsValid()) return;
    if (!ensureTensorFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0])) return;
    ex.EnsureGpu(*W);

    // The kernel reads scales as packed fp16 pairs (u32 containing two f16).
    // If the ONNX model stores scales as float32, convert them to float16.
    if (S->dtype == TensorDtype::Float32) {
        size_t count = (size_t)S->ElementCount();
        std::vector<float> f32Scales(count);
        const uint8_t* src = nullptr;
        std::vector<uint8_t> gpuReadback;
        if (!S->cpuData.empty() && S->cpuData.size() >= count * sizeof(float)) {
            src = S->cpuData.data();
        } else if (auto* init = ex.GetInitData(n.inputs.size() > 2 ? n.inputs[2] : std::string());
                   init && init->data && init->size >= count * sizeof(float)) {
            src = init->data;
        } else if (S->buffer.handle) {
            ex.FlushPendingWork();
            gpuReadback = ex.gpu->readBuffer(S->buffer, count * sizeof(float));
            src = gpuReadback.data();
        }
        if (src) {
            memcpy(f32Scales.data(), src, count * sizeof(float));
            std::vector<uint16_t> fp16Scales(count);
            for (size_t i = 0; i < count; i++) {
                // Convert f32 to f16 (simple truncation via bit manipulation)
                uint32_t bits;
                memcpy(&bits, &f32Scales[i], 4);
                uint32_t sign = (bits >> 31) & 1;
                int32_t exp = ((bits >> 23) & 0xFF) - 127;
                uint32_t mant = bits & 0x7FFFFF;
                uint16_t h;
                if (exp > 15) h = (uint16_t)((sign << 15) | 0x7C00);
                else if (exp < -14) h = (uint16_t)(sign << 15);
                else h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
                fp16Scales[i] = h;
            }
            GpuTensor rebuilt;
            rebuilt.shape = S->shape;
            rebuilt.dtype = TensorDtype::Float16;
            size_t bytes = count * sizeof(uint16_t);
            size_t bufSize = (bytes + 3) & ~(size_t)3;
            rebuilt.buffer = ex.gpu->createBuffer("mmnb_scales_f16", bufSize);
            ex.gpu->writeBuffer(rebuilt.buffer, fp16Scales.data(), bytes);
            rebuilt.isCpuOnly = false;
            *S = std::move(rebuilt);
        }
    }
    ex.EnsureGpu(*S);

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

    if (A->shape.size() > 2 || B->shape.size() > 2) {
        if (!ensureTensorFloat32(ex, *A, n.inputs.empty() ? std::string() : n.inputs[0]) ||
            !ensureTensorFloat32(ex, *B, n.inputs.size() > 1 ? n.inputs[1] : std::string())) {
            return;
        }
        GpuTensor batchedOut;
        if (runBatchedMatMulCpu(ex, n, *A, *B, batchedOut)) {
            *out[0] = std::move(batchedOut);
            return;
        }
    }

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

    if (A->dtype == TensorDtype::Float32 && B->dtype == TensorDtype::Float16 && ex.gpu->supportsShaderF16) {
        auto& pl = ex.GetPipeline("matmul_f16", WGSL_MATMUL_F16, 4);
        auto bg = ex.MakeBindGroup(pl, {
            {0, A->buffer}, {1, B->buffer}, {2, out[0]->buffer}, {3, paramBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)((N_out + 15) / 16), (uint32_t)((M + 15) / 16), 1, "matmul_f16"});
        return;
    }

    // Ensure both inputs are f32 for the basic kernel
    if (A->dtype == TensorDtype::Float16) {
        ensureTensorFloat32(ex, *A, n.inputs.empty() ? std::string() : n.inputs[0]);
    }
    if (B->dtype == TensorDtype::Float16) {
        ensureTensorFloat32(ex, *B, n.inputs.size() > 1 ? n.inputs[1] : std::string());
    }

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

    int64_t transB = n.GetInt("transB", 0);

    // Ensure inputs are on GPU (needed for all paths)
    ex.EnsureGpu(*A); ex.EnsureGpu(*B);

    // Try packed fp16 path: A=f32, B=f16, transB=1, K%4==0
    const auto& limits = effectiveLimits(*ex.gpu);
    int64_t K = A->shape.back();
    bool canUsePackedFp16 =
        A->dtype == TensorDtype::Float32 &&
        B->dtype == TensorDtype::Float16 &&
        transB == 1 &&
        ex.gpu->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 256u &&
        K > 0 && (K % 4) == 0;

    // If A is fp16 but B is fp16 with transB=1, convert A to f32 to enable packed fp16 path
    if (!canUsePackedFp16 && A->dtype == TensorDtype::Float16 &&
        B->dtype == TensorDtype::Float16 && transB == 1 &&
        ex.gpu->supportsSubgroups && K > 0 && (K % 4) == 0) {
        ensureTensorFloat32(ex, *A, n.inputs.empty() ? std::string() : n.inputs[0]);
        canUsePackedFp16 = (A->dtype == TensorDtype::Float32);
    }

    // If still not using packed fp16, ensure both inputs are f32 for the basic GEMM kernel
    if (!canUsePackedFp16) {
        if (A->dtype == TensorDtype::Float16)
            ensureTensorFloat32(ex, *A, n.inputs.empty() ? std::string() : n.inputs[0]);
        if (B->dtype == TensorDtype::Float16)
            ensureTensorFloat32(ex, *B, n.inputs.size() > 1 ? n.inputs[1] : std::string());
    }

    int64_t M = A->shape.size() >= 2 ? A->shape[0] : 1;
    int64_t N_out = transB ? B->shape[0] : B->shape.back();

    *out[0] = ex.AllocTensor({M, N_out}, TensorDtype::Float32);

    GPUBuffer biasBuf;
    if (in.size() > 2 && in[2] && in[2]->IsValid()) {
        if (in[2]->dtype == TensorDtype::Float16)
            ensureTensorFloat32(ex, *in[2], n.inputs.size() > 2 ? n.inputs[2] : std::string());
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

    if (canUsePackedFp16) {
        bool useWide = N_out >= 32 && limits.maxComputeInvocationsPerWorkgroup >= 256u;
        const char* kernelName = useWide ? "fp16_gemm_wide" : "fp16_gemm";
        const char* kernelSrc = useWide ? WGSL_FP16_GEMM_WIDE : WGSL_FP16_GEMM;
        uint32_t tileN = useWide ? 32u : 8u;
        uint32_t fp16Params[4] = {(uint32_t)K, (uint32_t)N_out, 0u, 0u};
        auto fp16ParamBuf = ex.gpu->createBuffer("gemm_fp16_p", 16);
        ex.gpu->writeBuffer(fp16ParamBuf, fp16Params, 16);

        auto& pl = ex.GetPipeline(kernelName, kernelSrc, 5);
        auto bg = ex.MakeBindGroup(pl, {
            {0, A->buffer}, {1, B->buffer}, {2, biasBuf},
            {3, out[0]->buffer}, {4, fp16ParamBuf}});
        ex.pendingDispatches_.push_back({pl.pipeline, bg,
            (uint32_t)M, (uint32_t)((N_out + tileN - 1) / tileN), 1,
            useWide ? "gemm_fp16_wide" : "gemm_fp16"});
        return;
    }

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

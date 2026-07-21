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
#include "../wgsl_template.h"
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

static bool ensureTensorFloat32(OpContext& ex, GpuTensor& tensor, const std::string& name) {
    if (tensor.dtype == TensorDtype::Float32) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }
    if (tensor.dtype != TensorDtype::Float16) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }

    // GPU fp16→f32 cast — no CPU readback needed
    ex.EnsureGpu(tensor);
    size_t count = (size_t)tensor.ElementCount();
    if (count <= 0 || !tensor.buffer.handle) return false;

    GpuTensor f32t;
    f32t.shape = tensor.shape;
    f32t.dtype = TensorDtype::Float32;
    f32t.buffer = ex.getGpu()->createBuffer(name.empty() ? "mmnb_f32_cast" : name, count * 4);
    f32t.isCpuOnly = false;

    uint32_t p[4] = {(uint32_t)count, 0, 0, 0};
    auto pb = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(pb, p, 16);
    auto& pl = ex.GetPipelineT("cast_f16_to_f32", 3, []() { return std::string(WGSL_CAST_F16_TO_F32); });
    auto bg = ex.MakeBindGroup(pl, {{0, tensor.buffer}, {1, f32t.buffer}, {2, pb}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((count + 255) / 256), 1, 1, "mmnb_cast");

    tensor = std::move(f32t);
    return tensor.buffer.handle != nullptr;
}

static bool readTensorFloats(OpContext& ex, const GpuTensor& tensor,
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
            auto raw = ex.getGpu()->readBuffer(tensor.buffer, bytes);
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
            raw = ex.getGpu()->readBuffer(tensor.buffer, bytes);
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

static bool runBatchedMatMulCpu(OpContext& ex, const OnnxGraphNode& n,
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
    ex.getGpu()->writeBuffer(outTensor.buffer, outData.data(), outData.size() * sizeof(float));
    return true;
}

static const WGPULimits& effectiveLimits(const GPUContext& gpu) {
    return gpu.deviceLimits.maxComputeInvocationsPerWorkgroup != 0
        ? gpu.deviceLimits
        : gpu.adapterLimits;
}

// ─── MatMulNBits ─────────────────────────────────────────────────────────────

static const char* kMatMulQ8Block32 = R"WGSL(
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
var<workgroup> sums: array<f32, 1024>;
fn byte_at(word: u32, index: u32) -> u32 {
    return (word >> ((index & 3u) * 8u)) & 255u;
}
fn scale_at(index: u32) -> f32 {
    let pair = unpack2x16float(scales[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) != 0u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let lane = lid3.x;
    let m = wid.y;
    let n0 = wid.x * 4u;
    let groups = p.K / 32u;
    var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;
    for (var k = lane; k < p.K; k += 256u) {
        let x = X[m * p.K + k];
        let byte_in_word = k & 3u;
        let word_k = k >> 2u;
        if (n0 < p.N) {
            let q = f32(i32(byte_at(W[n0 * (p.K / 4u) + word_k], byte_in_word)) - 128);
            acc0 += x * q * scale_at(n0 * groups + k / 32u);
        }
        if (n0 + 1u < p.N) {
            let n = n0 + 1u;
            let q = f32(i32(byte_at(W[n * (p.K / 4u) + word_k], byte_in_word)) - 128);
            acc1 += x * q * scale_at(n * groups + k / 32u);
        }
        if (n0 + 2u < p.N) {
            let n = n0 + 2u;
            let q = f32(i32(byte_at(W[n * (p.K / 4u) + word_k], byte_in_word)) - 128);
            acc2 += x * q * scale_at(n * groups + k / 32u);
        }
        if (n0 + 3u < p.N) {
            let n = n0 + 3u;
            let q = f32(i32(byte_at(W[n * (p.K / 4u) + word_k], byte_in_word)) - 128);
            acc3 += x * q * scale_at(n * groups + k / 32u);
        }
    }
    sums[lane] = acc0; sums[256u + lane] = acc1;
    sums[512u + lane] = acc2; sums[768u + lane] = acc3;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            sums[256u + lane] += sums[256u + lane + stride];
            sums[512u + lane] += sums[512u + lane + stride];
            sums[768u + lane] += sums[768u + lane + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        if (n0 < p.N) { Y[m * p.N + n0] = sums[0]; }
        if (n0 + 1u < p.N) { Y[m * p.N + n0 + 1u] = sums[256]; }
        if (n0 + 2u < p.N) { Y[m * p.N + n0 + 2u] = sums[512]; }
        if (n0 + 3u < p.N) { Y[m * p.N + n0 + 3u] = sums[768]; }
    }
}
)WGSL";

static void opMatMulNBits(OpContext& ex, const OnnxGraphNode& n,
                           const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0]; auto* W = in[1]; auto* S = in[2];
    if (!X || !W || !S || !X->IsValid() || !W->IsValid() || !S->IsValid()) return;
    // Keep input dtype (fp16 or f32) — the templated kernel handles both
    ex.EnsureGpu(*X);
    ex.EnsureGpu(*W);

    // The kernel reads scales as packed fp16 pairs (u32 containing two f16).
    // If the ONNX model stores scales as float32, convert them to float16.
    if (S->dtype == TensorDtype::Float32) {
        size_t count = (size_t)S->ElementCount();

        // Try CPU-side sources first (no GPU sync needed)
        const uint8_t* src = nullptr;
        if (!S->cpuData.empty() && S->cpuData.size() >= count * sizeof(float)) {
            src = S->cpuData.data();
        } else if (auto* init = ex.GetInitData(n.inputs.size() > 2 ? n.inputs[2] : std::string());
                   init && init->data && init->size >= count * sizeof(float)) {
            src = init->data;
        }

        if (src) {
            // CPU path: convert f32→f16 and upload
            std::vector<float> f32Scales(count);
            memcpy(f32Scales.data(), src, count * sizeof(float));
            std::vector<uint16_t> fp16Scales(count);
            for (size_t i = 0; i < count; i++) {
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
            rebuilt.buffer = ex.getGpu()->createBuffer("mmnb_scales_f16", bufSize);
            ex.getGpu()->writeBuffer(rebuilt.buffer, fp16Scales.data(), bytes);
            rebuilt.isCpuOnly = false;
            *S = std::move(rebuilt);
        } else if (S->buffer.handle) {
            // GPU path: cast f32→f16 on GPU (avoids expensive FlushPendingWork + readBuffer)
            GpuTensor f16t = ex.AllocTensor(S->shape, TensorDtype::Float16);
            uint32_t params[4] = {(uint32_t)count, 0, 0, 0};
            auto pb = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(pb, params, 16);
            auto& cpl = ex.GetPipelineT("cast_f32_to_f16", 3,
                []() { return std::string(WGSL_CAST_F32_TO_F16); });
            auto cbg = ex.MakeBindGroup(cpl, {{0, S->buffer}, {1, f16t.buffer}, {2, pb}});
            ex.QueueDispatch(cpl.pipeline, cbg,
                (uint32_t)((count + 255) / 256), 1, 1, "mmnb_scales_cast");
            *S = std::move(f16t);
        }
    }
    ex.EnsureGpu(*S);


    // Check for zero_point input (4th input)
    auto* ZP = (in.size() > 3 && in[3] && in[3]->IsValid()) ? in[3] : nullptr;
    if (ZP) ex.EnsureGpu(*ZP);

    uint32_t N = (uint32_t)n.GetInt("N");
    uint32_t K = (uint32_t)n.GetInt("K");
    int64_t M = 1;
    for (size_t i = 0; i + 1 < X->shape.size(); i++) M *= X->shape[i];

    // Output dtype matches input dtype (f32 or fp16)
    TensorDtype outDtype = X->dtype;
    if (outDtype != TensorDtype::Float16) outDtype = TensorDtype::Float32;

    auto outShape = X->shape;
    outShape.back() = N;
    *out[0] = ex.AllocTensor(outShape, outDtype);

    uint32_t params[4] = {(uint32_t)M, N, K, 0};
    auto paramBuf = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(paramBuf, params, 16);

    const int64_t bits = n.GetInt("bits", 4);
    if (bits == 8) {
        // ORT blockwise Q8 stores unsigned bytes with an implicit zero point
        // of 128 and one fp16 scale per 32 values.
        if (X->dtype == TensorDtype::Float16) {
            GpuTensor f32 = ex.AllocTensor(X->shape, TensorDtype::Float32);
            uint32_t castParams[4] = {(uint32_t)X->ElementCount(), 0, 0, 0};
            auto castParam = ex.getParamBuffer(16);
            ex.getGpu()->writeBuffer(castParam, castParams, 16);
            auto& cast = ex.GetPipelineT("cast_f16_to_f32", 3,
                []() { return std::string(WGSL_CAST_F16_TO_F32); });
            auto castGroup = ex.MakeBindGroup(cast,
                {{0, X->buffer}, {1, f32.buffer}, {2, castParam}});
            ex.QueueDispatch(cast.pipeline, castGroup,
                (castParams[0] + 255) / 256, 1, 1, "q8_input_cast");
            *X = std::move(f32);
            outDtype = TensorDtype::Float32;
            *out[0] = ex.AllocTensor(outShape, outDtype);
        }
        const bool useSubgroupDecode = M == 1 && (K % 32u) == 0u &&
            ex.getGpu()->backendType == WGPUBackendType_D3D12 &&
            ex.getGpu()->supportsSubgroups;
        auto& pipeline = useSubgroupDecode
            ? ex.GetPipelineT("matmul_q8_block32_subgroup", 5,
                []() { return std::string(WGSL_MATMUL_Q8_BLOCK32_SUBGROUP); })
            : ex.GetPipeline("matmul_q8_block32", kMatMulQ8Block32, 5);
        auto group = ex.MakeBindGroup(pipeline, {
            {0, X->buffer}, {1, W->buffer}, {2, S->buffer},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.QueueDispatch(pipeline.pipeline, group,
                         useSubgroupDecode ? (N + 7) / 8 : (N + 3) / 4,
                         static_cast<uint32_t>(M), 1,
                         useSubgroupDecode ? "matmul_q8_block32_subgroup" : "matmul_q8_block32");
        return;
    }

    if (ZP) {
        // Decode: K-parallel subgroup kernel (8 warps × 32 lanes, TILE_N=8).
        // Prefill: wide-tile kernel with shared-memory A reuse (TILE_M=8, TILE_N=128).
        if (M == 1) {
            std::string pname = "matmul_q4_zp_sub_t" + std::string(dtypeSuffix(outDtype));
            auto& pl = ex.GetPipelineT(pname, 6, [outDtype]() {
                return instantiateTemplate(WGSL_MATMUL_Q4_ZP_SUB_T, outDtype);
            });
            auto bg = ex.MakeBindGroup(pl, {
                {0, X->buffer}, {1, W->buffer}, {2, S->buffer},
                {3, out[0]->buffer}, {4, paramBuf}, {5, ZP->buffer}});
            ex.QueueDispatch(pl.pipeline, bg,
                (N + 7) / 8, 1, 1, "matmul_q4_zp");
        } else {
            std::string pname = "matmul_q4_zp_wide_t" + std::string(dtypeSuffix(outDtype));
            auto& pl = ex.GetPipelineT(pname, 6, [outDtype]() {
                return instantiateTemplate(WGSL_MATMUL_Q4_ZP_WIDE_T, outDtype);
            });
            auto bg = ex.MakeBindGroup(pl, {
                {0, X->buffer}, {1, W->buffer}, {2, S->buffer},
                {3, out[0]->buffer}, {4, paramBuf}, {5, ZP->buffer}});
            ex.QueueDispatch(pl.pipeline, bg,
                (N + 127) / 128, ((uint32_t)M + 7) / 8, 1, "matmul_q4_zp_wide");
        }
    } else {
        // The portable no-zero-point Q4 shader is an f32 kernel. Some Qwen
        // projections feed it fp16 tensors; interpreting those bytes as f32
        // also made the shader write past an fp16-sized output allocation.
        if (X->dtype == TensorDtype::Float16) {
            if (!ensureTensorFloat32(ex, *X,
                    n.inputs.empty() ? std::string() : n.inputs[0])) return;
            outDtype = TensorDtype::Float32;
            *out[0] = ex.AllocTensor(outShape, outDtype);
        }
        // Decode dominates Qwen runtime, and the scalar kernel below leaves
        // one invocation to dequantize and reduce an entire output row.  The
        // packed path mirrors ORT/llama.cpp: quantize each 32-value activation
        // block once, then use dot4I8Packed across 32 output columns per
        // workgroup.  Keep the scalar shader as the portable fallback.
        const bool usePackedDecode = M == 1 && (K % 256u) == 0u &&
            ex.getGpu()->backendType == WGPUBackendType_D3D12 &&
            ex.getGpu()->supportsSubgroups;
        const bool useTwoColumnDecode = usePackedDecode &&
            (K == 2048u || K == 6144u);
        const bool useOneColumnDecode = useTwoColumnDecode &&
            ex.getGpu()->adapterName.find("Intel") == std::string::npos;
        auto& pl = useTwoColumnDecode
            ? ex.GetPipelineT(useOneColumnDecode ? "matmul_q4_decode_1col" :
                                               "matmul_q4_decode_2col", 5,
              [useOneColumnDecode]() {
                std::string source(WGSL_MATMUL_Q4_DECODE);
                const std::string from = "COLS_PER_WARP: u32 = 4u";
                if (auto pos = source.find(from); pos != std::string::npos)
                    source.replace(pos, from.size(), useOneColumnDecode ?
                        "COLS_PER_WARP: u32 = 1u" : "COLS_PER_WARP: u32 = 2u");
                return source;
            })
            : usePackedDecode
                ? ex.GetPipelineT("matmul_q4_decode", 5,
                    []() { return std::string(WGSL_MATMUL_Q4_DECODE); })
            : ex.GetPipelineT("matmul_q4", 5,
                []() { return std::string(WGSL_MATMUL_Q4); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, X->buffer}, {1, W->buffer}, {2, S->buffer},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            useOneColumnDecode ? (N + 7) / 8
                : useTwoColumnDecode ? (N + 15) / 16
                : usePackedDecode ? (N + 31) / 32 : (N + 255) / 256,
            (uint32_t)M, 1, useOneColumnDecode ? "matmul_q4_decode_1col"
                : useTwoColumnDecode ? "matmul_q4_decode_2col"
                : usePackedDecode ? "matmul_q4_decode" : "matmul_q4");
    }
}

// ─── MatMul ──────────────────────────────────────────────────────────────────

static void opMatMul(OpContext& ex, const OnnxGraphNode& n,
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
    auto paramBuf = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(paramBuf, params, 16);

    if (A->dtype == TensorDtype::Float32 && B->dtype == TensorDtype::Float16 && ex.getGpu()->supportsShaderF16) {
        auto& pl = ex.GetPipelineT("matmul_f16", 4, []() { return std::string(WGSL_MATMUL_F16); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, A->buffer}, {1, B->buffer}, {2, out[0]->buffer}, {3, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)((N_out + 15) / 16), (uint32_t)((M + 15) / 16), 1, "matmul_f16");
        return;
    }

    // Packed fp16 path: reads fp16 weights as array<u32> with unpack2x16float.
    // Works on D3D12 without shader f16 support. N must be even for correct u32 packing.
    if (A->dtype == TensorDtype::Float32 && B->dtype == TensorDtype::Float16 && (N_out % 2) == 0) {
        auto& pl = ex.GetPipelineT("matmul_fp16_packed_nt", 4, []() { return std::string(WGSL_MATMUL_FP16_PACKED_NT); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, A->buffer}, {1, B->buffer}, {2, out[0]->buffer}, {3, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)((N_out + 15) / 16), (uint32_t)((M + 15) / 16), 1, "matmul_fp16_packed_nt");
        return;
    }

    // Ensure both inputs are f32 for the basic kernel
    if (A->dtype == TensorDtype::Float16) {
        ensureTensorFloat32(ex, *A, n.inputs.empty() ? std::string() : n.inputs[0]);
    }
    if (B->dtype == TensorDtype::Float16) {
        ensureTensorFloat32(ex, *B, n.inputs.size() > 1 ? n.inputs[1] : std::string());
    }

    auto& pl = ex.GetPipelineT("matmul_f32", 4, []() { return instantiateTemplate(WGSL_MATMUL_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, out[0]->buffer}, {3, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((N_out + 31) / 32), (uint32_t)((M + 15) / 16), 1, "matmul_f32");
}

// ─── Gemm ────────────────────────────────────────────────────────────────────

static void opGemm(OpContext& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;

    int64_t transB = n.GetInt("transB", 0);

    // Ensure inputs are on GPU (needed for all paths)
    ex.EnsureGpu(*A); ex.EnsureGpu(*B);

    // Try packed fp16 path: A=f32, B=f16, transB=1, K%4==0
    const auto& limits = effectiveLimits(*ex.getGpu());
    int64_t K = A->shape.back();
    bool canUsePackedFp16 =
        A->dtype == TensorDtype::Float32 &&
        B->dtype == TensorDtype::Float16 &&
        transB == 1 &&
        ex.getGpu()->supportsSubgroups &&
        limits.maxComputeInvocationsPerWorkgroup >= 256u &&
        K > 0 && (K % 4) == 0;

    // If A is fp16 but B is fp16 with transB=1, convert A to f32 to enable packed fp16 path
    if (!canUsePackedFp16 && A->dtype == TensorDtype::Float16 &&
        B->dtype == TensorDtype::Float16 && transB == 1 &&
        ex.getGpu()->supportsSubgroups && K > 0 && (K % 4) == 0) {
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
        biasBuf = ex.getGpu()->createBuffer("gemm_b0", N_out * 4);
        ex.getGpu()->writeBuffer(biasBuf, zeros.data(), N_out * 4);
    }

    uint32_t params[4] = {(uint32_t)M, (uint32_t)N_out, (uint32_t)K, (uint32_t)transB};
    auto paramBuf = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(paramBuf, params, 16);

    if (canUsePackedFp16) {
        bool useWide = N_out >= 32 && limits.maxComputeInvocationsPerWorkgroup >= 256u;
        const char* kernelName = useWide ? "fp16_gemm_wide" : "fp16_gemm";
        const char* kernelSrc = useWide ? WGSL_FP16_GEMM_WIDE : WGSL_FP16_GEMM;
        uint32_t tileN = useWide ? 32u : 8u;
        uint32_t fp16Params[4] = {(uint32_t)K, (uint32_t)N_out, 0u, 0u};
        auto fp16ParamBuf = ex.getParamBuffer(16);
        ex.getGpu()->writeBuffer(fp16ParamBuf, fp16Params, 16);

        auto& pl = ex.GetPipelineT(kernelName, 5, [kernelSrc]() { return std::string(kernelSrc); });
        auto bg = ex.MakeBindGroup(pl, {
            {0, A->buffer}, {1, B->buffer}, {2, biasBuf},
            {3, out[0]->buffer}, {4, fp16ParamBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (uint32_t)M, (uint32_t)((N_out + tileN - 1) / tileN), 1,
            useWide ? "gemm_fp16_wide" : "gemm_fp16");
        return;
    }

    auto& pl = ex.GetPipelineT("gemm", 5, []() { return instantiateTemplate(WGSL_GEMM_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});
    ex.QueueDispatch(pl.pipeline, bg,
        (uint32_t)((N_out + 15) / 16), (uint32_t)((M + 15) / 16), 1, "gemm");
}

// ─── GatherBlockQuantized ────────────────────────────────────────────────────

static void opGatherBlockQuantized(OpContext& ex, const OnnxGraphNode& n,
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
        // Q4: each byte has 2 nibbles, so actual K = packed_dim * 2
        K = (uint32_t)W->shape.back() * 2;
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

    // Check for zero_point input (4th input)
    auto* ZP = (in.size() > 3 && in[3] && in[3]->IsValid()) ? in[3] : nullptr;
    if (ZP) ex.EnsureGpu(*ZP);

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
            idxBuf = ex.getGpu()->createBuffer("gbq_idx32", nIdx * 4);
            ex.getGpu()->writeBuffer(idxBuf, i32.data(), nIdx * 4);
            // Only a scalar token-embedding gather follows the generated token.
            // Fixed int64 gathers (for example LM-head pruning tables) must not
            // be overwritten during capture replay.
            const bool tokenEmbeddingGather = nIdx == 1 && !n.inputs.empty() &&
                n.inputs[0].find("embed_tokens") != std::string::npos;
            if (ex.fastDecodeState() == ExecutionContext::FastDecodeState::Capturing &&
                tokenEmbeddingGather) {
                ex.exec.capturedTokenIdBufs_.push_back({idxBuf, nIdx});
            }
        }
    }

    uint32_t params[4] = {(uint32_t)nIdx, K, n_groups, bs};
    auto paramBuf = ex.getParamBuffer(16);
    ex.getGpu()->writeBuffer(paramBuf, params, 16);

    const char* kernelSrc = (bits == 4) ? (ZP ? WGSL_GATHER_BQ_Q4_ZP : WGSL_GATHER_BQ_Q4)
                                        : WGSL_GATHER_BQ_Q8;
    const char* plName = (bits == 4) ? (ZP ? "gather_bq_q4_zp" : "gather_bq_q4")
                                      : "gather_bq_q8";
    int numBindings = (bits == 4 && ZP) ? 6 : 5;
    auto& pl = ex.GetPipelineT(plName, numBindings, [kernelSrc]() { return std::string(kernelSrc); });
    if (ZP) {
        auto bg = ex.MakeBindGroup(pl, {
            {0, W->buffer}, {1, Scales->buffer}, {2, idxBuf},
            {3, out[0]->buffer}, {4, paramBuf}, {5, ZP->buffer}});
        ex.QueueDispatch(pl.pipeline, bg,
            (K + 255) / 256, (uint32_t)nIdx, 1, "gather_bq_zp");
    } else {
        auto bg = ex.MakeBindGroup(pl, {
            {0, W->buffer}, {1, Scales->buffer}, {2, idxBuf},
            {3, out[0]->buffer}, {4, paramBuf}});
        ex.QueueDispatch(pl.pipeline, bg,
            (K + 255) / 256, (uint32_t)nIdx, 1, "gather_bq");
    }
}

REGISTER_OP(MatMul, opMatMul)
REGISTER_OP(MatMulNBits, opMatMulNBits)
REGISTER_OP(Gemm, opGemm)
REGISTER_OP(GatherBlockQuantized, opGatherBlockQuantized)

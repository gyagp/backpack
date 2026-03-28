/**
 * ops/elementwise.cpp — Elementwise ONNX op implementations.
 *
 * Covers: Add, Sub, Mul, Div, Neg, Sqrt, Sigmoid, Tanh, Sin, Cos,
 *         Cast, Equal, GreaterOrEqual, Where, Softmax, ReduceSum.
 *
 * Uses embedded kernels from runtimes/cpp/kernels/shared/.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include "../wgsl_template.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <set>

// Binary and unary elementwise ops now use embedded kernels from
// runtimes/cpp/kernels/shared/binary_elementwise.wgsl and unary_elementwise.wgsl

// ─── Helpers ─────────────────────────────────────────────────────────────────

static int64_t tensorNel(const GpuTensor* t) {
    if (!t) return 0;
    return t->ElementCount();
}

static constexpr bool kDebugSmallIntOps = false;
static constexpr bool kDebugVaeCast = false;

static std::vector<int64_t> preferredIntOutputShape(const GpuTensor* A, const GpuTensor* B,
                                                    int64_t N_A, int64_t N_B) {
    const auto& preferredShape = (N_A >= N_B) ? A->shape : B->shape;
    if (!preferredShape.empty()) return preferredShape;
    return (std::max<int64_t>(N_A, N_B) <= 1) ? std::vector<int64_t>{}
                                               : std::vector<int64_t>{std::max<int64_t>(N_A, N_B)};
}

static GPUBuffer makeParamBuf(GraphExecutor& ex, uint32_t p0, uint32_t p1 = 0,
                               uint32_t p2 = 0, uint32_t p3 = 0) {
    uint32_t data[4] = {p0, p1, p2, p3};
    auto buf = ex.getParamBuffer(16);
    ex.gpu->writeBuffer(buf, data, 16);
    return buf;
}

static GPUBuffer makeScalarParamBuf(GraphExecutor& ex, uint32_t p0) {
    uint32_t data[4] = {p0, 0, 0, 0};
    auto buf = ex.getParamBuffer(16);
    ex.gpu->writeBuffer(buf, data, 16);
    return buf;
}

static bool isSmallIntTensor(const GpuTensor* t) {
    if (!t) return false;
    return (t->dtype == TensorDtype::Int64 || t->dtype == TensorDtype::Int32) &&
           tensorNel(t) <= 64;
}

static bool loadTensorBytes(GraphExecutor& ex, const GpuTensor* t,
                            const std::string& name, std::vector<uint8_t>& bytes) {
    if (!t) return false;
    size_t need = (size_t)t->ElementCount() * t->DtypeSize();
    if (need == 0) {
        bytes.clear();
        return true;
    }
    if (t->cpuData.size() >= need) {
        bytes.assign(t->cpuData.begin(), t->cpuData.begin() + need);
        return true;
    }
    if (auto* init = ex.GetInitData(name); init && init->data && init->size >= need) {
        bytes.resize(need);
        memcpy(bytes.data(), init->data, need);
        return true;
    }
    if (t->buffer.handle) {
        ex.FlushPendingWork();
        auto rb = ex.gpu->readBuffer(t->buffer, need);
        if (rb.size() >= need) {
            bytes.assign(rb.begin(), rb.begin() + need);
            return true;
        }
    }
    return false;
}

static bool readTensorIntValues(GraphExecutor& ex, const GpuTensor* t,
                                const std::string& name, std::vector<int64_t>& values) {
    values.clear();
    if (!t) return false;
    int64_t nel = t->ElementCount();
    if (nel < 0) return false;

    std::vector<uint8_t> raw;
    if (!loadTensorBytes(ex, t, name, raw)) return false;

    values.resize((size_t)nel);
    if (t->dtype == TensorDtype::Int64) {
        if (raw.size() < (size_t)nel * sizeof(int64_t)) return false;
        auto* src = reinterpret_cast<const int64_t*>(raw.data());
        for (int64_t i = 0; i < nel; i++) values[(size_t)i] = src[i];
        return true;
    }
    if (t->dtype == TensorDtype::Int32) {
        if (raw.size() < (size_t)nel * sizeof(int32_t)) return false;
        auto* src = reinterpret_cast<const int32_t*>(raw.data());
        for (int64_t i = 0; i < nel; i++) values[(size_t)i] = src[i];
        return true;
    }
    return false;
}

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

static bool ensureCpuBackedFloat32(GraphExecutor& ex, GpuTensor& tensor, const std::string& name) {
    if (auto* init = ex.GetInitData(name); init && init->data && init->dtype == TensorDtype::Float16 && tensor.cpuData.empty()) {
        size_t count = (size_t)tensor.ElementCount();
        if (count > 0 && init->size >= count * sizeof(uint16_t)) {
            std::vector<float> values(count, 0.0f);
            auto* srcFp16 = reinterpret_cast<const uint16_t*>(init->data);
            for (size_t i = 0; i < count; i++) values[i] = fp16ToFloat(srcFp16[i]);
            // Create a NON-persistent copy to avoid corrupting persistent initializers
            GpuTensor copy;
            copy.shape = tensor.shape;
            copy.dtype = TensorDtype::Float32;
            copy.cpuData.resize(values.size() * sizeof(float));
            memcpy(copy.cpuData.data(), values.data(), copy.cpuData.size());
            copy.buffer = ex.gpu->createBuffer("ecbf32", copy.cpuData.size());
            ex.gpu->writeBuffer(copy.buffer, copy.cpuData.data(), copy.cpuData.size());
            copy.isCpuOnly = false;
            tensor = std::move(copy);
            return true;
        }
    }
    if (tensor.dtype != TensorDtype::Float16) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }

    const uint8_t* src = nullptr;
    size_t bytes = 0;
    std::vector<uint8_t> gpuReadback;
    if (!tensor.cpuData.empty()) {
        src = tensor.cpuData.data();
        bytes = tensor.cpuData.size();
    } else if (auto* init = ex.GetInitData(name); init && init->data) {
        src = init->data;
        bytes = init->size;
    } else if (tensor.buffer.handle) {
        size_t count = (size_t)tensor.ElementCount();
        size_t readBytes = count * sizeof(uint16_t);
        ex.FlushPendingWork();
        gpuReadback = ex.gpu->readBuffer(tensor.buffer, readBytes);
        if (gpuReadback.size() >= readBytes) {
            src = gpuReadback.data();
            bytes = gpuReadback.size();
        }
    }

    if (!src) {
        ex.EnsureGpu(tensor);
        return tensor.buffer.handle != nullptr;
    }

    size_t count = (size_t)tensor.ElementCount();
    if (bytes < count * sizeof(uint16_t)) return false;
    std::vector<float> values(count, 0.0f);
    auto* srcFp16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < count; i++) values[i] = fp16ToFloat(srcFp16[i]);
    tensor = ex.AllocCpuTensor(tensor.shape, TensorDtype::Float32,
                               values.data(), values.size() * sizeof(float));
    ex.EnsureGpu(tensor);
    return tensor.buffer.handle != nullptr;
}

static void debugTensorFloatStats(GraphExecutor& ex, const char* label,
                                  const GpuTensor* tensor, const std::string& name) {
    if (!tensor) {
        fprintf(stderr, "    [dbg] %s: null\n", label);
        fflush(stderr);
        return;
    }
    std::vector<float> values;
    if (tensor->dtype == TensorDtype::Float16) {
        GpuTensor tmp = *tensor;
        if (!ensureCpuBackedFloat32(ex, tmp, name)) return;
        size_t count = (size_t)tmp.ElementCount();
        auto raw = ex.gpu->readBuffer(tmp.buffer, count * sizeof(float));
        if (raw.size() < count * sizeof(float)) return;
        values.resize(count);
        memcpy(values.data(), raw.data(), count * sizeof(float));
    } else if (tensor->dtype == TensorDtype::Float32) {
        size_t count = (size_t)tensor->ElementCount();
        if (count == 0) return;
        ex.FlushPendingWork();
        auto raw = ex.gpu->readBuffer(tensor->buffer, count * sizeof(float));
        if (raw.size() < count * sizeof(float)) return;
        values.resize(count);
        memcpy(values.data(), raw.data(), count * sizeof(float));
    } else {
        return;
    }
    float minV = 1e30f, maxV = -1e30f;
    double sum = 0.0;
    int nanCount = 0;
    for (float v : values) {
        if (!std::isfinite(v)) {
            nanCount++;
            continue;
        }
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        sum += v;
    }
    int valid = (int)values.size() - nanCount;
    fprintf(stderr, "    [dbg] %s: min=%.4f max=%.4f avg=%.4f nan=%d/%zu\n",
            label, minV, maxV, valid > 0 ? sum / valid : 0.0, nanCount, values.size());
    fflush(stderr);
}

// ─── CPU fallback for small int64 ops ────────────────────────────────────────
// Shape metadata ops (Div, Mul, Add on int64 scalars) must run on CPU
// because our GPU kernels only handle f32.

static void cpuBinaryInt64(GraphExecutor& ex, const OnnxGraphNode& node,
                            const std::vector<GpuTensor*>& inputs,
                            std::vector<GpuTensor*>& outputs, int op) {
    auto* A = inputs[0]; auto* B = inputs[1];
    int64_t N_A = tensorNel(A);
    int64_t N_B = tensorNel(B);
    int64_t N = std::max<int64_t>(1, std::max(N_A, N_B));
    std::vector<int64_t> outShape = preferredIntOutputShape(A, B, N_A, N_B);

        if (kDebugSmallIntOps) {
        fprintf(stderr, "    [binint] %s op=%d N_A=%lld N_B=%lld dtypeA=%d dtypeB=%d\n",
            node.opType.c_str(), op, (long long)N_A, (long long)N_B, (int)A->dtype, (int)B->dtype);
        fflush(stderr);
        }

    if (N_A <= 0 || N_B <= 0) {
        std::vector<int64_t> zeros((size_t)N, 0);
        *outputs[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64, zeros.data(), zeros.size() * sizeof(int64_t));
        if (kDebugSmallIntOps) {
            fprintf(stderr, "    [binint] degenerate zero-output N=%lld\n", (long long)N);
            fflush(stderr);
        }
        return;
    }

    std::vector<int64_t> a(N_A), b(N_B), c(N);

    if (!readTensorIntValues(ex, A, node.inputs[0], a)) {
        std::fill(a.begin(), a.end(), 0);
    }
    if (!readTensorIntValues(ex, B, node.inputs[1], b)) {
        std::fill(b.begin(), b.end(), 0);
    }

    for (int64_t i = 0; i < N; i++) {
        int64_t av = a[i % N_A], bv = b[i % N_B];
        switch (op) {
            case 0: c[i] = av + bv; break;
            case 1: c[i] = av - bv; break;
            case 2: c[i] = av * bv; break;
            case 3: c[i] = (bv != 0) ? av / bv : 0; break;
        }
    }

    *outputs[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64, c.data(), N * 8);
    if (kDebugSmallIntOps) {
        fprintf(stderr, "    [binint] done N=%lld\n", (long long)N);
        fflush(stderr);
    }
}

// ─── Binary elementwise dispatch (dtype-aware) ───────────────────────────────

// ─── Binary elementwise dispatch (dtype-aware) ───────────────────────────────

static void dispatchBinaryOp(GraphExecutor& ex, const OnnxGraphNode& node,
                              const std::vector<GpuTensor*>& inputs,
                              std::vector<GpuTensor*>& outputs, uint32_t opCode) {
    auto* A = inputs[0];
    auto* B = inputs[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;

    // CPU fallback for int64 ops (shape metadata)
    if (isSmallIntTensor(A) || isSmallIntTensor(B)) {
        cpuBinaryInt64(ex, node, inputs, outputs, (int)opCode);
        return;
    }

    // Determine dtype: use fp16 if both inputs are fp16, else f32
    TensorDtype dtype = TensorDtype::Float32;
    if (A->dtype == TensorDtype::Float16 && B->dtype == TensorDtype::Float16) {
        dtype = TensorDtype::Float16;
    } else {
        // Cast fp16 → f32 if mixed
        if (!ensureCpuBackedFloat32(ex, *A, node.inputs.empty() ? std::string() : node.inputs[0]) ||
            !ensureCpuBackedFloat32(ex, *B, node.inputs.size() > 1 ? node.inputs[1] : std::string())) {
            return;
        }
    }

    ex.EnsureGpu(*A);
    ex.EnsureGpu(*B);

    int64_t N_A = tensorNel(A);
    int64_t N_B = tensorNel(B);
    int64_t N = std::max(N_A, N_B);
    auto& outShape = (N_A >= N_B) ? A->shape : B->shape;
    *outputs[0] = ex.AllocTensor(outShape, dtype);

    auto params = makeParamBuf(ex, (uint32_t)N, opCode, (uint32_t)N_A, (uint32_t)N_B);
    std::string pipelineName = "binary_t" + std::string(dtypeSuffix(dtype));
    auto& pl = ex.GetPipelineT(pipelineName, 4, [dtype]() {
        return instantiateTemplate(WGSL_BINARY_ELEMENTWISE_T, dtype);
    });
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, outputs[0]->buffer}, {3, params}});
    // Each thread handles 2 elements
    uint32_t numWorkgroups = (uint32_t)(((N + 1) / 2 + 255) / 256);
    ex.SubmitAsync({{pl.pipeline, bg, numWorkgroups, 1, 1, node.opType}});
}

// ─── Unary elementwise dispatch ──────────────────────────────────────────────

static void dispatchUnaryOp(GraphExecutor& ex, const OnnxGraphNode& node,
                             const std::vector<GpuTensor*>& inputs,
                             std::vector<GpuTensor*>& outputs, uint32_t opCode) {
    auto* A = inputs[0];
    if (!A || !A->IsValid()) return;

    // Determine dtype: use fp16 if input is fp16, else f32
    TensorDtype dtype = A->dtype;
    if (dtype != TensorDtype::Float16) {
        dtype = TensorDtype::Float32;
        if (!ensureCpuBackedFloat32(ex, *A, node.inputs.empty() ? std::string() : node.inputs[0])) {
            return;
        }
    }

    ex.EnsureGpu(*A);
    int64_t N = tensorNel(A);
    *outputs[0] = ex.AllocTensor(A->shape, dtype);

    auto params = makeParamBuf(ex, (uint32_t)N, opCode);
    std::string pipelineName = "unary_t" + std::string(dtypeSuffix(dtype));
    auto& pl = ex.GetPipelineT(pipelineName, 3, [dtype]() {
        return instantiateTemplate(WGSL_UNARY_ELEMENTWISE_T, dtype);
    });
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, outputs[0]->buffer}, {2, params}});
    // Each thread handles 2 elements
    uint32_t numWorkgroups = (uint32_t)(((N + 1) / 2 + 255) / 256);
    ex.SubmitAsync({{pl.pipeline, bg, numWorkgroups, 1, 1, node.opType}});
}

// ─── Op Registrations ────────────────────────────────────────────────────────

// Binary ops
static void opAdd(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchBinaryOp(ex, n, in, out, 0);
}
static void opSub(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchBinaryOp(ex, n, in, out, 1);
}
static void opMul(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchBinaryOp(ex, n, in, out, 2);
}
static void opDiv(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchBinaryOp(ex, n, in, out, 3);
}

// Unary ops — also need dtype check for int64
static void opSigmoid(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && isSmallIntTensor(in[0])) { *out[0] = *in[0]; return; } // no-op for int
    dispatchUnaryOp(ex, n, in, out, 0);
}
static void opTanh(GraphExecutor& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchUnaryOp(ex, n, in, out, 1);
}
static void opNeg(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && isSmallIntTensor(in[0])) {
        std::vector<int64_t> values;
        int64_t count = tensorNel(in[0]);
        if (!readTensorIntValues(ex, in[0], n.inputs.empty() ? std::string() : n.inputs[0], values) || values.empty()) {
            values.assign((size_t)std::max<int64_t>(1, count), 0);
        }
        for (auto& value : values) value = -value;
        const auto& shape = in[0]->shape.empty() ? std::vector<int64_t>{1} : in[0]->shape;
        *out[0] = ex.AllocCpuTensor(shape, TensorDtype::Int64, values.data(), values.size() * sizeof(int64_t));
        return;
    }
    dispatchUnaryOp(ex, n, in, out, 2);
}
static void opSqrt(GraphExecutor& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchUnaryOp(ex, n, in, out, 3);
}
static void opSin(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchUnaryOp(ex, n, in, out, 4);
}
static void opCos(GraphExecutor& ex, const OnnxGraphNode& n,
                   const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    dispatchUnaryOp(ex, n, in, out, 5);
}

// Cast: actual type conversion
static void opCast(GraphExecutor& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0];
    if (!A || !A->IsValid()) return;
    bool isUnifiedResultsCast = std::find(n.outputs.begin(), n.outputs.end(), "unified_results") != n.outputs.end();
    bool isVaeInputCast = kDebugVaeCast &&
        std::find(n.outputs.begin(), n.outputs.end(), "latent_sample_to_fp16") != n.outputs.end();
    int64_t targetType = n.GetInt("to", 1);
    TensorDtype outDtype = A->dtype;
    switch (targetType) {
        case 1: outDtype = TensorDtype::Float32; break;
        case 6: outDtype = TensorDtype::Int32; break;
        case 7: outDtype = TensorDtype::Int64; break;
        case 10: outDtype = TensorDtype::Float16; break;
        case 9: outDtype = TensorDtype::Bool; break;
    }

    // Same type → alias
    if (outDtype == A->dtype) {
        *out[0] = *A;
        if (isVaeInputCast) debugTensorFloatStats(ex, "vae cast input(alias)", A, n.inputs.empty() ? std::string() : n.inputs[0]);
        return;
    }

    int64_t N = tensorNel(A);

    // For CPU-backed tensors, do type conversion on CPU.
    // Use cpuData whenever available for int↔float conversions (no GPU kernel exists for these).
    const uint8_t* srcPtr = nullptr;
    bool needsCpuConvert = (A->dtype == TensorDtype::Int64 || A->dtype == TensorDtype::Int32 ||
                            outDtype == TensorDtype::Int64 || outDtype == TensorDtype::Int32 ||
                            outDtype == TensorDtype::Bool);
    if (!A->cpuData.empty() && (A->isCpuOnly || needsCpuConvert)) {
        srcPtr = A->cpuData.data();
    } else if (auto* init = ex.GetInitData(n.inputs[0]); init && init->data) {
        srcPtr = init->data;
    }

    if (srcPtr) {
        // CPU type conversion — no GPU sync needed
    } else {
        if (ex.gpu->supportsShaderF16 && A->buffer.handle) {
            if (A->dtype == TensorDtype::Float32 && outDtype == TensorDtype::Float16) {
                ex.EnsureGpu(*A);
                *out[0] = ex.AllocTensor(A->shape, TensorDtype::Float16);
                auto params = makeScalarParamBuf(ex, (uint32_t)N);
                auto& pl = ex.GetPipelineT("cast_f32_to_f16", 3, []() { return std::string(WGSL_CAST_F32_TO_F16); });
                auto bg = ex.MakeBindGroup(pl, {
                    {0, A->buffer}, {1, out[0]->buffer}, {2, params}});
                ex.SubmitAsync({{pl.pipeline, bg, (uint32_t)((N + 255) / 256), 1, 1, "cast_f32_to_f16"}});
                return;
            }
            if (A->dtype == TensorDtype::Float16 && outDtype == TensorDtype::Float32) {
                ex.EnsureGpu(*A);
                *out[0] = ex.AllocTensor(A->shape, TensorDtype::Float32);
                auto params = makeScalarParamBuf(ex, (uint32_t)N);
                auto& pl = ex.GetPipelineT("cast_f16_to_f32", 3, []() { return std::string(WGSL_CAST_F16_TO_F32); });
                auto bg = ex.MakeBindGroup(pl, {
                    {0, A->buffer}, {1, out[0]->buffer}, {2, params}});
                ex.SubmitAsync({{pl.pipeline, bg, (uint32_t)((N + 255) / 256), 1, 1, "cast_f16_to_f32"}});
                return;
            }
        }
        if (isVaeInputCast) debugTensorFloatStats(ex, "vae cast input", A, n.inputs.empty() ? std::string() : n.inputs[0]);
        if (A->dtype == TensorDtype::Float16 && outDtype == TensorDtype::Float32) {
            ex.FlushPendingWork();
            size_t inBytes = (size_t)N * sizeof(uint16_t);
            auto rb = ex.gpu->readBuffer(A->buffer, inBytes);
            if (rb.size() >= inBytes) {
                std::vector<float> fp32((size_t)N);
                auto* src = reinterpret_cast<const uint16_t*>(rb.data());
                for (int64_t i = 0; i < N; i++) fp32[(size_t)i] = fp16ToFloat(src[i]);
                if (isUnifiedResultsCast) {
                    float minV = 1e30f, maxV = -1e30f;
                    double sumV = 0.0;
                    for (float v : fp32) {
                        minV = std::min(minV, v);
                        maxV = std::max(maxV, v);
                        sumV += v;
                    }
                    fprintf(stderr, "    [cast] unified_results N=%lld min=%.6f max=%.6f avg=%.6f\n",
                            (long long)N, minV, maxV, fp32.empty() ? 0.0 : (sumV / fp32.size()));
                    fflush(stderr);
                }
                *out[0] = ex.AllocTensor(A->shape, TensorDtype::Float32);
                ex.gpu->writeBuffer(out[0]->buffer, fp32.data(), fp32.size() * sizeof(float));
                return;
            }
        }
        // GPU tensor — all GPU compute uses fp32, so aliasing preserves data
        // DON'T change dtype to avoid ByteSize() mismatch
        // For conversions between float and integer types, must read back and convert
        if ((outDtype == TensorDtype::Int64 || outDtype == TensorDtype::Int32) &&
            (A->dtype == TensorDtype::Float32 || A->dtype == TensorDtype::Float16)) {
            ex.FlushPendingWork();
            size_t inBytes = (size_t)N * A->DtypeSize();
            auto rb = ex.gpu->readBuffer(A->buffer, inBytes);
            if (rb.size() >= inBytes) {
                if (outDtype == TensorDtype::Int64) {
                    std::vector<int64_t> result((size_t)N);
                    if (A->dtype == TensorDtype::Float32) {
                        auto* src = reinterpret_cast<const float*>(rb.data());
                        for (int64_t i = 0; i < N; i++) result[i] = (int64_t)src[i];
                    } else {
                        auto* src = reinterpret_cast<const uint16_t*>(rb.data());
                        for (int64_t i = 0; i < N; i++) result[i] = (int64_t)fp16ToFloat(src[i]);
                    }
                    *out[0] = ex.AllocCpuTensor(A->shape, TensorDtype::Int64, result.data(), N * 8);
                } else {
                    std::vector<int32_t> result((size_t)N);
                    if (A->dtype == TensorDtype::Float32) {
                        auto* src = reinterpret_cast<const float*>(rb.data());
                        for (int64_t i = 0; i < N; i++) result[i] = (int32_t)src[i];
                    } else {
                        auto* src = reinterpret_cast<const uint16_t*>(rb.data());
                        for (int64_t i = 0; i < N; i++) result[i] = (int32_t)fp16ToFloat(src[i]);
                    }
                    *out[0] = ex.AllocCpuTensor(A->shape, TensorDtype::Int32, result.data(), N * 4);
                }
                return;
            }
        }
        // Int→Float conversions that weren't handled above: read back and convert via CPU
        if ((A->dtype == TensorDtype::Int64 || A->dtype == TensorDtype::Int32) &&
            (outDtype == TensorDtype::Float32 || outDtype == TensorDtype::Float16) &&
            A->buffer.handle) {
            ex.FlushPendingWork();
            size_t inBytes = (size_t)N * A->DtypeSize();
            auto rb = ex.gpu->readBuffer(A->buffer, inBytes);
            if (rb.size() >= inBytes) {
                std::vector<double> vals(N);
                for (int64_t i = 0; i < N; i++) {
                    if (A->dtype == TensorDtype::Int64) {
                        int64_t v; memcpy(&v, rb.data() + i*8, 8); vals[i] = (double)v;
                    } else {
                        int32_t v; memcpy(&v, rb.data() + i*4, 4); vals[i] = (double)v;
                    }
                }
                if (outDtype == TensorDtype::Float32) {
                    std::vector<float> result((size_t)N);
                    for (int64_t i = 0; i < N; i++) result[i] = (float)vals[i];
                    *out[0] = ex.AllocTensor(A->shape, TensorDtype::Float32);
                    ex.gpu->writeBuffer(out[0]->buffer, result.data(), N * 4);
                } else {
                    // Float16
                    std::vector<uint16_t> result((size_t)N);
                    for (int64_t i = 0; i < N; i++) {
                        float fv = (float)vals[i];
                        uint32_t fb; memcpy(&fb, &fv, 4);
                        uint32_t s = (fb >> 16) & 0x8000;
                        int32_t e = ((fb >> 23) & 0xFF) - 112;
                        uint32_t m = (fb >> 13) & 0x3FF;
                        uint16_t h;
                        if (e <= 0) h = (uint16_t)s;
                        else if (e > 30) h = (uint16_t)(s | 0x7C00);
                        else h = (uint16_t)(s | (e << 10) | m);
                        result[i] = h;
                    }
                    bool isCpuResult = (N <= 64);
                    if (isCpuResult) {
                        *out[0] = ex.AllocCpuTensor(A->shape, TensorDtype::Float16, result.data(), N * 2);
                    } else {
                        *out[0] = ex.AllocTensor(A->shape, TensorDtype::Float16);
                        ex.gpu->writeBuffer(out[0]->buffer, result.data(), N * 2);
                    }
                }
                return;
            }
        }
        *out[0] = *A;
        // Only change dtype for int↔float conversions, not float↔float
        if ((outDtype == TensorDtype::Float32 || outDtype == TensorDtype::Float16) &&
            (A->dtype == TensorDtype::Float32 || A->dtype == TensorDtype::Float16)) {
            // Keep as Float32 since GPU buffers contain fp32 data
            out[0]->dtype = TensorDtype::Float32;
        } else {
            out[0]->dtype = outDtype;
        }
        if (isVaeInputCast) debugTensorFloatStats(ex, "vae cast output", out[0], "latent_sample_to_fp16");
        return;
    }

    // Read source as fp64 intermediate
    std::vector<double> vals(N);
    for (int64_t i = 0; i < N; i++) {
        switch (A->dtype) {
            case TensorDtype::Float32: { float v; memcpy(&v, srcPtr + i*4, 4); vals[i] = v; break; }
            case TensorDtype::Int64: { int64_t v; memcpy(&v, srcPtr + i*8, 8); vals[i] = (double)v; break; }
            case TensorDtype::Int32: { int32_t v; memcpy(&v, srcPtr + i*4, 4); vals[i] = (double)v; break; }
            case TensorDtype::Float16: {
                uint16_t h; memcpy(&h, srcPtr + i*2, 2);
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) f = (sign << 31) | (mant << 13);
                else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
                else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                float fv; memcpy(&fv, &f, 4); vals[i] = fv;
                break;
            }
            default: vals[i] = 0; break;
        }
    }

    // Write target type
    bool isCpuResult = (A->isCpuOnly || N <= 64);
    size_t outElemSize = 4;
    switch (outDtype) {
        case TensorDtype::Float32: case TensorDtype::Int32: outElemSize = 4; break;
        case TensorDtype::Float16: outElemSize = 2; break;
        case TensorDtype::Int64: outElemSize = 8; break;
        case TensorDtype::Bool: case TensorDtype::UInt8: case TensorDtype::Int8: outElemSize = 1; break;
    }
    std::vector<uint8_t> outBuf(N * outElemSize);
    for (int64_t i = 0; i < N; i++) {
        switch (outDtype) {
            case TensorDtype::Float32: { float v = (float)vals[i]; memcpy(outBuf.data() + i*4, &v, 4); break; }
            case TensorDtype::Int64: { int64_t v = (int64_t)vals[i]; memcpy(outBuf.data() + i*8, &v, 8); break; }
            case TensorDtype::Int32: { int32_t v = (int32_t)vals[i]; memcpy(outBuf.data() + i*4, &v, 4); break; }
            case TensorDtype::Float16: {
                float fv = (float)vals[i];
                uint32_t fb; memcpy(&fb, &fv, 4);
                uint32_t s = (fb >> 16) & 0x8000;
                int32_t e = ((fb >> 23) & 0xFF) - 112;
                uint32_t m = (fb >> 13) & 0x3FF;
                uint16_t h;
                if (e <= 0) h = (uint16_t)s;
                else if (e > 30) h = (uint16_t)(s | 0x7C00);
                else h = (uint16_t)(s | (e << 10) | m);
                memcpy(outBuf.data() + i*2, &h, 2);
                break;
            }
            case TensorDtype::Bool: { uint8_t v = vals[i] != 0 ? 1 : 0; outBuf[i] = v; break; }
            default: break;
        }
    }
    if (isCpuResult) {
        *out[0] = ex.AllocCpuTensor(A->shape, outDtype, outBuf.data(), outBuf.size());
    } else {
        *out[0] = ex.AllocTensor(A->shape, outDtype);
        ex.gpu->writeBuffer(out[0]->buffer, outBuf.data(), outBuf.size());
    }
}

// Where: condition ? X : Y — uses embedded where_select kernel
static void opWhere(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* Cond = in[0];
    auto* X = in.size() > 1 ? in[1] : nullptr;
    auto* Y = in.size() > 2 ? in[2] : nullptr;
    if (!Cond || !X || !Y || !Cond->IsValid() || !X->IsValid() || !Y->IsValid()) {
        // Fallback: pass X
        if (X && X->IsValid()) { ex.EnsureGpu(*X); *out[0] = *X; }
        else if (Y && Y->IsValid()) { ex.EnsureGpu(*Y); *out[0] = *Y; }
        return;
    }

    if (!ensureCpuBackedFloat32(ex, *X, n.inputs.size() > 1 ? n.inputs[1] : std::string()) ||
        !ensureCpuBackedFloat32(ex, *Y, n.inputs.size() > 2 ? n.inputs[2] : std::string())) {
        return;
    }
    ex.EnsureGpu(*Cond); ex.EnsureGpu(*X); ex.EnsureGpu(*Y);

    int64_t N = std::max({tensorNel(Cond), tensorNel(X), tensorNel(Y)});
    auto& outShape = (tensorNel(X) >= tensorNel(Y)) ? X->shape : Y->shape;
    TensorDtype outDtype = (X->dtype == TensorDtype::Float16 || X->dtype == TensorDtype::Float32 ||
                            Y->dtype == TensorDtype::Float16 || Y->dtype == TensorDtype::Float32)
                             ? TensorDtype::Float32
                             : X->dtype;
    if (tensorNel(Cond) > tensorNel(X) && tensorNel(Cond) > tensorNel(Y))
        *out[0] = ex.AllocTensor(Cond->shape, outDtype);
    else
        *out[0] = ex.AllocTensor(outShape, outDtype);

    auto params = makeParamBuf(ex, (uint32_t)N, (uint32_t)tensorNel(Cond),
                                (uint32_t)tensorNel(X), (uint32_t)tensorNel(Y));
    auto& pl = ex.GetPipelineT("where_select", 5, []() { return instantiateTemplate(WGSL_WHERE_SELECT_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, Cond->buffer}, {1, X->buffer}, {2, Y->buffer},
        {3, out[0]->buffer}, {4, params}});
    ex.SubmitAsync({{pl.pipeline, bg, (uint32_t)((N + 255) / 256), 1, 1, "where"}});
}

// Equal: compare two f32 tensors, output bool — uses embedded kernel
static void opEqual(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in.size() > 1 ? in[1] : nullptr;
    if (!A || !B || !A->IsValid() || !B->IsValid()) {
        if (A && A->IsValid()) {
            *out[0] = ex.AllocTensor(A->shape, TensorDtype::Bool);
            size_t bytes = (out[0]->ElementCount() + 3) / 4 * 4;
            std::vector<uint8_t> zeros(bytes, 0);
            ex.gpu->writeBuffer(out[0]->buffer, zeros.data(), bytes);
        }
        return;
    }

    if (!ensureCpuBackedFloat32(ex, *A, n.inputs.empty() ? std::string() : n.inputs[0]) ||
        !ensureCpuBackedFloat32(ex, *B, n.inputs.size() > 1 ? n.inputs[1] : std::string())) {
        return;
    }
    ex.EnsureGpu(*A); ex.EnsureGpu(*B);

    int64_t N = tensorNel(A);
    *out[0] = ex.AllocTensor(A->shape, TensorDtype::Bool);

    auto params = makeParamBuf(ex, (uint32_t)N, (uint32_t)tensorNel(B));
    auto& pl = ex.GetPipelineT("equal_op", 4, []() { return instantiateTemplate(WGSL_EQUAL_OP_T, TensorDtype::Float32); });
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, out[0]->buffer}, {3, params}});
    ex.SubmitAsync({{pl.pipeline, bg, (uint32_t)((N + 255) / 256), 1, 1, "equal"}});
}

static void opGreaterOrEqual(GraphExecutor& ex, const OnnxGraphNode& n,
                              const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in.size() > 1 ? in[1] : nullptr;
    if (!A || !A->IsValid()) return;
    int64_t N = tensorNel(A);
    *out[0] = ex.AllocTensor(A->shape, TensorDtype::Bool);
    // For now, produce all-true (GreaterOrEqual is used in scheduler for step selection)
    size_t bytes = (N + 3) / 4 * 4;
    std::vector<uint8_t> ones(bytes, 1);
    ex.gpu->writeBuffer(out[0]->buffer, ones.data(), bytes);
}

// Softmax: proper GPU kernel — uses embedded softmax kernel
static void opSoftmax(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];
    if (!X || !X->IsValid()) return;

    // Determine dtype
    TensorDtype dtype = X->dtype;
    if (dtype != TensorDtype::Float16) {
        dtype = TensorDtype::Float32;
        if (!ensureCpuBackedFloat32(ex, *X, n.inputs.empty() ? std::string() : n.inputs[0])) return;
    }
    ex.EnsureGpu(*X);

    int64_t axis = n.GetInt("axis", -1);
    if (axis < 0) axis += (int64_t)X->shape.size();
    int64_t rowLen = (axis < (int64_t)X->shape.size()) ? X->shape[axis] : X->shape.back();
    int64_t nRows = tensorNel(X) / rowLen;

    *out[0] = ex.AllocTensor(X->shape, dtype);

    auto params = makeParamBuf(ex, (uint32_t)nRows, (uint32_t)rowLen);
    std::string pname = "softmax_t" + std::string(dtypeSuffix(dtype));
    auto& pl = ex.GetPipelineT(pname, 3, [dtype]() {
        return instantiateTemplate(WGSL_SOFTMAX_T, dtype);
    });
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, out[0]->buffer}, {2, params}});
    ex.SubmitAsync({{pl.pipeline, bg, (uint32_t)((nRows + 255) / 256), 1, 1, "softmax"}});
}

// ReduceSum: CPU for small tensors, GPU readback for larger ones
static void opReduceSum(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0];
    if (!A || !A->IsValid()) return;
    int64_t N = tensorNel(A);
    int keepdims = (int)n.GetInt("keepdims", 1);

    // Read axes from input[1] or attribute
    std::vector<int64_t> axes;
    if (in.size() > 1 && in[1] && in[1]->IsValid()) {
        std::vector<int64_t> axVals;
        readTensorIntValues(ex, in[1], n.inputs.size() > 1 ? n.inputs[1] : "", axVals);
        axes = axVals;
    } else if (n.attrIntLists.count("axes")) {
        axes = n.attrIntLists.at("axes");
    }

    int ndim = (int)A->shape.size();
    // Normalize negative axes
    for (auto& a : axes) { if (a < 0) a += ndim; }
    // If no axes specified, reduce all
    if (axes.empty()) { for (int i = 0; i < ndim; i++) axes.push_back(i); }

    // Get data on CPU
    const uint8_t* ptr = nullptr;
    std::vector<uint8_t> gpuReadback;
    if (A->isCpuOnly && !A->cpuData.empty()) ptr = A->cpuData.data();
    else if (!A->cpuData.empty()) ptr = A->cpuData.data();
    else if (auto* init = ex.GetInitData(n.inputs[0]); init && init->data) ptr = init->data;
    else if (A->buffer.handle) {
        ex.FlushPendingWork();
        gpuReadback = ex.gpu->readBuffer(A->buffer, N * A->DtypeSize());
        if (!gpuReadback.empty()) ptr = gpuReadback.data();
    }

    if (!ptr) {
        float z = 0;
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &z, 4);
        return;
    }

    // Compute output shape
    std::vector<int64_t> outShape;
    std::set<int> reduceSet(axes.begin(), axes.end());
    for (int i = 0; i < ndim; i++) {
        if (reduceSet.count(i)) {
            if (keepdims) outShape.push_back(1);
        } else {
            outShape.push_back(A->shape[i]);
        }
    }
    if (outShape.empty()) outShape.push_back(1);

    int64_t outNel = 1;
    for (auto d : outShape) outNel *= d;

    // Compute strides for input
    std::vector<int64_t> inStrides(ndim, 1);
    for (int i = ndim - 2; i >= 0; i--) inStrides[i] = inStrides[i+1] * A->shape[i+1];

    if (A->dtype == TensorDtype::Float32) {
        const float* src = reinterpret_cast<const float*>(ptr);
        std::vector<float> result(outNel, 0.0f);

        // For each element in input, compute output index and accumulate
        for (int64_t flat = 0; flat < N; flat++) {
            // Decompose flat index into coordinates
            int64_t remaining = flat;
            int64_t outFlat = 0;
            int64_t outStride = 1;
            // Walk dims in reverse to compute output flat index
            for (int i = ndim - 1; i >= 0; i--) {
                int64_t coord = remaining % A->shape[i];
                remaining /= A->shape[i];
                if (!reduceSet.count(i)) {
                    // Find position of this dim in output
                    int outDimIdx = 0;
                    for (int j = 0; j < i; j++) {
                        if (!reduceSet.count(j) || keepdims) outDimIdx++;
                    }
                    // Compute output stride
                    int64_t os = 1;
                    for (int j = outDimIdx + 1; j < (int)outShape.size(); j++) os *= outShape[j];
                    outFlat += coord * os;
                }
            }
            result[outFlat] += src[flat];
        }
        *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Float32, result.data(), outNel * 4);
    } else if (A->dtype == TensorDtype::Int64) {
        const int64_t* src = reinterpret_cast<const int64_t*>(ptr);
        std::vector<int64_t> result(outNel, 0);
        for (int64_t flat = 0; flat < N; flat++) {
            int64_t remaining = flat;
            int64_t outFlat = 0;
            for (int i = ndim - 1; i >= 0; i--) {
                int64_t coord = remaining % A->shape[i];
                remaining /= A->shape[i];
                if (!reduceSet.count(i)) {
                    int outDimIdx = 0;
                    for (int j = 0; j < i; j++) {
                        if (!reduceSet.count(j) || keepdims) outDimIdx++;
                    }
                    int64_t os = 1;
                    for (int j = outDimIdx + 1; j < (int)outShape.size(); j++) os *= outShape[j];
                    outFlat += coord * os;
                }
            }
            result[outFlat] += src[flat];
        }
        *out[0] = ex.AllocCpuTensor(outShape, TensorDtype::Int64, result.data(), outNel * 8);
    } else {
        float z = 0;
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &z, 4);
    }
}

// ─── Register all elementwise ops ────────────────────────────────────────────

// Helper macro for simple unary dispatches
#define DEF_UNARY(name, code) \
    static void op##name(GraphExecutor& ex, const OnnxGraphNode& n, \
        const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) { \
        dispatchUnaryOp(ex, n, in, out, code); \
    }

DEF_UNARY(Gelu, 7)
DEF_UNARY(Silu, 8)
DEF_UNARY(Erf, 9)
DEF_UNARY(Relu, 10)
DEF_UNARY(Exp, 11)
DEF_UNARY(Log, 12)
DEF_UNARY(Abs, 13)
DEF_UNARY(Floor, 14)
DEF_UNARY(Ceil, 15)
DEF_UNARY(Round, 16)
DEF_UNARY(Softplus, 17)

REGISTER_OP(Add, opAdd)
REGISTER_OP(Sub, opSub)
REGISTER_OP(Mul, opMul)
REGISTER_OP(Div, opDiv)
REGISTER_OP(Sigmoid, opSigmoid)
REGISTER_OP(Tanh, opTanh)
REGISTER_OP(Neg, opNeg)
REGISTER_OP(Sqrt, opSqrt)
REGISTER_OP(Sin, opSin)
REGISTER_OP(Cos, opCos)
REGISTER_OP(Cast, opCast)
REGISTER_OP(Where, opWhere)
REGISTER_OP(Equal, opEqual)
REGISTER_OP(GreaterOrEqual, opGreaterOrEqual)
REGISTER_OP(Softmax, opSoftmax)
REGISTER_OP(ReduceSum, opReduceSum)
REGISTER_OP(Gelu, opGelu)
REGISTER_OP(FastGelu, opGelu)
REGISTER_OP(Silu, opSilu)
REGISTER_OP(Erf, opErf)
REGISTER_OP(Relu, opRelu)
REGISTER_OP(Exp, opExp)
REGISTER_OP(Log, opLog)
REGISTER_OP(Abs, opAbs)
REGISTER_OP(Floor, opFloor)
REGISTER_OP(Ceil, opCeil)
REGISTER_OP(Round, opRound)
REGISTER_OP(Softplus, opSoftplus)

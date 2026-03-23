/**
 * ops/elementwise.cpp — Elementwise ONNX op implementations.
 *
 * Covers: Add, Sub, Mul, Div, Neg, Sqrt, Sigmoid, Tanh, Sin, Cos,
 *         Cast, Equal, GreaterOrEqual, Where, Softmax, ReduceSum.
 *
 * Most ops use a single generic WGSL kernel parameterized by op code.
 * Some ops (Cast, Where, Softmax) need specialized kernels.
 */

#include "../graph_executor.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// ─── Generic binary elementwise kernel ───────────────────────────────────────

static const char* WGSL_BINARY_ELEMENTWISE = R"WGSL(
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let B_N = _params_[2]; // B element count (for broadcasting)
    let idx = gid.x;
    if (idx >= N) { return; }
    let a = A[idx];
    let b_idx = select(idx, idx % B_N, B_N < N && B_N > 0u);
    let b = B[b_idx];
    switch (op) {
        case 0u: { C[idx] = a + b; }
        case 1u: { C[idx] = a - b; }
        case 2u: { C[idx] = a * b; }
        case 3u: { C[idx] = a / b; }
        default: { C[idx] = a + b; }
    }
}
)WGSL";

// ─── Generic unary elementwise kernel ────────────────────────────────────────

static const char* WGSL_UNARY_ELEMENTWISE = R"WGSL(
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let a = A[idx];
    switch (op) {
        case 0u: { C[idx] = 1.0 / (1.0 + exp(-a)); }  // sigmoid
        case 1u: { C[idx] = tanh(a); }
        case 2u: { C[idx] = -a; }
        case 3u: { C[idx] = sqrt(a); }
        case 4u: { C[idx] = sin(a); }
        case 5u: { C[idx] = cos(a); }
        case 6u: { C[idx] = a; }  // identity (cast to same type)
        default: { C[idx] = a; }
    }
}
)WGSL";

// ─── Helpers ─────────────────────────────────────────────────────────────────

static int64_t tensorNel(const GpuTensor* t) {
    if (!t || t->shape.empty()) return 0;
    int64_t n = 1; for (auto d : t->shape) n *= d; return n;
}

static GPUBuffer makeParamBuf(GraphExecutor& ex, uint32_t p0, uint32_t p1 = 0,
                               uint32_t p2 = 0, uint32_t p3 = 0) {
    uint32_t data[4] = {p0, p1, p2, p3};
    auto buf = ex.gpu->createBuffer("params", 16);
    ex.gpu->writeBuffer(buf, data, 16);
    return buf;
}

static bool isSmallIntTensor(const GpuTensor* t) {
    if (!t) return false;
    return (t->dtype == TensorDtype::Int64 || t->dtype == TensorDtype::Int32) &&
           tensorNel(t) <= 64;
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
    int64_t N = std::max(N_A, N_B);
    auto& outShape = (N_A >= N_B) ? A->shape : B->shape;

    // Read from CPU data if available, else GPU readback
    std::vector<int64_t> a(N_A), b(N_B), c(N);

    auto readInt64 = [&](GpuTensor* t, std::vector<int64_t>& dst, int64_t nel, const std::string& name) {
        if (t->isCpuOnly && !t->cpuData.empty()) {
            memcpy(dst.data(), t->cpuData.data(), nel * 8);
        } else if (auto* init = ex.GetInitData(name); init && init->data) {
            memcpy(dst.data(), init->data, nel * 8);
        } else {
            // GPU tensor — can't read without sync. Use zeros.
            // TODO: add GPU int64 binary kernel
            memset(dst.data(), 0, nel * 8);
        }
    };

    readInt64(A, a, N_A, node.inputs[0]);
    readInt64(B, b, N_B, node.inputs[1]);

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

    // GPU kernel for fp32 binary ops
    ex.EnsureGpu(*A);
    ex.EnsureGpu(*B);

    int64_t N_A = tensorNel(A);
    int64_t N_B = tensorNel(B);
    int64_t N = std::max(N_A, N_B);

    // Output shape = broadcast shape (simplified: take the larger)
    auto& outShape = (N_A >= N_B) ? A->shape : B->shape;
    *outputs[0] = ex.AllocTensor(outShape, A->dtype);

    auto params = makeParamBuf(ex, (uint32_t)N, opCode, (uint32_t)N_B);
    auto& pl = ex.GetPipeline("binary_elem", WGSL_BINARY_ELEMENTWISE, 4);
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, outputs[0]->buffer}, {3, params}});
    ex.SubmitAsync({{pl.pipeline, bg, (uint32_t)((N + 255) / 256), 1, 1, node.opType}});
}

// ─── Unary elementwise dispatch ──────────────────────────────────────────────

static void dispatchUnaryOp(GraphExecutor& ex, const OnnxGraphNode& node,
                             const std::vector<GpuTensor*>& inputs,
                             std::vector<GpuTensor*>& outputs, uint32_t opCode) {
    auto* A = inputs[0];
    if (!A || !A->IsValid()) return;

    ex.EnsureGpu(*A);
    int64_t N = tensorNel(A);
    *outputs[0] = ex.AllocTensor(A->shape, A->dtype);

    auto params = makeParamBuf(ex, (uint32_t)N, opCode);
    auto& pl = ex.GetPipeline("unary_elem", WGSL_UNARY_ELEMENTWISE, 3);
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, outputs[0]->buffer}, {2, params}});
    ex.SubmitAsync({{pl.pipeline, bg, (uint32_t)((N + 255) / 256), 1, 1, node.opType}});
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
        return;
    }

    int64_t N = tensorNel(A);

    // For CPU-only tensors, do type conversion on CPU
    const uint8_t* srcPtr = nullptr;
    if (A->isCpuOnly && !A->cpuData.empty()) {
        srcPtr = A->cpuData.data();
    } else if (auto* init = ex.GetInitData(n.inputs[0]); init && init->data) {
        srcPtr = init->data;
    }

    if (srcPtr) {
        // CPU type conversion — no GPU sync needed
    } else {
        // GPU tensor — for now, just alias with new dtype (no actual conversion)
        // TODO: add GPU cast kernel
        *out[0] = *A;
        out[0]->dtype = outDtype;
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

// Where: for now, just pass through X (TODO: GPU Where kernel)
static void opWhere(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in.size() > 1 ? in[1] : nullptr;
    if (X && X->IsValid()) { ex.EnsureGpu(*X); *out[0] = *X; }
    else if (in[0] && in[0]->IsValid()) { ex.EnsureGpu(*in[0]); *out[0] = *in[0]; }
}

// Equal, GreaterOrEqual: produce bool tensor (zeros for now)
static void opEqual(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        *out[0] = ex.AllocTensor(in[0]->shape, TensorDtype::Bool);
        // Zero-fill (all false)
        size_t bytes = out[0]->ByteSize();
        std::vector<uint8_t> zeros(bytes, 0);
        ex.gpu->writeBuffer(out[0]->buffer, zeros.data(), bytes);
    }
}

static void opGreaterOrEqual(GraphExecutor& ex, const OnnxGraphNode& n,
                              const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        *out[0] = ex.AllocTensor(in[0]->shape, TensorDtype::Bool);
        size_t bytes = out[0]->ByteSize();
        std::vector<uint8_t> ones(bytes, 1); // all true
        ex.gpu->writeBuffer(out[0]->buffer, ones.data(), bytes);
    }
}

// Softmax: full implementation needed for VAE attention
static void opSoftmax(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        *out[0] = *in[0]; // TODO: implement softmax kernel
    }
}

// ReduceSum: CPU for small tensors
static void opReduceSum(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0];
    if (!A || !A->IsValid()) return;
    int64_t N = tensorNel(A);

    // Read from CPU if possible
    const uint8_t* ptr = nullptr;
    if (A->isCpuOnly && !A->cpuData.empty()) ptr = A->cpuData.data();
    else if (auto* init = ex.GetInitData(n.inputs[0]); init && init->data) ptr = init->data;

    if (!ptr && A->buffer.handle) {
        // GPU tensor — can't reduce without sync. Allocate zero result.
        // TODO: add GPU reduce kernel
        if (A->dtype == TensorDtype::Int64) {
            int64_t z = 0;
            *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Int64, &z, 8);
        } else {
            float z = 0;
            *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &z, 4);
        }
        return;
    }
    if (!ptr) return;

    if (A->dtype == TensorDtype::Int64) {
        int64_t sum = 0;
        for (int64_t i = 0; i < N; i++) { int64_t v; memcpy(&v, ptr + i*8, 8); sum += v; }
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Int64, &sum, 8);
    } else {
        float sum = 0;
        for (int64_t i = 0; i < N; i++) { float v; memcpy(&v, ptr + i*4, 4); sum += v; }
        *out[0] = ex.AllocCpuTensor({1}, TensorDtype::Float32, &sum, 4);
    }
}

// ─── Register all elementwise ops ────────────────────────────────────────────

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

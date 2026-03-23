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

// ─── Binary elementwise dispatch ─────────────────────────────────────────────

static void dispatchBinaryOp(GraphExecutor& ex, const OnnxGraphNode& node,
                              const std::vector<GpuTensor*>& inputs,
                              std::vector<GpuTensor*>& outputs, uint32_t opCode) {
    auto* A = inputs[0];
    auto* B = inputs[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;

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

// Unary ops
static void opSigmoid(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
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

// Cast: for now, just copy (same type or fp16↔fp32 handled by identity copy)
static void opCast(GraphExecutor& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // TODO: actual type conversion. For now, treat as identity copy.
    auto* A = in[0];
    if (!A || !A->IsValid()) return;
    int64_t targetType = n.GetInt("to", 1);
    TensorDtype outDtype = A->dtype;
    switch (targetType) {
        case 1: outDtype = TensorDtype::Float32; break;
        case 6: outDtype = TensorDtype::Int32; break;
        case 7: outDtype = TensorDtype::Int64; break;
        case 10: outDtype = TensorDtype::Float16; break;
    }
    // If same dtype, just alias
    if (outDtype == A->dtype) {
        *out[0] = *A;
        out[0]->shape = A->shape;
        return;
    }
    // TODO: implement GPU type conversion kernel
    // For now, allocate same-size buffer and copy raw bytes
    *out[0] = ex.AllocTensor(A->shape, outDtype);
}

// Where: C = cond ? X : Y
static void opWhere(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    // TODO: GPU kernel. For now, placeholder.
    if (in.size() >= 3 && in[1] && in[1]->IsValid()) {
        *out[0] = *in[1]; // just take X branch
    }
}

// Equal, GreaterOrEqual: comparison ops
static void opEqual(GraphExecutor& ex, const OnnxGraphNode& n,
                     const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        *out[0] = ex.AllocTensor(in[0]->shape, TensorDtype::Bool);
    }
}

static void opGreaterOrEqual(GraphExecutor& ex, const OnnxGraphNode& n,
                              const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        *out[0] = ex.AllocTensor(in[0]->shape, TensorDtype::Bool);
    }
}

// Softmax: full implementation needed for VAE attention
static void opSoftmax(GraphExecutor& ex, const OnnxGraphNode& n,
                       const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        *out[0] = *in[0]; // TODO: implement softmax kernel
    }
}

// ReduceSum: placeholder
static void opReduceSum(GraphExecutor& ex, const OnnxGraphNode& n,
                          const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    if (in[0] && in[0]->IsValid()) {
        // TODO: implement reduce kernel
        *out[0] = ex.AllocTensor({1}, in[0]->dtype);
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

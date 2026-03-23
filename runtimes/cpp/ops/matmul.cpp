/**
 * ops/matmul.cpp — Matrix multiplication ops with full GPU kernels.
 * MatMulNBits (Q4 quantized), MatMul (fp32), Gemm (matmul+bias),
 * GatherBlockQuantized (quantized embedding).
 */

#include "../graph_executor.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// ─── MatMulNBits: Q4 quantized matmul ────────────────────────────────────────
// Y[b,m,n] = X[b,m,k] * dequant(W_Q4[n,k/block_size,block_size/2], scales[n*n_blocks+block])
// Q4 format: each byte = 2 nibbles (low=elem[2j], high=elem[2j+1]), zero_point=8

static const char* WGSL_MATMUL_Q4 = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q4: array<u32>; // packed uint8 as u32
@group(0) @binding(2) var<storage, read> Scales: array<u32>; // packed fp16 as u32
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = _params_[0]; // batch * seq_len (number of input rows)
    let N = _params_[1]; // output dim
    let K = _params_[2]; // input dim
    let n_blocks = K / 32u; // blocks per row

    let row = gid.y; // which input row
    let col = gid.x; // which output column

    if (row >= M || col >= N) { return; }

    var acc: f32 = 0.0;
    let x_base = row * K;
    // W_Q4 layout: [N, n_blocks, 16] as uint8, packed 4 per u32
    // Each block: 16 bytes = 32 nibbles = 32 elements
    let w_base = col * n_blocks * 4u; // 16 bytes / 4 = 4 u32 per block

    for (var blk = 0u; blk < n_blocks; blk++) {
        // Read scale (fp16, packed 2 per u32)
        let scale_idx = col * n_blocks + blk;
        let scale_u32 = Scales[scale_idx / 2u];
        let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_idx & 1u) != 0u);
        let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;

        let w_off = w_base + blk * 4u;
        let x_off = x_base + blk * 32u;

        // Process 32 elements per block (4 u32 = 16 bytes = 32 nibbles)
        for (var j = 0u; j < 4u; j++) {
            let packed = W_Q4[w_off + j];
            // Each u32 has 4 bytes, each byte has 2 nibbles
            for (var b = 0u; b < 4u; b++) {
                let byte_val = (packed >> (b * 8u)) & 0xFFu;
                let lo = f32(i32(byte_val & 0xFu) - 8);
                let hi = f32(i32((byte_val >> 4u) & 0xFu) - 8);
                let k0 = j * 8u + b * 2u;
                acc += X[x_off + k0] * lo * scale;
                acc += X[x_off + k0 + 1u] * hi * scale;
            }
        }
    }

    Y[row * N + col] = acc;
}
)WGSL";

static void opMatMulNBits(GraphExecutor& ex, const OnnxGraphNode& n,
                           const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];     // [M, K] or [B, M, K]
    auto* W = in[1];     // [N, n_blocks, block_half] uint8
    auto* S = in[2];     // [N * n_blocks] fp16 (flattened or [N, n_blocks])
    if (!X || !W || !S || !X->IsValid() || !W->IsValid() || !S->IsValid()) return;

    uint32_t N = (uint32_t)n.GetInt("N");
    uint32_t K = (uint32_t)n.GetInt("K");

    // Compute M = total input rows (product of all dims except last)
    int64_t M = 1;
    for (size_t i = 0; i + 1 < X->shape.size(); i++) M *= X->shape[i];

    // Output shape = X shape with last dim replaced by N
    auto outShape = X->shape;
    outShape.back() = N;
    *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);

    // Params: [M, N, K, 0]
    uint32_t params[4] = {(uint32_t)M, N, K, 0};
    auto paramBuf = ex.gpu->createBuffer("mmnb_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("matmul_q4", WGSL_MATMUL_Q4, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, S->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (N + 255) / 256, (uint32_t)M, 1, "matmul_q4"});
}

// ─── MatMul: fp32 matrix multiply ────────────────────────────────────────────

static const char* WGSL_MATMUL_F32 = R"WGSL(
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }

    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
)WGSL";

static void opMatMul(GraphExecutor& ex, const OnnxGraphNode& n,
                      const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;

    // A: [..., M, K], B: [..., K, N] → C: [..., M, N]
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

// ─── Gemm: Y = alpha*A*B + beta*C ───────────────────────────────────────────

static const char* WGSL_GEMM = R"WGSL(
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];

    let row = gid.x / N;
    let col = gid.x % N;
    if (row >= M || col >= N) { return; }

    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        let b_val = select(B[k * N + col], B[col * K + k], transB != 0u);
        acc += A[row * K + k] * b_val;
    }
    Y[row * N + col] = acc + Bias[col];
}
)WGSL";

static void opGemm(GraphExecutor& ex, const OnnxGraphNode& n,
                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* A = in[0]; auto* B = in[1];
    if (!A || !B || !A->IsValid() || !B->IsValid()) return;

    int64_t transB = n.GetInt("transB", 0);
    int64_t M = A->shape.size() >= 2 ? A->shape[0] : 1;
    int64_t K = A->shape.back();
    int64_t N_out = transB ? B->shape[0] : B->shape.back();

    *out[0] = ex.AllocTensor({M, N_out}, TensorDtype::Float32);

    // Bias (optional third input)
    GPUBuffer biasBuf;
    if (in.size() > 2 && in[2] && in[2]->IsValid()) {
        biasBuf = in[2]->buffer;
    } else {
        // Zero bias
        std::vector<float> zeros((size_t)N_out, 0.0f);
        biasBuf = ex.gpu->createBuffer("gemm_bias0", N_out * 4);
        ex.gpu->writeBuffer(biasBuf, zeros.data(), N_out * 4);
    }

    uint32_t params[4] = {(uint32_t)M, (uint32_t)N_out, (uint32_t)K, (uint32_t)transB};
    auto paramBuf = ex.gpu->createBuffer("gemm_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("gemm", WGSL_GEMM, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, A->buffer}, {1, B->buffer}, {2, biasBuf},
        {3, out[0]->buffer}, {4, paramBuf}});

    int64_t total = M * N_out;
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (uint32_t)((total + 255) / 256), 1, 1, "gemm"});
}

// ─── GatherBlockQuantized: quantized embedding lookup ────────────────────────

static const char* WGSL_GATHER_BQ = R"WGSL(
@group(0) @binding(0) var<storage, read> W: array<u32>;  // [V, n_groups, bs] as packed uint8
@group(0) @binding(1) var<storage, read> Scales: array<u32>; // [V, n_groups] fp16 packed
@group(0) @binding(2) var<storage, read> Indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let K = _params_[1];      // embedding dim
    let n_groups = _params_[2];
    let bs = _params_[3];     // block_size

    let idx_i = gid.y;  // which index
    let k = gid.x;      // which element in the embedding
    if (idx_i >= nIdx || k >= K) { return; }

    let vocab_idx = u32(Indices[idx_i]);
    let group = k / bs;
    let elem = k % bs;

    // Read scale
    let scale_flat = vocab_idx * n_groups + group;
    let scale_u32 = Scales[scale_flat / 2u];
    let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
    let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;

    // Read uint8 value
    let byte_flat = vocab_idx * n_groups * bs + group * bs + elem;
    let byte_u32 = W[byte_flat / 4u];
    let byte_val = (byte_u32 >> ((byte_flat % 4u) * 8u)) & 0xFFu;
    let centered = f32(i32(byte_val) - 128);

    Y[idx_i * K + k] = centered * scale;
}
)WGSL";

static void opGatherBlockQuantized(GraphExecutor& ex, const OnnxGraphNode& n,
                                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* W = in[0];       // quantized embedding weights
    auto* Indices = in[1]; // token indices
    auto* Scales = in.size() > 2 ? in[2] : nullptr;
    if (!W || !Indices || !W->IsValid() || !Indices->IsValid()) return;

    int64_t nIdx = 1;
    for (auto d : Indices->shape) nIdx *= d;

    // Compute K from node attributes and weight dims
    int64_t bits = n.GetInt("bits", 8);
    int64_t block_size_attr = n.GetInt("block_size", 32);
    // For Q8 embedding: weight is [V, n_groups, block_size] raw uint8
    // K = n_groups * block_size = hidden_dim
    uint32_t n_groups, bs, K;
    if (W->shape.size() >= 3) {
        n_groups = (uint32_t)W->shape[1];
        bs = (uint32_t)W->shape[2];
        K = n_groups * bs;
    } else if (W->shape.size() == 2) {
        // Weight was reshaped to [V, hidden_dim]
        K = (uint32_t)W->shape[1];
        bs = (uint32_t)block_size_attr;
        n_groups = K / bs;
    } else {
        K = 1; bs = 1; n_groups = 1;
    }

    auto outShape = Indices->shape;
    outShape.push_back(K);
    *out[0] = ex.AllocTensor(outShape, TensorDtype::Float32);

    if (!Scales || !Scales->IsValid()) return;

    uint32_t params[4] = {(uint32_t)nIdx, K, n_groups, bs};
    auto paramBuf = ex.gpu->createBuffer("gbq_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    auto& pl = ex.GetPipeline("gather_bq", WGSL_GATHER_BQ, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, W->buffer}, {1, Scales->buffer}, {2, Indices->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (K + 255) / 256, (uint32_t)nIdx, 1, "gather_bq"});
}

REGISTER_OP(MatMul, opMatMul)
REGISTER_OP(MatMulNBits, opMatMulNBits)
REGISTER_OP(Gemm, opGemm)
REGISTER_OP(GatherBlockQuantized, opGatherBlockQuantized)

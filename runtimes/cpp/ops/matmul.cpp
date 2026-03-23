/**
 * ops/matmul.cpp — Matrix multiplication ops with optimized GPU kernels.
 *
 * MatMulNBits: Q4 quantized matmul (ORT-style tiled kernel)
 * MatMul: fp32 matmul (tiled)
 * Gemm: matmul + bias
 * GatherBlockQuantized: quantized embedding
 */

#include "../graph_executor.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// ─── MatMulNBits: Optimized Q4 matmul (ORT-style tiled) ─────────────────────
//
// Approach (inspired by ORT WebGPU):
//   - tile_size = 8: each workgroup computes 8 output columns
//   - K is divided into blocks of 32 (= Q4 block_size)
//   - Shared memory: tile A input into workgroup-shared array
//   - Each thread processes: K_chunk × 1 output column
//   - vec4 dot products: unpack4xU8 + dot for 8 Q4 values at once
//   - Inter-results: partial sums in shared memory, reduced at end
//
// Dispatch: (ceil(N/8), M, 1)
// Workgroup: 128 threads = 16 K-threads × 8 N-threads

static const char* WGSL_MATMUL_Q4_TILED = R"WGSL(
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_K_VEC: u32 = 16u;
const WG_SIZE: u32 = 128u;  // TILE_N * TILE_K_VEC

var<workgroup> tile_A: array<vec4<f32>, TILE_K_VEC>;
var<workgroup> inter_results: array<array<f32, TILE_K_VEC>, TILE_N>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let row = wid.y;
    let n_tile = wid.x;
    let local_id = lid.x;
    let k_idx = local_id % TILE_K_VEC;
    let n_idx = local_id / TILE_K_VEC;

    // Clear inter_results (all threads participate uniformly)
    inter_results[n_idx][k_idx] = 0.0;
    workgroupBarrier();

    let a_base = row * (K / 4u);
    let b_col = n_tile * TILE_N + n_idx;

    for (var k_start = 0u; k_start < K; k_start += TILE_K_VEC * 8u) {
        // Load A tile (uniform: all threads participate)
        if (local_id < TILE_K_VEC) {
            let a_offset = (k_start / 4u) + local_id * 2u;
            if (row < M && a_offset < K / 4u) {
                tile_A[local_id] = A[a_base + a_offset];
            } else {
                tile_A[local_id] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();

        // Compute (guard with conditionals, no early return)
        let k_elem = k_start + k_idx * 8u;
        if (b_col < N && k_elem < K && row < M) {
            let b_offset = b_col * blocks_per_col * 4u + k_elem / 8u;
            let b_packed = B[b_offset];

            let block_idx = k_elem / 32u;
            let scale_flat = b_col * blocks_per_col + block_idx;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let lo = unpack4xU8(b_packed & 0x0F0F0F0Fu);
            let hi = unpack4xU8((b_packed >> 4u) & 0x0F0F0F0Fu);
            let b0 = vec4<f32>(f32(lo[0]) - 8.0, f32(hi[0]) - 8.0,
                               f32(lo[1]) - 8.0, f32(hi[1]) - 8.0) * scale;
            let b1 = vec4<f32>(f32(lo[2]) - 8.0, f32(hi[2]) - 8.0,
                               f32(lo[3]) - 8.0, f32(hi[3]) - 8.0) * scale;

            let a_local_offset = k_idx * 2u;
            var sum: f32 = 0.0;
            if (a_local_offset < TILE_K_VEC) {
                sum += dot(tile_A[a_local_offset], b0);
            }
            if (a_local_offset + 1u < TILE_K_VEC) {
                sum += dot(tile_A[a_local_offset + 1u], b1);
            }
            inter_results[n_idx][k_idx] += sum;
        }
        workgroupBarrier();
    }

    // Reduce inter_results across K dimension
    if (k_idx == 0u && b_col < N && row < M) {
        var total: f32 = 0.0;
        for (var k = 0u; k < TILE_K_VEC; k++) {
            total += inter_results[n_idx][k];
        }
        Y[row * N + b_col] = total;
    }
}
)WGSL";

static void opMatMulNBits(GraphExecutor& ex, const OnnxGraphNode& n,
                           const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* X = in[0];     // [batch*M, K]
    auto* W = in[1];     // [N, n_blocks, block_half] uint8
    auto* S = in[2];     // [N * n_blocks] fp16 or [N, n_blocks]
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

    auto& pl = ex.GetPipeline("matmul_q4_tiled", WGSL_MATMUL_Q4_TILED, 5);
    auto bg = ex.MakeBindGroup(pl, {
        {0, X->buffer}, {1, W->buffer}, {2, S->buffer},
        {3, out[0]->buffer}, {4, paramBuf}});

    // Dispatch: (ceil(N/8), M, 1) — each workgroup handles 8 output columns
    ex.pendingDispatches_.push_back({pl.pipeline, bg,
        (N + 7) / 8, (uint32_t)M, 1, "matmul_q4"});
}

// ─── MatMul: fp32 matrix multiply (tiled) ────────────────────────────────────

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

// ─── Gemm: Y = alpha*A*B + beta*C ───────────────────────────────────────────

static const char* WGSL_GEMM = R"WGSL(
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];
    let row = gid.y;
    let col = gid.x;
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
// Supports both Q8 (bits=8, zp=128) and Q4 (bits=4, zp=8)

static const char* WGSL_GATHER_BQ_Q8 = R"WGSL(
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let K = _params_[1];
    let n_groups = _params_[2];
    let bs = _params_[3];
    let idx_i = gid.y;
    let k = gid.x;
    if (idx_i >= nIdx || k >= K) { return; }
    let vocab_idx = u32(Indices[idx_i]);
    let group = k / bs;
    let scale_flat = vocab_idx * n_groups + group;
    let scale_u32 = Scales[scale_flat / 2u];
    let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
    let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;
    let byte_flat = vocab_idx * n_groups * bs + k;
    let byte_u32 = W[byte_flat / 4u];
    let byte_val = (byte_u32 >> ((byte_flat % 4u) * 8u)) & 0xFFu;
    let centered = f32(i32(byte_val) - 128);
    Y[idx_i * K + k] = centered * scale;
}
)WGSL";

static const char* WGSL_GATHER_BQ_Q4 = R"WGSL(
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let K = _params_[1];       // output elements per row
    let n_groups = _params_[2]; // K / block_size
    let bs = _params_[3];      // block_size
    let idx_i = gid.y;
    let k = gid.x;
    if (idx_i >= nIdx || k >= K) { return; }
    let vocab_idx = u32(Indices[idx_i]);
    let group = k / bs;
    let scale_flat = vocab_idx * n_groups + group;
    let scale_u32 = Scales[scale_flat / 2u];
    let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
    let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;
    // Q4: 2 elements per byte. Element k is at byte k/2, nibble k%2
    let byte_flat = vocab_idx * (K / 2u) + k / 2u;
    let byte_u32 = W[byte_flat / 4u];
    let byte_val = (byte_u32 >> ((byte_flat % 4u) * 8u)) & 0xFFu;
    let nibble = select(byte_val & 0x0Fu, (byte_val >> 4u) & 0x0Fu, (k & 1u) != 0u);
    let centered = f32(i32(nibble)) - 8.0;
    Y[idx_i * K + k] = centered * scale;
}
)WGSL";

static void opGatherBlockQuantized(GraphExecutor& ex, const OnnxGraphNode& n,
                                    const std::vector<GpuTensor*>& in, std::vector<GpuTensor*>& out) {
    auto* W = in[0]; auto* Indices = in[1];
    auto* Scales = in.size() > 2 ? in[2] : nullptr;
    if (!W || !Indices || !W->IsValid() || !Indices->IsValid()) return;
    ex.EnsureGpu(*W); ex.EnsureGpu(*Indices);

    int64_t nIdx = 1;
    for (auto d : Indices->shape) nIdx *= d;

    int64_t bits = n.GetInt("bits", 8);
    int64_t block_size_attr = n.GetInt("block_size", 32);
    uint32_t n_groups, bs, K;

    if (bits == 4) {
        // Q4: weight dims [V, K] where K is the full output dim
        // Raw bytes = V * K / 2
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
        // Read from CPU if possible, convert to int32
        const uint8_t* idxPtr = nullptr;
        if (Indices->isCpuOnly && !Indices->cpuData.empty())
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
        } else {
            // GPU int64 tensor — need readback or GPU cast kernel
            // For now, flush and readback
            if (!ex.pendingDispatches_.empty()) {
                ex.gpu->submitOnly(ex.pendingDispatches_, false);
                ex.gpu->waitForQueue();
                ex.pendingDispatches_.clear();
            }
            auto rb = ex.gpu->readBuffer(Indices->buffer, nIdx * 8);
            std::vector<int32_t> i32(nIdx);
            const int64_t* src = (const int64_t*)rb.data();
            for (int64_t i = 0; i < nIdx; i++) i32[i] = (int32_t)src[i];
            idxBuf = ex.gpu->createBuffer("gbq_idx32", nIdx * 4);
            ex.gpu->writeBuffer(idxBuf, i32.data(), nIdx * 4);
        }
    }

    uint32_t params[4] = {(uint32_t)nIdx, K, n_groups, bs};
    auto paramBuf = ex.gpu->createBuffer("gbq_p", 16);
    ex.gpu->writeBuffer(paramBuf, params, 16);

    const char* kernel = (bits == 4) ? WGSL_GATHER_BQ_Q4 : WGSL_GATHER_BQ_Q8;
    const char* plName = (bits == 4) ? "gather_bq_q4" : "gather_bq_q8";
    auto& pl = ex.GetPipeline(plName, kernel, 5);
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

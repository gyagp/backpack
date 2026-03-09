#pragma once
/**
 * wgsl_shaders.h -- Auto-generated from compiler/kernels/*.wgsl
 * Do not edit manually. Regenerate with: python compiler/gen_wgsl_shaders.py
 */

#include <string>
#include <unordered_map>

struct ShaderInfo {
    const char* source;
    uint32_t numBindings;
    bool isTritonGenerated;  // true = Triton-compiled, false = hand-written WGSL
};

// [gguf_q8] q8_down_silu_add (6 bindings, hand)
static const char* WGSL_Q8_DOWN_SILU_ADD = R"WGSL(
enable subgroups;

// Fused: SiLU·mul + Q8_0 matmul + residual add.
// Reads gateUpBuf (2*IM elements), applies silu(gate)*up on-the-fly,
// then multiplies by W_down and adds to residual Y.
// Eliminates the separate silu_mul dispatch entirely.
//
// Input X layout: [gate_0..gate_{IM-1}, up_0..up_{IM-1}]
// Effective input after silu: silu(gate_i) * up_i for i in [0, IM)
//
// Bindings:
//   0: GateUp (read) — 2*IM floats (gate || up concatenated)
//   1: W_Q8 (read) — quantized weight matrix (N × K/4 u32)
//   2: Scales (read) — fp16 scales packed as u32
//   3: Bias (read) — per-output bias (zeros for no bias)
//   4: Y (read_write) — output += matmul result (residual add)
//   5: _params_ — [K=IM, N=E, IM, 0]

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const STRIDE: u32 = 256u;
const MAX_STRIDES: u32 = 24u;  // ceil(6144 / 256)

var<workgroup> smem_x: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];   // IM (intermediate_size)
    let N = _params_[1];   // E (n_embd)
    let IM = _params_[2];  // same as K, used for up offset

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let valid = col < N;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    var acc: f32 = 0.0;

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * STRIDE;
        let in_range = k_off < K;

        // Load gate and up values, apply silu·mul, store to shared memory
        if (in_range) {
            let idx = k_off + tid;
            let gate = GateUp[row * 2u * IM + idx];         // gate[idx]
            let up   = GateUp[row * 2u * IM + IM + idx];    // up[idx]
            // silu(gate) * up = gate / (1 + exp(-gate)) * up
            let silu_gate = gate / (1.0 + exp(-gate));
            smem_x[tid] = silu_gate * up;
        }
        workgroupBarrier();

        if (valid && in_range) {
            let k_base = lane * 8u;
            let xv0 = vec4<f32>(smem_x[k_base], smem_x[k_base+1u],
                                smem_x[k_base+2u], smem_x[k_base+3u]);
            let xv1 = vec4<f32>(smem_x[k_base+4u], smem_x[k_base+5u],
                                smem_x[k_base+6u], smem_x[k_base+7u]);

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_down_silu_add_batched (6 bindings, hand)
static const char* WGSL_Q8_DOWN_SILU_ADD_BATCHED = R"WGSL(
enable subgroups;

// Batched fused: SiLU·mul + Q8_0 down-proj + residual add for prefill.
// Y[T×N] += (silu(gate) * up) × W_down^T
// GateUp layout: [T rows × 2*IM cols] where gate=[0..IM), up=[IM..2*IM)
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 24u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];   // IM
    let N = _params_[1];   // E
    let IM = _params_[2];  // same as K
    let T = _params_[3];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    var acc: array<f32, 8>;
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        var wv0: vec4<f32>;
        var wv1: vec4<f32>;
        var scale0: f32 = 0.0;
        var scale1: f32 = 0.0;

        if (valid && in_k) {
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
            wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);
        }

        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                // Apply silu·mul on-the-fly: silu(gate[k]) * up[k]
                let gu_base = row * 2u * IM;
                var sv0: vec4<f32>;
                var sv1: vec4<f32>;
                for (var e = 0u; e < 4u; e++) {
                    let idx = k_base + e;
                    let gate = GateUp[gu_base + idx];
                    let up   = GateUp[gu_base + IM + idx];
                    sv0[e] = gate / (1.0 + exp(-gate)) * up;
                }
                for (var e = 0u; e < 4u; e++) {
                    let idx = k_base + 4u + e;
                    let gate = GateUp[gu_base + idx];
                    let up   = GateUp[gu_base + IM + idx];
                    sv1[e] = gate / (1.0 + exp(-gate)) * up;
                }
                acc[m] += dot(sv0, wv0) * scale0 + dot(sv1, wv1) * scale1;
            }
        }
    }

    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] += warp_sum + Bias[col];
        }
    }
}
)WGSL";

// [gguf_q8] q8_down_silu_add_mma (6 bindings, hand)
static const char* WGSL_Q8_DOWN_SILU_ADD_MMA = R"WGSL(
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// Double-buffered MMA fused SiLU+down+residual:
//   Y[M×N] += silu_mul(GateUp[M×2*IM]) × W[N×K]^T + Bias[N]
//
// Same double-buffer + TILE_K=32 as q8_matmul_mma but with SiLU
// activation computed during tile_A staging.
//
// params: [K=IM, N=E, IM, M=T]
// Grid: (ceil(N/32), ceil(M/32))

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { K: u32, N: u32, IM: u32, M: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const MK: u32 = 16u;
const SB: u32 = 32u;
const WG: u32 = 512u;
const AB_A: u32 = 2048u;
const AB_B: u32 = 2048u;

var<workgroup> tA: array<array<f16, 2048>, 2>;
var<workgroup> tB: array<array<f16, 2048>, 2>;
var<workgroup> tC: array<f32, 4096>;

fn load_silu(gu: ptr<storage, array<f32>, read_write>, row: u32, k: u32, IM: u32) -> f16 {
    let base = row * 2u * IM;
    let gate = (*gu)[base + k];
    let up = (*gu)[base + IM + k];
    return f16(gate / (1.0 + exp(-gate)) * up);
}

@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let K = params.K;  let N = params.N;
    let IM = params.IM;  let M = params.M;
    let lx = lid.x;
    let rb = wid.y * TM;
    let cb = wid.x * TN;
    let sr = sg_id & 3u;
    let sc = sg_id >> 2u;
    let ws = K / 4u;
    let nkt = K / TK;

    var mC: subgroup_matrix_result<f32, 16, 16>;

    // Prefetch tile 0
    for (var i = lx; i < AB_A; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gr = rb + r;
        tA[0][i] = select(0.0h, load_silu(&GateUp, gr, c, IM), gr < M);
    }
    for (var i = lx; i < AB_B; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gc = cb + r;
        if (gc < N) {
            let pk = W_Q8[gc * ws + c / 4u];
            let q = f32(extractBits(i32(pk), (c & 3u) * 8u, 8u));
            let si = gc * (K / SB) + c / SB;
            let sp = unpack2x16float(Scales[si / 2u]);
            tB[0][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
        } else { tB[0][i] = 0.0h; }
    }
    workgroupBarrier();

    for (var ti = 0u; ti < nkt; ti++) {
        let cur = ti & 1u;
        let nxt = 1u - cur;
        let kn = (ti + 1u) * TK;

        if (ti + 1u < nkt) {
            for (var i = lx; i < AB_A; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gr = rb + r;
                tA[nxt][i] = select(0.0h, load_silu(&GateUp, gr, kn + c, IM), gr < M);
            }
            for (var i = lx; i < AB_B; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gc = cb + r;  let gk = kn + c;
                if (gc < N) {
                    let pk = W_Q8[gc * ws + gk / 4u];
                    let q = f32(extractBits(i32(pk), (gk & 3u) * 8u, 8u));
                    let si = gc * (K / SB) + gk / SB;
                    let sp = unpack2x16float(Scales[si / 2u]);
                    tB[nxt][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
                } else { tB[nxt][i] = 0.0h; }
            }
        }

        let ao = sr * 16u * TK;
        let bo = sc * 16u * TK;
        let mA0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&tA[cur], ao, false, TK);
        let mB0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&tB[cur], bo, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA0, mB0, mC);

        let mA1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&tA[cur], ao + MK, false, TK);
        let mB1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&tB[cur], bo + MK, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA1, mB1, mC);

        workgroupBarrier();
    }

    subgroupMatrixStore(&tC, sr * 16u * TN + sc * 16u, mC, false, TN);
    workgroupBarrier();

    for (var i = lx; i < TM * TN; i += WG) {
        let r = i / TN;  let c = i % TN;
        let gr = rb + r;  let gc = cb + c;
        if (gr < M && gc < N) {
            Y[gr * N + gc] += tC[i] + Bias[gc];
        }
    }
}
)WGSL";

// [gguf_q8] q8_down_silu_add_tiled (6 bindings, hand)
static const char* WGSL_Q8_DOWN_SILU_ADD_TILED = R"WGSL(
enable subgroups;

// Tiled fused: SiLU·mul + Q8_0 down-proj + residual add for prefill.
//   Y[T×N] += silu_mul(GateUp[T×2*IM]) × W_down[N×K]^T
//
// Tile: TILE_M(8) rows × TILE_N(32) columns per workgroup.
// 256 threads = 8 warps, each warp processes 4 output columns.
//
// Shared memory caches the SiLU-activated input: silu(gate) * up.
// This is computed once cooperatively per K-block, then reused
// across all 32 output columns — eliminates redundant SiLU eval.
//
// GateUp layout: [T × 2*IM], gate=[0..IM), up=[IM..2*IM)
// params: [K=IM, N=E, IM, T]
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK_LOAD: u32 = 256u;
const MAX_ITERS: u32 = 24u;  // ceil(6144/256)

// Shared memory: SiLU-activated values [TILE_M × BK_LOAD]
var<workgroup> smem_s: array<f32, 2048>;  // 8 × 256

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];   // IM
    let N = _params_[1];   // E
    let IM = _params_[2];  // same as K
    let T = _params_[3];

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    var acc: array<array<f32, 4>, 8>;
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) {
            acc[m][c] = 0.0;
        }
    }

    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    var w_bases: array<u32, 4>;
    var s_bases: array<u32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = tile_col * TILE_N + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
        w_bases[c] = select(0u, cols[c] * stride_w, col_valid[c]);
        s_bases[c] = select(0u, cols[c] * n_blocks, col_valid[c]);
    }

    for (var g = 0u; g < MAX_ITERS; g = g + 1u) {
        let k_off = g * BK_LOAD;
        let in_k = k_off < K;

        // ── Cooperative load: compute SiLU·mul into smem ────────────
        // 256 threads, 8 rows × 256 K elements per iteration
        for (var r = 0u; r < TILE_M; r++) {
            let row = tile_row * TILE_M + r;
            let smem_idx = r * BK_LOAD + tid;
            if (in_k && row < T) {
                let idx = k_off + tid;
                let gu_base = row * 2u * IM;
                let gate = GateUp[gu_base + idx];
                let up   = GateUp[gu_base + IM + idx];
                smem_s[smem_idx] = gate / (1.0 + exp(-gate)) * up;
            } else {
                smem_s[smem_idx] = 0.0;
            }
        }
        workgroupBarrier();

        // ── Compute: dot SiLU-activated rows (smem) with W (global) ─
        if (in_k) {
            let k_base = lane * 8u;

            for (var c = 0u; c < COLS_PER_WARP; c++) {
                var wv0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var wv1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var scale0: f32 = 0.0;
                var scale1: f32 = 0.0;

                if (col_valid[c]) {
                    let w_off = w_bases[c] + g * 64u + lane * 2u;
                    let pw0 = W_Q8[w_off];
                    let pw1 = W_Q8[w_off + 1u];

                    wv0 = vec4<f32>(
                        f32(extractBits(i32(pw0), 0u, 8u)),
                        f32(extractBits(i32(pw0), 8u, 8u)),
                        f32(extractBits(i32(pw0), 16u, 8u)),
                        f32(extractBits(i32(pw0), 24u, 8u)));
                    wv1 = vec4<f32>(
                        f32(extractBits(i32(pw1), 0u, 8u)),
                        f32(extractBits(i32(pw1), 8u, 8u)),
                        f32(extractBits(i32(pw1), 16u, 8u)),
                        f32(extractBits(i32(pw1), 24u, 8u)));

                    let block0 = g * 8u + (lane * 8u) / 32u;
                    let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                    let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                    scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                    let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                    scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
                }

                for (var m = 0u; m < TILE_M; m++) {
                    let smem_base = m * BK_LOAD + k_base;
                    let sv0 = vec4<f32>(smem_s[smem_base],
                                        smem_s[smem_base + 1u],
                                        smem_s[smem_base + 2u],
                                        smem_s[smem_base + 3u]);
                    let sv1 = vec4<f32>(smem_s[smem_base + 4u],
                                        smem_s[smem_base + 5u],
                                        smem_s[smem_base + 6u],
                                        smem_s[smem_base + 7u]);
                    acc[m][c] += dot(sv0, wv0) * scale0 + dot(sv1, wv1) * scale1;
                }
            }
        }

        workgroupBarrier();
    }

    // Reduce + residual add
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < T) {
                Y[row * N + cols[c]] += warp_sum + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [gguf_q8] q8_matmul (6 bindings, hand)
static const char* WGSL_Q8_MATMUL = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // K_PER_ITER=8: each lane processes 8 elements per 256-element stride
    // (32 lanes × 8 elem/lane = 256 elements = 8 Q8_0 blocks per stride)
    let n_strides = K / 256u;
    let stride_w = K / 4u;  // u32 per weight row

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations (two vec4 loads)
            let k_base = g * 256u + lane * 8u;
            let xv0 = vec4<f32>(X[x_base + k_base],
                                X[x_base + k_base + 1u],
                                X[x_base + k_base + 2u],
                                X[x_base + k_base + 3u]);
            let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                X[x_base + k_base + 5u],
                                X[x_base + k_base + 6u],
                                X[x_base + k_base + 7u]);

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 and form vec4 for dot product
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales: 2 blocks per 8 elements
            // block_idx = (g * 256 + lane * 8) / 32
            let bi = g * 8u + lane / 4u;
            let si0 = s_base + bi;
            let si1 = si0 + 1u;
            // Correction: lane * 8 spans TWO 32-element blocks when lane >= 4
            // Block for first 4 elements: (g*256 + lane*8) / 32 = g*8 + lane/4
            // Block for second 4 elements: (g*256 + lane*8 + 4) / 32
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            // vec4 dot products with per-block scaling
            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_add (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_ADD = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // K_PER_ITER=8: each lane processes 8 elements per 256-element stride
    // (32 lanes × 8 elem/lane = 256 elements = 8 Q8_0 blocks per stride)
    let n_strides = K / 256u;
    let stride_w = K / 4u;  // u32 per weight row

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations (two vec4 loads)
            let k_base = g * 256u + lane * 8u;
            let xv0 = vec4<f32>(X[x_base + k_base],
                                X[x_base + k_base + 1u],
                                X[x_base + k_base + 2u],
                                X[x_base + k_base + 3u]);
            let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                X[x_base + k_base + 5u],
                                X[x_base + k_base + 6u],
                                X[x_base + k_base + 7u]);

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 and form vec4 for dot product
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales: 2 blocks per 8 elements
            // block_idx = (g * 256 + lane * 8) / 32
            let bi = g * 8u + lane / 4u;
            let si0 = s_base + bi;
            let si1 = si0 + 1u;
            // Correction: lane * 8 spans TWO 32-element blocks when lane >= 4
            // Block for first 4 elements: (g*256 + lane*8) / 32 = g*8 + lane/4
            // Block for second 4 elements: (g*256 + lane*8 + 4) / 32
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            // vec4 dot products with per-block scaling
            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_add_batched (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_ADD_BATCHED = R"WGSL(
enable subgroups;

// Batched Q8_0 matmul + residual add for prefill.
// Y[T×N] += X[T×K] × W[K×N]^T + Bias
// Same as q8_matmul_batched but adds to output (residual connection).

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 24u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    var acc: array<f32, 8>;
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        var wv0: vec4<f32>;
        var wv1: vec4<f32>;
        var scale0: f32 = 0.0;
        var scale1: f32 = 0.0;

        if (valid && in_k) {
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
            wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);
        }

        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                let x_base = row * K;
                let xv0 = vec4<f32>(X[x_base + k_base],
                                    X[x_base + k_base + 1u],
                                    X[x_base + k_base + 2u],
                                    X[x_base + k_base + 3u]);
                let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                    X[x_base + k_base + 5u],
                                    X[x_base + k_base + 6u],
                                    X[x_base + k_base + 7u]);
                acc[m] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
            }
        }
    }

    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] += warp_sum + Bias[col];
        }
    }
}
)WGSL";

// [gguf_q8] q8_matmul_add_fast (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_ADD_FAST = R"WGSL(
enable subgroups;
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;
const TILE_N: u32 = 8u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x; let tile_col = wid.y; let tid = lid.x;
    let K = _params_[0]; let N = _params_[1]; let x_base = row * K;
    let warp_id = tid/32u; let lane = tid%32u;
    let col = tile_col * TILE_N + warp_id;
    let n_strides = K / 512u; let stride_w = K / 4u;
    var acc: f32 = 0.0;
    if (col < N) {
        let w_base = col * stride_w; let n_blocks = K / 32u; let s_base = col * n_blocks;
        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g*512u + lane*16u;
            let xv0 = vec4<f32>(X[x_base+k_base],X[x_base+k_base+1u],X[x_base+k_base+2u],X[x_base+k_base+3u]);
            let xv1 = vec4<f32>(X[x_base+k_base+4u],X[x_base+k_base+5u],X[x_base+k_base+6u],X[x_base+k_base+7u]);
            let xv2 = vec4<f32>(X[x_base+k_base+8u],X[x_base+k_base+9u],X[x_base+k_base+10u],X[x_base+k_base+11u]);
            let xv3 = vec4<f32>(X[x_base+k_base+12u],X[x_base+k_base+13u],X[x_base+k_base+14u],X[x_base+k_base+15u]);
            let w_off = w_base + g*128u + lane*4u;
            let pw0=W_Q8[w_off]; let pw1=W_Q8[w_off+1u]; let pw2=W_Q8[w_off+2u]; let pw3=W_Q8[w_off+3u];
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));
            let wv2 = vec4<f32>(f32(extractBits(i32(pw2),0u,8u)),f32(extractBits(i32(pw2),8u,8u)),f32(extractBits(i32(pw2),16u,8u)),f32(extractBits(i32(pw2),24u,8u)));
            let wv3 = vec4<f32>(f32(extractBits(i32(pw3),0u,8u)),f32(extractBits(i32(pw3),8u,8u)),f32(extractBits(i32(pw3),16u,8u)),f32(extractBits(i32(pw3),24u,8u)));
            let b0=g*16u+(lane*16u)/32u; let b1=g*16u+(lane*16u+4u)/32u;
            let b2=g*16u+(lane*16u+8u)/32u; let b3=g*16u+(lane*16u+12u)/32u;
            let sp0=unpack2x16float(Scales[(s_base+b0)/2u]); let sc0=select(sp0.x,sp0.y,((s_base+b0)&1u)!=0u);
            let sp1=unpack2x16float(Scales[(s_base+b1)/2u]); let sc1=select(sp1.x,sp1.y,((s_base+b1)&1u)!=0u);
            let sp2=unpack2x16float(Scales[(s_base+b2)/2u]); let sc2=select(sp2.x,sp2.y,((s_base+b2)&1u)!=0u);
            let sp3=unpack2x16float(Scales[(s_base+b3)/2u]); let sc3=select(sp3.x,sp3.y,((s_base+b3)&1u)!=0u);
            acc += dot(xv0,wv0)*sc0+dot(xv1,wv1)*sc1+dot(xv2,wv2)*sc2+dot(xv3,wv3)*sc3;
        }
    }
    let warp_sum = subgroupAdd(acc);
    if (lane==0u && col<N) { Y[row*N+col] += warp_sum + Bias[col]; }
}
)WGSL";

// [gguf_q8] q8_matmul_add_lite (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_ADD_LITE = R"WGSL(
enable subgroups;

// Q8_0 matmul + residual add with single output per workgroup (TILE_N=1, WG=32).
// Same as q8_matmul_lite but adds the result to Y (residual connection).

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let col = wid.y;
    let lane = lid.x;

    let K = _params_[0];
    let N = _params_[1];

    let valid = col < N;

    let x_base = row * K;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    let n_strides = K / 256u;

    var acc: f32 = 0.0;

    for (var g = 0u; g < n_strides; g = g + 1u) {
        if (valid) {
        let k_base = g * 256u + lane * 8u;
        let xv0 = vec4<f32>(X[x_base + k_base],
                            X[x_base + k_base + 1u],
                            X[x_base + k_base + 2u],
                            X[x_base + k_base + 3u]);
        let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                            X[x_base + k_base + 5u],
                            X[x_base + k_base + 6u],
                            X[x_base + k_base + 7u]);

        let w_off = w_base + g * 64u + lane * 2u;
        let pw0 = W_Q8[w_off];
        let pw1 = W_Q8[w_off + 1u];

        let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
        let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

        let block0 = g * 8u + (lane * 8u) / 32u;
        let block1 = g * 8u + (lane * 8u + 4u) / 32u;
        let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
        let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
        let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
        let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

        acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && valid) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_add_smem (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_ADD_SMEM = R"WGSL(
enable subgroups;

// Q8_0 matmul + residual add with shared-memory X caching.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const STRIDE: u32 = 256u;
const MAX_STRIDES: u32 = 24u;

var<workgroup> smem_x: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;
    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let valid = col < N;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    var acc: f32 = 0.0;

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * STRIDE;
        let in_range = k_off < K;

        if (in_range) {
            smem_x[tid] = X[x_base + k_off + tid];
        }
        workgroupBarrier();

        if (valid && in_range) {
            let k_base = lane * 8u;
            let xv0 = vec4<f32>(smem_x[k_base], smem_x[k_base+1u],
                                smem_x[k_base+2u], smem_x[k_base+3u]);
            let xv1 = vec4<f32>(smem_x[k_base+4u], smem_x[k_base+5u],
                                smem_x[k_base+6u], smem_x[k_base+7u]);

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_batched (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_BATCHED = R"WGSL(
enable subgroups;

// Batched Q8_0 matmul for prefill: Y[T×N] = X[T×K] × W[K×N]^T
// Each workgroup computes a TILE_M × TILE_N output tile.
// Weight row is Q8_0: read once, reused across all T (M) rows.
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)
// WG: 256 threads = 8 warps
//
// Tile: TILE_M=8 rows × TILE_N=8 cols, K iterated in blocks of 32
// Each warp handles one output column, all TILE_M rows.
//
// Bindings:
//   0: X — fp32 activations [T × K]
//   1: W_Q8 — int8 weights packed as u32 [N × K/4]
//   2: Scales — fp16 weight scales packed as u32
//   3: Bias — per-output bias [N]
//   4: Y — output [T × N]
//   5: _params_ — [K, N, T, 0]

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 24u;  // ceil(6144/256)

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;   // which TILE_M block of rows
    let tile_col = wid.y;   // which TILE_N block of columns
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    // Each warp accumulates TILE_M rows for its output column
    var acc: array<f32, 8>;  // TILE_M accumulators
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        // Read 2 packed u32 weights (8 int8 values) — same for all rows
        var wv0: vec4<f32>;
        var wv1: vec4<f32>;
        var scale0: f32 = 0.0;
        var scale1: f32 = 0.0;

        if (valid && in_k) {
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
            wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);
        }

        // For each of TILE_M rows, read X and dot with same weights
        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                let x_base = row * K;
                let xv0 = vec4<f32>(X[x_base + k_base],
                                    X[x_base + k_base + 1u],
                                    X[x_base + k_base + 2u],
                                    X[x_base + k_base + 3u]);
                let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                    X[x_base + k_base + 5u],
                                    X[x_base + k_base + 6u],
                                    X[x_base + k_base + 7u]);
                acc[m] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
            }
        }
    }

    // Reduce across lanes within each warp
    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] = warp_sum + Bias[col];
        }
    }
}
)WGSL";

// [gguf_q8] q8_matmul_dp4a (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Q8_0 matmul using DP4A (dot4I8Packed) for integer dot products.
// W8A32: INT8 weights × FP32 activations.
//
// Strategy: quantize the fp32 activation vector to int8 per-block
// (same blocking as Q8_0 weights: blocks of 32), then use dp4a
// for the integer dot product, and rescale with both scales.
//
// Each lane processes 8 elements (2 dp4a calls per iteration).

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const MAX_STRIDES: u32 = 8u;   // ceil(2048 / 256) — supports up to K=2048

var<workgroup> smem_x_q: array<u32, 64>;   // quantized X: 256 int8 = 64 u32
var<workgroup> smem_x_s: array<f32, 8>;    // per-block scales for X: 256/32 = 8

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;

    var acc: f32 = 0.0;
    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let n_blocks = K / 32u;
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * 256u;
        let in_range = k_off < K;

        // ── Phase 1: Cooperatively quantize X to int8 ──────────────────
        {
            let block_id = tid / 32u;  // 0..7
            let elem_in_block = tid % 32u;
            let x_val = select(0.0, X[x_base + k_off + tid], in_range);
            let abs_val = abs(x_val);

            // Reduce absmax within warp
            var max_val = abs_val;
            max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

            let x_scale = max_val / 127.0;
            if (elem_in_block == 0u) {
                smem_x_s[block_id] = x_scale;
            }

            // Quantize this element to int8
            let safe_scale = select(1.0, x_scale, x_scale != 0.0);
            let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

            // Pack 4 int8 values into u32 using subgroup shuffle
            let pack_lane = elem_in_block % 4u;
            let pack_group = elem_in_block / 4u;
            let byte_val = u32(q_val & 0xFF);
            let shifted = byte_val << (pack_lane * 8u);

            var packed = shifted;
            packed = packed | subgroupShuffleXor(packed, 1u);
            packed = packed | subgroupShuffleXor(packed, 2u);

            if (pack_lane == 0u) {
                smem_x_q[block_id * 8u + pack_group] = packed;
            }
        }
        workgroupBarrier();

        // ── Phase 2: DP4A matmul using quantized X ─────────────────────
        if (valid && in_range) {
            let k_base_in_stride = lane * 8u;
            let xq_off = k_base_in_stride / 4u;

            let xq0 = smem_x_q[xq_off];
            let xq1 = smem_x_q[xq_off + 1u];

            let w_off = w_base + g * 64u + lane * 2u;
            let wq0 = W_Q8[w_off];
            let wq1 = W_Q8[w_off + 1u];

            // DP4A: dot product of 4 packed int8 values
            let idot0 = dot4I8Packed(xq0, wq0);
            let idot1 = dot4I8Packed(xq1, wq1);

            // Rescale: result = int_sum * x_scale * w_scale
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;

            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let w_scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let w_scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            let x_block0 = k_base_in_stride / 32u;
            let x_block1 = (k_base_in_stride + 4u) / 32u;
            let x_scale0 = smem_x_s[x_block0];
            let x_scale1 = smem_x_s[x_block1];

            acc += f32(idot0) * w_scale0 * x_scale0
                 + f32(idot1) * w_scale1 * x_scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_fast (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_FAST = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;
    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let n_strides = K / 512u;
    let stride_w = K / 4u;
    var acc: f32 = 0.0;
    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;
        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g * 512u + lane * 16u;
            let xv0 = vec4<f32>(X[x_base+k_base], X[x_base+k_base+1u],
                                X[x_base+k_base+2u], X[x_base+k_base+3u]);
            let xv1 = vec4<f32>(X[x_base+k_base+4u], X[x_base+k_base+5u],
                                X[x_base+k_base+6u], X[x_base+k_base+7u]);
            let xv2 = vec4<f32>(X[x_base+k_base+8u], X[x_base+k_base+9u],
                                X[x_base+k_base+10u], X[x_base+k_base+11u]);
            let xv3 = vec4<f32>(X[x_base+k_base+12u], X[x_base+k_base+13u],
                                X[x_base+k_base+14u], X[x_base+k_base+15u]);
            let w_off = w_base + g * 128u + lane * 4u;
            let pw0 = W_Q8[w_off]; let pw1 = W_Q8[w_off+1u];
            let pw2 = W_Q8[w_off+2u]; let pw3 = W_Q8[w_off+3u];
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));
            let wv2 = vec4<f32>(f32(extractBits(i32(pw2),0u,8u)),f32(extractBits(i32(pw2),8u,8u)),
                                f32(extractBits(i32(pw2),16u,8u)),f32(extractBits(i32(pw2),24u,8u)));
            let wv3 = vec4<f32>(f32(extractBits(i32(pw3),0u,8u)),f32(extractBits(i32(pw3),8u,8u)),
                                f32(extractBits(i32(pw3),16u,8u)),f32(extractBits(i32(pw3),24u,8u)));
            let block0 = g*16u + (lane*16u)/32u;
            let block1 = g*16u + (lane*16u+4u)/32u;
            let block2 = g*16u + (lane*16u+8u)/32u;
            let block3 = g*16u + (lane*16u+12u)/32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let sc0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let sc1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);
            let sp2 = unpack2x16float(Scales[(s_base+block2)/2u]);
            let sc2 = select(sp2.x, sp2.y, ((s_base+block2)&1u)!=0u);
            let sp3 = unpack2x16float(Scales[(s_base+block3)/2u]);
            let sc3 = select(sp3.x, sp3.y, ((s_base+block3)&1u)!=0u);
            acc += dot(xv0,wv0)*sc0 + dot(xv1,wv1)*sc1
                 + dot(xv2,wv2)*sc2 + dot(xv3,wv3)*sc3;
        }
    }
    let warp_sum = subgroupAdd(acc);
    if (lane==0u && col<N) { Y[row*N+col] = warp_sum + Bias[col]; }
}
)WGSL";

// [gguf_q8] q8_matmul_lite (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_LITE = R"WGSL(
enable subgroups;

// Q8_0 matmul with single output per workgroup (TILE_N=1, WG=32).
// Optimized for small K where high workgroup count = more GPU occupancy.
// Each workgroup = 1 warp = 32 threads computing 1 output element.
// K_PER_ITER=8: each thread processes 8 elements per 256-element stride.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let col = wid.y;
    let lane = lid.x;

    let K = _params_[0];
    let N = _params_[1];

    let valid = col < N;

    let x_base = row * K;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    let n_strides = K / 256u;

    var acc: f32 = 0.0;

    for (var g = 0u; g < n_strides; g = g + 1u) {
        if (valid) {
        // Read 8 fp32 activations
        let k_base = g * 256u + lane * 8u;
        let xv0 = vec4<f32>(X[x_base + k_base],
                            X[x_base + k_base + 1u],
                            X[x_base + k_base + 2u],
                            X[x_base + k_base + 3u]);
        let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                            X[x_base + k_base + 5u],
                            X[x_base + k_base + 6u],
                            X[x_base + k_base + 7u]);

        // Read 2 packed u32 weights (8 int8 values)
        let w_off = w_base + g * 64u + lane * 2u;
        let pw0 = W_Q8[w_off];
        let pw1 = W_Q8[w_off + 1u];

        let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
        let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

        // Per-block scales
        let block0 = g * 8u + (lane * 8u) / 32u;
        let block1 = g * 8u + (lane * 8u + 4u) / 32u;
        let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
        let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
        let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
        let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

        acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_mma (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_MMA = R"WGSL(
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// Double-buffered subgroup-matrix Q8_0 GEMM:
//   Y[M×N] = X[M×K] × W[N×K]^T + Bias[N]
//
// Optimizations:
//   - Double-buffered smem: load next tile while computing current
//   - TILE_K=32: 2 MMA calls per K-block, halves barrier count
//   - Uniform params: enable early loop exit at actual K
//   - Vectorized 4-at-a-time Q8 dequant from packed u32
//
// 4 subgroups, 32×32 output tile. Grid: (ceil(N/32), ceil(M/32))

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { M: u32, N: u32, K: u32, pad: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const MK: u32 = 16u;     // MMA tile K
const SB: u32 = 32u;     // Q8 scale block size
const WG: u32 = 512u;
const AB_A: u32 = 2048u; // TM * TK = 64*32
const AB_B: u32 = 2048u; // TN * TK = 64*32

var<workgroup> tA: array<array<f16, 2048>, 2>;  // double-buf A [64×32]
var<workgroup> tB: array<array<f16, 2048>, 2>;  // double-buf B [64×32]
var<workgroup> tC: array<f32, 4096>;             // output [64×64]

@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let M = params.M;  let N = params.N;  let K = params.K;
    let lx = lid.x;
    let rb = wid.y * TM;
    let cb = wid.x * TN;
    let sr = sg_id & 3u;   // 4 rows of 16 = 64
    let sc = sg_id >> 2u;  // 4 cols of 16 = 64
    let ws = K / 4u;
    let nkt = K / TK;

    var mC: subgroup_matrix_result<f32, 16, 16>;

    // ── Prefetch tile 0 into buffer 0 ────────────────────────────────
    for (var i = lx; i < AB_A; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gr = rb + r;
        tA[0][i] = select(0.0h, f16(X[gr * K + c]), gr < M);
    }
    for (var i = lx; i < AB_B; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gc = cb + r;
        if (gc < N) {
            let pk = W_Q8[gc * ws + c / 4u];
            let q = f32(extractBits(i32(pk), (c & 3u) * 8u, 8u));
            let si = gc * (K / SB) + c / SB;
            let sp = unpack2x16float(Scales[si / 2u]);
            tB[0][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
        } else { tB[0][i] = 0.0h; }
    }
    workgroupBarrier();

    // ── Main loop: compute cur tile, load next ───────────────────────
    for (var ti = 0u; ti < nkt; ti++) {
        let cur = ti & 1u;
        let nxt = 1u - cur;
        let kn = (ti + 1u) * TK;

        // Load next tile (if not last iteration)
        if (ti + 1u < nkt) {
            for (var i = lx; i < AB_A; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gr = rb + r;
                tA[nxt][i] = select(0.0h, f16(X[gr * K + kn + c]), gr < M);
            }
            for (var i = lx; i < AB_B; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gc = cb + r;  let gk = kn + c;
                if (gc < N) {
                    let pk = W_Q8[gc * ws + gk / 4u];
                    let q = f32(extractBits(i32(pk), (gk & 3u) * 8u, 8u));
                    let si = gc * (K / SB) + gk / SB;
                    let sp = unpack2x16float(Scales[si / 2u]);
                    tB[nxt][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
                } else { tB[nxt][i] = 0.0h; }
            }
        }

        // Compute: 2 MMA calls (K=32 = 2×16)
        let ao = sr * 16u * TK;
        let bo = sc * 16u * TK;

        let mA0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tA[cur], ao, false, TK);
        let mB0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tB[cur], bo, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA0, mB0, mC);

        let mA1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tA[cur], ao + MK, false, TK);
        let mB1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tB[cur], bo + MK, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA1, mB1, mC);

        workgroupBarrier();
    }

    // ── Store result ─────────────────────────────────────────────────
    subgroupMatrixStore(&tC, sr * 16u * TN + sc * 16u, mC, false, TN);
    workgroupBarrier();

    for (var i = lx; i < TM * TN; i += WG) {
        let r = i / TN;  let c = i % TN;
        let gr = rb + r;  let gc = cb + c;
        if (gr < M && gc < N) {
            Y[gr * N + gc] = tC[i] + Bias[gc];
        }
    }
}
)WGSL";

// [gguf_q8] q8_matmul_norm (7 bindings, hand)
static const char* WGSL_Q8_MATMUL_NORM = R"WGSL(
enable subgroups;

// Fused: RMSNorm + Q8_0 matmul.
// Reads raw X, computes RMSNorm inline (two-pass over X — second from L1 cache),
// then multiplies normalized X by Q8 weight matrix.
// Eliminates the separate rms_norm/rms_next dispatch.
//
// Pass 1: Load X, accumulate sum_sq per warp via subgroupAdd
// Pass 2: Re-load X (L1 cached, 8KB), multiply by rstd * NormW, dot with W_Q8
//
// Bindings:
//   0: X (read) — raw input vector (pre-norm), E floats
//   1: W_Q8 (read) — quantized weight matrix (N × K/4 u32)
//   2: Scales (read) — fp16 scales packed as u32
//   3: Bias (read) — per-output bias
//   4: Y (write) — output
//   5: _params_ — [K, N, 0, eps_as_u32]
//   6: NormW (read) — norm weight vector, E floats

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;
@group(0) @binding(6) var<storage, read_write> NormW: array<f32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let eps = bitcast<f32>(_params_[3]);
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;

    // ── Pass 1: Compute sum of squares for RMSNorm ──────────────────────
    // Each warp independently reads the full X and computes sum_sq.
    // X is 8KB (K=2048 × 4 bytes) — fits in L1 cache for pass 2.
    var sum_sq: f32 = 0.0;
    for (var g = 0u; g < n_strides; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let xv0 = vec4<f32>(X[x_base + k_base],
                            X[x_base + k_base + 1u],
                            X[x_base + k_base + 2u],
                            X[x_base + k_base + 3u]);
        let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                            X[x_base + k_base + 5u],
                            X[x_base + k_base + 6u],
                            X[x_base + k_base + 7u]);
        sum_sq += dot(xv0, xv0) + dot(xv1, xv1);
    }
    let total_sq = subgroupAdd(sum_sq);
    let rstd = 1.0 / sqrt(total_sq / f32(K) + eps);

    // ── Pass 2: Normalized matmul ───────────────────────────────────────
    // Re-read X (L1 cached), apply RMSNorm weight, dot with Q8 weights.
    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g * 256u + lane * 8u;

            // Re-read X from L1 cache + apply norm
            let raw0 = vec4<f32>(X[x_base + k_base],
                                 X[x_base + k_base + 1u],
                                 X[x_base + k_base + 2u],
                                 X[x_base + k_base + 3u]);
            let raw1 = vec4<f32>(X[x_base + k_base + 4u],
                                 X[x_base + k_base + 5u],
                                 X[x_base + k_base + 6u],
                                 X[x_base + k_base + 7u]);
            let nw0 = vec4<f32>(NormW[k_base],     NormW[k_base + 1u],
                                NormW[k_base + 2u], NormW[k_base + 3u]);
            let nw1 = vec4<f32>(NormW[k_base + 4u], NormW[k_base + 5u],
                                NormW[k_base + 6u], NormW[k_base + 7u]);
            let xv0 = raw0 * rstd * nw0;
            let xv1 = raw1 * rstd * nw1;

            // Read Q8 weights
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_smem (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_SMEM = R"WGSL(
enable subgroups;

// Q8_0 matmul with shared-memory X caching.
// All 256 threads cooperatively load X, all participate in barriers.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const STRIDE: u32 = 256u;
const MAX_STRIDES: u32 = 24u;

var<workgroup> smem_x: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;
    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let valid = col < N;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    var acc: f32 = 0.0;

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * STRIDE;
        // Guard: K is uniform across workgroup so this is safe
        let in_range = k_off < K;

        // All threads participate in shared memory load + barrier
        if (in_range) {
            smem_x[tid] = X[x_base + k_off + tid];
        }
        workgroupBarrier();

        if (valid && in_range) {
            let k_base = lane * 8u;
            let xv0 = vec4<f32>(smem_x[k_base], smem_x[k_base+1u],
                                smem_x[k_base+2u], smem_x[k_base+3u]);
            let xv1 = vec4<f32>(smem_x[k_base+4u], smem_x[k_base+5u],
                                smem_x[k_base+6u], smem_x[k_base+7u]);

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [gguf_q8] q8_matmul_tiled (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_TILED = R"WGSL(
enable subgroups;

// Shared-memory tiled Q8_0 GEMM for batched prefill:
//   Y[T×N] = X[T×K] × W[N×K]^T + Bias[N]
//
// Tile: TILE_M(8) rows × TILE_N(32) columns per workgroup.
// 256 threads = 8 warps, each warp processes 4 output columns.
// X rows are loaded into shared memory and reused across all 32 columns.
//
// vs. q8_matmul_batched (TILE_N=8): 4× fewer workgroups, 4× more X reuse.
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)
// Workgroup: 256 threads
//
// Shared memory: TILE_M × 256 = 2048 floats = 8 KB per K-block.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;  // output columns per tile (4 per warp)
const TILE_M: u32 = 8u;   // output rows per tile
const COLS_PER_WARP: u32 = 4u;
const BK_LOAD: u32 = 256u;
const MAX_ITERS: u32 = 24u;

// Shared memory: TILE_M rows × BK_LOAD columns of X
var<workgroup> smem_x: array<f32, 2048>;  // 8 × 256

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    // Each warp handles COLS_PER_WARP=4 output columns
    var acc: array<array<f32, 4>, 8>;  // [TILE_M][COLS_PER_WARP]
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) {
            acc[m][c] = 0.0;
        }
    }

    // Pre-compute column indices and validity
    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    var w_bases: array<u32, 4>;
    var s_bases: array<u32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = tile_col * TILE_N + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
        w_bases[c] = select(0u, cols[c] * stride_w, col_valid[c]);
        s_bases[c] = select(0u, cols[c] * n_blocks, col_valid[c]);
    }

    for (var g = 0u; g < MAX_ITERS; g = g + 1u) {
        let k_off = g * BK_LOAD;
        let in_k = k_off < K;

        // ── Cooperative load: X tile [TILE_M × BK_LOAD] into smem ───
        for (var r = 0u; r < TILE_M; r++) {
            let row = tile_row * TILE_M + r;
            let smem_idx = r * BK_LOAD + tid;
            if (in_k && row < T) {
                smem_x[smem_idx] = X[row * K + k_off + tid];
            } else {
                smem_x[smem_idx] = 0.0;
            }
        }
        workgroupBarrier();

        // ── Compute: each warp processes 4 columns from global W,
        //    dotting with X rows from shared memory ───────────────────
        if (in_k) {
            let k_base = lane * 8u;

            for (var c = 0u; c < COLS_PER_WARP; c++) {
                // Guard invalid columns but keep all threads active
                // (no divergent control flow before subgroupAdd)
                var wv0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var wv1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var scale0: f32 = 0.0;
                var scale1: f32 = 0.0;

                if (col_valid[c]) {
                    let w_off = w_bases[c] + g * 64u + lane * 2u;
                    let pw0 = W_Q8[w_off];
                    let pw1 = W_Q8[w_off + 1u];

                    wv0 = vec4<f32>(
                        f32(extractBits(i32(pw0), 0u, 8u)),
                        f32(extractBits(i32(pw0), 8u, 8u)),
                        f32(extractBits(i32(pw0), 16u, 8u)),
                        f32(extractBits(i32(pw0), 24u, 8u)));
                    wv1 = vec4<f32>(
                        f32(extractBits(i32(pw1), 0u, 8u)),
                        f32(extractBits(i32(pw1), 8u, 8u)),
                        f32(extractBits(i32(pw1), 16u, 8u)),
                        f32(extractBits(i32(pw1), 24u, 8u)));

                    let block0 = g * 8u + (lane * 8u) / 32u;
                    let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                    let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                    scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                    let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                    scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
                }

                for (var m = 0u; m < TILE_M; m++) {
                    let smem_base = m * BK_LOAD + k_base;
                    let xv0 = vec4<f32>(smem_x[smem_base],
                                        smem_x[smem_base + 1u],
                                        smem_x[smem_base + 2u],
                                        smem_x[smem_base + 3u]);
                    let xv1 = vec4<f32>(smem_x[smem_base + 4u],
                                        smem_x[smem_base + 5u],
                                        smem_x[smem_base + 6u],
                                        smem_x[smem_base + 7u]);
                    acc[m][c] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
                }
            }
        }

        workgroupBarrier();
    }

    // Reduce across lanes within each warp, write results
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < T) {
                Y[row * N + cols[c]] = warp_sum + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [gguf_q8] q8_matmul_vec4 (6 bindings, hand)
static const char* WGSL_Q8_MATMUL_VEC4 = R"WGSL(
enable subgroups;

// Optimized Q8_0 matmul with vec4 loads for X (activation vector).
// Uses array<vec4<f32>> binding to enable 128-bit coalesced memory reads.
// K_PER_ITER=8: each lane processes 8 elements per 256-element stride.
// TILE_N=8: each workgroup computes 8 output elements (one per warp).

@group(0) @binding(0) var<storage, read_write> X_v4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        // X base in vec4 units: row * K / 4
        let x_base_v4 = row * (K / 4u);

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations via 2 × vec4 loads (128-bit each)
            let k_v4_base = g * 64u + lane * 2u;  // 256 elements / 4 = 64 vec4s per stride
            let xv0 = X_v4[x_base_v4 + k_v4_base];
            let xv1 = X_v4[x_base_v4 + k_v4_base + 1u];

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 via extractBits (sign-extended)
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales (2 blocks per 8 elements)
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [shared] add_rms_norm (6 bindings, triton)
static const char* WGSL_ADD_RMS_NORM = R"WGSL(
enable subgroups;

// Auto-generated by Triton WebGPU Backend
// Kernel: add_rms_norm_loop_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read_write> buf0: array<f32>;  // X
@group(0) @binding(1) var<storage, read> buf1: array<f32>;  // Residual
@group(0) @binding(2) var<storage, read_write> buf2: array<f32>;  // Y
@group(0) @binding(3) var<storage, read> buf3: array<f32>;  // W
@group(0) @binding(4) var<storage, read_write> buf4: array<f32>;  // Rstd

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(5) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v11: i32 = i32(_wg_id.x);
    let v12: i32 = params.N + 127;
    let v13: i32 = v12 / 128;
    var v15: i32 = 0;
    var v16_0: f32 = f32(0);
    loop {
        let v17: bool = v15 < v13;
        if !v17 { break; }
        let v19: i32 = v15 * 128;
        let v20: i32 = i32(_lid.x);
        let v21: i32 = v20 & 127;
        let v22: i32 = i32(u32(v21) % u32(32));
        let v23: i32 = i32(_lid.x);
        let v24: i32 = i32(u32(v23) / u32(32));
        let v25: i32 = v22 << u32(0);
        let v26: i32 = 0 | v25;
        let v27: i32 = v24 << u32(5);
        let v28: i32 = v26 | v27;
        let v29: i32 = v28 & 127;
        let v30: i32 = i32(u32(v29) >> u32(0));
        let v31: i32 = v30 | 0;
        let v32: i32 = 0 ^ v31;
        let v33: i32 = v32 ^ 0;
        let v34: i32 = v33 + 0;
        let v35: i32 = v19 + v34;
        let v36: bool = v35 < params.N;
        let v37: i32 = v11 * params.stride;
        let v40: f32 = buf0[u32((v37 + v35))];
        let v41: f32 = select(f32(0.0), v40, v36);
        let v44: f32 = buf1[u32((v37 + v35))];
        let v45: f32 = select(f32(0.0), v44, v36);
        let v46: f32 = v41 + v45;
        let v47: f32 = select(f32(0), v46, v36);
        if v36 { buf0[u32((v37 + v35))] = v47; }
        let v48: f32 = v46 * v46;
        let v50: f32 = v16_0 + v48;
        let v52: i32 = v15 + 1;
        v15 = v52;
        v16_0 = v50;
    }
    let v55: i32 = bitcast<i32>(v16_0);
    let v56: i32 = subgroupShuffleXor(v55, u32(16));
    let v57: f32 = bitcast<f32>(v56);
    let v58: f32 = v16_0 + v57;
    let v59: i32 = bitcast<i32>(v58);
    let v60: i32 = subgroupShuffleXor(v59, u32(8));
    let v61: f32 = bitcast<f32>(v60);
    let v62: f32 = v58 + v61;
    let v63: i32 = bitcast<i32>(v62);
    let v64: i32 = subgroupShuffleXor(v63, u32(4));
    let v65: f32 = bitcast<f32>(v64);
    let v66: f32 = v62 + v65;
    let v67: i32 = bitcast<i32>(v66);
    let v68: i32 = subgroupShuffleXor(v67, u32(2));
    let v69: f32 = bitcast<f32>(v68);
    let v70: f32 = v66 + v69;
    let v71: i32 = bitcast<i32>(v70);
    let v72: i32 = subgroupShuffleXor(v71, u32(1));
    let v73: f32 = bitcast<f32>(v72);
    let v74: f32 = v70 + v73;
    let v75: i32 = i32(_lid.x);
    let v76: i32 = v75 & 127;
    let v77: i32 = i32(u32(v76) % u32(32));
    let v78: i32 = i32(_lid.x);
    let v79: i32 = i32(u32(v78) / u32(32));
    let v80: i32 = v77 << u32(0);
    let v81: i32 = 0 | v80;
    let v82: i32 = v79 << u32(5);
    let v83: i32 = v81 | v82;
    let v84: i32 = v83 & 96;
    let v85: i32 = i32(u32(v84) >> u32(3));
    let v86: i32 = 0 | v85;
    let v87: i32 = 0 ^ v86;
    let v88: i32 = v87 ^ 0;
    let v89: i32 = v88 ^ 0;
    let v90: i32 = v89 + 0;
    _smem[u32(v90) >> 2u] = bitcast<i32>(v74);
    workgroupBarrier();
    let v94: i32 = i32(_lid.x);
    let v95: i32 = v94 & 127;
    let v96: i32 = i32(u32(v95) % u32(32));
    let v97: i32 = i32(_lid.x);
    let v98: i32 = i32(u32(v97) / u32(32));
    let v99: i32 = v96 << u32(0);
    let v100: i32 = 0 | v99;
    let v101: i32 = v98 << u32(5);
    let v102: i32 = v100 | v101;
    let v103: i32 = v102 & 3;
    let v104: i32 = v103 << u32(2);
    let v105: i32 = v104 | 0;
    let v106: i32 = 0 ^ v105;
    let v107: i32 = v106 ^ 0;
    let v108: i32 = v107 ^ 0;
    let v109: i32 = v108 + 0;
    let v111: f32 = bitcast<f32>(_smem[u32(v109) >> 2u]);
    let v114: i32 = bitcast<i32>(v111);
    let v115: i32 = subgroupShuffleXor(v114, u32(2));
    let v116: f32 = bitcast<f32>(v115);
    let v117: f32 = v111 + v116;
    let v118: i32 = bitcast<i32>(v117);
    let v119: i32 = subgroupShuffleXor(v118, u32(1));
    let v120: f32 = bitcast<f32>(v119);
    let v121: f32 = v117 + v120;
    let v122: f32 = f32(params.N);
    let v123: f32 = v121 / v122;
    let v124: f32 = v123 + params.eps;
    let v125: f32 = sqrt(v124);
    let v126: f32 = f32(1.0) / v125;
    var v128: i32 = 0;
    loop {
        let v129: bool = v128 < v13;
        if !v129 { break; }
        let v131: i32 = v128 * 128;
        let v132: i32 = i32(_lid.x);
        let v133: i32 = v132 & 127;
        let v134: i32 = i32(u32(v133) % u32(32));
        let v135: i32 = i32(_lid.x);
        let v136: i32 = i32(u32(v135) / u32(32));
        let v137: i32 = v134 << u32(0);
        let v138: i32 = 0 | v137;
        let v139: i32 = v136 << u32(5);
        let v140: i32 = v138 | v139;
        let v141: i32 = v140 & 127;
        let v142: i32 = i32(u32(v141) >> u32(0));
        let v143: i32 = v142 | 0;
        let v144: i32 = 0 ^ v143;
        let v145: i32 = v144 ^ 0;
        let v146: i32 = v145 + 0;
        let v147: i32 = v131 + v146;
        let v148: bool = v147 < params.N;
        let v149: i32 = v11 * params.stride;
        let v152: f32 = buf0[u32((v149 + v147))];
        let v153: f32 = select(f32(0.0), v152, v148);
        let v155: f32 = buf3[u32(v147)];
        let v156: f32 = select(f32(1.0), v155, v148);
        let v159: f32 = v153 * v126;
        let v160: f32 = v159 * v156;
        let v161: f32 = select(f32(0), v160, v148);
        if v148 { buf2[u32((v149 + v147))] = v161; }
        let v162: i32 = v128 + 1;
        v128 = v162;
    }
    if _lid.x == 0u { buf4[u32(v11)] = v126; }
}
)WGSL";

// [shared] add_rms_norm_batched (6 bindings, hand)
static const char* WGSL_ADD_RMS_NORM_BATCHED = R"WGSL(
enable subgroups;

// Batched add + RMSNorm for T rows:
//   X[t] = X[t] + A[t]  (residual add, in-place)
//   Y[t] = X[t] * W / rms(X[t])  (RMSNorm for next op)
// Grid: (T, 1, 1)
// WG: 128 threads

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;  // residual [T × N], updated in-place
@group(0) @binding(1) var<storage, read> A: array<f32>;  // addend [T × N]
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;  // normed output [T × N]
@group(0) @binding(3) var<storage, read> W: array<f32>;  // norm weight [N]
@group(0) @binding(4) var<storage, read_write> Rstd: array<f32>;  // [T]

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(5) var<storage, read> params: Params;

const MAX_N: u32 = 8192u;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
) {
    let row = i32(_wg_id.x);
    let tid = i32(_lid.x);
    let N = params.N;
    let stride = params.stride;
    let base = row * stride;

    // Pass 1: Add residual and compute sum of squares
    var sum_sq: f32 = 0.0;
    var idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let pos = u32(base + idx);
            let v = X[pos] + A[pos];
            X[pos] = v;
            sum_sq += v * v;
        }
    }

    let warp_sum = subgroupAdd(sum_sq);
    let warp_id = tid / 32;
    let lane_id = tid % 32;
    if (lane_id == 0) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    var total: f32 = 0.0;
    if (tid < 4) {
        total = bitcast<f32>(_smem[tid]);
    }
    let final_sum = subgroupAdd(total);
    let rstd = 1.0 / sqrt(final_sum / f32(N) + params.eps);
    if (tid == 0) {
        Rstd[u32(row)] = rstd;
    }

    // Pass 2: Apply norm + weight
    idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let pos = u32(base + idx);
            Y[pos] = X[pos] * rstd * W[u32(idx)];
        }
    }
}
)WGSL";

// [shared] argmax (3 bindings, hand)
static const char* WGSL_ARGMAX = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> Result: array<i32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

var<workgroup> wg_max_val: array<f32, 8>;
var<workgroup> wg_max_idx: array<i32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each thread scans N/256 elements
    var local_max: f32 = -1e30;
    var local_idx: i32 = 0;
    var i = tid;
    for (; i < N; i = i + 256u) {
        let v = Logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i32(i);
        }
    }

    // Warp reduce: find max within 32 lanes
    for (var offset = 16u; offset > 0u; offset = offset >> 1u) {
        let other_val = subgroupShuffleXor(bitcast<i32>(local_max), offset);
        let other_idx = subgroupShuffleXor(local_idx, offset);
        let ov = bitcast<f32>(other_val);
        if (ov > local_max) {
            local_max = ov;
            local_idx = other_idx;
        }
    }

    // Write warp results to shared memory
    if (lane == 0u) {
        wg_max_val[warp_id] = local_max;
        wg_max_idx[warp_id] = local_idx;
    }
    workgroupBarrier();

    // Final reduce across 8 warps (done by thread 0)
    if (tid == 0u) {
        var best_val = wg_max_val[0];
        var best_idx = wg_max_idx[0];
        for (var w = 1u; w < 8u; w = w + 1u) {
            if (wg_max_val[w] > best_val) {
                best_val = wg_max_val[w];
                best_idx = wg_max_idx[w];
            }
        }
        Result[0] = best_idx;
    }
}
)WGSL";

// [shared] causal_attn (5 bindings, hand)
static const char* WGSL_CAUSAL_ATTN = R"WGSL(
enable f16;
enable subgroups;

// Multi-query causal attention with fp16 KV + early exit.
// 4 query positions per WG. Uniform params for data-dependent loop bound.
// Used on D3D12 (no MMA) and as Vulkan fallback.
//
// Grid: (n_head, ceil(T/4), 1)
// WG: 128 threads (4 warps × 32 threads)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

struct AttnParams {
    kv_stride: u32,
    n_rep: u32,
    T_total: u32,
    cache_offset: u32,
    T_prefill: u32,
    scale_bits: u32,
    neg_inf_bits: u32,
    pad1: u32,
};
@group(0) @binding(4) var<uniform> params: AttnParams;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;
const QUERIES_PER_WG: u32 = 4u;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let q_block = wid.y;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let kv_stride = params.kv_stride;
    let n_rep = params.n_rep;
    let T_total = params.T_total;
    let cache_offset = params.cache_offset;
    let scale = bitcast<f32>(params.scale_bits);
    let neg_inf = bitcast<f32>(params.neg_inf_bits);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let n_head_total = kv_stride / HD * n_rep;

    let q_idx = q_block * QUERIES_PER_WG + warp_id;
    let q_abs_pos = cache_offset + q_idx;

    let q_base = q_idx * n_head_total * HD + head * HD;
    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    // Early exit: uniform loop bound from var<uniform> params
    let max_causal = min(cache_offset + q_block * QUERIES_PER_WG + QUERIES_PER_WG, T_total);

    for (var t = 0u; t < max_causal; t = t + 1u) {
        let causal_valid = t <= q_abs_pos;

        let k_base = t * kv_stride + kv_off;
        let k_off = k_base + lane * HD_PER_THREAD;
        let k0 = select(0.0, f32(K_cache[k_off]), causal_valid);
        let k1 = select(0.0, f32(K_cache[k_off + 1u]), causal_valid);
        let k2 = select(0.0, f32(K_cache[k_off + 2u]), causal_valid);
        let k3 = select(0.0, f32(K_cache[k_off + 3u]), causal_valid);

        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot_qk = subgroupAdd(partial);
        let score = select(neg_inf, dot_qk * scale, causal_valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_off = k_base + lane * HD_PER_THREAD;
        let v0 = select(0.0, f32(V_cache[v_off]), causal_valid);
        let v1 = select(0.0, f32(V_cache[v_off + 1u]), causal_valid);
        let v2 = select(0.0, f32(V_cache[v_off + 2u]), causal_valid);
        let v3 = select(0.0, f32(V_cache[v_off + 3u]), causal_valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    let out_base = q_idx * n_head_total * HD + head * HD;
    Out[out_base + lane * HD_PER_THREAD]      = acc0;
    Out[out_base + lane * HD_PER_THREAD + 1u] = acc1;
    Out[out_base + lane * HD_PER_THREAD + 2u] = acc2;
    Out[out_base + lane * HD_PER_THREAD + 3u] = acc3;
}
)WGSL";

// [shared] embed_gather (4 bindings, hand)
static const char* WGSL_EMBED_GATHER = R"WGSL(
@group(0) @binding(0) var<storage, read> EmbeddingTable: array<f32>;
@group(0) @binding(1) var<storage, read> TokenId: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let idx = gid.x;
    if (idx >= E) { return; }
    let token = u32(TokenId[0]);
    X[idx] = EmbeddingTable[token * E + idx];
}
)WGSL";

// [shared] flash_attn_mma (5 bindings, hand)
static const char* WGSL_FLASH_ATTN_MMA = R"WGSL(
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// MMA Flash Attention for batched prefill.
//
// Processes BQ=16 query positions per WG. Iterates KV in blocks of
// BK=16. Uses 16×16×16 fp16→f32 MMA for Q·K^T and P·V.
//
// Per KV block:
//   1. MMA: S[16×16] = Q[16×HD] · K[16×HD]^T  (8 MMA calls over HD)
//   2. Causal mask + online softmax on S (per-row)
//   3. MMA: O[16×16] += P[16×16] · V[16×HD]   (8 MMA calls over HD)
//
// Grid: (n_head, ceil(T/16), 1)
// WG: 128 threads (4 subgroups). Subgroup 0 does MMA, all help load.
//
// KV cache: fp16 (array<f16>)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> OutBuf: array<f32>;

struct AttnParams {
    kv_stride: u32,
    n_rep: u32,
    T_total: u32,
    cache_offset: u32,
    T_prefill: u32,
    scale_bits: u32,
    neg_inf_bits: u32,
    pad1: u32,
};
@group(0) @binding(4) var<uniform> params: AttnParams;

const BQ: u32 = 16u;
const BK: u32 = 16u;
const HD: u32 = 128u;
const MK: u32 = 16u;
const HD_TILES: u32 = 8u;  // HD / MK
const WG: u32 = 128u;

// Shared memory
var<workgroup> tile_Q: array<f16, 256>;    // 16 × 16 (Q chunk)
var<workgroup> tile_K: array<f16, 256>;    // 16 × 16 (K chunk)
var<workgroup> tile_S: array<f32, 256>;    // 16 × 16 (score matrix)
var<workgroup> tile_P: array<f16, 256>;    // 16 × 16 (attn weights)
var<workgroup> tile_V: array<f16, 256>;    // 16 × 16 (V chunk)
var<workgroup> tile_O: array<f32, 256>;    // 16 × 16 (output chunk)
var<workgroup> row_m: array<f32, 16>;      // per-row max
var<workgroup> row_l: array<f32, 16>;      // per-row sum
var<workgroup> row_rescale: array<f32, 16>; // rescale factor for old acc
// Full output: [BQ × HD] stored as 8 chunks of [16×16] in smem
var<workgroup> out_acc: array<f32, 2048>;  // 16 × 128

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let head = wid.x;
    let q_block = wid.y;
    let lx = lid.x;

    let kv_stride = params.kv_stride;
    let n_rep = params.n_rep;
    let T_total = params.T_total;
    let cache_offset = params.cache_offset;
    let scale = bitcast<f32>(params.scale_bits);
    let neg_inf = bitcast<f32>(params.neg_inf_bits);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let n_head_total = kv_stride / HD * n_rep;
    let q_base = q_block * BQ;

    // Initialize output accumulator and softmax state
    for (var i = lx; i < BQ * HD; i += WG) {
        out_acc[i] = 0.0;
    }
    if (lx < BQ) {
        row_m[lx] = neg_inf;
        row_l[lx] = 0.0;
    }
    workgroupBarrier();

    // Max KV position for any query in this block (uniform)
    let max_kv = min(cache_offset + q_base + BQ, T_total);
    let n_kv_blocks = (max_kv + BK - 1u) / BK;

    for (var kb = 0u; kb < n_kv_blocks; kb++) {
        let kv_start = kb * BK;

        // ── Phase 1: S = Q · K^T via MMA ────────────────────────────
        // Zero score tile
        if (lx < 256u) { tile_S[lx] = 0.0; }
        workgroupBarrier();

        for (var hd = 0u; hd < HD_TILES; hd++) {
            let hd_off = hd * MK;

            // Load Q[BQ × MK] and K[BK × MK] as fp16
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                let gq = q_base + r;
                tile_Q[i] = select(0.0h,
                    f16(Q[gq * n_head_total * HD + head * HD + hd_off + c]),
                    gq < T_total);
            }
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                let gk = kv_start + r;
                tile_K[i] = select(0.0h,
                    K_cache[gk * kv_stride + kv_off + hd_off + c],
                    gk < T_total);
            }
            workgroupBarrier();

            // MMA: S += Q × K^T  (subgroup 0)
            if (sg_id == 0u) {
                let mQ = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
                    &tile_Q, 0u, false, MK);
                let mK = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
                    &tile_K, 0u, true, MK);
                var mS = subgroupMatrixLoad<subgroup_matrix_result<f32, 16, 16>>(
                    &tile_S, 0u, false, BK);
                mS = subgroupMatrixMultiplyAccumulate(mQ, mK, mS);
                subgroupMatrixStore(&tile_S, 0u, mS, false, BK);
            }
            workgroupBarrier();
        }

        // ── Phase 2: Scale + causal mask + online softmax ───────────
        // 128 threads process 256 elements (2 each)
        for (var i = lx; i < BQ * BK; i += WG) {
            let qr = i / BK;
            let kc = i % BK;
            let gq = q_base + qr;
            let gk = kv_start + kc;
            var s = tile_S[i] * scale;
            if (gk >= T_total || gk > cache_offset + gq) {
                s = neg_inf;
            }
            tile_S[i] = s;
        }
        workgroupBarrier();

        // Per-row max and rescale factor (16 threads, one per row)
        if (lx < BQ) {
            var rm = neg_inf;
            for (var j = 0u; j < BK; j++) {
                rm = max(rm, tile_S[lx * BK + j]);
            }
            let m_old = row_m[lx];
            let m_new = max(m_old, rm);
            let rescale = exp(m_old - m_new);
            row_m[lx] = m_new;
            row_l[lx] *= rescale;
            row_rescale[lx] = rescale;

            var ls = 0.0f;
            for (var j = 0u; j < BK; j++) {
                let e = exp(tile_S[lx * BK + j] - m_new);
                tile_P[lx * BK + j] = f16(e);
                ls += e;
            }
            row_l[lx] += ls;
        }
        workgroupBarrier();

        // Rescale existing output accumulator
        for (var i = lx; i < BQ * HD; i += WG) {
            let qr = i / HD;
            out_acc[i] *= row_rescale[qr];
        }
        workgroupBarrier();

        // ── Phase 3: O += P · V via MMA ─────────────────────────────
        for (var hd = 0u; hd < HD_TILES; hd++) {
            let hd_off = hd * MK;

            // Load V[BK × MK] as fp16
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                let gv = kv_start + r;
                tile_V[i] = select(0.0h,
                    V_cache[gv * kv_stride + kv_off + hd_off + c],
                    gv < T_total);
            }

            // Load current output chunk [16×16]
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                tile_O[i] = out_acc[r * HD + hd_off + c];
            }
            workgroupBarrier();

            // MMA: O += P × V  (subgroup 0)
            if (sg_id == 0u) {
                let mP = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
                    &tile_P, 0u, false, BK);
                let mV = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
                    &tile_V, 0u, false, MK);
                var mO = subgroupMatrixLoad<subgroup_matrix_result<f32, 16, 16>>(
                    &tile_O, 0u, false, MK);
                mO = subgroupMatrixMultiplyAccumulate(mP, mV, mO);
                subgroupMatrixStore(&tile_O, 0u, mO, false, MK);
            }
            workgroupBarrier();

            // Write back to out_acc
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                out_acc[r * HD + hd_off + c] = tile_O[i];
            }
            workgroupBarrier();
        }
    }

    // ── Normalize by row_l and write output ─────────────────────────
    for (var i = lx; i < BQ * HD; i += WG) {
        let qr = i / HD;
        let hd = i % HD;
        let gq = q_base + qr;
        if (gq < T_total) {
            let l = max(row_l[qr], 1e-10);
            OutBuf[gq * n_head_total * HD + head * HD + hd] = out_acc[i] / l;
        }
    }
}
)WGSL";

// [shared] fp16_gemm (5 bindings, hand)
static const char* WGSL_FP16_GEMM = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // Each lane handles K/32 elements, strided by 4 (vec4 processing)
    // lane processes elements: lane*4, lane*4 + 32*4, lane*4 + 64*4, ...
    var acc: f32 = 0.0;

    if (col < N) {
        // W layout: (N, K) row-major, stored as u32 (2 fp16 per u32)
        // w_base = col * (K/2) — index into u32 array
        let w_base = col * (K / 2u);

        // Each lane processes 4 consecutive elements starting at lane*4
        // Stride between lane iterations: 32 lanes * 4 elements = 128
        let stride = 128u;
        var k = lane * 4u;

        for (; k + 3u < K; k = k + stride) {
            // Load 4 fp32 activations
            let x0 = X[x_base + k];
            let x1 = X[x_base + k + 1u];
            let x2 = X[x_base + k + 2u];
            let x3 = X[x_base + k + 3u];

            // Load 2 u32 = 4 fp16 weights, unpack to fp32
            let w01 = unpack2x16float(W[w_base + k / 2u]);
            let w23 = unpack2x16float(W[w_base + k / 2u + 1u]);

            // dot(vec4<f32>, vec4<f32>) — 4 FMAs
            acc += dot(vec4<f32>(x0, x1, x2, x3),
                       vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Reduce across 32 lanes in warp
    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [shared] fp16_gemm_wide (5 bindings, hand)
static const char* WGSL_FP16_GEMM_WIDE = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const COLS_PER_WARP: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each warp handles COLS_PER_WARP=4 columns
    let base_col = tile_col * TILE_N + warp_id * COLS_PER_WARP;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    let half_K = K / 2u;
    let stride = 128u;  // 32 lanes * 4 elements = 128
    var k = lane * 4u;

    for (; k + 3u < K; k = k + stride) {
        // Load X once, reuse for all 4 columns
        let x0 = X[x_base + k];
        let x1 = X[x_base + k + 1u];
        let x2 = X[x_base + k + 2u];
        let x3 = X[x_base + k + 3u];
        let xv = vec4<f32>(x0, x1, x2, x3);

        // Column 0
        if (base_col < N) {
            let w_base0 = base_col * half_K;
            let w01 = unpack2x16float(W[w_base0 + k / 2u]);
            let w23 = unpack2x16float(W[w_base0 + k / 2u + 1u]);
            acc0 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 1
        if (base_col + 1u < N) {
            let w_base1 = (base_col + 1u) * half_K;
            let w01 = unpack2x16float(W[w_base1 + k / 2u]);
            let w23 = unpack2x16float(W[w_base1 + k / 2u + 1u]);
            acc1 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 2
        if (base_col + 2u < N) {
            let w_base2 = (base_col + 2u) * half_K;
            let w01 = unpack2x16float(W[w_base2 + k / 2u]);
            let w23 = unpack2x16float(W[w_base2 + k / 2u + 1u]);
            acc2 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 3
        if (base_col + 3u < N) {
            let w_base3 = (base_col + 3u) * half_K;
            let w01 = unpack2x16float(W[w_base3 + k / 2u]);
            let w23 = unpack2x16float(W[w_base3 + k / 2u + 1u]);
            acc3 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Warp reduce
    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);
    let sum2 = subgroupAdd(acc2);
    let sum3 = subgroupAdd(acc3);

    if (lane == 0u) {
        if (base_col      < N) { Y[row * N + base_col]      = sum0 + Bias[base_col]; }
        if (base_col + 1u < N) { Y[row * N + base_col + 1u] = sum1 + Bias[base_col + 1u]; }
        if (base_col + 2u < N) { Y[row * N + base_col + 2u] = sum2 + Bias[base_col + 2u]; }
        if (base_col + 3u < N) { Y[row * N + base_col + 3u] = sum3 + Bias[base_col + 3u]; }
    }
}
)WGSL";

// [shared] fused_qknorm_rope (9 bindings, triton)
static const char* WGSL_FUSED_QKNORM_ROPE = R"WGSL(
enable f16;
enable subgroups;

// Auto-generated by Triton WebGPU Backend
// Kernel: fused_qknorm_rope_qkv_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // QKV
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Q_out
@group(0) @binding(2) var<storage, read_write> buf2: array<f16>;  // K_cache
@group(0) @binding(3) var<storage, read_write> buf3: array<f16>;  // V_cache
@group(0) @binding(4) var<storage, read> buf4: array<f32>;  // CosTable
@group(0) @binding(5) var<storage, read> buf5: array<f32>;  // SinTable
@group(0) @binding(6) var<storage, read> buf6: array<f32>;  // NormQ
@group(0) @binding(7) var<storage, read> buf7: array<f32>;  // NormK

struct Params {
    n_head: i32,
    q_size: i32,
    kv_size: i32,
    pos: i32,
    half_rot: i32,
    cache_offset: i32,
    eps: f32,
};
@group(0) @binding(8) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v18: i32 = i32(_wg_id.x);
    let v19: i32 = i32(_lid.x);
    let v20: i32 = v19 & 127;
    let v21: i32 = i32(u32(v20) % u32(32));
    let v22: i32 = i32(_lid.x);
    let v23: i32 = i32(u32(v22) / u32(32));
    let v24: i32 = v21 << u32(0);
    let v25: i32 = 0 | v24;
    let v26: i32 = v23 << u32(5);
    let v27: i32 = v25 | v26;
    let v28: i32 = v27 & 127;
    let v29: i32 = i32(u32(v28) >> u32(0));
    let v30: i32 = v29 | 0;
    let v31: i32 = 0 ^ v30;
    let v32: i32 = v31 ^ 0;
    let v33: i32 = v32 + 0;
    let v34: bool = v18 < params.n_head;
    let v35: i32 = v18 - params.n_head;
    let v36: i32 = v18 * 128;
    let v37: i32 = v35 * 128;
    let v38: i32 = params.q_size + v37;
    let v39: i32 = select(v38, v36, v34);
    let v42: f32 = buf0[u32((v39 + v33))];
    let v43: f32 = v42 * v42;
    let v44: i32 = bitcast<i32>(v43);
    let v45: i32 = subgroupShuffleXor(v44, u32(16));
    let v46: f32 = bitcast<f32>(v45);
    let v47: f32 = v43 + v46;
    let v48: i32 = bitcast<i32>(v47);
    let v49: i32 = subgroupShuffleXor(v48, u32(8));
    let v50: f32 = bitcast<f32>(v49);
    let v51: f32 = v47 + v50;
    let v52: i32 = bitcast<i32>(v51);
    let v53: i32 = subgroupShuffleXor(v52, u32(4));
    let v54: f32 = bitcast<f32>(v53);
    let v55: f32 = v51 + v54;
    let v56: i32 = bitcast<i32>(v55);
    let v57: i32 = subgroupShuffleXor(v56, u32(2));
    let v58: f32 = bitcast<f32>(v57);
    let v59: f32 = v55 + v58;
    let v60: i32 = bitcast<i32>(v59);
    let v61: i32 = subgroupShuffleXor(v60, u32(1));
    let v62: f32 = bitcast<f32>(v61);
    let v63: f32 = v59 + v62;
    let v64: i32 = i32(_lid.x);
    let v65: i32 = v64 & 127;
    let v66: i32 = i32(u32(v65) % u32(32));
    let v67: i32 = i32(_lid.x);
    let v68: i32 = i32(u32(v67) / u32(32));
    let v69: i32 = v66 << u32(0);
    let v70: i32 = 0 | v69;
    let v71: i32 = v68 << u32(5);
    let v72: i32 = v70 | v71;
    let v73: i32 = v72 & 96;
    let v74: i32 = i32(u32(v73) >> u32(3));
    let v75: i32 = 0 | v74;
    let v76: i32 = 0 ^ v75;
    let v77: i32 = v76 ^ 0;
    let v78: i32 = v77 ^ 0;
    let v79: i32 = v78 + 0;
    _smem[u32(v79) >> 2u] = bitcast<i32>(v63);
    workgroupBarrier();
    let v83: i32 = i32(_lid.x);
    let v84: i32 = v83 & 127;
    let v85: i32 = i32(u32(v84) % u32(32));
    let v86: i32 = i32(_lid.x);
    let v87: i32 = i32(u32(v86) / u32(32));
    let v88: i32 = v85 << u32(0);
    let v89: i32 = 0 | v88;
    let v90: i32 = v87 << u32(5);
    let v91: i32 = v89 | v90;
    let v92: i32 = v91 & 3;
    let v93: i32 = v92 << u32(2);
    let v94: i32 = v93 | 0;
    let v95: i32 = 0 ^ v94;
    let v96: i32 = v95 ^ 0;
    let v97: i32 = v96 ^ 0;
    let v98: i32 = v97 + 0;
    let v100: f32 = bitcast<f32>(_smem[u32(v98) >> 2u]);
    let v103: i32 = bitcast<i32>(v100);
    let v104: i32 = subgroupShuffleXor(v103, u32(2));
    let v105: f32 = bitcast<f32>(v104);
    let v106: f32 = v100 + v105;
    let v107: i32 = bitcast<i32>(v106);
    let v108: i32 = subgroupShuffleXor(v107, u32(1));
    let v109: f32 = bitcast<f32>(v108);
    let v110: f32 = v106 + v109;
    let v111: f32 = v110 / f32(128.0);
    let v112: f32 = v111 + params.eps;
    let v113: f32 = sqrt(v112);
    let v114: f32 = f32(1.0) / v113;
    let v116: f32 = buf6[u32(v33)];
    let v118: f32 = buf7[u32(v33)];
    let v119: f32 = select(v118, v116, v34);
    let v120: f32 = v42 * v114;
    let v121: f32 = v120 * v119;
    let v122: i32 = v33 % params.half_rot;
    let v123: i32 = params.half_rot * 2;
    let v124: bool = v33 < v123;
    let v125: i32 = params.pos * params.half_rot;
    let v128: f32 = buf4[u32((v125 + v122))];
    let v129: f32 = select(f32(1.0), v128, v124);
    let v132: f32 = buf5[u32((v125 + v122))];
    let v133: f32 = select(f32(0.0), v132, v124);
    let v134: bool = v33 < params.half_rot;
    let v135: f32 = select(f32(1.0), f32(-1.0), v134);
    let v136: i32 = v33 + params.half_rot;
    let v137: i32 = v33 - params.half_rot;
    let v138: i32 = select(v137, v136, v134);
    let v139: i32 = select(v33, v138, v124);
    let v141: f32 = buf0[u32((v39 + v139))];
    let v143: f32 = buf6[u32(v139)];
    let v145: f32 = buf7[u32(v139)];
    let v146: f32 = select(v145, v143, v34);
    let v147: f32 = v141 * v114;
    let v148: f32 = v147 * v146;
    let v149: f32 = v121 * v129;
    let v150: f32 = v135 * v148;
    let v151: f32 = v150 * v133;
    let v152: f32 = v149 + v151;
    let v153: f32 = select(v121, v152, v124);
    let v156: f32 = select(f32(0), v153, v34);
    if v34 { buf1[u32((v36 + v33))] = v156; }
    let v157: i32 = select(0, 1, v34);
    let v158: bool = v157 == 0;
    let v162: f32 = select(f32(0), v153, v158);
    if v158 { buf2[u32(((params.cache_offset + v37) + v33))] = f16(v162); }
    let v163: i32 = params.q_size + params.kv_size;
    let v164: i32 = v163 + v37;
    let v167: f32 = buf0[u32((v164 + v33))];
    let v168: f32 = select(f32(0.0), v167, v158);
    let v172: f32 = select(f32(0), v168, v158);
    if v158 { buf3[u32(((params.cache_offset + v37) + v33))] = f16(v172); }
}
)WGSL";

// [shared] fused_qknorm_rope_batched (9 bindings, hand)
static const char* WGSL_FUSED_QKNORM_ROPE_BATCHED = R"WGSL(
enable subgroups;

// Batched fused QKnorm + RoPE + KV cache scatter for prefill.
// Processes T tokens: applies QK-norm, rotary embeddings, and writes K/V to cache.
//
// QKV layout: [T × (qDim + 2*kvDim)]
//   Q: [T × qDim], K: [T × kvDim], V: [T × kvDim]
//
// Grid: ((n_head + n_kv) * T, 1, 1)
//   First n_head*T WGs handle Q heads (norm + RoPE → qRotBuf)
//   Next n_kv*T WGs handle KV heads (norm + RoPE + scatter to cache)
//
// Params: [n_head, qDim, kvDim, pos_offset, half_dim, 0, eps, T]

@group(0) @binding(0) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QNormW: array<f32>;
@group(0) @binding(7) var<storage, read> KNormW: array<f32>;
@group(0) @binding(8) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat_id = wid.x;
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let eps = bitcast<f32>(_params_[6]);
    let T = _params_[7];
    let n_kv = kvDim / HD;

    let total_q_wgs = n_head * T;

    if (flat_id < total_q_wgs) {
        // ── Q head processing ────────────────────────────────────────
        let t = flat_id / n_head;
        let head = flat_id % n_head;
        let qkv_stride = qDim + 2u * kvDim;

        // Source: QKV[t, head*HD .. (head+1)*HD]
        let src_base = t * qkv_stride + head * HD;

        // Load Q values
        var q: array<f32, 128>;
        for (var i = tid; i < HD; i += 128u) {
            q[i] = QKV[src_base + i];
        }

        // QK-norm: compute rstd across HD
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            sum_sq += q[i] * q[i];
        }
        let total_sq = subgroupAdd(sum_sq);
        // Cross-warp (128 threads = 4 warps)
        var _smem: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        // Use subgroup to collect
        if (lane == 0u) { _smem[warp_id] = total_sq; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = _smem[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply norm weight + RoPE
        let pos = pos_offset + t;
        let cos_base = pos * half_dim;
        let sin_base = pos * half_dim;

        let dst_base = t * n_head * HD + head * HD;
        for (var i = tid; i < HD; i += 128u) {
            let normed = q[i] * rstd * QNormW[i];
            if (i < half_dim) {
                let j = i + half_dim;
                let nj = q[j] * rstd * QNormW[j];
                let c = Cos[cos_base + i];
                let s = Sin[sin_base + i];
                QRot[dst_base + i] = normed * c - nj * s;
                QRot[dst_base + j] = nj * c + normed * s;
            }
        }
    } else {
        // ── KV head processing ────────────────────────────────────────
        let kv_flat = flat_id - total_q_wgs;
        let t = kv_flat / n_kv;
        let kv_head = kv_flat % n_kv;
        let qkv_stride = qDim + 2u * kvDim;

        // K source: QKV[t, qDim + kv_head*HD ..]
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        // V source: QKV[t, qDim + kvDim + kv_head*HD ..]
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        // Load K
        var k: array<f32, 128>;
        for (var i = tid; i < HD; i += 128u) {
            k[i] = QKV[k_src + i];
        }

        // K-norm
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            sum_sq += k[i] * k[i];
        }
        let total_sq = subgroupAdd(sum_sq);
        var _smem2: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { _smem2[warp_id] = total_sq; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = _smem2[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply K-norm + RoPE + scatter to cache
        let pos = pos_offset + t;
        let cos_base = pos * (HD / 2u);
        let sin_base = pos * (HD / 2u);
        let kv_stride = _params_[5];   // cache_offset * n_kv * HD (not used, derive from pos)
        // KV cache layout: [T_total × n_kv × HD]
        let cache_pos = pos;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;
        let half = HD / 2u;

        for (var i = tid; i < HD; i += 128u) {
            let normed = k[i] * rstd * KNormW[i];
            if (i < half) {
                let j = i + half;
                let nj = k[j] * rstd * KNormW[j];
                let c = Cos[cos_base + i];
                let s = Sin[sin_base + i];
                K_cache[k_dst + i] = normed * c - nj * s;
                K_cache[k_dst + j] = nj * c + normed * s;
            }
            // V: just copy to cache (no RoPE)
            V_cache[v_dst + i] = QKV[v_src + i];
        }
    }
}
)WGSL";

// [shared] gqa_chunked_pass1 (5 bindings, hand)
static const char* WGSL_GQA_CHUNKED_PASS1 = R"WGSL(
enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const CHUNK: u32 = 64u;
// Each thread processes HD_PER_THREAD elements, total 32 * 4 = 128
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let chunk_id = wid.y;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let n_chunks = _params_[4];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    // Load Q into registers (4 elements per thread)
    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    let t_start = chunk_id * CHUNK;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var i = 0u; i < CHUNK; i = i + 1u) {
        let t = t_start + i;
        let valid = t < T_total;

        let k_base = select(0u, t * kv_stride + kv_off, valid);
        let k0 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD]), valid);
        let k1 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD + 1u]), valid);
        let k2 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD + 2u]), valid);
        let k3 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD + 3u]), valid);

        // Full 128-element dot product via 4-element partial + subgroupAdd
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot = subgroupAdd(partial);
        let raw_score = dot * scale;
        let score = select(neg_inf, raw_score, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_base = select(0u, t * kv_stride + kv_off, valid);
        let v0 = select(0.0, f32(V_cache[v_base + lane * HD_PER_THREAD]), valid);
        let v1 = select(0.0, f32(V_cache[v_base + lane * HD_PER_THREAD + 1u]), valid);
        let v2 = select(0.0, f32(V_cache[v_base + lane * HD_PER_THREAD + 2u]), valid);
        let v3 = select(0.0, f32(V_cache[v_base + lane * HD_PER_THREAD + 3u]), valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    // Use max_chunks for the stride to prevent overlapping writes
    // across heads. Only write if this chunk is within n_chunks.
    let partial_stride = HD + 2u;
    let base = head * max_chunks * partial_stride + chunk_id * partial_stride;
    if (chunk_id < n_chunks) {
        if (lane == 0u) {
            Partials[base] = m_prev;
            Partials[base + 1u] = l_prev;
        }
        Partials[base + 2u + lane * HD_PER_THREAD] = acc0;
        Partials[base + 2u + lane * HD_PER_THREAD + 1u] = acc1;
        Partials[base + 2u + lane * HD_PER_THREAD + 2u] = acc2;
        Partials[base + 2u + lane * HD_PER_THREAD + 3u] = acc3;
    }
}
)WGSL";

// [shared] gqa_chunked_pass2 (3 bindings, hand)
static const char* WGSL_GQA_CHUNKED_PASS2 = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;

    let n_chunks = _params_[4];
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let partial_stride = HD + 2u;
    let head_base = head * max_chunks * partial_stride;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var c = 0u; c < n_chunks; c = c + 1u) {
        let base = head_base + c * partial_stride;
        let m_chunk = Partials[base];
        let l_chunk = Partials[base + 1u];
        let v0 = Partials[base + 2u + lane * HD_PER_THREAD];
        let v1 = Partials[base + 2u + lane * HD_PER_THREAD + 1u];
        let v2 = Partials[base + 2u + lane * HD_PER_THREAD + 2u];
        let v3 = Partials[base + 2u + lane * HD_PER_THREAD + 3u];

        if (l_chunk > 0.0) {
            let m_new = max(m_prev, m_chunk);
            let exp_prev = exp(m_prev - m_new);
            let exp_chunk = exp(m_chunk - m_new);
            let l_new = l_prev * exp_prev + l_chunk * exp_chunk;
            let rescale = l_prev * exp_prev / max(l_new, 1e-10);
            let w = l_chunk * exp_chunk / max(l_new, 1e-10);

            acc0 = acc0 * rescale + v0 * w;
            acc1 = acc1 * rescale + v1 * w;
            acc2 = acc2 * rescale + v2 * w;
            acc3 = acc3 * rescale + v3 * w;

            m_prev = m_new;
            l_prev = l_new;
        }
    }

    Out[head * HD + lane * HD_PER_THREAD] = acc0;
    Out[head * HD + lane * HD_PER_THREAD + 1u] = acc1;
    Out[head * HD + lane * HD_PER_THREAD + 2u] = acc2;
    Out[head * HD + lane * HD_PER_THREAD + 3u] = acc3;
}
)WGSL";

// [shared] gqa_fused_attn (5 bindings, hand)
static const char* WGSL_GQA_FUSED_ATTN = R"WGSL(
enable f16;
enable subgroups;

// Fused single-dispatch GQA decode attention with fp16 KV cache.
// Replaces the 2-dispatch chunked approach (attn_p1 + attn_p2).
//
// Each workgroup handles one Q head for T=1 decode.
// Grid: (n_head, 1, 1)
// WG: 32 threads (1 warp)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;
const MAX_SEQ: u32 = 4096u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var t = 0u; t < MAX_SEQ; t = t + 1u) {
        let valid = t < T_total;
        let k_base = select(0u, t * kv_stride + kv_off, valid);
        let k0 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD]), valid);
        let k1 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD + 1u]), valid);
        let k2 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD + 2u]), valid);
        let k3 = select(0.0, f32(K_cache[k_base + lane * HD_PER_THREAD + 3u]), valid);

        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot_qk = subgroupAdd(partial);
        let score = select(neg_inf, dot_qk * scale, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v0 = select(0.0, f32(V_cache[k_base + lane * HD_PER_THREAD]), valid);
        let v1 = select(0.0, f32(V_cache[k_base + lane * HD_PER_THREAD + 1u]), valid);
        let v2 = select(0.0, f32(V_cache[k_base + lane * HD_PER_THREAD + 2u]), valid);
        let v3 = select(0.0, f32(V_cache[k_base + lane * HD_PER_THREAD + 3u]), valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    Out[head * HD + lane * HD_PER_THREAD]      = acc0;
    Out[head * HD + lane * HD_PER_THREAD + 1u] = acc1;
    Out[head * HD + lane * HD_PER_THREAD + 2u] = acc2;
    Out[head * HD + lane * HD_PER_THREAD + 3u] = acc3;
}
)WGSL";

// [shared] rms_norm (5 bindings, triton)
static const char* WGSL_RMS_NORM = R"WGSL(
enable subgroups;

// Auto-generated by Triton WebGPU Backend
// Kernel: rms_norm_loop_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // X
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Y
@group(0) @binding(2) var<storage, read> buf2: array<f32>;  // W
@group(0) @binding(3) var<storage, read_write> buf3: array<f32>;  // Rstd

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v10: i32 = i32(_wg_id.x);
    let v11: i32 = params.N + 127;
    let v12: i32 = v11 / 128;
    var v14: i32 = 0;
    var v15_0: f32 = f32(0);
    loop {
        let v16: bool = v14 < v12;
        if !v16 { break; }
        let v18: i32 = v14 * 128;
        let v19: i32 = i32(_lid.x);
        let v20: i32 = v19 & 127;
        let v21: i32 = i32(u32(v20) % u32(32));
        let v22: i32 = i32(_lid.x);
        let v23: i32 = i32(u32(v22) / u32(32));
        let v24: i32 = v21 << u32(0);
        let v25: i32 = 0 | v24;
        let v26: i32 = v23 << u32(5);
        let v27: i32 = v25 | v26;
        let v28: i32 = v27 & 127;
        let v29: i32 = i32(u32(v28) >> u32(0));
        let v30: i32 = v29 | 0;
        let v31: i32 = 0 ^ v30;
        let v32: i32 = v31 ^ 0;
        let v33: i32 = v32 + 0;
        let v34: i32 = v18 + v33;
        let v35: bool = v34 < params.N;
        let v36: i32 = v10 * params.stride;
        let v39: f32 = buf0[u32((v36 + v34))];
        let v40: f32 = select(f32(0.0), v39, v35);
        let v41: f32 = v40 * v40;
        let v43: f32 = v15_0 + v41;
        let v45: i32 = v14 + 1;
        v14 = v45;
        v15_0 = v43;
    }
    let v48: i32 = bitcast<i32>(v15_0);
    let v49: i32 = subgroupShuffleXor(v48, u32(16));
    let v50: f32 = bitcast<f32>(v49);
    let v51: f32 = v15_0 + v50;
    let v52: i32 = bitcast<i32>(v51);
    let v53: i32 = subgroupShuffleXor(v52, u32(8));
    let v54: f32 = bitcast<f32>(v53);
    let v55: f32 = v51 + v54;
    let v56: i32 = bitcast<i32>(v55);
    let v57: i32 = subgroupShuffleXor(v56, u32(4));
    let v58: f32 = bitcast<f32>(v57);
    let v59: f32 = v55 + v58;
    let v60: i32 = bitcast<i32>(v59);
    let v61: i32 = subgroupShuffleXor(v60, u32(2));
    let v62: f32 = bitcast<f32>(v61);
    let v63: f32 = v59 + v62;
    let v64: i32 = bitcast<i32>(v63);
    let v65: i32 = subgroupShuffleXor(v64, u32(1));
    let v66: f32 = bitcast<f32>(v65);
    let v67: f32 = v63 + v66;
    let v68: i32 = i32(_lid.x);
    let v69: i32 = v68 & 127;
    let v70: i32 = i32(u32(v69) % u32(32));
    let v71: i32 = i32(_lid.x);
    let v72: i32 = i32(u32(v71) / u32(32));
    let v73: i32 = v70 << u32(0);
    let v74: i32 = 0 | v73;
    let v75: i32 = v72 << u32(5);
    let v76: i32 = v74 | v75;
    let v77: i32 = v76 & 96;
    let v78: i32 = i32(u32(v77) >> u32(3));
    let v79: i32 = 0 | v78;
    let v80: i32 = 0 ^ v79;
    let v81: i32 = v80 ^ 0;
    let v82: i32 = v81 ^ 0;
    let v83: i32 = v82 + 0;
    _smem[u32(v83) >> 2u] = bitcast<i32>(v67);
    workgroupBarrier();
    let v87: i32 = i32(_lid.x);
    let v88: i32 = v87 & 127;
    let v89: i32 = i32(u32(v88) % u32(32));
    let v90: i32 = i32(_lid.x);
    let v91: i32 = i32(u32(v90) / u32(32));
    let v92: i32 = v89 << u32(0);
    let v93: i32 = 0 | v92;
    let v94: i32 = v91 << u32(5);
    let v95: i32 = v93 | v94;
    let v96: i32 = v95 & 3;
    let v97: i32 = v96 << u32(2);
    let v98: i32 = v97 | 0;
    let v99: i32 = 0 ^ v98;
    let v100: i32 = v99 ^ 0;
    let v101: i32 = v100 ^ 0;
    let v102: i32 = v101 + 0;
    let v104: f32 = bitcast<f32>(_smem[u32(v102) >> 2u]);
    let v107: i32 = bitcast<i32>(v104);
    let v108: i32 = subgroupShuffleXor(v107, u32(2));
    let v109: f32 = bitcast<f32>(v108);
    let v110: f32 = v104 + v109;
    let v111: i32 = bitcast<i32>(v110);
    let v112: i32 = subgroupShuffleXor(v111, u32(1));
    let v113: f32 = bitcast<f32>(v112);
    let v114: f32 = v110 + v113;
    let v115: f32 = f32(params.N);
    let v116: f32 = v114 / v115;
    let v117: f32 = v116 + params.eps;
    let v118: f32 = sqrt(v117);
    let v119: f32 = f32(1.0) / v118;
    var v121: i32 = 0;
    loop {
        let v122: bool = v121 < v12;
        if !v122 { break; }
        let v124: i32 = v121 * 128;
        let v125: i32 = i32(_lid.x);
        let v126: i32 = v125 & 127;
        let v127: i32 = i32(u32(v126) % u32(32));
        let v128: i32 = i32(_lid.x);
        let v129: i32 = i32(u32(v128) / u32(32));
        let v130: i32 = v127 << u32(0);
        let v131: i32 = 0 | v130;
        let v132: i32 = v129 << u32(5);
        let v133: i32 = v131 | v132;
        let v134: i32 = v133 & 127;
        let v135: i32 = i32(u32(v134) >> u32(0));
        let v136: i32 = v135 | 0;
        let v137: i32 = 0 ^ v136;
        let v138: i32 = v137 ^ 0;
        let v139: i32 = v138 + 0;
        let v140: i32 = v124 + v139;
        let v141: bool = v140 < params.N;
        let v142: i32 = v10 * params.stride;
        let v145: f32 = buf0[u32((v142 + v140))];
        let v146: f32 = select(f32(0.0), v145, v141);
        let v148: f32 = buf2[u32(v140)];
        let v149: f32 = select(f32(1.0), v148, v141);
        let v152: f32 = v146 * v119;
        let v153: f32 = v152 * v149;
        let v154: f32 = select(f32(0), v153, v141);
        if v141 { buf1[u32((v142 + v140))] = v154; }
        let v155: i32 = v121 + 1;
        v121 = v155;
    }
    if _lid.x == 0u { buf3[u32(v10)] = v119; }
}
)WGSL";

// [shared] rms_norm_batched (5 bindings, hand)
static const char* WGSL_RMS_NORM_BATCHED = R"WGSL(
enable subgroups;

// Batched RMSNorm for T rows: Y[t] = X[t] * W / rms(X[t]) for each row t.
// Grid: (T, 1, 1) — one workgroup per row.
// WG: 128 threads.

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // X [T × N]
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Y [T × N]
@group(0) @binding(2) var<storage, read> buf2: array<f32>;  // W [N]
@group(0) @binding(3) var<storage, read_write> buf3: array<f32>;  // Rstd [T]

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

const MAX_N: u32 = 8192u;  // supports up to E=8192

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
) {
    let row = i32(_wg_id.x);
    let tid = i32(_lid.x);
    let N = params.N;
    let stride = params.stride;
    let base = row * stride;

    var sum_sq: f32 = 0.0;
    var idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let v = buf0[u32(base + idx)];
            sum_sq += v * v;
        }
    }

    // Warp reduce
    let warp_sum = subgroupAdd(sum_sq);

    // Cross-warp reduce via shared memory
    let warp_id = tid / 32;
    let lane_id = tid % 32;
    if (lane_id == 0) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    var total: f32 = 0.0;
    if (tid < 4) {
        total = bitcast<f32>(_smem[tid]);
    }
    let final_sum = subgroupAdd(total);

    let rstd = 1.0 / sqrt(final_sum / f32(N) + params.eps);
    if (tid == 0) {
        buf3[u32(row)] = rstd;
    }

    // Apply norm + weight
    idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let v = buf0[u32(base + idx)];
            buf1[u32(base + idx)] = v * rstd * buf2[u32(idx)];
        }
    }
}
)WGSL";

// [shared] rope_batched_simple (9 bindings, hand)
static const char* WGSL_ROPE_BATCHED_SIMPLE = R"WGSL(
enable f16;
enable subgroups;

// Simple batched RoPE + QK-norm + KV cache scatter for prefill.
// Each workgroup handles ONE head for ONE token.
// Grid: (n_heads_total, T, 1) where n_heads_total = n_head + n_kv
//   WG (head < n_head, t): processes Q head
//   WG (head >= n_head, t): processes K head + V scatter
//
// QKV layout: [T × (qDim + 2*kvDim)] row-major
// Output Q: QRot[T × qDim] row-major (T × n_head × HD)
// Output K/V: scattered into KV cache at positions pos_offset..pos_offset+T-1
//
// Params: [n_head, qDim, kvDim, pos_offset, half_dim, cache_len, eps_u32, n_kv]

@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QNormW: array<f32>;
@group(0) @binding(7) var<storage, read> KNormW: array<f32>;
@group(0) @binding(8) var<storage, read> _params_: array<u32>;

const HD: u32 = 128u;
const HALF: u32 = 64u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = wid.x;   // which head (0..n_head-1 for Q, n_head..n_head+n_kv-1 for K)
    let t = wid.y;           // which token in the batch
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let cache_len = _params_[5];
    let eps = bitcast<f32>(_params_[6]);
    let n_kv = _params_[7];

    let qkv_stride = qDim + 2u * kvDim;
    let pos = pos_offset + t;

    if (head_idx < n_head) {
        // ── Q head processing ────────────────────────────────────────
        let src_base = t * qkv_stride + head_idx * HD;

        // Load Q values and compute RMS norm
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            let v = QKV[src_base + i];
            sum_sq += v * v;
        }
        // Warp reduce
        let warp_sum = subgroupAdd(sum_sq);
        var smem: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { smem[warp_id] = warp_sum; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = smem[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply norm + RoPE, write to QRot
        let dst_base = t * n_head * HD + head_idx * HD;
        let cos_base = pos * half_dim;
        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let qi = QKV[src_base + i] * rstd * QNormW[i];
            let qj = QKV[src_base + j] * rstd * QNormW[j];
            let c = Cos[cos_base + i];
            let s = Sin[cos_base + i];
            QRot[dst_base + i] = qi * c - qj * s;
            QRot[dst_base + j] = qj * c + qi * s;
        }
    } else {
        // ── KV head processing ───────────────────────────────────────
        let kv_head = head_idx - n_head;
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        // K: load, norm, RoPE, scatter to cache
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            let v = QKV[k_src + i];
            sum_sq += v * v;
        }
        let warp_sum = subgroupAdd(sum_sq);
        var smem2: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { smem2[warp_id] = warp_sum; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = smem2[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // KV cache layout: [MAX_SEQ × n_kv × HD]
        let cache_pos = cache_len + t;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;

        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let ki = QKV[k_src + i] * rstd * KNormW[i];
            let kj = QKV[k_src + j] * rstd * KNormW[j];
            let c = Cos[pos * half_dim + i];
            let s = Sin[pos * half_dim + i];
            K_cache[k_dst + i] = f16(ki * c - kj * s);
            K_cache[k_dst + j] = f16(kj * c + ki * s);
        }

        // V: just copy to cache (no RoPE, no norm)
        for (var i = tid; i < HD; i += 128u) {
            V_cache[v_dst + i] = f16(QKV[v_src + i]);
        }
    }
}
)WGSL";

// [shared] silu_mul_fused (3 bindings, triton)
static const char* WGSL_SILU_MUL_FUSED = R"WGSL(
// Auto-generated by Triton WebGPU Backend
// Kernel: silu_mul_fused_kernel
// Workgroup size: 128 (4 warps x 32 threads)

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // GateUp
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Out

struct Params {
    N: i32,
};
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v6: i32 = i32(_wg_id.x);
    let v7: i32 = v6 * 128;
    let v8: i32 = i32(_lid.x);
    let v9: i32 = v8 & 127;
    let v10: i32 = i32(u32(v9) % u32(32));
    let v11: i32 = i32(_lid.x);
    let v12: i32 = i32(u32(v11) / u32(32));
    let v13: i32 = v10 << u32(0);
    let v14: i32 = 0 | v13;
    let v15: i32 = v12 << u32(5);
    let v16: i32 = v14 | v15;
    let v17: i32 = v16 & 127;
    let v18: i32 = i32(u32(v17) >> u32(0));
    let v19: i32 = v18 | 0;
    let v20: i32 = 0 ^ v19;
    let v21: i32 = v20 ^ 0;
    let v22: i32 = v21 + 0;
    let v23: i32 = v7 + v22;
    let v24: bool = v23 < params.N;
    let v26: f32 = buf0[u32(v23)];
    let v27: f32 = select(f32(0.0), v26, v24);
    let v30: f32 = buf0[u32((params.N + v23))];
    let v31: f32 = select(f32(0.0), v30, v24);
    let v32: f32 = f32(0.0) - v27;
    let v33: f32 = exp(v32);
    let v34: f32 = v33 + f32(1.0);
    let v35: f32 = v27 / v34;
    let v37: f32 = v35 * v31;
    let v38: f32 = select(f32(0), v37, v24);
    if v24 { buf1[u32(v23)] = v38; }
}
)WGSL";


inline const std::unordered_map<std::string, ShaderInfo>& getEmbeddedKernels() {
    static const std::unordered_map<std::string, ShaderInfo> kernels = {
        {"q8_down_silu_add", {WGSL_Q8_DOWN_SILU_ADD, 6, false}},
        {"q8_down_silu_add_batched", {WGSL_Q8_DOWN_SILU_ADD_BATCHED, 6, false}},
        {"q8_down_silu_add_mma", {WGSL_Q8_DOWN_SILU_ADD_MMA, 6, false}},
        {"q8_down_silu_add_tiled", {WGSL_Q8_DOWN_SILU_ADD_TILED, 6, false}},
        {"q8_matmul", {WGSL_Q8_MATMUL, 6, false}},
        {"q8_matmul_add", {WGSL_Q8_MATMUL_ADD, 6, false}},
        {"q8_matmul_add_batched", {WGSL_Q8_MATMUL_ADD_BATCHED, 6, false}},
        {"q8_matmul_add_fast", {WGSL_Q8_MATMUL_ADD_FAST, 6, false}},
        {"q8_matmul_add_lite", {WGSL_Q8_MATMUL_ADD_LITE, 6, false}},
        {"q8_matmul_add_smem", {WGSL_Q8_MATMUL_ADD_SMEM, 6, false}},
        {"q8_matmul_batched", {WGSL_Q8_MATMUL_BATCHED, 6, false}},
        {"q8_matmul_dp4a", {WGSL_Q8_MATMUL_DP4A, 6, false}},
        {"q8_matmul_fast", {WGSL_Q8_MATMUL_FAST, 6, false}},
        {"q8_matmul_lite", {WGSL_Q8_MATMUL_LITE, 6, false}},
        {"q8_matmul_mma", {WGSL_Q8_MATMUL_MMA, 6, false}},
        {"q8_matmul_norm", {WGSL_Q8_MATMUL_NORM, 7, false}},
        {"q8_matmul_smem", {WGSL_Q8_MATMUL_SMEM, 6, false}},
        {"q8_matmul_tiled", {WGSL_Q8_MATMUL_TILED, 6, false}},
        {"q8_matmul_vec4", {WGSL_Q8_MATMUL_VEC4, 6, false}},
        {"add_rms_norm", {WGSL_ADD_RMS_NORM, 6, true}},
        {"add_rms_norm_batched", {WGSL_ADD_RMS_NORM_BATCHED, 6, false}},
        {"argmax", {WGSL_ARGMAX, 3, false}},
        {"causal_attn", {WGSL_CAUSAL_ATTN, 5, false}},
        {"embed_gather", {WGSL_EMBED_GATHER, 4, false}},
        {"flash_attn_mma", {WGSL_FLASH_ATTN_MMA, 5, false}},
        {"fp16_gemm", {WGSL_FP16_GEMM, 5, false}},
        {"fp16_gemm_wide", {WGSL_FP16_GEMM_WIDE, 5, false}},
        {"fused_qknorm_rope", {WGSL_FUSED_QKNORM_ROPE, 9, true}},
        {"fused_qknorm_rope_batched", {WGSL_FUSED_QKNORM_ROPE_BATCHED, 9, false}},
        {"gqa_chunked_pass1", {WGSL_GQA_CHUNKED_PASS1, 5, false}},
        {"gqa_chunked_pass2", {WGSL_GQA_CHUNKED_PASS2, 3, false}},
        {"gqa_fused_attn", {WGSL_GQA_FUSED_ATTN, 5, false}},
        {"rms_norm", {WGSL_RMS_NORM, 5, true}},
        {"rms_norm_batched", {WGSL_RMS_NORM_BATCHED, 5, false}},
        {"rope_batched_simple", {WGSL_ROPE_BATCHED_SIMPLE, 9, false}},
        {"silu_mul_fused", {WGSL_SILU_MUL_FUSED, 3, true}},
    };
    return kernels;
}

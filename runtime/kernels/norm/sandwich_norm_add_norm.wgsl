enable subgroups;

// Fused sandwich norm: RMSNorm(A, W1) → add to X → RMSNorm(X, W2) → Y
// Replaces 3 dispatches: post_attn_norm + attn_residual_add + pre_ffn_norm
// Grid: (1, 1, 1) — single workgroup for single-row decode
// WG: 128 threads (4 warps)
//
// Uses shared memory to cache updated X for pass 2, avoiding
// storage buffer re-reads on D3D12/Dawn.
// Pass 1 uses subgroupAdd; Pass 2 uses tree reduction (avoids
// D3D12 issue with multiple subgroupAdd calls in one kernel).

const STRIDE: u32 = 128u;

var<workgroup> _smem: array<f32, 4096>; // dual-use: X cache + tree reduction

@group(0) @binding(0) var<storage, read> A: array<f32>;       // projOutBuf
@group(0) @binding(1) var<storage, read_write> X: array<f32>; // xBuf
@group(0) @binding(2) var<storage, read_write> Y: array<f32>; // normOutBuf
@group(0) @binding(3) var<storage, read> W1: array<f32>;      // post_attn_norm weight
@group(0) @binding(4) var<storage, read> W2: array<f32>;      // pre_ffn_norm weight
@group(0) @binding(5) var<storage, read> params: array<u32>;  // [N, 0, eps_as_u32, 0]

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let N = params[0];
    let eps = bitcast<f32>(params[2]);
    let warp_id = tid / 32u;
    let lane_id = tid % 32u;

    // ── Pass 1: Compute rstd1 from A ──
    var sum_sq1: f32 = 0.0;
    for (var idx = tid; idx < N; idx += STRIDE) {
        let v = A[idx];
        sum_sq1 += v * v;
    }
    let ws1 = subgroupAdd(sum_sq1);
    if (lane_id == 0u) { _smem[tid] = ws1; }
    workgroupBarrier();
    let fs1 = _smem[0] + _smem[32] + _smem[64] + _smem[96];
    let rstd1 = 1.0 / sqrt(fs1 / f32(N) + eps);

    // ── Fused: norm(A)*W1 + X → write to X and cache in smem, accumulate pass-2 sum ──
    var sum_sq2: f32 = 0.0;
    for (var idx = tid; idx < N; idx += STRIDE) {
        let new_x = X[idx] + A[idx] * rstd1 * W1[idx];
        X[idx] = new_x;       // update storage
        _smem[idx] = new_x;   // cache in shared memory
        sum_sq2 += new_x * new_x;
    }
    workgroupBarrier();

    // ── Pass 2: tree reduction for rstd2 ──
    // Use smem slots [4080..4095+] for tree reduction (above X cache)
    let red_base = 3968u; // 4096 - 128
    _smem[red_base + tid] = sum_sq2;
    workgroupBarrier();
    if (tid < 64u) { _smem[red_base + tid] += _smem[red_base + tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { _smem[red_base + tid] += _smem[red_base + tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { _smem[red_base + tid] += _smem[red_base + tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { _smem[red_base + tid] += _smem[red_base + tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { _smem[red_base + tid] += _smem[red_base + tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { _smem[red_base + tid] += _smem[red_base + tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { _smem[red_base] += _smem[red_base + 1u]; }
    workgroupBarrier();

    let rstd2 = 1.0 / sqrt(_smem[red_base] / f32(N) + eps);

    // ── Write Y from cached smem ──
    for (var idx = tid; idx < N; idx += STRIDE) {
        Y[idx] = _smem[idx] * rstd2 * W2[idx];
    }
}

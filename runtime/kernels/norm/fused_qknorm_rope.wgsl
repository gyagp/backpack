// @meta generated=true
enable f16;

// Fused QK-norm + RoPE kernel for single-token decode.
// Applies RMSNorm (with learned weights) to Q/K, then RoPE, then scatters K/V to cache.
// Optionally applies V-norm (RMSNorm without learned scale) to V before cache write.
//
// Grid: (n_head + n_kv, 1, 1)
//   WG x < n_head  → Q head
//   WG x >= n_head → K head + V scatter
//
// Workgroup: 128 threads, each handles HD/WG_SIZE elements
// For HD=128: 1 element/thread. HD=256: 2. HD=512: 4.

const WG_SIZE: u32 = 128u;

var<workgroup> _q_sums: array<f32, 128>;
var<workgroup> _v_sums: array<f32, 128>;

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
    v_norm: i32,   // 1 = apply RMSNorm (no scale) to V before cache write
};
@group(0) @binding(8) var<storage, read> params: Params;

const HD: u32 = 128u;
const EPT: u32 = (HD + WG_SIZE - 1u) / WG_SIZE;  // elements per thread

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head_idx = i32(wid.x);
    let tid = lid.x;
    let is_q = head_idx < params.n_head;
    let kv_head = head_idx - params.n_head;

    // Source offset into QKV buffer
    let q_off = head_idx * i32(HD);
    let k_off = params.q_size + kv_head * i32(HD);
    let src_off = select(k_off, q_off, is_q);

    // V offset (only used by KV heads)
    let v_off_base = params.q_size + params.kv_size + kv_head * i32(HD);
    let do_vnorm = params.v_norm != 0 && !is_q;

    // Step 1: Compute sum of squares for RMSNorm (Q/K) and optionally V
    var sum_sq: f32 = 0.0;
    var v_sum_sq: f32 = 0.0;
    for (var e: u32 = 0u; e < EPT; e++) {
        let d = tid + e * WG_SIZE;
        if (d < HD) {
            let val = buf0[u32(src_off + i32(d))];
            sum_sq += val * val;
            if (do_vnorm) {
                let vval = buf0[u32(v_off_base + i32(d))];
                v_sum_sq += vval * vval;
            }
        }
    }

    // Workgroup reduction. Do not assume a particular hardware subgroup width:
    // AMD commonly exposes 64-wide subgroups while other adapters use 32.
    _q_sums[tid] = sum_sq;
    _v_sums[tid] = v_sum_sq;
    workgroupBarrier();
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            _q_sums[tid] += _q_sums[tid + stride];
            _v_sums[tid] += _v_sums[tid + stride];
        }
        workgroupBarrier();
    }

    // RMSNorm scale: 1 / sqrt(mean_sq + eps)
    let rms_scale = 1.0 / sqrt(_q_sums[0] / f32(HD) + params.eps);

    // V RMSNorm scale (only for KV heads with v_norm enabled)
    var v_rms_scale: f32 = 1.0;
    if (do_vnorm) {
        v_rms_scale = 1.0 / sqrt(_v_sums[0] / f32(HD) + params.eps);
    }

    // Step 2: Apply RMSNorm + RoPE + scatter
    for (var e: u32 = 0u; e < EPT; e++) {
        let d = i32(tid + e * WG_SIZE);
        if (u32(d) >= HD) { continue; }

        // Load and apply RMSNorm
        let val = buf0[u32(src_off + d)];
        let norm_w = select(buf7[u32(d)], buf6[u32(d)], is_q);
        let normed = val * rms_scale * norm_w;

        // RoPE
        let half_rot = params.half_rot;
        let rot_dim = half_rot * 2;
        let in_rot = d < rot_dim;

        let d_mod = d % half_rot;
        let cos_sin_idx = params.pos * half_rot + d_mod;
        let c = select(f32(1.0), buf4[u32(cos_sin_idx)], in_rot);
        let s = select(f32(0.0), buf5[u32(cos_sin_idx)], in_rot);

        // Pair element for rotation
        let in_first_half = d < half_rot;
        let pair_d = select(d - half_rot, d + half_rot, in_first_half);
        let pair_d_final = select(d, pair_d, in_rot);
        let pair_val = buf0[u32(src_off + pair_d_final)];
        let pair_norm_w = select(buf7[u32(pair_d_final)], buf6[u32(pair_d_final)], is_q);
        let pair_normed = pair_val * rms_scale * pair_norm_w;

        let sign = select(f32(1.0), f32(-1.0), in_first_half);
        let rotated = select(normed, normed * c + sign * pair_normed * s, in_rot);

        // Write Q output
        if (is_q) {
            buf1[u32(head_idx * i32(HD) + d)] = rotated;
        }

        // Write K/V to cache
        if (!is_q) {
            buf2[u32(params.cache_offset + kv_head * i32(HD) + d)] = f16(rotated);
            let v_val = buf0[u32(v_off_base + d)];
            let v_normed = select(v_val, v_val * v_rms_scale, do_vnorm);
            buf3[u32(params.cache_offset + kv_head * i32(HD) + d)] = f16(v_normed);
        }
    }
}

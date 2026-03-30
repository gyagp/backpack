// @meta triton=true
enable f16;
enable subgroups;

// Hand-written fused RoPE kernel for single-token decode.
// Applies RoPE to Q and K heads, scatters K/V into cache.
// No QK-norm is applied (norm weight buffers are identity=1.0).
//
// Grid: (n_head + n_kv, 1, 1)
//   WG x < n_head  → Q head
//   WG x >= n_head  → K head + V scatter
//
// Must declare all 9 bindings to match bind group layout (auto-layout).

@group(0) @binding(0) var<storage, read> buf0: array<f32>;       // QKV
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>; // Q_out
@group(0) @binding(2) var<storage, read_write> buf2: array<f16>; // K_cache
@group(0) @binding(3) var<storage, read_write> buf3: array<f16>; // V_cache
@group(0) @binding(4) var<storage, read> buf4: array<f32>;       // CosTable
@group(0) @binding(5) var<storage, read> buf5: array<f32>;       // SinTable
@group(0) @binding(6) var<storage, read> buf6: array<f32>;       // NormQ (identity)
@group(0) @binding(7) var<storage, read> buf7: array<f32>;       // NormK (identity)

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

const HD: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = i32(wid.x);
    let d = i32(lid.x);  // 0..127 = element within head

    let is_q = head_idx < params.n_head;
    let kv_head = head_idx - params.n_head;

    // Source offset into QKV buffer
    let q_off = head_idx * 128;
    let k_off = params.q_size + kv_head * 128;
    let src_off = select(k_off, q_off, is_q);

    // Load value
    let val = buf0[u32(src_off + d)];

    // Read norm weight (identity=1.0 for non-QK-norm models)
    let norm_w = select(buf7[u32(d)], buf6[u32(d)], is_q);
    // Apply identity norm weight (no-op multiply by 1.0)
    let normed = val * norm_w;

    // RoPE parameters
    let half_rot = params.half_rot;
    let rot_dim = half_rot * 2;
    let in_rot = d < rot_dim;

    // Cos/sin lookup
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
    let pair_normed = pair_val * pair_norm_w;

    // Apply rotation: first half uses -sin, second half uses +sin
    let sign = select(f32(1.0), f32(-1.0), in_first_half);
    let rotated = select(normed, normed * c + sign * pair_normed * s, in_rot);

    // Write Q output
    if (is_q) {
        buf1[u32(head_idx * 128 + d)] = rotated;
    }

    // Write K/V to cache
    if (!is_q) {
        buf2[u32(params.cache_offset + kv_head * 128 + d)] = f16(rotated);
        // V: straight copy, no RoPE
        let v_off = params.q_size + params.kv_size + kv_head * 128 + d;
        buf3[u32(params.cache_offset + kv_head * 128 + d)] = f16(buf0[u32(v_off)]);
    }
}

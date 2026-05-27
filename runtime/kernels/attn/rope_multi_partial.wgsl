// Partial-rotation Multi-axis RoPE for qwen35moe
//
// qwen35moe has head_dim=256 but rope_dim=64 — only the first 64 elements
// per head rotate (across 4 axis sections [11,11,10,0]). The remaining
// 192 elements pass through unchanged.
//
// Bindings:
//   0: X         [n_head * head_dim]  in-place
//   1: cos_sin   [2 * rope_dim/2 * 4_sections, but only s0+s1+s2+s3 pairs used]
//   2: _params_  [n_head, head_dim, s0, s1, s2, s3, pos]
//
// rot_dim total pairs = s0 + s1 + s2 + s3 (each pair = 2 elements; total
// rotated = 2 * total_pairs which must equal rope_dim).

@group(0) @binding(0) var<storage, read_write> X:        array<f32>;
@group(0) @binding(1) var<storage, read>       cos_sin:  array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head    = _params_[0];
    let head_dim  = _params_[1];
    let s0        = _params_[2];
    let s1        = _params_[3];
    let s2        = _params_[4];
    let s3        = _params_[5];
    let total_pairs = s0 + s1 + s2 + s3;
    let rope_dim    = 2u * total_pairs;

    let head_idx = gid.y;
    let pair_idx = gid.x;
    if (head_idx >= n_head) { return; }
    if (pair_idx >= total_pairs) { return; }

    // Determine axis-section
    var section: u32 = 0u;
    var local_idx: u32 = pair_idx;
    if (local_idx >= s0) { local_idx = local_idx - s0; section = 1u; }
    if (section == 1u && local_idx >= s1) { local_idx = local_idx - s1; section = 2u; }
    if (section == 2u && local_idx >= s2) { local_idx = local_idx - s2; section = 3u; }

    var sec_off: u32 = 0u;
    if (section >= 1u) { sec_off = sec_off + 2u * s0; }
    if (section >= 2u) { sec_off = sec_off + 2u * s1; }
    if (section >= 3u) { sec_off = sec_off + 2u * s2; }
    let c = cos_sin[sec_off + 2u * local_idx + 0u];
    let s = cos_sin[sec_off + 2u * local_idx + 1u];

    // Apply rotation to first rope_dim elements of this head only.
    // pair (x_i, x_{i + total_pairs}) — neox-style — within first rope_dim of head.
    let head_base = head_idx * head_dim;
    let x0 = X[head_base + pair_idx];
    let x1 = X[head_base + pair_idx + total_pairs];
    X[head_base + pair_idx]               = x0 * c - x1 * s;
    X[head_base + pair_idx + total_pairs] = x0 * s + x1 * c;
    // Elements [rope_dim .. head_dim) per head are untouched.
}

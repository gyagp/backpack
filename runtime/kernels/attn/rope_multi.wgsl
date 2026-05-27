// Multi-axis RoPE (MRoPE) — used by qwen35moe attention layers for multimodal
//
// Standard RoPE rotates pairs (x_i, x_{i+nrot/2}) by angle theta_i = pos * base^(-2i/d).
// MRoPE splits the rotation dimensions into multiple sections (e.g. T/H/W axes
// for vision tokens) with each section getting position from a different axis.
//
// Sections: 4 ints [s0, s1, s2, s3] from rope.dimension_sections (qwen35moe = [11,11,10,0]).
// Each section uses its own position; for text decode all sections see the same pos.
//
// Bindings:
//   0: Q          [n_head * head_dim]       in-place (read+write)
//   1: cos_sin    [4 sections, each (rot_dim/2) cos + sin] precomputed per position
//   2: _params_  — [n_head, head_dim, s0, s1, s2, s3, pos]

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
    // s0+s1+s2+s3 = rot_dim/2 (total pairs to rotate per head)

    let head_idx = gid.y;
    let pair_idx = gid.x;  // 0..(rot_dim/2)-1
    if (head_idx >= n_head) { return; }
    let total_pairs = s0 + s1 + s2 + s3;
    if (pair_idx >= total_pairs) { return; }

    // Find which axis-section this pair belongs to.
    var section: u32 = 0u;
    var local_idx: u32 = pair_idx;
    if (local_idx >= s0) { local_idx = local_idx - s0; section = 1u; }
    if (section == 1u && local_idx >= s1) { local_idx = local_idx - s1; section = 2u; }
    if (section == 2u && local_idx >= s2) { local_idx = local_idx - s2; section = 3u; }

    // cos/sin table is laid out per section: section i has 2*s_i floats (cos+sin alternating).
    var sec_off: u32 = 0u;
    if (section >= 1u) { sec_off = sec_off + 2u * s0; }
    if (section >= 2u) { sec_off = sec_off + 2u * s1; }
    if (section >= 3u) { sec_off = sec_off + 2u * s2; }
    let c = cos_sin[sec_off + 2u * local_idx + 0u];
    let s = cos_sin[sec_off + 2u * local_idx + 1u];

    // Apply rotation to the pair (x_i, x_{i+total_pairs}) — neox-style
    let base = head_idx * head_dim + pair_idx;
    let x0 = X[base];
    let x1 = X[base + total_pairs];
    X[base]               = x0 * c - x1 * s;
    X[base + total_pairs] = x0 * s + x1 * c;
}

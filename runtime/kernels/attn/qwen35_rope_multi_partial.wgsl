// Qwen3.5 partial multi-section RoPE for decode.
//
// Text-only Qwen3.5 uses MRoPE sections over the rotated prefix. For the
// current GGUFs the 64 rotary dimensions are split as 32 pairs: [11, 11, 10, 0].
// Each section restarts the frequency index, matching ggml_rope_multi.
//
// Bindings:
//   0: X        [n_head * head_dim] f32, in-place
//   1: cos      [max_seq * rope_half] f32, standard RoPE cos table
//   2: sin      [max_seq * rope_half] f32, standard RoPE sin table
//   3: params   [n_head, head_dim, s0, s1, s2, s3, pos, rope_half]

@group(0) @binding(0) var<storage, read_write> X:      array<f32>;
@group(0) @binding(1) var<storage, read>       Cos:    array<f32>;
@group(0) @binding(2) var<storage, read>       Sin:    array<f32>;
@group(0) @binding(3) var<storage, read>       params: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head    = params[0];
    let head_dim  = params[1];
    let s0        = params[2];
    let s1        = params[3];
    let s2        = params[4];
    let s3        = params[5];
    let pos       = params[6];
    let rope_half = params[7];

    let total_pairs = s0 + s1 + s2 + s3;
    let head_idx = gid.y;
    let pair_idx = gid.x;
    if (head_idx >= n_head || pair_idx >= total_pairs) { return; }

    var section: u32 = 0u;
    var local_idx: u32 = pair_idx;
    if (local_idx >= s0) {
        local_idx = local_idx - s0;
        section = 1u;
    }
    if (section == 1u && local_idx >= s1) {
        local_idx = local_idx - s1;
        section = 2u;
    }
    if (section == 2u && local_idx >= s2) {
        local_idx = local_idx - s2;
        section = 3u;
    }

    let table_idx = pos * rope_half + local_idx;
    let c = Cos[table_idx];
    let s = Sin[table_idx];

    let head_base = head_idx * head_dim;
    let x0_idx = head_base + pair_idx;
    let x1_idx = head_base + pair_idx + total_pairs;
    let x0 = X[x0_idx];
    let x1 = X[x1_idx];

    X[x0_idx] = x0 * c - x1 * s;
    X[x1_idx] = x0 * s + x1 * c;
}

// Qwen3.5 Q RoPE decode path, writing the attention-ready Q buffer.
//
// Bindings:
//   0: X        [n_head * head_dim] f32, normalized Q input
//   1: Out      [n_head * head_dim] f32, rotated/copy output for attention
//   2: cos      [max_seq * rope_half] f32, standard RoPE cos table
//   3: sin      [max_seq * rope_half] f32, standard RoPE sin table
//   4: params   [n_head, head_dim, s0, s1, s2, s3, pos, rope_half]

@group(0) @binding(0) var<storage, read>       X:      array<f32>;
@group(0) @binding(1) var<storage, read_write> Out:    array<f32>;
@group(0) @binding(2) var<storage, read>       Cos:    array<f32>;
@group(0) @binding(3) var<storage, read>       Sin:    array<f32>;
@group(0) @binding(4) var<storage, read>       params: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head    = params[0];
    let head_dim  = params[1];
    let s0        = params[2];
    let s1        = params[3];
    let s2        = params[4];
    let s3        = params[5];
    let pos       = params[6];
    let rope_half = params[7];

    let head_idx = gid.y;
    let elem_idx = gid.x;
    if (head_idx >= n_head || elem_idx >= head_dim) { return; }

    let total_pairs = s0 + s1 + s2 + s3;
    let head_base = head_idx * head_dim;
    let out_idx = head_base + elem_idx;

    if (elem_idx < total_pairs) {
        let pair_idx = elem_idx;
        let table_idx = pos * rope_half + pair_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = X[head_base + pair_idx];
        let x1 = X[head_base + pair_idx + total_pairs];
        Out[out_idx] = x0 * c - x1 * s;
    } else if (elem_idx < total_pairs * 2u) {
        let pair_idx = elem_idx - total_pairs;
        let table_idx = pos * rope_half + pair_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = X[head_base + pair_idx];
        let x1 = X[head_base + pair_idx + total_pairs];
        Out[out_idx] = x0 * s + x1 * c;
    } else {
        Out[out_idx] = X[out_idx];
    }
}

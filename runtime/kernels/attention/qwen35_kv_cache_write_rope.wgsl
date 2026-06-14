enable f16;

// Qwen3.5 K RoPE decode path fused with fp16 KV cache write.
//
// Bindings:
//   0: K        [kv_dim] f32, normalized K input
//   1: V        [kv_dim] f32
//   2: K_cache  [seq * kv_dim] f16
//   3: V_cache  [seq * kv_dim] f16
//   4: cos      [max_seq * rope_half] f32, standard RoPE cos table
//   5: sin      [max_seq * rope_half] f32, standard RoPE sin table
//   6: rope_p   [n_kv_head, head_dim, s0, s1, s2, s3, pos, rope_half]
//   7: kv_p     [kv_dim, cache_offset_words]

@group(0) @binding(0) var<storage, read>       K:       array<f32>;
@group(0) @binding(1) var<storage, read>       V:       array<f32>;
@group(0) @binding(2) var<storage, read_write> KCache:  array<f16>;
@group(0) @binding(3) var<storage, read_write> VCache:  array<f16>;
@group(0) @binding(4) var<storage, read>       Cos:     array<f32>;
@group(0) @binding(5) var<storage, read>       Sin:     array<f32>;
@group(0) @binding(6) var<storage, read>       rope_p:  array<u32>;
@group(0) @binding(7) var<storage, read>       kv_p:    array<u32>;

fn section_local_idx(pair_idx: u32, s0: u32, s1: u32, s2: u32) -> u32 {
    var local_idx = pair_idx;
    var section = 0u;
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
    }
    return local_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_dim = kv_p[0];
    let off = kv_p[1];
    let n_kv_head = rope_p[0];
    let head_dim  = rope_p[1];
    let head_idx = gid.y;
    let elem_idx = gid.x;
    if (head_idx >= n_kv_head || elem_idx >= head_dim) { return; }

    let i = head_idx * head_dim + elem_idx;
    if (i >= kv_dim) { return; }

    let s0        = rope_p[2];
    let s1        = rope_p[3];
    let s2        = rope_p[4];
    let s3        = rope_p[5];
    let pos       = rope_p[6];
    let rope_half = rope_p[7];

    let head_base = head_idx * head_dim;
    let total_pairs = s0 + s1 + s2 + s3;

    var k_val = K[i];
    if (elem_idx < total_pairs) {
        let pair_idx = elem_idx;
        let local_idx = section_local_idx(pair_idx, s0, s1, s2);
        let table_idx = pos * rope_half + local_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = K[head_base + pair_idx];
        let x1 = K[head_base + pair_idx + total_pairs];
        k_val = x0 * c - x1 * s;
    } else if (elem_idx < total_pairs * 2u) {
        let pair_idx = elem_idx - total_pairs;
        let local_idx = section_local_idx(pair_idx, s0, s1, s2);
        let table_idx = pos * rope_half + local_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = K[head_base + pair_idx];
        let x1 = K[head_base + pair_idx + total_pairs];
        k_val = x0 * s + x1 * c;
    }

    KCache[off + i] = f16(k_val);
    VCache[off + i] = f16(V[i]);
}

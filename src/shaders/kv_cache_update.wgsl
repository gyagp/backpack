// KV-cache update: append M new K/V entries, packing f32 input to f16 (stored as u32 pairs)
// Cache layout: K_cache[(head * max_seq_len + seq) * half_hd + d_pair] where each u32 = 2 x f16
// new_K/new_V: [M * num_kv_heads * head_dim] in f32
// params[0]: (num_kv_heads, max_seq_len, head_dim, seq_pos_start)
// params[1].x: M (number of tokens to write)

@group(0) @binding(0) var<storage, read_write> K_cache : array<u32>;
@group(0) @binding(1) var<storage, read_write> V_cache : array<u32>;
@group(0) @binding(2) var<storage, read> new_K : array<f32>;
@group(0) @binding(3) var<storage, read> new_V : array<f32>;
@group(0) @binding(4) var<uniform> params : array<vec4u, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let num_kv_heads = params[0].x;
    let max_seq_len = params[0].y;
    let head_dim = params[0].z;
    let seq_pos_start = params[0].w;
    let M = params[1].x;

    // head_dim must be even (always true for power-of-2 LLM head dims)
    let half_hd = head_dim / 2u;
    let pairs_per_token = num_kv_heads * half_hd;
    let total_pairs = M * pairs_per_token;
    let idx = gid.x;
    if (idx >= total_pairs) {
        return;
    }

    let token_idx = idx / pairs_per_token;
    let within_token = idx % pairs_per_token;
    let head = within_token / half_hd;
    let d_pair = within_token % half_hd;

    let seq_pos = seq_pos_start + token_idx;
    let cache_offset = (head * max_seq_len + seq_pos) * half_hd + d_pair;

    let d = d_pair * 2u;
    let elements_per_token = num_kv_heads * head_dim;
    let src_base = token_idx * elements_per_token + head * head_dim + d;

    K_cache[cache_offset] = pack2x16float(vec2f(new_K[src_base], new_K[src_base + 1u]));
    V_cache[cache_offset] = pack2x16float(vec2f(new_V[src_base], new_V[src_base + 1u]));
}

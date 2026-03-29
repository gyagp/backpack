// KV cache write — Write new K or V token into static cache at position offset.
// Unlike kv_cache_append, this does NOT copy past data — the cache buffer is reused.
// new_kv: [kv_heads * head_dim]  (from current step, after RoPE)
// cache:  [kv_heads, max_seq, head_dim]  (static buffer, read_write)
// Params: [0]=kv_heads, [1]=head_dim, [2]=write_pos, [3]=max_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> new_kv: array<${T}>;
@group(0) @binding(1) var<storage, read_write> cache: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let write_pos = _params_[2];
    let max_seq = _params_[3];

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;

    // Write new value at cache[h, write_pos, d]
    t_write(&cache, (h * max_seq + write_pos) * head_dim + d, t_read(&new_kv, h * head_dim + d));
}

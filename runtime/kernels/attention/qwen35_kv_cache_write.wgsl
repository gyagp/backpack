enable f16;

// Write already-normalized/rotated Qwen3.5 K and raw V into fp16 KV cache.
//
// Bindings:
//   0: K        [kv_dim] f32
//   1: V        [kv_dim] f32
//   2: K_cache  [seq * kv_dim] f16
//   3: V_cache  [seq * kv_dim] f16
//   4: params   [kv_dim, cache_offset_words]

@group(0) @binding(0) var<storage, read>       K:       array<f32>;
@group(0) @binding(1) var<storage, read>       V:       array<f32>;
@group(0) @binding(2) var<storage, read_write> KCache:  array<f16>;
@group(0) @binding(3) var<storage, read_write> VCache:  array<f16>;
@group(0) @binding(4) var<storage, read>       params:  array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_dim = params[0];
    let off = params[1];
    let i = gid.x;
    if (i >= kv_dim) { return; }
    KCache[off + i] = f16(K[i]);
    VCache[off + i] = f16(V[i]);
}

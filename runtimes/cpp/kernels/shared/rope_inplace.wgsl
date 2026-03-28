// RoPE in-place — Rotary positional embedding applied in-place.
// One workgroup per head. Sequential loop over rotary dimensions.
// Params: [0]=num_heads, [1]=head_dim, [2]=rotary_dim, [3]=pos
// Dispatch: (ceil(num_heads/64), 1, 1)

${T_READ}
${T_READ_RW}
${T_WRITE}

@group(0) @binding(0) var<storage, read_write> data: array<${T}>;
@group(0) @binding(1) var<storage, read> cos_cache: array<${T}>;
@group(0) @binding(2) var<storage, read> sin_cache: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let rotary_dim = _params_[2];
    let pos = _params_[3];

    let h = gid.x;
    if (h >= num_heads) { return; }

    let half_rot = rotary_dim / 2u;
    let head_base = h * head_dim;
    let cache_base = pos * half_rot;

    for (var d = 0u; d < half_rot; d++) {
        let cos_val = t_read(&cos_cache, cache_base + d);
        let sin_val = t_read(&sin_cache, cache_base + d);
        let x0 = t_read_rw(&data, head_base + d);
        let x1 = t_read_rw(&data, head_base + half_rot + d);
        t_write(&data, head_base + d, x0 * cos_val - x1 * sin_val);
        t_write(&data, head_base + half_rot + d, x1 * cos_val + x0 * sin_val);
    }
}

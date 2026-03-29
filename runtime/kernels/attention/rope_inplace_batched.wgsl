// Batched RoPE — T tokens via workgroup_id.y.
// data: [T, num_heads, head_dim], positions [posOffset, posOffset+T)
// Params: [0]=num_heads, [1]=head_dim, [2]=rotary_dim, [3]=posOffset
// Dispatch: (ceil(num_heads/64), T, 1)

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let rotary_dim = _params_[2];
    let posOffset = _params_[3];
    let tok = wid.y;

    let h = gid.x;
    if (h >= num_heads) { return; }

    let pos = posOffset + tok;
    let half_rot = rotary_dim / 2u;
    let head_base = tok * num_heads * head_dim + h * head_dim;
    let cache_base = pos * half_rot;

    for (var d = 0u; d < half_rot; d++) {
        let cos_val = cos_cache[cache_base + d];
        let sin_val = sin_cache[cache_base + d];
        let x0 = data[head_base + d];
        let x1 = data[head_base + half_rot + d];
        data[head_base + d] = x0 * cos_val - x1 * sin_val;
        data[head_base + half_rot + d] = x1 * cos_val + x0 * sin_val;
    }
}

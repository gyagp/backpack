struct Params {
    axis_dim: u32,
    inner_dim: u32,
    index_count: u32,
}

@group(0) @binding(0) var<storage, read> data : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    let total = params.index_count * params.inner_dim;
    if (i >= total) {
        return;
    }
    let idx_pos = i / params.inner_dim;
    let inner_pos = i % params.inner_dim;
    let gather_idx = u32(indices[idx_pos]);
    output[i] = data[gather_idx * params.inner_dim + inner_pos];
}

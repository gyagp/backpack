// Scale — in-place element-wise multiply by scalar
// data[i] *= scale
// Dispatch: (ceil(N/256), 1, 1)
// Params: [N, bitcast<u32>(scale)]

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let scale = bitcast<f32>(params[1]);
    data[idx] *= scale;
}

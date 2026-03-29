// Weighted accumulate: out[i] += weight * src[i]
// Params: [0]=N, [1]=weight_as_u32
// Dispatch: (ceil(N/256), 1, 1)

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let weight = bitcast<f32>(_params_[1]);
    dst[idx] = dst[idx] + weight * src[idx];
}

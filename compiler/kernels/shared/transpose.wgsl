// Transpose — general N-dimensional permutation
// Works on u32 elements (f32 or packed fp16 pairs).
// Params: [total, ndim, 0, 0, out_strides[ndim], in_strides[ndim]]
//
// Dispatch: (ceil(total/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    var out_idx = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let coord = out_idx / out_stride;
        out_idx = out_idx % out_stride;
        in_flat += coord * in_stride;
    }
    Y[idx] = X[in_flat];
}

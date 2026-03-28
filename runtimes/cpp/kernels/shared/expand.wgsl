// Expand — broadcast tensor to larger shape
// Dispatch: (ceil(total/512), 1, 1) — each thread handles 2 elements
// Params: [total, ndim, 0, 0, out_strides[ndim], in_dims[ndim], in_strides[ndim]]

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_in_idx(out_idx: u32, ndim: u32) -> u32 {
    var remaining = out_idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_dim = _params_[4u + ndim + d];
        let in_stride = _params_[4u + 2u * ndim + d];
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;
        in_flat += (coord % in_dim) * in_stride;
    }
    return in_flat;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let ndim = _params_[1];

    let base = gid.x * 2u;
    if (base >= total) { return; }

    let v0 = t_read(&X, compute_in_idx(base, ndim));
    var v1: f32 = 0.0;
    if (base + 1u < total) {
        v1 = t_read(&X, compute_in_idx(base + 1u, ndim));
    }
    t_write2(&Y, base, v0, v1);
}

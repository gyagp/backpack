${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= total) { return; }

    var remaining = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let start = _params_[4u + 2u * ndim + d];
        let step = _params_[4u + 3u * ndim + d];
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;
        let in_coord = start + coord * step;
        in_flat += in_coord * in_stride;
    }
    let val = t_read(&X, in_flat);
    t_write(&Y, idx, val);
}


${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> src: array<${T}>;
@group(0) @binding(1) var<storage, read_write> dst: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let src_offset = _params_[0];
    t_write(&dst, idx, t_read(&src, src_offset + idx));
}

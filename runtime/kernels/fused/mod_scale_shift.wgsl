
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> x: array<${T}>;
@group(0) @binding(1) var<storage, read> scale: array<${T}>;
@group(0) @binding(2) var<storage, read> shift: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let D = _params_[0];
    let mod_idx = idx % D;
    let xv = t_read(&x, idx);
    let s = t_read(&scale, mod_idx);
    let sh = t_read(&shift, mod_idx);
    t_write(&output, idx, (1.0 + s) * xv + sh);
}

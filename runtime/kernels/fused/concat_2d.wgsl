
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> a: array<${T}>;
@group(0) @binding(1) var<storage, read> b: array<${T}>;
@group(0) @binding(2) var<storage, read_write> output: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_total = _params_[1];
    let idx = gid.x;
    if (idx >= N_total) { return; }
    let N_a = _params_[0];
    if (idx < N_a) {
        t_write(&output, idx, t_read(&a, idx));
    } else {
        t_write(&output, idx, t_read(&b, idx - N_a));
    }
}

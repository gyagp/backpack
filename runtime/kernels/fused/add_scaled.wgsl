
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read_write> acc: array<${T}>;
@group(0) @binding(1) var<storage, read> x: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let alpha = bitcast<f32>(_params_[1]);
    let a = t_read(&acc, idx);
    let v = t_read(&x, idx);
    t_write(&acc, idx, a + alpha * v);
}


${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> gate_up: array<${T}>;
@group(0) @binding(1) var<storage, read_write> output: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let gate = t_read(&gate_up, idx * 2u);
    let up = t_read(&gate_up, idx * 2u + 1u);
    let x = gate * 1.702;
    t_write(&output, idx, (up + 1.0) * gate * (1.0 / (1.0 + exp(-x))));
}

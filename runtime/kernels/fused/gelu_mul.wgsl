
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> gate: array<${T}>;
@group(0) @binding(1) var<storage, read> up: array<${T}>;
@group(0) @binding(2) var<storage, read_write> output: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let g = t_read(&gate, idx);
    let u = t_read(&up, idx);
    let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
    let tanh_val = 1.0 - 2.0 / (exp(2.0 * inner) + 1.0);
    let gelu = 0.5 * g * (1.0 + tanh_val);
    t_write(&output, idx, gelu * u);
}

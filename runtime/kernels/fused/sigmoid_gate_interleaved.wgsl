
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read_write> output: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let HD = _params_[1];
    let h = idx / HD;
    let d = idx % HD;
    let gate_idx = h * HD * 2u + d;
    let val_idx = gate_idx + HD;
    let g = t_read(&input, gate_idx);
    let v = t_read(&input, val_idx);
    t_write(&output, idx, (1.0 / (1.0 + exp(-g))) * v);
}

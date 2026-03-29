// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
// Input is [N*2] with interleaved layout: [gate[0], up[0], gate[1], up[1], ...]
// Output is [N].
// Params: [0]=N (half_size = moe_intermediate_size)
// Dispatch: (ceil(N/256), 1, 1)

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
    let silu = gate / (1.0 + exp(-gate));
    t_write(&output, idx, silu * up);
}

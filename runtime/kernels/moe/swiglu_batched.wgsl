// Batched SwiGLU: T tokens via workgroup_id.y.
// gate_up[tok * N*2 + i], output[tok * N + i]
// Params: [0]=N (half_size)
// Dispatch: (ceil(N/256), nTokens, 1)

@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let tok = wid.y;
    let gate = gate_up[tok * N * 2u + idx * 2u];
    let up = gate_up[tok * N * 2u + idx * 2u + 1u];
    let silu = gate / (1.0 + exp(-gate));
    output[tok * N + idx] = silu * up;
}

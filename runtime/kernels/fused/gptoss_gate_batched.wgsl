// Batched GPT-OSS gated activation: T tokens via workgroup_id.y.
// y = (up + 1.0) * gate * sigmoid(gate * 1.702)
// Input: gate_up[T * 2*N], output[T * N], interleaved gate/up per token.
// Params: [0]=N
// Dispatch: (ceil(N/256), T, 1)

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
    let in_base = tok * N * 2u;
    let out_base = tok * N;
    let gate = gate_up[in_base + idx * 2u];
    let up = gate_up[in_base + idx * 2u + 1u];
    let x = gate * 1.702;
    output[out_base + idx] = (up + 1.0) * gate * (1.0 / (1.0 + exp(-x)));
}

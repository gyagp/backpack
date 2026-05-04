// @meta bindings=2
// Logit softcapping: Y[i] = tanh(Y[i] / cap) * cap
// Applied in-place after LM head matmul, before argmax.
//
// Dispatch: (ceil(N/256), 1, 1)
//
// Bindings:
//   0: Y (read_write) — logits [N] fp32, modified in-place
//   1: _params_ — [N, cap_as_u32, 0, 0]

@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let cap = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= N) { return; }
    Y[i] = tanh(Y[i] / cap) * cap;
}

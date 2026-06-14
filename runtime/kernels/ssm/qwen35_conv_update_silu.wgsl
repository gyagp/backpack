// qwen35 SSM decode fused rolling-state update + depthwise conv + SiLU.
//
// Equivalent to:
//   conv_state_update(state, x)
//   y = conv1d_decode(state, weights, bias)
//   out = silu(y)
//
// State layout is [channels, K], oldest to newest.
// Bindings:
//   0: state   [channels * K] read_write
//   1: x       [channels] new sample
//   2: weights [channels * K]
//   3: bias    [channels]
//   4: out     [channels]
//   5: params  [channels, K]

@group(0) @binding(0) var<storage, read_write> state:   array<f32>;
@group(0) @binding(1) var<storage, read>       x:       array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read>       bias:    array<f32>;
@group(0) @binding(4) var<storage, read_write> out:     array<f32>;
@group(0) @binding(5) var<storage, read>       params:  array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channels = params[0];
    let K = params[1];
    let c = gid.x;
    if (c >= channels) { return; }

    let base = c * K;
    var acc = bias[c];

    for (var k = 0u; k + 1u < K; k = k + 1u) {
        let v = state[base + k + 1u];
        state[base + k] = v;
        acc = acc + weights[base + k] * v;
    }

    let newest = x[c];
    state[base + K - 1u] = newest;
    acc = acc + weights[base + K - 1u] * newest;

    out[c] = acc / (1.0 + exp(-acc));
}

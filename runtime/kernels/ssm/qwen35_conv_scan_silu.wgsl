// @meta bindings=6
// Batched causal depthwise convolution for Qwen 3.5 prefill. One invocation
// owns a channel and scans prompt rows in order, leaving State ready for decode.

@group(0) @binding(0) var<storage, read_write> State: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channels = P[0]; let convK = P[1]; let T = P[2];
    let c = gid.x;
    if (c >= channels) { return; }
    let base = c * convK;
    for (var t = 0u; t < T; t++) {
        var acc = Bias[c];
        for (var k = 0u; k + 1u < convK; k++) {
            let old = State[base + k + 1u];
            State[base + k] = old;
            acc += W[base + k] * old;
        }
        let newest = X[t * channels + c];
        State[base + convK - 1u] = newest;
        acc += W[base + convK - 1u] * newest;
        Y[t * channels + c] = acc / (1.0 + exp(-acc));
    }
}

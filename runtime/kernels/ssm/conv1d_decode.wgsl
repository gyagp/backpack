// SSM conv1d (depthwise) — Mamba decode primitive
//
// For decode mode (single new token), the SSM conv state is a rolling
// buffer of the last `conv_k` input vectors per channel, ordered oldest to
// newest like llama.cpp's ggml_ssm_conv input. This kernel
// performs the FIR filter:
//   out[c] = sum_{k=0..K-1} weights[c, k] * state[c, k]
// per channel c in 0..d_inner.
//
// Bindings:
//   0: state   [d_inner * K]  — rolling buffer (fp32)
//   1: weights [d_inner * K]  — depthwise conv1d kernel (fp32, from ssm_conv1d.weight)
//   2: bias    [d_inner]      — optional bias (fp32, may be zero buffer)
//   3: out     [d_inner]      — fp32 output
//   4: _params_                — [d_inner, K]
//
// Workgroup: 256 threads, each handles ceil(d_inner / 256) channels.

@group(0) @binding(0) var<storage, read>       state:    array<f32>;
@group(0) @binding(1) var<storage, read>       weights:  array<f32>;
@group(0) @binding(2) var<storage, read>       bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> out:      array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let K       = _params_[1];
    let c = gid.x;
    if (c >= d_inner) { return; }
    var acc: f32 = bias[c];
    // weights laid out as [d_inner, K]: weights[c*K + k]
    // state laid out as [d_inner, K], oldest to newest: state[c*K + k]
    let base = c * K;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        acc = acc + weights[base + k] * state[base + k];
    }
    out[c] = acc;
}

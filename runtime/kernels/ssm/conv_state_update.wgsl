// SSM conv state update — shift-in new sample
//
// For decode mode, the conv1d state is a rolling buffer of the last K input
// vectors per channel. Before each conv1d_decode dispatch, we shift the
// state by 1 (drop oldest, append new x) per channel.
//
// Layout: state[c*K + k] where k=0 is the newest sample, k=K-1 is the oldest.
// After update: state[c*K + 0] = x[c], state[c*K + k] = old_state[c*K + (k-1)] for k>=1.
//
// Bindings:
//   0: state  [d_inner * K]   (read+write)
//   1: x      [d_inner]       new sample
//   2: _params_              — [d_inner, K]

@group(0) @binding(0) var<storage, read_write> state:   array<f32>;
@group(0) @binding(1) var<storage, read>       x:       array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let K       = _params_[1];
    let c = gid.x;
    if (c >= d_inner) { return; }
    let base = c * K;
    // Shift right (k=K-1..1 = state[k-1])
    for (var k: u32 = K; k > 1u; k = k - 1u) {
        state[base + k - 1u] = state[base + k - 2u];
    }
    state[base + 0u] = x[c];
}

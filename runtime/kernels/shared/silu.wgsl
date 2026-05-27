// SiLU activation (in-place or to separate output) — used by Mamba gate
//   out[i] = x[i] * sigmoid(x[i])
//
// Bindings:
//   0: x         array<f32>  (read)
//   1: out       array<f32>  (write — may alias x for in-place)
//   2: _params_              — [N]

@group(0) @binding(0) var<storage, read>       x:       array<f32>;
@group(0) @binding(1) var<storage, read_write> out:     array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i >= N) { return; }
    let v = x[i];
    let sig = 1.0 / (1.0 + exp(-v));
    out[i] = v * sig;
}

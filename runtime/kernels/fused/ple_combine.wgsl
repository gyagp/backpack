// @meta bindings=3
// PLE: In-place combine — adds per-layer token embedding and scales.
// Computes: Y[i] = (Y[i] + B[i]) * scale
// Used after ple_slice_rms_norm to merge normalized projection (already in Y)
// with per-layer token embeddings (B), with scaling by pleInputScale.
//
// Dispatch: (ceil(count/256), 1, 1)
//
// Bindings:
//   0: Y (read_write) — [count] fp32, normalized projection (modified in-place)
//   1: B (read) — [count] fp32, per-layer token embedding (CPU-uploaded)
//   2: _params_ — [count, scale_bits, 0, 0]

@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = _params_[0];
    let scale = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= count) { return; }
    Y[i] = (Y[i] + B[i]) * scale;
}

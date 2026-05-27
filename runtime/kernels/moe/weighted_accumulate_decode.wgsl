// MoE per-expert weighted accumulate
//
// After computing one expert's down-projection output, accumulate into the
// per-token output with weight from the softmax routing:
//   out[c] += weights[k_slot] * src[c]   for c in 0..d_model
//
// k_slot is the slot index (0..numExpertsPerTok-1) — host code does one
// dispatch per active expert.
//
// Bindings:
//   0: out      [d_model]                  (read+write)
//   1: src      [d_model]
//   2: weights  [numExpertsPerTok]
//   3: _params_                           — [d_model, k_slot]

@group(0) @binding(0) var<storage, read_write> out:      array<f32>;
@group(0) @binding(1) var<storage, read>       src:      array<f32>;
@group(0) @binding(2) var<storage, read>       weights:  array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = _params_[0];
    let k_slot = _params_[1];
    let c = gid.x;
    if (c >= d) { return; }
    let w = weights[k_slot];
    out[c] = out[c] + w * src[c];
}

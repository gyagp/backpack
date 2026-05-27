// Gated attention output — qwen35moe-style modulation
//
// qwen35moe replaces the standard `attn_output @ W_O` with a learned gate:
//   gated_out[c] = attn_out[c] * sigmoid(gate_proj[c])
// where gate_proj = x @ attn_gate.weight (computed separately).
//
// Bindings:
//   0: attn_out  [d]  — attention output (will be modulated)
//   1: gate_in   [d]  — pre-sigmoid gate
//   2: result    [d]  — output
//   3: _params_      — [d]

@group(0) @binding(0) var<storage, read>       attn_out: array<f32>;
@group(0) @binding(1) var<storage, read>       gate_in:  array<f32>;
@group(0) @binding(2) var<storage, read_write> result:   array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = _params_[0];
    let c = gid.x;
    if (c >= d) { return; }
    let g = gate_in[c];
    let sig = 1.0 / (1.0 + exp(-g));
    result[c] = attn_out[c] * sig;
}

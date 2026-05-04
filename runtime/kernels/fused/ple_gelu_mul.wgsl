// @meta bindings=3
// PLE: GELU activation + elementwise multiply with per-layer input.
// Computes: Gate[i] = GELU(Gate[i]) * PleInput[offset + i]
// Used after the gate projection (E → ple_dim) in Per-Layer Embeddings.
// The offset allows indexing into a concatenated per-layer input buffer.
//
// Dispatch: (ceil(N/256), 1, 1) where N = ple_dim
//
// Bindings:
//   0: Gate (read_write) — [N] fp32, gate projection output (modified in-place)
//   1: PleInput (read) — [nLayer*N] fp32, concatenated per-layer inputs
//   2: _params_ — [N, offset, 0, 0]

@group(0) @binding(0) var<storage, read_write> Gate: array<f32>;
@group(0) @binding(1) var<storage, read> PleInput: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let offset = _params_[1];
    let i = gid.x;
    if (i >= N) { return; }
    let g = Gate[i];
    let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
    let gelu = 0.5 * g * (1.0 + tanh(inner));
    Gate[i] = gelu * PleInput[offset + i];
}

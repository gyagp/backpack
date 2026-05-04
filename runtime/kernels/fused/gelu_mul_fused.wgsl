// @meta bindings=3
// GELU-mul activation for fused GateUp buffer.
// Reads gate[i] from buf[i], up[i] from buf[N + i].
// Computes: out[i] = GELU(gate[i]) * up[i]
// Uses GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Dispatch: (ceil(N/256), 1, 1) where N = intermediate_size
//
// Bindings:
//   0: GateUp (read) — 2*N floats (gate || up concatenated)
//   1: Out (write) — N floats
//   2: _params_ — [N, 0, 0, 0]

@group(0) @binding(0) var<storage, read> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i >= N) { return; }
    let g = GateUp[i];
    let u = GateUp[N + i];
    let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
    let tanh_val = tanh(inner);
    let gelu = 0.5 * g * (1.0 + tanh_val);
    Out[i] = gelu * u;
}

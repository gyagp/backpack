// Unary elementwise ops: Sigmoid(0), Tanh(1), Neg(2), Sqrt(3), Sin(4), Cos(5)
// Dispatch: (ceil(N/256), 1, 1)

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let a = A[idx];
    switch (op) {
        case 0u: { C[idx] = 1.0 / (1.0 + exp(-a)); }
        case 1u: { C[idx] = tanh(a); }
        case 2u: { C[idx] = -a; }
        case 3u: { C[idx] = sqrt(a); }
        case 4u: { C[idx] = sin(a); }
        case 5u: { C[idx] = cos(a); }
        case 6u: { C[idx] = a; }
        default: { C[idx] = a; }
    }
}

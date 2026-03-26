// Unary elementwise ops:
//   Sigmoid(0), Tanh(1), Neg(2), Sqrt(3), Sin(4), Cos(5), Identity(6),
//   Gelu(7), Silu(8), Erf(9), Relu(10), Exp(11), Log(12), Abs(13),
//   Floor(14), Ceil(15), Round(16)
// Dispatch: (ceil(N/256), 1, 1)

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

// Approximate erf via Abramowitz & Stegun (max error ~1.5e-7)
fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let e = 1.0 - p * exp(-a * a);
    return select(-e, e, x >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let a = A[idx];
    switch (op) {
        case 0u: { C[idx] = 1.0 / (1.0 + exp(-a)); }                            // Sigmoid
        case 1u: { C[idx] = tanh(a); }                                           // Tanh
        case 2u: { C[idx] = -a; }                                                // Neg
        case 3u: { C[idx] = sqrt(a); }                                           // Sqrt
        case 4u: { C[idx] = sin(a); }                                            // Sin
        case 5u: { C[idx] = cos(a); }                                            // Cos
        case 6u: { C[idx] = a; }                                                 // Identity
        case 7u: { C[idx] = 0.5 * a * (1.0 + erf_approx(a * 0.7071067811865476)); } // GELU
        case 8u: { C[idx] = a / (1.0 + exp(-a)); }                               // SiLU (Swish)
        case 9u: { C[idx] = erf_approx(a); }                                     // Erf
        case 10u: { C[idx] = max(a, 0.0); }                                      // ReLU
        case 11u: { C[idx] = exp(a); }                                           // Exp
        case 12u: { C[idx] = log(a); }                                           // Log
        case 13u: { C[idx] = abs(a); }                                           // Abs
        case 14u: { C[idx] = floor(a); }                                         // Floor
        case 15u: { C[idx] = ceil(a); }                                          // Ceil
        case 16u: { C[idx] = round(a); }                                         // Round
        default: { C[idx] = a; }
    }
}

// Unary elementwise ops:
//   Sigmoid(0), Tanh(1), Neg(2), Sqrt(3), Sin(4), Cos(5), Identity(6),
//   Gelu(7), Silu(8), Erf(9), Relu(10), Exp(11), Log(12), Abs(13),
//   Floor(14), Ceil(15), Round(16), Softplus(18)
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read_write> C: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_unary(x: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return 1.0 / (1.0 + exp(-x)); }                               // Sigmoid
        case 1u: { return tanh(x); }                                              // Tanh
        case 2u: { return -x; }                                                   // Neg
        case 3u: { return sqrt(max(x, 0.0)); }                                   // Sqrt
        case 4u: { return sin(x); }                                               // Sin
        case 5u: { return cos(x); }                                               // Cos
        case 6u: { return x; }                                                    // Identity
        case 7u: { return x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); } // GELU
        case 8u: { return x / (1.0 + exp(-x)); }                                 // SiLU (Swish)
        case 10u: { return max(x, 0.0); }                                        // ReLU
        case 11u: { return exp(x); }                                              // Exp
        case 12u: { return log(max(x, 1e-10)); }                                 // Log
        case 13u: { return abs(x); }                                              // Abs
        case 14u: { return floor(x); }                                            // Floor
        case 15u: { return ceil(x); }                                             // Ceil
        case 16u: { return round(x); }                                            // Round
        case 18u: { if (x > 20.0) { return x; } return log(exp(x) + 1.0); }     // Softplus
        default: { return x; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let r0 = compute_unary(t_read(&A, base), op);
    var r1: f32 = 0.0;
    if (base + 1u < N) {
        r1 = compute_unary(t_read(&A, base + 1u), op);
    }
    t_write2(&C, base, r0, r1);
}

enable f16;

@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read_write> C: array<f16>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

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
    let a = f32(A[idx]);
    var result: f32;
    switch (op) {
        case 0u: { result = 1.0 / (1.0 + exp(-a)); }
        case 1u: { result = tanh(a); }
        case 2u: { result = -a; }
        case 3u: { result = sqrt(a); }
        case 4u: { result = sin(a); }
        case 5u: { result = cos(a); }
        case 6u: { result = a; }
        case 7u: { result = 0.5 * a * (1.0 + erf_approx(a * 0.7071067811865476)); }
        case 8u: { result = a / (1.0 + exp(-a)); }
        case 9u: { result = erf_approx(a); }
        case 10u: { result = max(a, 0.0); }
        case 11u: { result = exp(a); }
        case 12u: { result = log(a); }
        case 13u: { result = abs(a); }
        case 14u: { result = floor(a); }
        case 15u: { result = ceil(a); }
        case 16u: { result = round(a); }
        case 17u: { result = log(1.0 + exp(a)); }
        default: { result = a; }
    }
    C[idx] = f16(result);
}

enable f16;

@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f16>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let N_A = _params_[2];
    let N_B = _params_[3];
    let idx = gid.x;
    if (idx >= N) { return; }
    let a_idx = select(idx, idx % N_A, N_A < N && N_A > 0u);
    let b_idx = select(idx, idx % N_B, N_B < N && N_B > 0u);
    let a = f32(A[a_idx]);
    let b = f32(B[b_idx]);
    var result: f32;
    switch (op) {
        case 0u: { result = a + b; }
        case 1u: { result = a - b; }
        case 2u: { result = a * b; }
        case 3u: { result = a / b; }
        default: { result = a; }
    }
    C[idx] = f16(result);
}

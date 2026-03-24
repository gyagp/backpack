// Binary elementwise ops: Add(0), Sub(1), Mul(2), Div(3)
// With broadcasting support.
// Dispatch: (ceil(N/256), 1, 1)

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let B_N = _params_[2];
    let idx = gid.x;
    if (idx >= N) { return; }
    let a = A[idx];
    let b_idx = select(idx, idx % B_N, B_N < N && B_N > 0u);
    let b = B[b_idx];
    switch (op) {
        case 0u: { C[idx] = a + b; }
        case 1u: { C[idx] = a - b; }
        case 2u: { C[idx] = a * b; }
        case 3u: { C[idx] = a / b; }
        default: { C[idx] = a + b; }
    }
}

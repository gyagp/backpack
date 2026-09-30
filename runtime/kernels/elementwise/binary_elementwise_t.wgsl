// @meta noregistry=true raw_template=true
${T_READ}
${T_WRITE2}
@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> C: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;
fn compute_op(a: f32, b: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return a + b; } case 1u: { return a - b; }
        case 2u: { return a * b; } case 3u: { return a / b; }
        default: { return a + b; }
    }
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0]; let op = _params_[1];
    let A_N = _params_[2]; let B_N = _params_[3];
    let base = gid.x * 2u;
    if (base >= N) { return; }
    let a0 = select(base, base % A_N, A_N < N);
    let b0 = select(base, base % B_N, B_N < N);
    let r0 = compute_op(t_read(&A, a0), t_read(&B, b0), op);
    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i = base + 1u;
        let a1 = select(i, i % A_N, A_N < N);
        let b1 = select(i, i % B_N, B_N < N);
        r1 = compute_op(t_read(&A, a1), t_read(&B, b1), op);
    }
    t_write2(&C, base, r0, r1);
}

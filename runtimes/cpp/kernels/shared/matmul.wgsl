// MatMul — matrix multiplication
// C[m,n] = sum_k A[m,k] * B[k,n]
// Dispatch: (ceil(N/32), ceil(M/16), 1) — each thread handles 2 adjacent columns

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> C: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x * 2u;
    if (row >= M || col >= N) { return; }

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        let a_val = t_read(&A, row * K + k);
        acc0 += a_val * t_read(&B, k * N + col);
        if (col + 1u < N) {
            acc1 += a_val * t_read(&B, k * N + col + 1u);
        }
    }
    t_write2(&C, row * N + col, acc0, acc1);
}

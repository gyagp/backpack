// Gemm — Y = A * B^T + Bias (or A * B + Bias)
// Dispatch: (ceil(N/16), ceil(M/16), 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read> Bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }
    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        let b_val = select(t_read(&B, k * N + col), t_read(&B, col * K + k), transB != 0u);
        acc += t_read(&A, row * K + k) * b_val;
    }
    t_write(&Y, row * N + col, acc + t_read(&Bias, col));
}

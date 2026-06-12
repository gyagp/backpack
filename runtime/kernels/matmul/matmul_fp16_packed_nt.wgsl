// MatMul - fp32 activations by packed fp16 weights without requiring shader-f16.
// C[m,n] = sum_k A[m,k] * B[k,n]
// B is stored as row-major fp16 and viewed as array<u32>, two fp16 values per word.

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

fn read_b(k: u32, n: u32, N: u32) -> f32 {
    let linear = k * N + n;
    let pair = unpack2x16float(B[linear / 2u]);
    return select(pair.x, pair.y, (linear & 1u) == 1u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }

    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        acc += A[row * K + k] * read_b(k, col, N);
    }
    C[row * N + col] = acc;
}

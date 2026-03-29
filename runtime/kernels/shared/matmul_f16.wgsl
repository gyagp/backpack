enable f16;

// MatMul — fp32 activations by fp16 weights
// C[m,n] = sum_k A[m,k] * B[k,n]

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

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
        acc += A[row * K + k] * f32(B[k * N + col]);
    }
    C[row * N + col] = acc;
}

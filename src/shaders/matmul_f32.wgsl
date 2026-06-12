struct Params {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if row >= params.M || col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        sum = sum + A[row * params.K + k] * B[k * params.N + col];
    }

    C[row * params.N + col] = sum;
}

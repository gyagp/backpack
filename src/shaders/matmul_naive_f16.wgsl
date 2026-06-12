struct Params {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn load_f16(buf: ptr<storage, array<u32>, read>, index: u32) -> f32 {
    let word = (*buf)[index / 2u];
    let bits = (word >> ((index % 2u) * 16u)) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if row >= params.M || col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        let a_val = load_f16(&A, row * params.K + k);
        let b_val = load_f16(&B, k * params.N + col);
        sum = sum + a_val * b_val;
    }

    C[row * params.N + col] = sum;
}

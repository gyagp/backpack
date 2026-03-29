// @meta bindings=5
// MatMulNBits Q4 — simple per-element kernel
// Y[m,n] = sum_k X[m,k] * dequant(W[n,k])
// Weight: W[N, K/2] packed uint8 (2 Q4 values per byte)
// Scale: [N * blocks_per_col] fp16 packed into u32
// block_size=32, blocks_per_col = K/32
// Dispatch: (ceil(N/8), M, 1)

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let n = gid.x;
    let m = gid.y;
    if (n >= N || m >= M) { return; }

    var acc: f32 = 0.0;
    let a_base = m * K;
    let w_base = n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        for (var j = 0u; j < 16u; j++) {
            let byte_idx = w_blk_base + j;
            let byte_u32 = B[byte_idx / 4u];
            let byte_val = (byte_u32 >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            let lo = f32(byte_val & 0xFu) - 8.0;
            let hi = f32((byte_val >> 4u) & 0xFu) - 8.0;
            acc += A[a_base + k_base + j * 2u] * lo * scale;
            acc += A[a_base + k_base + j * 2u + 1u] * hi * scale;
        }
    }

    Y[m * N + n] = acc;
}

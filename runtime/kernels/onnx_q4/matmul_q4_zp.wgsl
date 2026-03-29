${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

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

        let zp_byte_idx = scale_flat / 2u;
        let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
        let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        // Process 4 bytes (8 Q4 values) per iteration for better ILP
        for (var j = 0u; j < 4u; j++) {
            let packed = B[w_blk_base / 4u + j];
            let k0 = k_base + j * 8u;
            let b0 = packed & 0xFFu;
            let b1 = (packed >> 8u) & 0xFFu;
            let b2 = (packed >> 16u) & 0xFFu;
            let b3 = (packed >> 24u) & 0xFFu;
            acc += t_read(&A, a_base + k0)      * (f32(b0 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 1u) * (f32(b0 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 2u) * (f32(b1 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 3u) * (f32(b1 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 4u) * (f32(b2 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 5u) * (f32(b2 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 6u) * (f32(b3 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 7u) * (f32(b3 >> 4u)  - zp) * scale;
        }
    }

    t_write(&Y, m * N + n, acc);
}

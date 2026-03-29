// K-parallel subgroup Q4+ZP matmul for prefill with weight reuse.
// 256 threads = 8 warps x 32 lanes. TILE_N=8, TILE_M=4.
// Each warp computes 1 output N × TILE_M rows, reusing weights across rows.
// 32 lanes split blocks_per_col for K-parallel reduction.
// Dispatch: (ceil(N/8), ceil(M/TILE_M), 1)

enable subgroups;

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const TILE_M: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let warp_id = lid.x / 32u;     // 0..7
    let lane = lid.x % 32u;        // 0..31
    let n = wid.x * 8u + warp_id;
    let m_base = wid.y * TILE_M;
    let n_valid = n < N;

    let w_base = n * (K / 2u);

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    if (n_valid) {
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                let w0 = (f32(b0 & 0xFu) - zp) * scale;
                let w1 = (f32(b0 >> 4u)  - zp) * scale;
                let w2 = (f32(b1 & 0xFu) - zp) * scale;
                let w3 = (f32(b1 >> 4u)  - zp) * scale;
                let w4 = (f32(b2 & 0xFu) - zp) * scale;
                let w5 = (f32(b2 >> 4u)  - zp) * scale;
                let w6 = (f32(b3 & 0xFu) - zp) * scale;
                let w7 = (f32(b3 >> 4u)  - zp) * scale;

                // Row 0
                if (m_base < M) {
                    let ab = m_base * K;
                    acc0 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 1
                if (m_base + 1u < M) {
                    let ab = (m_base + 1u) * K;
                    acc1 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 2
                if (m_base + 2u < M) {
                    let ab = (m_base + 2u) * K;
                    acc2 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 3
                if (m_base + 3u < M) {
                    let ab = (m_base + 3u) * K;
                    acc3 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
            }
        }
    }

    // Subgroup reduction across 32 K-parallel lanes
    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);
    let sum2 = subgroupAdd(acc2);
    let sum3 = subgroupAdd(acc3);

    if (lane == 0u && n_valid) {
        if (m_base < M)      { t_write(&Y, m_base * N + n, sum0); }
        if (m_base + 1u < M) { t_write(&Y, (m_base + 1u) * N + n, sum1); }
        if (m_base + 2u < M) { t_write(&Y, (m_base + 2u) * N + n, sum2); }
        if (m_base + 3u < M) { t_write(&Y, (m_base + 3u) * N + n, sum3); }
    }
}

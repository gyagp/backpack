// Wide-tile Q4+ZP matmul for prefill with shared-memory A reuse.
// 128 threads, TILE_M=8, TILE_N=128. Each thread owns one N-column.
// A tile [8 x 32] is cooperatively loaded into shared memory per Q4 block,
// then reused by all 128 threads for weight-activation dot products.
// Dispatch: (ceil(N/128), ceil(M/8), 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const TILE_M: u32 = 8u;

var<workgroup> smem_a: array<f32, 256>;  // TILE_M(8) * 32

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let tid = lid.x;
    let n = wid.x * 128u + tid;
    let m_base = wid.y * TILE_M;
    let n_valid = n < N;

    let w_base = select(0u, n * (K / 2u), n_valid);

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    var acc4: f32 = 0.0;
    var acc5: f32 = 0.0;
    var acc6: f32 = 0.0;
    var acc7: f32 = 0.0;

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let k_off = blk * 32u;

        // Cooperative A load: 128 threads load 256 elements (8 rows x 32 cols)
        {
            let idx0 = tid;           // 0..127
            let row0 = idx0 / 32u;    // 0..3
            let col0 = idx0 % 32u;
            let g_row0 = m_base + row0;
            smem_a[idx0] = select(0.0, t_read(&A, g_row0 * K + k_off + col0), g_row0 < M);
        }
        {
            let idx1 = tid + 128u;    // 128..255
            let row1 = idx1 / 32u;    // 4..7
            let col1 = idx1 % 32u;
            let g_row1 = m_base + row1;
            smem_a[idx1] = select(0.0, t_read(&A, g_row1 * K + k_off + col1), g_row1 < M);
        }

        workgroupBarrier();

        if (n_valid) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let w_blk_base = w_base + k_off / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let kl = j * 8u;
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
                acc0 += smem_a[kl]      * w0 + smem_a[kl + 1u] * w1
                      + smem_a[kl + 2u] * w2 + smem_a[kl + 3u] * w3
                      + smem_a[kl + 4u] * w4 + smem_a[kl + 5u] * w5
                      + smem_a[kl + 6u] * w6 + smem_a[kl + 7u] * w7;
                // Row 1
                acc1 += smem_a[32u + kl]      * w0 + smem_a[32u + kl + 1u] * w1
                      + smem_a[32u + kl + 2u] * w2 + smem_a[32u + kl + 3u] * w3
                      + smem_a[32u + kl + 4u] * w4 + smem_a[32u + kl + 5u] * w5
                      + smem_a[32u + kl + 6u] * w6 + smem_a[32u + kl + 7u] * w7;
                // Row 2
                acc2 += smem_a[64u + kl]      * w0 + smem_a[64u + kl + 1u] * w1
                      + smem_a[64u + kl + 2u] * w2 + smem_a[64u + kl + 3u] * w3
                      + smem_a[64u + kl + 4u] * w4 + smem_a[64u + kl + 5u] * w5
                      + smem_a[64u + kl + 6u] * w6 + smem_a[64u + kl + 7u] * w7;
                // Row 3
                acc3 += smem_a[96u + kl]      * w0 + smem_a[96u + kl + 1u] * w1
                      + smem_a[96u + kl + 2u] * w2 + smem_a[96u + kl + 3u] * w3
                      + smem_a[96u + kl + 4u] * w4 + smem_a[96u + kl + 5u] * w5
                      + smem_a[96u + kl + 6u] * w6 + smem_a[96u + kl + 7u] * w7;
                // Row 4
                acc4 += smem_a[128u + kl]      * w0 + smem_a[128u + kl + 1u] * w1
                      + smem_a[128u + kl + 2u] * w2 + smem_a[128u + kl + 3u] * w3
                      + smem_a[128u + kl + 4u] * w4 + smem_a[128u + kl + 5u] * w5
                      + smem_a[128u + kl + 6u] * w6 + smem_a[128u + kl + 7u] * w7;
                // Row 5
                acc5 += smem_a[160u + kl]      * w0 + smem_a[160u + kl + 1u] * w1
                      + smem_a[160u + kl + 2u] * w2 + smem_a[160u + kl + 3u] * w3
                      + smem_a[160u + kl + 4u] * w4 + smem_a[160u + kl + 5u] * w5
                      + smem_a[160u + kl + 6u] * w6 + smem_a[160u + kl + 7u] * w7;
                // Row 6
                acc6 += smem_a[192u + kl]      * w0 + smem_a[192u + kl + 1u] * w1
                      + smem_a[192u + kl + 2u] * w2 + smem_a[192u + kl + 3u] * w3
                      + smem_a[192u + kl + 4u] * w4 + smem_a[192u + kl + 5u] * w5
                      + smem_a[192u + kl + 6u] * w6 + smem_a[192u + kl + 7u] * w7;
                // Row 7
                acc7 += smem_a[224u + kl]      * w0 + smem_a[224u + kl + 1u] * w1
                      + smem_a[224u + kl + 2u] * w2 + smem_a[224u + kl + 3u] * w3
                      + smem_a[224u + kl + 4u] * w4 + smem_a[224u + kl + 5u] * w5
                      + smem_a[224u + kl + 6u] * w6 + smem_a[224u + kl + 7u] * w7;
            }
        }

        workgroupBarrier();
    }

    if (n_valid) {
        if (m_base < M)      { t_write(&Y, m_base * N + n, acc0); }
        if (m_base + 1u < M) { t_write(&Y, (m_base + 1u) * N + n, acc1); }
        if (m_base + 2u < M) { t_write(&Y, (m_base + 2u) * N + n, acc2); }
        if (m_base + 3u < M) { t_write(&Y, (m_base + 3u) * N + n, acc3); }
        if (m_base + 4u < M) { t_write(&Y, (m_base + 4u) * N + n, acc4); }
        if (m_base + 5u < M) { t_write(&Y, (m_base + 5u) * N + n, acc5); }
        if (m_base + 6u < M) { t_write(&Y, (m_base + 6u) * N + n, acc6); }
        if (m_base + 7u < M) { t_write(&Y, (m_base + 7u) * N + n, acc7); }
    }
}

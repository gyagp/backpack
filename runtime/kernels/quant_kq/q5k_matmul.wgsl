enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q5K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 44u; // 176 bytes / 4

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q5K[wi] >> sh) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = wid.y * TILE_N + warp_id;

    let K = _params_[0];
    let N = _params_[1];
    let n_blocks = _params_[2];
    let row_stride_words = _params_[3];

    var acc: f32 = 0.0;
    if (col < N) {
        let x_base = row * K;

        for (var b = 0u; b < n_blocks; b = b + 1u) {
            let block_base = col * row_stride_words + b * BLOCK_WORDS;

            let dd = unpack2x16float(W_Q5K[block_base]);
            let d = dd.x;
            let dmin = dd.y;

            // Scale/min bytes: bytes 4..15 (same layout as Q4_K)
            let d0 = get_u8(block_base, 4u);
            let d1 = get_u8(block_base, 5u);
            let d2 = get_u8(block_base, 6u);
            let d3 = get_u8(block_base, 7u);
            let m0 = get_u8(block_base, 8u);
            let m1 = get_u8(block_base, 9u);
            let m2 = get_u8(block_base, 10u);
            let m3 = get_u8(block_base, 11u);
            let md0 = get_u8(block_base, 12u);
            let md1 = get_u8(block_base, 13u);
            let md2 = get_u8(block_base, 14u);
            let md3 = get_u8(block_base, 15u);

            for (var sb = 0u; sb < 8u; sb = sb + 1u) {
                var sc_u: u32;
                var mn_u: u32;
                if (sb < 4u) {
                    let dv = select(select(select(d0, d1, sb == 1u), d2, sb == 2u), d3, sb == 3u);
                    let mv = select(select(select(m0, m1, sb == 1u), m2, sb == 2u), m3, sb == 3u);
                    sc_u = dv & 0x3Fu;
                    mn_u = mv & 0x3Fu;
                } else {
                    let j = sb - 4u;
                    let dv = select(select(select(d0, d1, j == 1u), d2, j == 2u), d3, j == 3u);
                    let mv = select(select(select(m0, m1, j == 1u), m2, j == 2u), m3, j == 3u);
                    let mdv = select(select(select(md0, md1, j == 1u), md2, j == 2u), md3, j == 3u);
                    sc_u = (mdv & 0x0Fu) | ((dv >> 2u) & 0x30u);
                    mn_u = (mdv >> 4u) | ((mv >> 2u) & 0x30u);
                }

                let sc = d * f32(sc_u);
                let mn = dmin * f32(mn_u);
                let g = sb / 2u;
                let hi = (sb & 1u) == 1u;

                let i = lane;
                let kidx = b * QK_K + sb * 32u + i;
                if (kidx < K) {
                    // Q5_K: 4 low bits from qs + 1 high bit from qh
                    let qb = get_u8(block_base, 48u + g * 32u + i);
                    let q_lo = select(qb & 0x0Fu, (qb >> 4u) & 0x0Fu, hi);
                    // High bit: bytes 16..47 = 32 bytes of qh
                    let qh_byte = get_u8(block_base, 16u + i);
                    let q_hi = (qh_byte >> sb) & 1u;
                    let q = q_lo | (q_hi << 4u);
                    let w = sc * f32(q) - mn;
                    acc = acc + X[x_base + kidx] * w;
                }
            }
        }
    }

    let sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col] = sum + Bias[col];
    }
}

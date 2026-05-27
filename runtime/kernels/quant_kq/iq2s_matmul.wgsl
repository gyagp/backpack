// IQ2_S matmul — Phase 4 (WIP, not yet validated)
//
// Block: 82 bytes / 256 elements
//   [0..1]   fp16 d
//   [2..33]  qs[32]      — 2-bit grid index low byte (4 per ib32)
//   [34..65] signs[32]   — 1 sign bit per element (overlapping qs region)
//   [66..73] qh[8]       — high bits (2 bits per ib32)
//   [74..81] scales[8]   — 4-bit sub-block scales (2 packed per byte = 16 sub-blocks? actually 8)
//
// Per sub-block ib32 in 0..7:
//   db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25
//   db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25
//   For l in 0..3:
//     grid_idx = qs[ib32*4 + l] | ((qh[ib32] << (8 - 2*l)) & 0x300)   // 10-bit, 0..1023
//     grid64 = iq2s_grid[grid_idx]   // 8 packed uint8 magnitudes
//     For j in 0..7:
//       mag = (grid64 >> (8*j)) & 0xff
//       sign_bit = (signs[ib32*4 + l] >> j) & 1
//       val = db[l/2] * f32(mag) * (sign_bit ? -1 : +1)
//
// Codebook: 1024 entries of uint64. Stored as 2048 uint32 (low/high pairs).
//
// NOTE: not yet registered or dispatched.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ2S:   array<u32>;
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>;  // iq2s_grid as 2048 u32s (2 per entry)
@group(0) @binding(3) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(5) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 21u;  // host packs 82-byte IQ2_S blocks padded to 84 bytes

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ2S[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) {
        f = (sign << 31u) | (mant << 13u);
    } else if (exp == 31u) {
        f = (sign << 31u) | 0x7F800000u | (mant << 13u);
    } else {
        f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];

    let col = col_base + warp_id;
    if (col >= N) {
        return;
    }

    let x_base = row * K;
    var acc: f32 = 0.0;

    let row_w_base = col * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        // Cooperative X load
        let x_idx = b * QK_K + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx];
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        let blk_word_base = row_w_base + b * BLOCK_WORDS;

        // Super-block scale d (fp16 at bytes 0..1)
        let d_u32 = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let d     = fp16_to_f32(d_u32);

        // One lane per ib32 (8 used, 24 idle)
        let ib32 = lane;
        if (ib32 < 8u) {
            let scales_byte = get_u8(blk_word_base, 74u + ib32);
            let db0 = d * (0.5 + f32(scales_byte & 0xfu)) * 0.25;
            let db1 = d * (0.5 + f32(scales_byte >> 4u)) * 0.25;
            let qh_byte = get_u8(blk_word_base, 66u + ib32);

            var partial: f32 = 0.0;
            for (var l = 0u; l < 4u; l = l + 1u) {
                let qs_byte = get_u8(blk_word_base, 2u + ib32 * 4u + l);
                let grid_idx = qs_byte | (((qh_byte << (8u - 2u * l)) & 0x300u));
                // Read 64-bit codebook entry (2 u32s)
                let cb_lo = Codebook[grid_idx * 2u + 0u];
                let cb_hi = Codebook[grid_idx * 2u + 1u];
                let signs_byte = get_u8(blk_word_base, 34u + ib32 * 4u + l);
                let dl = select(db1, db0, l < 2u);  // l in 0..3, db0 covers l/2==0, db1 covers l/2==1

                for (var j = 0u; j < 4u; j = j + 1u) {
                    let mag_lo = (cb_lo >> (8u * j)) & 0xFFu;
                    let mag_hi = (cb_hi >> (8u * j)) & 0xFFu;
                    let sign_lo = (signs_byte >> j) & 1u;
                    let sign_hi = (signs_byte >> (j + 4u)) & 1u;
                    let v_lo = dl * f32(mag_lo) * select(1.0, -1.0, sign_lo == 1u);
                    let v_hi = dl * f32(mag_hi) * select(1.0, -1.0, sign_hi == 1u);
                    let k_off = ib32 * 32u + l * 8u + j;
                    partial = partial + v_lo * smem_x[k_off];
                    partial = partial + v_hi * smem_x[k_off + 4u];
                }
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}

// IQ2_S MoE matmul — indirect-via-buffer expert offset
// Same pattern as iq3s_matmul_moe but for IQ2_S blocks.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ2S:   array<u32>;
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>;  // iq2s_grid as 2 u32 per entry
@group(0) @binding(3) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(5) var<storage, read>       offsets:  array<u32>;
@group(0) @binding(6) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 21u;

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
    if (exp == 0u) { f = (sign << 31u) | (mant << 13u); }
    else if (exp == 31u) { f = (sign << 31u) | 0x7F800000u | (mant << 13u); }
    else { f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u); }
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
    let slot_idx       = _params_[5];
    let expert_row_off = offsets[slot_idx];

    let col = col_base + warp_id;
    if (col >= N) { return; }

    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_word_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let d     = fp16_to_f32(d_u32);

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
                let cb_lo = Codebook[grid_idx * 2u + 0u];
                let cb_hi = Codebook[grid_idx * 2u + 1u];
                let signs_byte = get_u8(blk_word_base, 34u + ib32 * 4u + l);
                let dl = select(db1, db0, l < 2u);

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

// IQ4_XS matmul — Phase 4
//
// Block: 136 bytes / 256 elements
//   [0..1]   fp16 d
//   [2..3]   scales_h (u16)
//   [4..7]   scales_l[4]    — packed 4-bit + 4-bit per sub-block
//   [8..135] qs[128]        — 4-bit quants, packed 2 per byte
//
// Per sub-block ib in 0..7:
//   ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)
//   dl = d * (ls - 32)
//   For 32 quants in this sub-block: y[j] = dl * kvalues_iq4nl[nibble]
//
// kvalues_iq4nl is a 16-entry table; embedded as const array.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ4XS:  array<u32>;
@group(0) @binding(2) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 34u;  // 136 bytes / 4

var<workgroup> smem_x: array<f32, 256>;

// kvalues_iq4nl
const KV0:f32=-127.0; const KV1:f32=-104.0; const KV2:f32=-83.0; const KV3:f32=-65.0;
const KV4:f32=-49.0;  const KV5:f32=-35.0;  const KV6:f32=-22.0; const KV7:f32=-10.0;
const KV8:f32=1.0;    const KV9:f32=13.0;   const KVA:f32=25.0;  const KVB:f32=38.0;
const KVC:f32=53.0;   const KVD:f32=69.0;   const KVE:f32=89.0;  const KVF:f32=113.0;

fn iq4nl(n: u32) -> f32 {
    let arr = array<f32, 16>(KV0,KV1,KV2,KV3,KV4,KV5,KV6,KV7,KV8,KV9,KVA,KVB,KVC,KVD,KVE,KVF);
    return arr[n & 0xfu];
}

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ4XS[wi] >> sh) & 0xFFu;
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
    let col = col_base + warp_id;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];
    let expert_row_off = _params_[5];

    if (col >= N) { return; }

    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_base, 0u) | (get_u8(blk_base, 1u) << 8u);
        let d = fp16_to_f32(d_u32);
        let scales_h = get_u8(blk_base, 2u) | (get_u8(blk_base, 3u) << 8u);

        // One lane per sub-block (8 sub-blocks × 32 elements)
        let ib = lane;
        if (ib < 8u) {
            let scales_l_byte = get_u8(blk_base, 4u + ib / 2u);
            let ls_lo = (scales_l_byte >> (4u * (ib & 1u))) & 0xfu;
            let ls_hi = (scales_h >> (2u * ib)) & 3u;
            let ls = ls_lo | (ls_hi << 4u);
            let dl = d * (f32(ls) - 32.0);
            var partial: f32 = 0.0;
            for (var j = 0u; j < 16u; j = j + 1u) {
                let q_byte = get_u8(blk_base, 8u + ib * 16u + j);
                let n_lo = q_byte & 0xfu;
                let n_hi = q_byte >> 4u;
                let v_lo = dl * iq4nl(n_lo);
                let v_hi = dl * iq4nl(n_hi);
                partial = partial + v_lo * smem_x[ib * 32u + j];
                partial = partial + v_hi * smem_x[ib * 32u + 16u + j];
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

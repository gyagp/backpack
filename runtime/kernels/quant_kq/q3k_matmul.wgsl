// Q3_K matmul — Phase 4
//
// Block: 110 bytes / 256 elements
//   [0..1]    fp16 d
//   [2..33]   hmask[32]     — 1 high bit per quant
//   [34..97]  qs[64]        — 2-bit low quants
//   [98..109] scales[12]    — packed 16 × 6-bit signed scales (offset by 32)
//
// Sub-blocks: 16 of 16 elements each. Per element:
//   sc = unpacked_scale[is] - 32  (signed)
//   dl = d * sc
//   val = dl * ( (q[l] >> shift) & 3 - (hm[l] & mask ? 0 : 4) )
//
// NOTE: not yet registered or dispatched.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_Q3K:    array<u32>;
@group(0) @binding(2) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 28u;  // 110 bytes padded to 112

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q3K[wi] >> sh) & 0xFFu;
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

// Unpack 12 bytes of scales into 16 signed 6-bit values stored as u32.
// Mirrors llama.cpp aux[] computation. Returns scale[is] in 0..63 (need to subtract 32).
fn unpack_scale(blk_base: u32, is: u32) -> u32 {
    // Read aux[0..3] = scales[0..11] reorganized into 4 u32s.
    let s0_a = get_u8(blk_base, 98u + 0u);
    let s0_b = get_u8(blk_base, 98u + 1u);
    let s0_c = get_u8(blk_base, 98u + 2u);
    let s0_d = get_u8(blk_base, 98u + 3u);
    let s1_a = get_u8(blk_base, 98u + 4u);
    let s1_b = get_u8(blk_base, 98u + 5u);
    let s1_c = get_u8(blk_base, 98u + 6u);
    let s1_d = get_u8(blk_base, 98u + 7u);
    let s2_a = get_u8(blk_base, 98u + 8u);
    let s2_b = get_u8(blk_base, 98u + 9u);
    let s2_c = get_u8(blk_base, 98u + 10u);
    let s2_d = get_u8(blk_base, 98u + 11u);
    let aux0 = s0_a | (s0_b << 8u) | (s0_c << 16u) | (s0_d << 24u);
    let aux1 = s1_a | (s1_b << 8u) | (s1_c << 16u) | (s1_d << 24u);
    let aux2 = s2_a | (s2_b << 8u) | (s2_c << 16u) | (s2_d << 24u);
    let kmask1: u32 = 0x03030303u;
    let kmask2: u32 = 0x0f0f0f0fu;
    let tmp = aux2;
    let n0 = (aux0 & kmask2) | (((tmp >>  0u) & kmask1) << 4u);
    let n1 = (aux1 & kmask2) | (((tmp >>  2u) & kmask1) << 4u);
    let n2 = ((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    let n3 = ((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
    // Pick byte `is` from {n0, n1, n2, n3}
    let nibbles = array<u32, 4>(n0, n1, n2, n3);
    let group = is / 4u;          // 0..3
    let within = is & 3u;
    let n = nibbles[group];
    return (n >> (within * 8u)) & 0xFFu;
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

    if (col >= N) { return; }

    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = col * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_base, 0u) | (get_u8(blk_base, 1u) << 8u);
        let d = fp16_to_f32(d_u32);

        // One lane per sub-block (16 sub-blocks × 16 elements)
        let isb = lane;
        if (isb < 16u) {
            let raw_scale = unpack_scale(blk_base, isb);
            let dl = d * (f32(raw_scale) - 32.0);
            // Following the reference's nested loop: per 128-elt outer (n+=128)
            // with 4 sub-blocks (j=0..3) × 2 inner (l+0, l+16) × shift+=2
            // and m doubles each j. Map isb in 0..15 → (n_group, j, half):
            //   n_group = isb / 8  (0 or 1)
            //   sub_in_group = (isb % 8)        // 0..7
            //   j = sub_in_group / 2            // 0..3 — picks shift
            //   half = sub_in_group & 1         // 0=low (l 0..15), 1=high (l 16..31)
            //   m = 1u << ((isb % 8) / 2 + n_group * 4)   — hmask bit for THIS group
            let n_group = isb / 8u;
            let sub_in_group = isb % 8u;
            let j = sub_in_group / 2u;
            let half = sub_in_group & 1u;
            let shift = j * 2u;
            let m: u32 = 1u << (j + n_group * 4u);
            let q_base_off = 34u + n_group * 32u;
            let hm_base_off = 2u;
            var partial: f32 = 0.0;
            for (var l = 0u; l < 16u; l = l + 1u) {
                let q_off = half * 16u + l;
                let q_byte = get_u8(blk_base, q_base_off + q_off);
                let hm_byte = get_u8(blk_base, hm_base_off + q_off);
                let q_low = (q_byte >> shift) & 3u;
                let h_sub = select(4u, 0u, (hm_byte & m) != 0u);
                let q_signed = i32(q_low) - i32(h_sub);
                let val = dl * f32(q_signed);
                let k_off = n_group * 128u + j * 32u + half * 16u + l;
                partial = partial + val * smem_x[k_off];
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

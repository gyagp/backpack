// Q2_K matmul — Phase 4
//
// Block: 84 bytes / 256 elements
//   [0..1]   fp16 d
//   [2..3]   fp16 dmin
//   [4..19]  scales[16]    — 4-bit scale + 4-bit min per 16-element sub-block
//   [20..83] qs[64]        — 2-bit quants, packed 4 per byte
//
// Per llama.cpp dequant: 8 sub-blocks per super-block. Each sub-block uses
//   sc = scales[is]
//   dl = d * (sc & 0xf), ml = dmin * (sc >> 4)
//   val = dl * (int8)((q[l] >> shift) & 3) - ml
// with sub-blocks of 16 elements organized in groups of 128 with 4 shifts.
//
// NOTE: not yet registered or dispatched.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_Q2K:    array<u32>;
@group(0) @binding(2) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 21u;  // 84 bytes / 4

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q2K[wi] >> sh) & 0xFFu;
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


    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = col * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_word_base = row_w_base + b * BLOCK_WORDS;
        let d_u32    = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let dmin_u32 = get_u8(blk_word_base, 2u) | (get_u8(blk_word_base, 3u) << 8u);
        let d    = fp16_to_f32(d_u32);
        let dmin = fp16_to_f32(dmin_u32);

        // One lane per sub-block (16 sub-blocks of 16 elements each)
        let isb = lane;
        if (isb < 16u) {
            let sc = get_u8(blk_word_base, 4u + isb);
            let dl = d    * f32(sc & 0xfu);
            let ml = dmin * f32(sc >> 4u);
            // shift selects which 2-bit field (0,2,4,6) based on isb's chunk
            let chunk = isb / 4u;          // 0..3
            let shift = (chunk * 2u);      // 0,2,4,6
            let within = isb % 4u;         // 0..3
            // qs layout: 4 chunks × 16-byte groups; within-chunk picks high vs low halves
            // Following llama.cpp: each chunk has 32 bytes of qs covering 128 elements
            // (4 sub-blocks × 16 elements × 2 bits = 128 bits = 16 bytes... wait).
            // Reference walks q ptr by 32 each n+=128 iter.
            //
            // Use simpler indexing: for sub-block isb, the 16 quants are at
            //   q[(isb/8)*32 + offset]  with shift = (isb%8)/2 * 2
            //   offset = (isb & 1) * 16
            let group = isb / 8u;          // 0 or 1 (which 128-elt group)
            let sub_in_group = isb % 8u;
            let q_shift = (sub_in_group / 2u) * 2u;
            let q_offset = (sub_in_group & 1u) * 16u;
            var partial: f32 = 0.0;
            for (var l = 0u; l < 16u; l = l + 1u) {
                let q_byte = get_u8(blk_word_base, 20u + group * 32u + q_offset + l);
                let q2 = (q_byte >> q_shift) & 3u;
                // q is unsigned 0..3 but cast to int8 in reference. Here non-negative so just f32.
                let val = dl * f32(q2) - ml;
                let k_off = group * 128u + sub_in_group * 16u + l;
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

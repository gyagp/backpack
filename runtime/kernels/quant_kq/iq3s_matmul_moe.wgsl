// IQ3_S MoE matmul — indirect-via-buffer expert offset
//
// Differs from iq3s_matmul: reads expert_row_off from a separate GPU buffer
// indexed by slot_idx. This lets the dispatch fix slot_idx at build-time
// while the actual expert offset is computed per-decode by moe_compute_offsets
// from the routing decision.
//
// Bindings (7):
//   0: X        f32         — input row
//   1: W_IQ3S   u32         — fused-expert IQ3_S weights
//   2: Codebook u32         — iq3s_grid
//   3: Bias     f32
//   4: Y        f32         — output (per-expert result, accumulated by host)
//   5: offsets  u32         — [k] row offsets, one per slot
//   6: _params_ u32         — [K, N, n_blocks, row_stride_words, y_offset, slot_idx]

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ3S:   array<u32>;
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>;
@group(0) @binding(3) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(5) var<storage, read>       offsets:  array<u32>;
@group(0) @binding(6) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 28u;

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ3S[wi] >> sh) & 0xFFu;
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

fn dq_one_iq3s(base_word: u32, d_scale: f32, ib32: u32, inner: u32) -> f32 {
    let lp = inner / 4u;
    let l  = lp / 2u;
    let p  = lp & 1u;
    let j  = inner & 3u;
    let qs_byte = get_u8(base_word, 2u + ib32 * 8u + 2u * l + p);
    let qh_byte = get_u8(base_word, 66u + ib32);
    let qh_bit  = (qh_byte >> (2u * l + p)) & 1u;
    let grid_idx = qs_byte | (qh_bit << 8u);
    let grid32   = Codebook[grid_idx];
    let mag      = (grid32 >> (8u * j)) & 0xFFu;
    let signs_byte = get_u8(base_word, 74u + ib32 * 4u + l);
    let sign_bit = (signs_byte >> (4u * p + j)) & 1u;
    let scales_byte = get_u8(base_word, 106u + ib32 / 2u);
    let sub_4bit    = (scales_byte >> (4u * (ib32 & 1u))) & 0xFu;
    let db          = d_scale * (1.0 + 2.0 * f32(sub_4bit));
    let val_unsigned = db * f32(mag);
    return select(val_unsigned, -val_unsigned, sign_bit == 1u);
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
    let expert_row_off = offsets[slot_idx];  // buffer-driven, set per-decode

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
            var partial: f32 = 0.0;
            for (var inner = 0u; inner < 32u; inner = inner + 1u) {
                let w = dq_one_iq3s(blk_word_base, d, ib32, inner);
                partial = partial + w * smem_x[ib32 * 32u + inner];
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

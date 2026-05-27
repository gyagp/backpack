// IQ3_S matmul — Phase 4 (WIP, untested at runtime)
//
// Per-element formula (port of dequantize_row_iq3_s from llama.cpp):
//   block: 110 bytes for QK_K=256 elements
//     [0..1]    fp16 d
//     [2..65]   qs[64]
//     [66..73]  qh[8]
//     [74..105] signs[32]
//     [106..109] scales[4]   (4-bit sub-block scales, 2 per byte)
//   Per sub-block of 32 elements (ib32 in 0..7):
//     sub_scale_4bit = (scales[ib32/2] >> (4*(ib32 & 1))) & 0xf
//     db             = d * (1 + 2*sub_scale_4bit)
//     For l in 0..3, p in 0..1:
//       qs_byte = qs[ib32*8 + 2*l + p]
//       qh_bit  = (qh[ib32] >> (2*l + p)) & 1
//       grid_idx = qs_byte | (qh_bit << 8)              (9-bit, 0..511)
//       grid32  = Codebook[grid_idx]                    (4 packed uint8)
//       For j in 0..3:
//         mag = (grid32 >> (8*j)) & 0xff                (uint8 magnitude)
//         sign_bit = (signs[ib32*4 + l] >> (4*p + j)) & 1
//         val = db * f32(mag) * (sign_bit ? -1.0 : 1.0)
//
// NOTE: still wiring up — not yet registered or dispatched by model_runner.cpp.
// Use op_test_runner to validate against the CPU dq_iq3_s reference.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:       array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ3S:  array<u32>;  // raw 110-byte blocks packed as u32 words
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>; // iq3s_grid[512]
@group(0) @binding(3) var<storage, read>       Bias:    array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:       array<f32>;
@group(0) @binding(5) var<storage, read>       _params_: array<u32>; // [K, N, n_blocks, row_stride_words, y_offset, expert_row_offset]

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;

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
    if (exp == 0u) {
        f = (sign << 31u) | (mant << 13u);
    } else if (exp == 31u) {
        f = (sign << 31u) | 0x7F800000u | (mant << 13u);
    } else {
        f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f);
}

// Decode one element at (ib32 in 0..7, inner in 0..31). Returns dequantized
// fp32 value (without the X multiply).
fn dq_one_iq3s(base_word: u32, d_scale: f32, ib32: u32, inner: u32) -> f32 {
    // inner = 4*l + j  with l in 0..3 splitting into a pair (p=0 first 4 elts,
    // p=1 next 4 elts of the 8-elt mini-group; ib32 groups of 8 cover 32).
    // Actually llama.cpp's loop is l=0..3, p=0..1, j=0..3 → 32 elts.
    // Here inner = (2*l + p)*4 + j  for canonical addressing.
    let lp = inner / 4u;          // 0..7  → splits into l (0..3) and p (0..1)
    let l  = lp / 2u;
    let p  = lp & 1u;
    let j  = inner & 3u;

    // qs region starts at byte 2; one qs byte per (2*l+p) within this ib32.
    let qs_byte = get_u8(base_word, 2u + ib32 * 8u + 2u * l + p);
    // qh region starts at byte 66; one byte per ib32.
    let qh_byte = get_u8(base_word, 66u + ib32);
    let qh_bit  = (qh_byte >> (2u * l + p)) & 1u;
    let grid_idx = qs_byte | (qh_bit << 8u);              // 0..511
    let grid32   = Codebook[grid_idx];
    let mag      = (grid32 >> (8u * j)) & 0xFFu;          // uint8 magnitude

    // signs region starts at byte 74; 4 bytes per ib32, one byte per l.
    let signs_byte = get_u8(base_word, 74u + ib32 * 4u + l);
    let sign_bit = (signs_byte >> (4u * p + j)) & 1u;

    // sub-block scale: 8 sub-blocks → 4 bytes, 2 sub-blocks per byte (low / high nibble)
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
    let expert_row_off = _params_[5];  // 0 for non-MoE; expert_idx * N for indirect MoE expert dispatch

    // One row of X (size K) is shared across all TILE_N output columns; load once.
    let x_base = row * K;
    var acc: f32 = 0.0;

    let col = col_base + warp_id;
    if (col >= N) {
        return;
    }

    // Each row of W_IQ3S has n_blocks * (110/4) words but actually packed as
    // row_stride_w words per row. Block n in row `col` starts at:
    //   row_word_base + n * (110/4) = col * row_stride_w + n * 28? (110 isn't u32 aligned)
    // The packing convention in backpack is row-aligned to u32 strides; the
    // host-side packer (TODO: add pack_iq3s) sets row_stride_w. For now we
    // assume row_stride_w covers n_blocks blocks with no padding between blocks.
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        // Cooperative X load for this super-block — all 256 threads each load 1 float
        let x_idx = b * QK_K + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx];
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        // The block starts at (row_w_base + b * 28) words.
        // 28 words = 112 bytes (we pad 110 → 112 host-side).
        let blk_word_base = row_w_base + b * 28u;

        // Read super-block scale d (fp16 in bytes 0..1)
        let d_u32 = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let d     = fp16_to_f32(d_u32);

        // Each lane within a warp does 8 elements (8 lanes per 32-element ib32).
        // ib32 = warp_id_inside_block; we use lane / 8 to pick the inner offset (0..3).
        // Simpler: each of the 32 lanes does one ib32, computes 8 dequant values, MACs.
        let ib32 = lane;
        if (ib32 < 8u) {
            // unrolled-ish; let WGSL inline the inner work
            var partial: f32 = 0.0;
            for (var inner = 0u; inner < 32u; inner = inner + 1u) {
                let w = dq_one_iq3s(blk_word_base, d, ib32, inner);
                let k_offset = b * QK_K + ib32 * 32u + inner;
                partial = partial + w * smem_x[ib32 * 32u + inner];
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    // Reduce 32 lane partials in this warp (only first 8 lanes had work)
    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}

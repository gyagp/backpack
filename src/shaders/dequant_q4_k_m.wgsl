// Q4_K_M dequantization shader matching GGUF block_q4_K layout.
// Each super-block contains 256 values (QK_K=256):
//   - 2 x f16 (d, dmin) stored as u16 bit patterns (4 bytes)
//   - 12 bytes of packed 6-bit scales/mins (K_SCALE_SIZE=12)
//   - 128 bytes of 4-bit quantized values (QK_K/2)
// Total block size: 144 bytes

struct Params {
    n_blocks: u32,
}

@group(0) @binding(0) var<storage, read> quant_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const QK_K: u32 = 256u;
const BLOCK_SIZE_BYTES: u32 = 144u;

fn read_u8(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_pos = byte_offset % 4u;
    return (quant_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

fn read_u16(byte_offset: u32) -> u32 {
    return read_u8(byte_offset) | (read_u8(byte_offset + 1u) << 8u);
}

fn u16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let frac = bits & 0x3FFu;

    if (exp == 0u) {
        if (frac == 0u) {
            if (sign == 1u) { return -0.0; }
            return 0.0;
        }
        let val = f32(frac) / 1024.0 * pow(2.0, -14.0);
        if (sign == 1u) { return -val; }
        return val;
    }
    if (exp == 31u) {
        if (frac == 0u) {
            if (sign == 1u) { return -65504.0; }
            return 65504.0;
        }
        return 0.0;
    }
    let val = (1.0 + f32(frac) / 1024.0) * pow(2.0, f32(exp) - 15.0);
    if (sign == 1u) { return -val; }
    return val;
}

fn get_scale_min_k4(j: u32, block_byte_offset: u32, scale: ptr<function, u32>, min: ptr<function, u32>) {
    // 12-byte scales array at offset 4 in the block.
    // Bytes 0-7: lower 6 bits of scales[0..3] and mins[0..3]
    // Bytes 8-11: upper 2 bits of scales[4..7] and mins[4..7], lower 4 bits are scale/min values
    let scales_offset = block_byte_offset + 4u;
    if (j < 4u) {
        // Sub-blocks 0-3: 6-bit scale and min stored directly
        *scale = read_u8(scales_offset + j) & 63u;
        *min = read_u8(scales_offset + j + 4u) & 63u;
    } else {
        // Sub-blocks 4-7: byte q[j+4] packs scale (low nibble) and min (high nibble)
        // Upper 2 bits come from bits 6-7 of q[j-4] (scale) and q[j] (min)
        let packed = read_u8(scales_offset + j + 4u);
        let hi_s = read_u8(scales_offset + j - 4u);
        *scale = (packed & 0xFu) | ((hi_s >> 6u) << 4u);
        let hi_m = read_u8(scales_offset + j);
        *min = ((packed >> 4u) & 0xFu) | ((hi_m >> 6u) << 4u);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    if (block_idx >= params.n_blocks) {
        return;
    }

    let block_byte_offset = block_idx * BLOCK_SIZE_BYTES;

    let d_bits = read_u16(block_byte_offset);
    let dmin_bits = read_u16(block_byte_offset + 2u);
    let d = u16_to_f32(d_bits);
    let dmin = u16_to_f32(dmin_bits);

    let scales_offset = block_byte_offset + 4u;
    let qs_offset = block_byte_offset + 16u;

    let out_base = block_idx * QK_K;

    var is: u32 = 0u;
    for (var j: u32 = 0u; j < QK_K / 32u; j = j + 1u) {
        var sc: u32;
        var m: u32;
        get_scale_min_k4(is, block_byte_offset, &sc, &m);
        is = is + 1u;

        let d_sc = d * f32(sc);
        let dm = dmin * f32(m);

        for (var l: u32 = 0u; l < 16u; l = l + 1u) {
            let qs_byte_offset = qs_offset + j * 16u + l;
            let qs_val = read_u8(qs_byte_offset);

            let low_nibble = qs_val & 0xFu;
            let high_nibble = (qs_val >> 4u) & 0xFu;

            output[out_base + j * 32u + l] = d_sc * f32(low_nibble) - dm;
            output[out_base + j * 32u + l + 16u] = d_sc * f32(high_nibble) - dm;
        }
    }
}

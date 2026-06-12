struct Params {
    n_blocks: u32,
}

@group(0) @binding(0) var<storage, read> quant_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const BLOCK_SIZE_BYTES: u32 = 34u;

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

fn read_i8(byte_offset: u32) -> i32 {
    let val = read_u8(byte_offset);
    if (val >= 128u) {
        return i32(val) - 256;
    }
    return i32(val);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    if (block_idx >= params.n_blocks) {
        return;
    }

    let block_byte_offset = block_idx * BLOCK_SIZE_BYTES;
    let d_bits = read_u16(block_byte_offset);
    let d = u16_to_f32(d_bits);

    let qs_offset = block_byte_offset + 2u;
    let out_base = block_idx * BLOCK_SIZE;

    for (var i: u32 = 0u; i < BLOCK_SIZE; i = i + 1u) {
        let q = read_i8(qs_offset + i);
        output[out_base + i] = d * f32(q);
    }
}

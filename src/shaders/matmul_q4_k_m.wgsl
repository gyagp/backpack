// Quantized matmul: C[M,N] = A[M,K] (f16) × B[K,N] (Q4_K_M)
// A is f16 activations packed as u32 (two f16 per word).
// B is Q4_K_M quantized weights laid out as N columns, each column
//   contains (K/256) super-blocks of 144 bytes.
// C is f32 output.

const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;
const QK_K: u32 = 256u;
const BLOCK_SIZE_BYTES: u32 = 144u;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    batch_size: u32,
    stride_A: u32,
    stride_C: u32,
}

@group(0) @binding(0) var<storage, read> A: array<u32>;          // f16 activations [M × K]
@group(0) @binding(1) var<storage, read> B: array<u32>;          // Q4_K_M weights [N columns × blocks]
@group(0) @binding(2) var<storage, read_write> C: array<f32>;    // f32 output [M × N]
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 256>;  // TILE_M * TILE_K
var<workgroup> tileB: array<f32, 256>;  // TILE_N * TILE_K

fn load_f16_A(index: u32) -> f32 {
    let word = A[index / 2u];
    let bits = (word >> ((index % 2u) * 16u)) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

fn read_b_u8(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_pos = byte_offset % 4u;
    return (B[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

fn read_b_u16(byte_offset: u32) -> u32 {
    return read_b_u8(byte_offset) | (read_b_u8(byte_offset + 1u) << 8u);
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

fn get_scale_min_k4(j: u32, block_byte_offset: u32, scale: ptr<function, u32>, min_val: ptr<function, u32>) {
    let scales_offset = block_byte_offset + 4u;
    if (j < 4u) {
        *scale = read_b_u8(scales_offset + j) & 63u;
        *min_val = read_b_u8(scales_offset + j + 4u) & 63u;
    } else {
        let packed = read_b_u8(scales_offset + j + 4u);
        let hi_s = read_b_u8(scales_offset + j - 4u);
        *scale = (packed & 0xFu) | ((hi_s >> 6u) << 4u);
        let hi_m = read_b_u8(scales_offset + j);
        *min_val = ((packed >> 4u) & 0xFu) | ((hi_m >> 6u) << 4u);
    }
}

// Dequantize a single value at position `pos` (0..255) within a Q4_K_M super-block.
fn dequant_q4_k_m_value(block_byte_offset: u32, pos: u32) -> f32 {
    let d_bits = read_b_u16(block_byte_offset);
    let dmin_bits = read_b_u16(block_byte_offset + 2u);
    let d = u16_to_f32(d_bits);
    let dmin = u16_to_f32(dmin_bits);

    let sub_block = pos / 32u;
    let pos_in_sub = pos % 32u;

    var sc: u32;
    var m: u32;
    get_scale_min_k4(sub_block, block_byte_offset, &sc, &m);

    let d_sc = d * f32(sc);
    let dm = dmin * f32(m);

    let qs_offset = block_byte_offset + 16u;
    // Within a sub-block of 32 values: first 16 are low nibbles, next 16 are high nibbles
    var nibble_val: u32;
    if (pos_in_sub < 16u) {
        let byte_off = qs_offset + sub_block * 16u + pos_in_sub;
        nibble_val = read_b_u8(byte_off) & 0xFu;
    } else {
        let byte_off = qs_offset + sub_block * 16u + (pos_in_sub - 16u);
        nibble_val = (read_b_u8(byte_off) >> 4u) & 0xFu;
    }

    return d_sc * f32(nibble_val) - dm;
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let batch = wgid.z;
    if batch >= params.batch_size {
        return;
    }

    let row = gid.x;
    let col = gid.y;
    let lr = lid.x;
    let lc = lid.y;

    let a_offset = batch * params.stride_A;
    let c_offset = batch * params.stride_C;

    let blocks_per_col = params.K / QK_K;

    var sum: f32 = 0.0;
    let numTiles = (params.K + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE_K + lc;
        if (row < params.M && aCol < params.K) {
            tileA[lr * TILE_K + lc] = load_f16_A(a_offset + row * params.K + aCol);
        } else {
            tileA[lr * TILE_K + lc] = 0.0;
        }

        let bK = t * TILE_K + lr;
        if (col < params.N && bK < params.K) {
            let block_idx = col * blocks_per_col + bK / QK_K;
            let pos_in_block = bK % QK_K;
            let block_byte_offset = block_idx * BLOCK_SIZE_BYTES;
            tileB[lc * TILE_K + lr] = dequant_q4_k_m_value(block_byte_offset, pos_in_block);
        } else {
            tileB[lc * TILE_K + lr] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            sum = sum + tileA[lr * TILE_K + k] * tileB[lc * TILE_K + k];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        C[c_offset + row * params.N + col] = sum;
    }
}

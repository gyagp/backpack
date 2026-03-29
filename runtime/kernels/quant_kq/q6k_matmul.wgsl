enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q6K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q6K[wi] >> sh) & 0xFFu;
}

fn get_i8(base_word: u32, byte_off: u32) -> i32 {
    let u = get_u8(base_word, byte_off);
    return select(i32(u), i32(u) - 256, u >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = wid.y * TILE_N + warp_id;

    let K = _params_[0];
    let N = _params_[1];
    let n_blocks = _params_[2];
    let row_stride_words = _params_[3];

    var acc: f32 = 0.0;
    if (col < N) {
        let x_base = row * K;

        for (var b = 0u; b < n_blocks; b = b + 1u) {
            let block_base = col * row_stride_words + b * row_stride_words / n_blocks;

            // Q6_K block layout (210 bytes):
            //   [0:2]     = fp16 d (super-block scale)
            //   [2:130]   = ql: 128 bytes, low 4 bits of 256 quants
            //   [130:194] = qh: 64 bytes, high 2 bits of 256 quants
            //   [194:210] = scales: 16 int8 sub-block scales

            let d_u16 = W_Q6K[block_base] & 0xFFFFu;
            let d = unpack2x16float(d_u16).x;

            for (var sb = 0u; sb < 8u; sb = sb + 1u) {
                let i = lane;
                let kidx = b * QK_K + sb * 32u + i;
                if (kidx < K) {
                    // ql: 4 low bits, packed as nibbles
                    let ql_idx = sb * 32u + i;
                    let ql_byte = get_u8(block_base, 2u + ql_idx / 2u);
                    let ql = select(ql_byte & 0x0Fu, (ql_byte >> 4u) & 0x0Fu, (ql_idx & 1u) == 1u);

                    // qh: 2 high bits
                    let qh_byte = get_u8(block_base, 130u + (sb * 32u + i) / 4u);
                    let qh_shift = ((sb * 32u + i) % 4u) * 2u;
                    let qh = (qh_byte >> qh_shift) & 0x03u;

                    // Combine: 6-bit value = ql | (qh << 4), range [0, 63], bias by -32
                    let q6 = i32(ql | (qh << 4u)) - 32;

                    // Per-sub-block scale (int8)
                    let sc = f32(get_i8(block_base, 194u + sb));

                    let w = d * sc * f32(q6);
                    acc = acc + X[x_base + kidx] * w;
                }
            }
        }
    }

    let sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col] = sum + Bias[col];
    }
}

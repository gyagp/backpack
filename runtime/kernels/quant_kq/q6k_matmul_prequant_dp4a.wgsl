// @meta bindings=6 generated=false registry=q6k_matmul_prequant_dp4a
requires packed_4x8_integer_dot_product;
enable subgroups;

@group(0) @binding(0) var<storage, read> XQ: array<u32>;
@group(0) @binding(1) var<storage, read> XS: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;

fn load_u8(base_byte: u32, off: u32) -> u32 {
    let addr = base_byte + off;
    return (W[addr / 4u] >> ((addr & 3u) * 8u)) & 255u;
}

fn load_i8(base_byte: u32, off: u32) -> i32 {
    let v = load_u8(base_byte, off);
    return select(i32(v), i32(v) - 256, v >= 128u);
}

fn load_u32(base_byte: u32, off: u32) -> u32 {
    let addr = base_byte + off;
    let wi = addr / 4u;
    let shift = (addr & 3u) * 8u;
    if (shift == 0u) { return W[wi]; }
    return (W[wi] >> shift) | (W[wi + 1u] << (32u - shift));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let warp = tid / 32u;
    let lane = tid & 31u;
    let K = P[0];
    let N = P[1];
    let nb = P[2];
    let row_stride_words = P[3];
    let y_offset = P[4];
    let col = wid.y * 8u + warp;
    var acc = 0.0;

    if (col < N) {
        for (var b = 0u; b < nb; b++) {
            let block_base = col * row_stride_words * 4u + b * 210u;
            let dh = load_u8(block_base, 208u) |
                     (load_u8(block_base, 209u) << 8u);
            let d = unpack2x16float(dh).x;

            // Each lane consumes two packed groups of four activations.  The
            // 32 lanes therefore cover the full 256-value Q6_K super-block.
            for (var half = 0u; half < 2u; half++) {
                let pack = lane + half * 32u;
                let index = pack * 4u;
                let group = index / 128u;
                let within = index - group * 128u;
                let quarter = within / 32u;
                let local = within & 31u;
                let ql_off = group * 64u +
                    select(local, 32u + local, quarter == 1u || quarter == 3u);
                let qh_off = 128u + group * 32u + local;
                let ql = load_u32(block_base, ql_off);
                let qh = load_u32(block_base, qh_off);
                let low = select(ql & 0x0F0F0F0Fu,
                    (ql >> 4u) & 0x0F0F0F0Fu, quarter >= 2u);
                let values = low |
                    (((qh >> (quarter * 2u)) & 0x03030303u) << 4u);
                // Subtract 32 independently in each byte without repacking.
                let signed_values = ((values ^ 0x80808080u) -
                                     0x20202020u) ^ 0x80808080u;
                let scale_index = group * 8u + quarter * 2u + local / 16u;
                let weight_scale = f32(load_i8(block_base, 192u + scale_index));
                let activation_scale = XS[b * 8u + index / 32u];
                acc += f32(dot4I8Packed(XQ[b * 64u + pack], signed_values)) *
                       activation_scale * d * weight_scale;
            }
        }
    }

    // AMD exposes wave64, but adjacent logical halves own different logits.
    // Restrict the reduction to the five XOR stages inside each 32-lane half.
    var total = acc;
    total += subgroupShuffleXor(total, 16u);
    total += subgroupShuffleXor(total, 8u);
    total += subgroupShuffleXor(total, 4u);
    total += subgroupShuffleXor(total, 2u);
    total += subgroupShuffleXor(total, 1u);
    if (lane == 0u && col < N) {
        Y[wid.x * N + col + y_offset] = total + Bias[col];
    }
}

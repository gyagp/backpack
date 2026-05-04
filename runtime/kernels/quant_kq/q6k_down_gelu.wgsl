enable subgroups;

// Fused GELU-mul + Q6_K matmul (down projection)
// Reads gate/up from GateUp buffer, computes GELU(gate)*up on-the-fly
// into shared memory, then performs Q6_K matrix-vector multiply.
// Replaces 2 dispatches: gelu_mul_fused + q6k_matmul
//
// Grid: (1, ceil(N/8), 1) where N = nEmbd (output dim)
// WG: 256 threads = 8 warps, each warp handles one output column

@group(0) @binding(0) var<storage, read> GateUp: array<f32>; // 2*K floats (gate || up)
@group(0) @binding(1) var<storage, read> W_Q6K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;

var<workgroup> smem_x: array<f32, 256>;

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

    let K = _params_[0];  // intermediate_size (input dim = gate/up size)
    let N = _params_[1];  // nEmbd (output dim)
    let n_blocks = _params_[2];
    let row_stride_words = _params_[3];
    let y_offset = _params_[4];

    let x_base = row * K;
    var acc: f32 = 0.0;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let k_start = b * QK_K;

        // Fused GELU-mul + cooperative X load:
        // Instead of reading pre-computed X, compute GELU(gate)*up on-the-fly
        let x_idx = k_start + tid;
        if (x_idx < K) {
            let g = GateUp[x_idx];          // gate value
            let u = GateUp[K + x_idx];      // up value
            let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
            let gelu = 0.5 * g * (1.0 + tanh(inner));
            smem_x[tid] = gelu * u;
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        if (col < N) {
            let block_base = col * row_stride_words + b * row_stride_words / n_blocks;

            let d_u16 = get_u8(block_base, 208u) | (get_u8(block_base, 209u) << 8u);
            let d = unpack2x16float(d_u16).x;
            let l = lane;

            for (var group = 0u; group < 2u; group = group + 1u) {
                let ql_off = group * 64u;
                let qh_off = 128u + group * 32u;
                let sc_off = 192u + group * 8u;
                let local_base = group * 128u;

                let is_ = l / 16u;

                let ql0 = get_u8(block_base, ql_off + l);
                let ql1 = get_u8(block_base, ql_off + 32u + l);
                let qh_byte = get_u8(block_base, qh_off + l);

                let q1 = i32((ql0 & 0x0Fu) | (((qh_byte >> 0u) & 3u) << 4u)) - 32;
                let q2 = i32((ql1 & 0x0Fu) | (((qh_byte >> 2u) & 3u) << 4u)) - 32;
                let q3 = i32(((ql0 >> 4u) & 0x0Fu) | (((qh_byte >> 4u) & 3u) << 4u)) - 32;
                let q4 = i32(((ql1 >> 4u) & 0x0Fu) | (((qh_byte >> 6u) & 3u) << 4u)) - 32;

                let sc0 = f32(get_i8(block_base, sc_off + is_));
                let sc1 = f32(get_i8(block_base, sc_off + is_ + 2u));
                let sc2 = f32(get_i8(block_base, sc_off + is_ + 4u));
                let sc3 = f32(get_i8(block_base, sc_off + is_ + 6u));

                let li0 = local_base + l;
                let li1 = local_base + 32u + l;
                let li2 = local_base + 64u + l;
                let li3 = local_base + 96u + l;

                acc += smem_x[li0] * d * sc0 * f32(q1);
                acc += smem_x[li1] * d * sc1 * f32(q2);
                acc += smem_x[li2] * d * sc2 * f32(q3);
                acc += smem_x[li3] * d * sc3 * f32(q4);
            }
        }
        workgroupBarrier();
    }

    let sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col + y_offset] = sum + Bias[col];
    }
}

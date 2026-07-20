@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q4K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 36u; // 144 bytes / 4

// Shared memory: cooperative X load (256 floats = 1 K-block)
var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q4K[wi] >> sh) & 0xFFu;
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

    let x_base = row * K;
    var acc: f32 = 0.0;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        // Cooperative X load: all 256 threads load 1 float each
        let k_start = b * QK_K;
        let x_idx = k_start + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        if (col < N) {
            let block_base = col * row_stride_words + b * BLOCK_WORDS;

            let dd = unpack2x16float(W_Q4K[block_base]);
            let d = dd.x;
            let dmin = dd.y;

            for (var sb = 0u; sb < 8u; sb = sb + 1u) {
                var sc_u: u32;
                var mn_u: u32;
                if (sb < 4u) {
                    sc_u = get_u8(block_base, 4u + sb) & 0x3Fu;
                    mn_u = get_u8(block_base, 8u + sb) & 0x3Fu;
                } else {
                    let j = sb - 4u;
                    let dv = get_u8(block_base, 4u + j);
                    let mv = get_u8(block_base, 8u + j);
                    let mdv = get_u8(block_base, 12u + j);
                    sc_u = (mdv & 0x0Fu) | ((dv >> 2u) & 0x30u);
                    mn_u = (mdv >> 4u) | ((mv >> 2u) & 0x30u);
                }

                let sc = d * f32(sc_u);
                let mn = dmin * f32(mn_u);
                let g = sb / 2u;
                let hi = (sb & 1u) == 1u;

                let i = lane;
                let local_idx = sb * 32u + i;
                let qb = get_u8(block_base, 16u + g * 32u + i);
                let q = select(qb & 0x0Fu, (qb >> 4u) & 0x0Fu, hi);
                let w = sc * f32(q) - mn;
                acc += smem_x[local_idx] * w;
            }
        }
        workgroupBarrier();
    }

    // Reduce each logical 32-lane column independently. Hardware subgroup
    // width is vendor-dependent (AMD may expose 64), so subgroupAdd would
    // incorrectly combine two columns.
    smem_x[tid] = acc;
    workgroupBarrier();
    for (var offset = 16u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) { smem_x[tid] += smem_x[tid + offset]; }
        workgroupBarrier();
    }
    let sum = smem_x[warp_id * 32u];
    if (lane == 0u && col < N) {
        Y[row * N + col] = sum + Bias[col];
    }
}

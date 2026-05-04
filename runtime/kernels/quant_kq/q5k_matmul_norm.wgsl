enable subgroups;

// Fused RMSNorm + Q5_K matmul for decode (M=1)
// Eliminates the separate rms_norm / rms_next dispatch before each layer's QKV matmul.
// Pass 1: Cooperative RMSNorm of X → rstd
// Pass 2: Q5_K matmul with inline RMSNorm from shared memory
//
// Dispatch: (1, ceil(N/8), 1)
// WG: 256 threads = 8 warps, TILE_N=8

var<workgroup> smem_x: array<f32, 256>;
var<workgroup> _smem_reduce: array<i32, 8>;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q5K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;  // [K, N, n_blocks, row_stride_words, y_offset]
@group(0) @binding(5) var<storage, read> NormW: array<f32>;     // RMSNorm weight [K]
@group(0) @binding(6) var<storage, read> _norm_params_: array<u32>;  // [K, 0, 0, eps_as_u32]

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 44u;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q5K[wi] >> sh) & 0xFFu;
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
    let y_offset = _params_[4];
    let eps = bitcast<f32>(_norm_params_[3]);

    let x_base = row * K;

    // ── Pass 1: Compute RMSNorm rstd ──
    var sum_sq: f32 = 0.0;
    var k = tid;
    for (; k < K; k += 256u) {
        let v = X[x_base + k];
        sum_sq += v * v;
    }

    let warp_sum = subgroupAdd(sum_sq);
    if (lane == 0u) {
        _smem_reduce[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    var total: f32 = 0.0;
    if (tid < 8u) {
        total = bitcast<f32>(_smem_reduce[tid]);
    }
    let final_sum = subgroupAdd(total);
    // Broadcast rstd from warp 0 to all warps via shared memory
    if (tid == 0u) {
        _smem_reduce[0] = bitcast<i32>(1.0 / sqrt(final_sum / f32(K) + eps));
    }
    workgroupBarrier();
    let rstd = bitcast<f32>(_smem_reduce[0]);
    var acc: f32 = 0.0;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let k_start = b * QK_K;

        // Cooperative load with RMSNorm applied
        let x_idx = k_start + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx] * rstd * NormW[x_idx];
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        if (col < N) {
            let block_base = col * row_stride_words + b * BLOCK_WORDS;

            let dd = unpack2x16float(W_Q5K[block_base]);
            let d = dd.x;
            let dmin = dd.y;

            let d0 = get_u8(block_base, 4u);
            let d1 = get_u8(block_base, 5u);
            let d2 = get_u8(block_base, 6u);
            let d3 = get_u8(block_base, 7u);
            let m0 = get_u8(block_base, 8u);
            let m1 = get_u8(block_base, 9u);
            let m2 = get_u8(block_base, 10u);
            let m3 = get_u8(block_base, 11u);
            let md0 = get_u8(block_base, 12u);
            let md1 = get_u8(block_base, 13u);
            let md2 = get_u8(block_base, 14u);
            let md3 = get_u8(block_base, 15u);

            for (var sb = 0u; sb < 8u; sb = sb + 1u) {
                var sc_u: u32;
                var mn_u: u32;
                if (sb < 4u) {
                    let dv = select(select(select(d0, d1, sb == 1u), d2, sb == 2u), d3, sb == 3u);
                    let mv = select(select(select(m0, m1, sb == 1u), m2, sb == 2u), m3, sb == 3u);
                    sc_u = dv & 0x3Fu;
                    mn_u = mv & 0x3Fu;
                } else {
                    let j = sb - 4u;
                    let dv = select(select(select(d0, d1, j == 1u), d2, j == 2u), d3, j == 3u);
                    let mv = select(select(select(m0, m1, j == 1u), m2, j == 2u), m3, j == 3u);
                    let mdv = select(select(select(md0, md1, j == 1u), md2, j == 2u), md3, j == 3u);
                    sc_u = (mdv & 0x0Fu) | ((dv >> 2u) & 0x30u);
                    mn_u = (mdv >> 4u) | ((mv >> 2u) & 0x30u);
                }

                let sc = d * f32(sc_u);
                let mn = dmin * f32(mn_u);
                let g = sb / 2u;
                let hi = (sb & 1u) == 1u;

                let i = lane;
                let local_idx = sb * 32u + i;
                let qb = get_u8(block_base, 48u + g * 32u + i);
                let q_lo = select(qb & 0x0Fu, (qb >> 4u) & 0x0Fu, hi);
                let qh_byte = get_u8(block_base, 16u + i);
                let q_hi = (qh_byte >> sb) & 1u;
                let q = q_lo | (q_hi << 4u);
                let w = sc * f32(q) - mn;
                acc = acc + smem_x[local_idx] * w;
            }
        }
        workgroupBarrier();
    }

    let sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col + y_offset] = sum + Bias[col];
    }
}

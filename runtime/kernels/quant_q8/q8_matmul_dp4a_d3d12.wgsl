// @meta bindings=6
requires packed_4x8_integer_dot_product;
enable subgroups;

// D3D12 DP4A-accelerated Q8_0 GEMM for batched prefill.
//   Y[T×N] = X[T×K] × W[N×K]^T + Bias[N]
//
// Quantizes X activations to int8 per Q8 block (32 elements), then uses
// dot4I8Packed for 4× compute throughput vs scalar f32 dot products.
// Removes all extractBits — weights used as packed u32 directly.
//
// Double-buffered quantized X in smem (2.3KB per buffer vs 4KB scalar).
// TILE_M=8, TILE_N=32 (4 cols per warp), 256 threads.
// params struct: {M=T, N, K, pad}
//
// Grid: (ceil(N/TILE_N), ceil(T/TILE_M), 1)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { M: u32, N: u32, K: u32, pad: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK: u32 = 256u;
const WG: u32 = 256u;

// Double-buffered quantized X: 8 rows × 64 packed u32 = 512 u32 per buf
var<workgroup> smem_xq: array<array<u32, 512>, 2>;
// Double-buffered X scales: 8 rows × 8 blocks = 64 f32 per buf
var<workgroup> smem_xs: array<array<f32, 64>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_col = wid.x;
    let tile_row = wid.y;
    let tid = lid.x;

    let T = params.M;
    let N = params.N;
    let K = params.K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    // Accumulators
    var acc: array<array<f32, 4>, 8>;
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) { acc[m][c] = 0.0; }
    }

    // Pre-compute column info
    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    var w_bases: array<u32, 4>;
    var s_bases: array<u32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = tile_col * TILE_N + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
        w_bases[c] = select(0u, cols[c] * stride_w, col_valid[c]);
        s_bases[c] = select(0u, cols[c] * n_blocks, col_valid[c]);
    }

    // Quantization layout: tid 0..255 → K element 0..255
    let block_id = tid / 32u;           // 0..7 — which Q8 block within BK
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u; // byte position within u32
    let pack_group = elem_in_block / 4u; // which packed u32 within block

    // ── Quantize first X tile into buffer 0 ──────────────────────────
    let nk = K / BK;
    for (var r = 0u; r < TILE_M; r++) {
        let row = tile_row * TILE_M + r;
        let x_val = select(0.0, X[row * K + tid], row < T);

        // Subgroup reduce for absmax within 32-element block
        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[0][r * 8u + block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[0][r * 64u + block_id * 8u + pack_group] = packed;
        }
    }
    workgroupBarrier();

    // ── Main loop ────────────────────────────────────────────────────
    for (var g = 0u; g < nk; g++) {
        let cur = g & 1u;
        let nxt = 1u - cur;
        let k_next = (g + 1u) * BK;

        // Quantize next X tile into nxt buffer
        if (g + 1u < nk) {
            for (var r = 0u; r < TILE_M; r++) {
                let row = tile_row * TILE_M + r;
                let x_val = select(0.0, X[row * K + k_next + tid], row < T);

                var max_val = abs(x_val);
                max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
                max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
                max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
                max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
                max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

                let x_scale = max_val / 127.0;
                if (elem_in_block == 0u) {
                    smem_xs[nxt][r * 8u + block_id] = x_scale;
                }

                let safe_scale = select(1.0, x_scale, x_scale != 0.0);
                let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

                let byte_val = u32(q_val & 0xFF);
                let shifted = byte_val << (pack_lane * 8u);
                var packed = shifted;
                packed = packed | subgroupShuffleXor(packed, 1u);
                packed = packed | subgroupShuffleXor(packed, 2u);

                if (pack_lane == 0u) {
                    smem_xq[nxt][r * 64u + block_id * 8u + pack_group] = packed;
                }
            }
        }

        // DP4A compute from cur buffer
        let x_block = lane / 4u;  // which Q8 block this lane's K-elements belong to

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let w_off = w_bases[c] + g * 64u + lane * 2u;
                let wq0 = W_Q8[w_off];
                let wq1 = W_Q8[w_off + 1u];

                // Weight scale (all 8 K-elems per lane share one block)
                let w_block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(s_bases[c] + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((s_bases[c] + w_block) & 1u) != 0u);

                for (var m = 0u; m < TILE_M; m++) {
                    let xq0 = smem_xq[cur][m * 64u + lane * 2u];
                    let xq1 = smem_xq[cur][m * 64u + lane * 2u + 1u];
                    let x_scale = smem_xs[cur][m * 8u + x_block];

                    let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                    acc[m][c] += f32(idot) * w_scale * x_scale;
                }
            }
        }

        workgroupBarrier();
    }

    // ── Write output ─────────────────────────────────────────────────
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < T) {
                Y[row * N + cols[c]] = warp_sum + Bias[cols[c]];
            }
        }
    }
}

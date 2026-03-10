requires packed_4x8_integer_dot_product;
enable subgroups;

// D3D12 DP4A batched matmul over pre-quantized activations with residual add.
//   Y[M×N] += Xq[M×K] × Wq[N×K]^T + Bias[N]
// Grid: (ceil(N/32), ceil(M/8), 1)

@group(0) @binding(0) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(1) var<storage, read_write> XS: array<f32>;
@group(0) @binding(2) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(3) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

struct Params { M: u32, N: u32, K: u32, pad: u32, };
@group(0) @binding(6) var<uniform> params: Params;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK: u32 = 256u;

var<workgroup> smem_xq: array<u32, 512>;
var<workgroup> smem_xs: array<f32, 64>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_col = wid.x;
    let tile_row = wid.y;
    let tid = lid.x;

    let M = params.M;
    let N = params.N;
    let K = params.K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let stride_xq = K / 4u;
    let stride_xs = K / 32u;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let ng = K / BK;
    let x_block = lane / 4u;

    var acc: array<array<f32, 4>, 8>;
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) {
            acc[m][c] = 0.0;
        }
    }

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

    for (var g = 0u; g < ng; g++) {
        let x_pack_off = g * 64u;
        let x_scale_off = g * 8u;

        for (var i = tid; i < 512u; i += 256u) {
            let m = i / 64u;
            let j = i % 64u;
            let row = tile_row * TILE_M + m;
            if (row < M) {
                smem_xq[i] = XQ[row * stride_xq + x_pack_off + j];
            } else {
                smem_xq[i] = 0u;
            }
        }

        if (tid < 64u) {
            let m = tid / 8u;
            let j = tid % 8u;
            let row = tile_row * TILE_M + m;
            if (row < M) {
                smem_xs[tid] = XS[row * stride_xs + x_scale_off + j];
            } else {
                smem_xs[tid] = 0.0;
            }
        }
        workgroupBarrier();

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let w_off = w_bases[c] + x_pack_off + lane * 2u;
                let wq0 = W_Q8[w_off];
                let wq1 = W_Q8[w_off + 1u];
                let block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(s_bases[c] + block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((s_bases[c] + block) & 1u) != 0u);

                for (var m = 0u; m < TILE_M; m++) {
                    let xq0 = smem_xq[m * 64u + lane * 2u];
                    let xq1 = smem_xq[m * 64u + lane * 2u + 1u];
                    let xsc = smem_xs[m * 8u + x_block];
                    let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                    acc[m][c] += f32(idot) * w_scale * xsc;
                }
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < M) {
                Y[row * N + cols[c]] += warp_sum + Bias[cols[c]];
            }
        }
    }
}
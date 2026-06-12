const TILE_SIZE: u32 = 16u;
const TM: u32 = 8u;
const TN: u32 = 8u;
const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 16u;
const BM_PAD: u32 = 132u;
const BUF_A_SIZE: u32 = 2112u;  // BK * BM_PAD = 16 * 132
const BUF_B_SIZE: u32 = 2048u;  // BK * BN = 16 * 128
const A_EPT: u32 = 8u;  // (BK * BM) / (TILE_SIZE * TILE_SIZE) = 16*128/256
const B_EPT: u32 = 8u;  // (BK * BN) / (TILE_SIZE * TILE_SIZE) = 16*128/256

struct Params {
    M: u32,
    N: u32,
    K: u32,
    batch_size: u32,
    stride_A: u32,
    stride_C: u32,
}

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 4224>;  // 2 * BUF_A_SIZE
var<workgroup> tileB: array<f32, 4096>;  // 2 * BUF_B_SIZE

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

    let lr = lid.x;
    let lc = lid.y;
    let thread_id = lr * TILE_SIZE + lc;

    let a_offset = batch * params.stride_A;
    let c_offset = batch * params.stride_C;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var acc: array<f32, 64>;
    for (var i = 0u; i < 64u; i++) { acc[i] = 0.0; }

    let numTiles = (params.K + BK - 1u) / BK;
    let a_base = lr * TM;
    let b_base = lc * TN;

    // Load tile 0 into buffer 0
    {
        let tile_k = 0u;
        for (var e = 0u; e < A_EPT; e++) {
            let flat = thread_id * A_EPT + e;
            let a_r0 = flat / BK;
            let a_c0 = flat % BK;
            let gr = block_row + a_r0;
            let gc = tile_k + a_c0;
            var val = 0.0f;
            if gr < params.M && gc < params.K {
                let abs_idx = a_offset + gr * params.K + gc;
                let word = A[abs_idx / 2u];
                let pair = unpack2x16float(word);
                val = pair[abs_idx % 2u];
            }
            tileA[a_c0 * BM_PAD + a_r0] = val;
        }
        for (var e = 0u; e < B_EPT; e++) {
            let flat = thread_id * B_EPT + e;
            let b_r = flat / BN;
            let b_c = flat % BN;
            let gr = tile_k + b_r;
            let gc = block_col + b_c;
            var val = 0.0f;
            if gr < params.K && gc < params.N {
                let b_idx = gr * params.N + gc;
                let word = B[b_idx / 2u];
                let pair = unpack2x16float(word);
                val = pair[b_idx % 2u];
            }
            tileB[b_r * BN + b_c] = val;
        }
    }
    workgroupBarrier();

    for (var t = 0u; t < numTiles; t++) {
        let cur_a_off = (t % 2u) * BUF_A_SIZE;
        let cur_b_off = (t % 2u) * BUF_B_SIZE;

        // Prefetch next tile into alternate buffer
        if t + 1u < numTiles {
            let nxt_a_off = ((t + 1u) % 2u) * BUF_A_SIZE;
            let nxt_b_off = ((t + 1u) % 2u) * BUF_B_SIZE;
            let next_tile_k = (t + 1u) * BK;
            for (var e = 0u; e < A_EPT; e++) {
                let flat = thread_id * A_EPT + e;
                let a_r0 = flat / BK;
                let a_c0 = flat % BK;
                let gr = block_row + a_r0;
                let gc = next_tile_k + a_c0;
                var val = 0.0f;
                if gr < params.M && gc < params.K {
                    let abs_idx = a_offset + gr * params.K + gc;
                    let word = A[abs_idx / 2u];
                    let pair = unpack2x16float(word);
                    val = pair[abs_idx % 2u];
                }
                tileA[nxt_a_off + a_c0 * BM_PAD + a_r0] = val;
            }
            for (var e = 0u; e < B_EPT; e++) {
                let flat = thread_id * B_EPT + e;
                let b_r = flat / BN;
                let b_c = flat % BN;
                let gr = next_tile_k + b_r;
                let gc = block_col + b_c;
                var val = 0.0f;
                if gr < params.K && gc < params.N {
                    let b_idx = gr * params.N + gc;
                    let word = B[b_idx / 2u];
                    let pair = unpack2x16float(word);
                    val = pair[b_idx % 2u];
                }
                tileB[nxt_b_off + b_r * BN + b_c] = val;
            }
        }

        // Compute on current buffer
        for (var k = 0u; k < BK; k++) {
            let ak = cur_a_off + k * BM_PAD + a_base;
            let bk = cur_b_off + k * BN + b_base;

            let a0 = tileA[ak];
            let a1 = tileA[ak + 1u];
            let a2 = tileA[ak + 2u];
            let a3 = tileA[ak + 3u];
            let a4 = tileA[ak + 4u];
            let a5 = tileA[ak + 5u];
            let a6 = tileA[ak + 6u];
            let a7 = tileA[ak + 7u];

            let b0 = tileB[bk];
            let b1 = tileB[bk + 1u];
            let b2 = tileB[bk + 2u];
            let b3 = tileB[bk + 3u];
            let b4 = tileB[bk + 4u];
            let b5 = tileB[bk + 5u];
            let b6 = tileB[bk + 6u];
            let b7 = tileB[bk + 7u];

            acc[0] += a0*b0; acc[1] += a0*b1; acc[2] += a0*b2; acc[3] += a0*b3; acc[4] += a0*b4; acc[5] += a0*b5; acc[6] += a0*b6; acc[7] += a0*b7;
            acc[8] += a1*b0; acc[9] += a1*b1; acc[10] += a1*b2; acc[11] += a1*b3; acc[12] += a1*b4; acc[13] += a1*b5; acc[14] += a1*b6; acc[15] += a1*b7;
            acc[16] += a2*b0; acc[17] += a2*b1; acc[18] += a2*b2; acc[19] += a2*b3; acc[20] += a2*b4; acc[21] += a2*b5; acc[22] += a2*b6; acc[23] += a2*b7;
            acc[24] += a3*b0; acc[25] += a3*b1; acc[26] += a3*b2; acc[27] += a3*b3; acc[28] += a3*b4; acc[29] += a3*b5; acc[30] += a3*b6; acc[31] += a3*b7;
            acc[32] += a4*b0; acc[33] += a4*b1; acc[34] += a4*b2; acc[35] += a4*b3; acc[36] += a4*b4; acc[37] += a4*b5; acc[38] += a4*b6; acc[39] += a4*b7;
            acc[40] += a5*b0; acc[41] += a5*b1; acc[42] += a5*b2; acc[43] += a5*b3; acc[44] += a5*b4; acc[45] += a5*b5; acc[46] += a5*b6; acc[47] += a5*b7;
            acc[48] += a6*b0; acc[49] += a6*b1; acc[50] += a6*b2; acc[51] += a6*b3; acc[52] += a6*b4; acc[53] += a6*b5; acc[54] += a6*b6; acc[55] += a6*b7;
            acc[56] += a7*b0; acc[57] += a7*b1; acc[58] += a7*b2; acc[59] += a7*b3; acc[60] += a7*b4; acc[61] += a7*b5; acc[62] += a7*b6; acc[63] += a7*b7;
        }

        workgroupBarrier();
    }

    for (var rm = 0u; rm < TM; rm++) {
        let row = block_row + lr * TM + rm;
        if row >= params.M { continue; }
        let row_off = c_offset + row * params.N;
        for (var rn = 0u; rn < TN; rn++) {
            let col = block_col + lc * TN + rn;
            if col < params.N {
                C[row_off + col] = acc[rm * TN + rn];
            }
        }
    }
}

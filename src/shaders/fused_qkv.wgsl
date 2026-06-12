const TILE_SIZE: u32 = 16u;
const TM: u32 = 8u;
const TN: u32 = 8u;
const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 32u;
const BM_PAD: u32 = 132u;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    batch_size: u32,
    stride_input: u32,
    stride_out: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> Wk: array<u32>;
@group(0) @binding(3) var<storage, read> Wv: array<u32>;
@group(0) @binding(4) var<storage, read_write> outQ: array<f32>;
@group(0) @binding(5) var<storage, read_write> outK: array<f32>;
@group(0) @binding(6) var<storage, read_write> outV: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 4224>;  // BK * BM_PAD = 32 * 132
var<workgroup> tileB: array<f32, 4096>;  // BK * BN = 32 * 128

fn accumulate(a_base: u32, b_base: u32, acc: ptr<function, array<f32, 64>>) {
    for (var k: u32 = 0u; k < BK; k = k + 1u) {
        let ak = k * BM_PAD + a_base;
        let bk = k * BN + b_base;

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

        (*acc)[0] += a0*b0; (*acc)[1] += a0*b1; (*acc)[2] += a0*b2; (*acc)[3] += a0*b3; (*acc)[4] += a0*b4; (*acc)[5] += a0*b5; (*acc)[6] += a0*b6; (*acc)[7] += a0*b7;
        (*acc)[8] += a1*b0; (*acc)[9] += a1*b1; (*acc)[10] += a1*b2; (*acc)[11] += a1*b3; (*acc)[12] += a1*b4; (*acc)[13] += a1*b5; (*acc)[14] += a1*b6; (*acc)[15] += a1*b7;
        (*acc)[16] += a2*b0; (*acc)[17] += a2*b1; (*acc)[18] += a2*b2; (*acc)[19] += a2*b3; (*acc)[20] += a2*b4; (*acc)[21] += a2*b5; (*acc)[22] += a2*b6; (*acc)[23] += a2*b7;
        (*acc)[24] += a3*b0; (*acc)[25] += a3*b1; (*acc)[26] += a3*b2; (*acc)[27] += a3*b3; (*acc)[28] += a3*b4; (*acc)[29] += a3*b5; (*acc)[30] += a3*b6; (*acc)[31] += a3*b7;
        (*acc)[32] += a4*b0; (*acc)[33] += a4*b1; (*acc)[34] += a4*b2; (*acc)[35] += a4*b3; (*acc)[36] += a4*b4; (*acc)[37] += a4*b5; (*acc)[38] += a4*b6; (*acc)[39] += a4*b7;
        (*acc)[40] += a5*b0; (*acc)[41] += a5*b1; (*acc)[42] += a5*b2; (*acc)[43] += a5*b3; (*acc)[44] += a5*b4; (*acc)[45] += a5*b5; (*acc)[46] += a5*b6; (*acc)[47] += a5*b7;
        (*acc)[48] += a6*b0; (*acc)[49] += a6*b1; (*acc)[50] += a6*b2; (*acc)[51] += a6*b3; (*acc)[52] += a6*b4; (*acc)[53] += a6*b5; (*acc)[54] += a6*b6; (*acc)[55] += a6*b7;
        (*acc)[56] += a7*b0; (*acc)[57] += a7*b1; (*acc)[58] += a7*b2; (*acc)[59] += a7*b3; (*acc)[60] += a7*b4; (*acc)[61] += a7*b5; (*acc)[62] += a7*b6; (*acc)[63] += a7*b7;
    }
}

fn load_tile_a_fast(abs_idx: u32, a_c0: u32, a_r0: u32) {
    let w0 = input[abs_idx / 2u];
    let w1 = input[(abs_idx + 2u) / 2u];
    let w2 = input[(abs_idx + 4u) / 2u];
    let w3 = input[(abs_idx + 6u) / 2u];
    let w4 = input[(abs_idx + 8u) / 2u];
    let w5 = input[(abs_idx + 10u) / 2u];
    let w6 = input[(abs_idx + 12u) / 2u];
    let w7 = input[(abs_idx + 14u) / 2u];
    let p0 = unpack2x16float(w0);
    let p1 = unpack2x16float(w1);
    let p2 = unpack2x16float(w2);
    let p3 = unpack2x16float(w3);
    let p4 = unpack2x16float(w4);
    let p5 = unpack2x16float(w5);
    let p6 = unpack2x16float(w6);
    let p7 = unpack2x16float(w7);
    tileA[(a_c0) * BM_PAD + a_r0] = p0.x;
    tileA[(a_c0 + 1u) * BM_PAD + a_r0] = p0.y;
    tileA[(a_c0 + 2u) * BM_PAD + a_r0] = p1.x;
    tileA[(a_c0 + 3u) * BM_PAD + a_r0] = p1.y;
    tileA[(a_c0 + 4u) * BM_PAD + a_r0] = p2.x;
    tileA[(a_c0 + 5u) * BM_PAD + a_r0] = p2.y;
    tileA[(a_c0 + 6u) * BM_PAD + a_r0] = p3.x;
    tileA[(a_c0 + 7u) * BM_PAD + a_r0] = p3.y;
    tileA[(a_c0 + 8u) * BM_PAD + a_r0] = p4.x;
    tileA[(a_c0 + 9u) * BM_PAD + a_r0] = p4.y;
    tileA[(a_c0 + 10u) * BM_PAD + a_r0] = p5.x;
    tileA[(a_c0 + 11u) * BM_PAD + a_r0] = p5.y;
    tileA[(a_c0 + 12u) * BM_PAD + a_r0] = p6.x;
    tileA[(a_c0 + 13u) * BM_PAD + a_r0] = p6.y;
    tileA[(a_c0 + 14u) * BM_PAD + a_r0] = p7.x;
    tileA[(a_c0 + 15u) * BM_PAD + a_r0] = p7.y;
}

fn unpack_tile_b(base: u32, w0: u32, w1: u32, w2: u32, w3: u32, w4: u32, w5: u32, w6: u32, w7: u32) {
    let p0 = unpack2x16float(w0);
    let p1 = unpack2x16float(w1);
    let p2 = unpack2x16float(w2);
    let p3 = unpack2x16float(w3);
    let p4 = unpack2x16float(w4);
    let p5 = unpack2x16float(w5);
    let p6 = unpack2x16float(w6);
    let p7 = unpack2x16float(w7);
    tileB[base] = p0.x;
    tileB[base + 1u] = p0.y;
    tileB[base + 2u] = p1.x;
    tileB[base + 3u] = p1.y;
    tileB[base + 4u] = p2.x;
    tileB[base + 5u] = p2.y;
    tileB[base + 6u] = p3.x;
    tileB[base + 7u] = p3.y;
    tileB[base + 8u] = p4.x;
    tileB[base + 9u] = p4.y;
    tileB[base + 10u] = p5.x;
    tileB[base + 11u] = p5.y;
    tileB[base + 12u] = p6.x;
    tileB[base + 13u] = p6.y;
    tileB[base + 14u] = p7.x;
    tileB[base + 15u] = p7.y;
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

    let lr = lid.x;
    let lc = lid.y;
    let thread_id = lr * TILE_SIZE + lc;

    let a_offset = batch * params.stride_input;
    let c_offset = batch * params.stride_out;

    let block_row = wgid.x * BM;
    let block_col = wgid.y * BN;

    var accQ: array<f32, 64>;
    var accK: array<f32, 64>;
    var accV: array<f32, 64>;
    for (var i = 0u; i < 64u; i++) {
        accQ[i] = 0.0;
        accK[i] = 0.0;
        accV[i] = 0.0;
    }

    let numTiles = (params.K + BK - 1u) / BK;
    let a_base = lr * TM;
    let b_base = lc * TN;

    let a_r0 = (thread_id * 16u) / BK;
    let a_c0 = (thread_id * 16u) % BK;
    let b_r = (thread_id * 16u) / BN;
    let b_c0 = (thread_id * 16u) % BN;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tile_k = t * BK;

        // Load tileA from input (shared across Q, K, V projections)
        {
            let global_a_row = block_row + a_r0;
            let global_a_col = tile_k + a_c0;
            if global_a_row < params.M && global_a_col + 15u < params.K {
                let abs_idx = a_offset + global_a_row * params.K + global_a_col;
                load_tile_a_fast(abs_idx, a_c0, a_r0);
            } else {
                for (var i = 0u; i < 16u; i++) {
                    let ac = a_c0 + i;
                    let gr = block_row + a_r0;
                    let gc = tile_k + ac;
                    if gr < params.M && gc < params.K {
                        let abs_idx = a_offset + gr * params.K + gc;
                        let word = input[abs_idx / 2u];
                        let pair = unpack2x16float(word);
                        tileA[ac * BM_PAD + a_r0] = pair[abs_idx % 2u];
                    } else {
                        tileA[ac * BM_PAD + a_r0] = 0.0;
                    }
                }
            }
        }

        // --- Q projection: load Wq tile into tileB, accumulate ---
        {
            let global_b_row = tile_k + b_r;
            let global_b_col = block_col + b_c0;
            if global_b_row < params.K && global_b_col + 15u < params.N {
                let b_idx = global_b_row * params.N + global_b_col;
                unpack_tile_b(
                    b_r * BN + b_c0,
                    Wq[b_idx / 2u], Wq[(b_idx + 2u) / 2u],
                    Wq[(b_idx + 4u) / 2u], Wq[(b_idx + 6u) / 2u],
                    Wq[(b_idx + 8u) / 2u], Wq[(b_idx + 10u) / 2u],
                    Wq[(b_idx + 12u) / 2u], Wq[(b_idx + 14u) / 2u],
                );
            } else {
                for (var i = 0u; i < 16u; i++) {
                    let bc = b_c0 + i;
                    let gr = tile_k + b_r;
                    let gc = block_col + bc;
                    if gr < params.K && gc < params.N {
                        let b_idx = gr * params.N + gc;
                        let word = Wq[b_idx / 2u];
                        let pair = unpack2x16float(word);
                        tileB[b_r * BN + bc] = pair[b_idx % 2u];
                    } else {
                        tileB[b_r * BN + bc] = 0.0;
                    }
                }
            }
        }
        workgroupBarrier();
        accumulate(a_base, b_base, &accQ);
        workgroupBarrier();

        // --- K projection: load Wk tile into tileB, accumulate ---
        {
            let global_b_row = tile_k + b_r;
            let global_b_col = block_col + b_c0;
            if global_b_row < params.K && global_b_col + 15u < params.N {
                let b_idx = global_b_row * params.N + global_b_col;
                unpack_tile_b(
                    b_r * BN + b_c0,
                    Wk[b_idx / 2u], Wk[(b_idx + 2u) / 2u],
                    Wk[(b_idx + 4u) / 2u], Wk[(b_idx + 6u) / 2u],
                    Wk[(b_idx + 8u) / 2u], Wk[(b_idx + 10u) / 2u],
                    Wk[(b_idx + 12u) / 2u], Wk[(b_idx + 14u) / 2u],
                );
            } else {
                for (var i = 0u; i < 16u; i++) {
                    let bc = b_c0 + i;
                    let gr = tile_k + b_r;
                    let gc = block_col + bc;
                    if gr < params.K && gc < params.N {
                        let b_idx = gr * params.N + gc;
                        let word = Wk[b_idx / 2u];
                        let pair = unpack2x16float(word);
                        tileB[b_r * BN + bc] = pair[b_idx % 2u];
                    } else {
                        tileB[b_r * BN + bc] = 0.0;
                    }
                }
            }
        }
        workgroupBarrier();
        accumulate(a_base, b_base, &accK);
        workgroupBarrier();

        // --- V projection: load Wv tile into tileB, accumulate ---
        {
            let global_b_row = tile_k + b_r;
            let global_b_col = block_col + b_c0;
            if global_b_row < params.K && global_b_col + 15u < params.N {
                let b_idx = global_b_row * params.N + global_b_col;
                unpack_tile_b(
                    b_r * BN + b_c0,
                    Wv[b_idx / 2u], Wv[(b_idx + 2u) / 2u],
                    Wv[(b_idx + 4u) / 2u], Wv[(b_idx + 6u) / 2u],
                    Wv[(b_idx + 8u) / 2u], Wv[(b_idx + 10u) / 2u],
                    Wv[(b_idx + 12u) / 2u], Wv[(b_idx + 14u) / 2u],
                );
            } else {
                for (var i = 0u; i < 16u; i++) {
                    let bc = b_c0 + i;
                    let gr = tile_k + b_r;
                    let gc = block_col + bc;
                    if gr < params.K && gc < params.N {
                        let b_idx = gr * params.N + gc;
                        let word = Wv[b_idx / 2u];
                        let pair = unpack2x16float(word);
                        tileB[b_r * BN + bc] = pair[b_idx % 2u];
                    } else {
                        tileB[b_r * BN + bc] = 0.0;
                    }
                }
            }
        }
        workgroupBarrier();
        accumulate(a_base, b_base, &accV);
        workgroupBarrier();
    }

    // Store Q, K, V results
    for (var rm = 0u; rm < TM; rm++) {
        let row = block_row + lr * TM + rm;
        if row >= params.M { continue; }
        let row_off = c_offset + row * params.N;
        for (var rn = 0u; rn < TN; rn++) {
            let col = block_col + lc * TN + rn;
            if col < params.N {
                outQ[row_off + col] = accQ[rm * TN + rn];
                outK[row_off + col] = accK[rm * TN + rn];
                outV[row_off + col] = accV[rm * TN + rn];
            }
        }
    }
}

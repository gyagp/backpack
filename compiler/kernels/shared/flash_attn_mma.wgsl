enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// MMA Flash Attention for batched prefill.
//
// Processes BQ=16 query positions per WG. Iterates KV in blocks of
// BK=16. Uses 16×16×16 fp16→f32 MMA for Q·K^T and P·V.
//
// Per KV block:
//   1. MMA: S[16×16] = Q[16×HD] · K[16×HD]^T  (8 MMA calls over HD)
//   2. Causal mask + online softmax on S (per-row)
//   3. MMA: O[16×16] += P[16×16] · V[16×HD]   (8 MMA calls over HD)
//
// Grid: (n_head, ceil(T/16), 1)
// WG: 128 threads (4 subgroups). Subgroup 0 does MMA, all help load.
//
// KV cache: fp16 (array<f16>)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> OutBuf: array<f32>;

struct AttnParams {
    kv_stride: u32,
    n_rep: u32,
    T_total: u32,
    cache_offset: u32,
    T_prefill: u32,
    scale_bits: u32,
    neg_inf_bits: u32,
    pad1: u32,
};
@group(0) @binding(4) var<uniform> params: AttnParams;

const BQ: u32 = 16u;
const BK: u32 = 16u;
const HD: u32 = 128u;
const MK: u32 = 16u;
const HD_TILES: u32 = 8u;  // HD / MK
const WG: u32 = 128u;

// Shared memory
var<workgroup> tile_Q: array<f16, 256>;    // 16 × 16 (Q chunk)
var<workgroup> tile_K: array<f16, 256>;    // 16 × 16 (K chunk)
var<workgroup> tile_S: array<f32, 256>;    // 16 × 16 (score matrix)
var<workgroup> tile_P: array<f16, 256>;    // 16 × 16 (attn weights)
var<workgroup> tile_V: array<f16, 256>;    // 16 × 16 (V chunk)
var<workgroup> tile_O: array<f32, 256>;    // 16 × 16 (output chunk)
var<workgroup> row_m: array<f32, 16>;      // per-row max
var<workgroup> row_l: array<f32, 16>;      // per-row sum
var<workgroup> row_rescale: array<f32, 16>; // rescale factor for old acc
// Full output: [BQ × HD] stored as 8 chunks of [16×16] in smem
var<workgroup> out_acc: array<f32, 2048>;  // 16 × 128

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let head = wid.x;
    let q_block = wid.y;
    let lx = lid.x;

    let kv_stride = params.kv_stride;
    let n_rep = params.n_rep;
    let T_total = params.T_total;
    let cache_offset = params.cache_offset;
    let scale = bitcast<f32>(params.scale_bits);
    let neg_inf = bitcast<f32>(params.neg_inf_bits);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let n_head_total = kv_stride / HD * n_rep;
    let q_base = q_block * BQ;

    // Initialize output accumulator and softmax state
    for (var i = lx; i < BQ * HD; i += WG) {
        out_acc[i] = 0.0;
    }
    if (lx < BQ) {
        row_m[lx] = neg_inf;
        row_l[lx] = 0.0;
    }
    workgroupBarrier();

    // Max KV position for any query in this block (uniform)
    let max_kv = min(cache_offset + q_base + BQ, T_total);
    let n_kv_blocks = (max_kv + BK - 1u) / BK;

    for (var kb = 0u; kb < n_kv_blocks; kb++) {
        let kv_start = kb * BK;

        // ── Phase 1: S = Q · K^T via MMA ────────────────────────────
        // Zero score tile
        if (lx < 256u) { tile_S[lx] = 0.0; }
        workgroupBarrier();

        for (var hd = 0u; hd < HD_TILES; hd++) {
            let hd_off = hd * MK;

            // Load Q[BQ × MK] and K[BK × MK] as fp16
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                let gq = q_base + r;
                tile_Q[i] = select(0.0h,
                    f16(Q[gq * n_head_total * HD + head * HD + hd_off + c]),
                    gq < T_total);
            }
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                let gk = kv_start + r;
                tile_K[i] = select(0.0h,
                    K_cache[gk * kv_stride + kv_off + hd_off + c],
                    gk < T_total);
            }
            workgroupBarrier();

            // MMA: S += Q × K^T  (subgroup 0)
            if (sg_id == 0u) {
                let mQ = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
                    &tile_Q, 0u, false, MK);
                let mK = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
                    &tile_K, 0u, true, MK);
                var mS = subgroupMatrixLoad<subgroup_matrix_result<f32, 16, 16>>(
                    &tile_S, 0u, false, BK);
                mS = subgroupMatrixMultiplyAccumulate(mQ, mK, mS);
                subgroupMatrixStore(&tile_S, 0u, mS, false, BK);
            }
            workgroupBarrier();
        }

        // ── Phase 2: Scale + causal mask + online softmax ───────────
        // 128 threads process 256 elements (2 each)
        for (var i = lx; i < BQ * BK; i += WG) {
            let qr = i / BK;
            let kc = i % BK;
            let gq = q_base + qr;
            let gk = kv_start + kc;
            var s = tile_S[i] * scale;
            if (gk >= T_total || gk > cache_offset + gq) {
                s = neg_inf;
            }
            tile_S[i] = s;
        }
        workgroupBarrier();

        // Per-row max and rescale factor (16 threads, one per row)
        if (lx < BQ) {
            var rm = neg_inf;
            for (var j = 0u; j < BK; j++) {
                rm = max(rm, tile_S[lx * BK + j]);
            }
            let m_old = row_m[lx];
            let m_new = max(m_old, rm);
            let rescale = exp(m_old - m_new);
            row_m[lx] = m_new;
            row_l[lx] *= rescale;
            row_rescale[lx] = rescale;

            var ls = 0.0f;
            for (var j = 0u; j < BK; j++) {
                let e = exp(tile_S[lx * BK + j] - m_new);
                tile_P[lx * BK + j] = f16(e);
                ls += e;
            }
            row_l[lx] += ls;
        }
        workgroupBarrier();

        // Rescale existing output accumulator
        for (var i = lx; i < BQ * HD; i += WG) {
            let qr = i / HD;
            out_acc[i] *= row_rescale[qr];
        }
        workgroupBarrier();

        // ── Phase 3: O += P · V via MMA ─────────────────────────────
        for (var hd = 0u; hd < HD_TILES; hd++) {
            let hd_off = hd * MK;

            // Load V[BK × MK] as fp16
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                let gv = kv_start + r;
                tile_V[i] = select(0.0h,
                    V_cache[gv * kv_stride + kv_off + hd_off + c],
                    gv < T_total);
            }

            // Load current output chunk [16×16]
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                tile_O[i] = out_acc[r * HD + hd_off + c];
            }
            workgroupBarrier();

            // MMA: O += P × V  (subgroup 0)
            if (sg_id == 0u) {
                let mP = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
                    &tile_P, 0u, false, BK);
                let mV = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
                    &tile_V, 0u, false, MK);
                var mO = subgroupMatrixLoad<subgroup_matrix_result<f32, 16, 16>>(
                    &tile_O, 0u, false, MK);
                mO = subgroupMatrixMultiplyAccumulate(mP, mV, mO);
                subgroupMatrixStore(&tile_O, 0u, mO, false, MK);
            }
            workgroupBarrier();

            // Write back to out_acc
            for (var i = lx; i < 256u; i += WG) {
                let r = i / MK;  let c = i % MK;
                out_acc[r * HD + hd_off + c] = tile_O[i];
            }
            workgroupBarrier();
        }
    }

    // ── Normalize by row_l and write output ─────────────────────────
    for (var i = lx; i < BQ * HD; i += WG) {
        let qr = i / HD;
        let hd = i % HD;
        let gq = q_base + qr;
        if (gq < T_total) {
            let l = max(row_l[qr], 1e-10);
            OutBuf[gq * n_head_total * HD + head * HD + hd] = out_acc[i] / l;
        }
    }
}

#pragma once
// wgsl_shaders.h -- Auto-generated from runtime/kernels/
// Do not edit manually.  Regenerate with:
//     python runtime/gen_wgsl_shaders.py

#include <string>
#include <unordered_map>

// [onnx_q4] matmul_q4_zp_batched_dp4a
static const char* WGSL_MATMUL_Q4_ZP_BATCHED_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Asymmetric ONNX Q4G32 prefill matmul. Four activation rows share each
// packed weight tile; Q4 values are centered by their per-block zero point
// and accumulated with DP4A. Grid: (ceil(N/32), ceil(M/4), 1).
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const ROWS: u32 = 4u;
const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;
var<workgroup> xq: array<u32, ROWS * 64u>;
var<workgroup> xs: array<f32, ROWS * 8u>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M=P[0]; let N=P[1]; let K=P[2]; let tid=lid.x;
    let warp=tid/32u; let lane=tid&31u; let row0=wid.y*ROWS;
    let nblocks=K/32u; let words=K/8u;
    var cols: array<u32,COLS_PER_WARP>;
    var valid: array<bool,COLS_PER_WARP>;
    var acc: array<f32,ROWS*COLS_PER_WARP>;
    for(var c=0u;c<COLS_PER_WARP;c++){
        cols[c]=wid.x*32u+warp*COLS_PER_WARP+c; valid[c]=cols[c]<N;
    }
    for(var i=0u;i<ROWS*COLS_PER_WARP;i++){acc[i]=0.0;}
    for(var kb=0u;kb<K;kb+=BK){
        let block=tid/32u; let elem=tid&31u;
        let packlane=elem&3u; let packgroup=elem/4u;
        for(var r=0u;r<ROWS;r++){
            let row=row0+r; let xv=select(0.0,X[row*K+kb+tid],row<M);
            var amax=abs(xv);
            amax=max(amax,subgroupShuffleXor(amax,16u));
            amax=max(amax,subgroupShuffleXor(amax,8u));
            amax=max(amax,subgroupShuffleXor(amax,4u));
            amax=max(amax,subgroupShuffleXor(amax,2u));
            amax=max(amax,subgroupShuffleXor(amax,1u));
            let scale=amax/127.0;
            if(elem==0u){xs[r*8u+block]=scale;}
            let safe=select(1.0,scale,scale!=0.0);
            let q=u32(clamp(i32(round(xv/safe)),-127,127))&255u;
            var packed=q<<(packlane*8u);
            packed|=subgroupShuffleXor(packed,1u);
            packed|=subgroupShuffleXor(packed,2u);
            if(packlane==0u){xq[r*64u+block*8u+packgroup]=packed;}
        }
        workgroupBarrier();
        for(var c=0u;c<COLS_PER_WARP;c++){
            if(valid[c]){
                let col=cols[c]; let q4=B[col*words+kb/8u+lane];
                let xblock=lane/4u; let wb=kb/32u+xblock;
                let si=col*nblocks+wb;
                let zbyte=(ZeroPoints[(si/2u)/4u]>>(((si/2u)%4u)*8u))&255u;
                let zp=select(zbyte&15u,(zbyte>>4u)&15u,(si&1u)!=0u);
                let b0=q4&255u; let b1=(q4>>8u)&255u;
                let b2=(q4>>16u)&255u; let b3=(q4>>24u)&255u;
                let wq0=(((b0&15u)-zp)&255u)|((((b0>>4u)-zp)&255u)<<8u)|
                    ((((b1&15u)-zp)&255u)<<16u)|((((b1>>4u)-zp)&255u)<<24u);
                let wq1=(((b2&15u)-zp)&255u)|((((b2>>4u)-zp)&255u)<<8u)|
                    ((((b3&15u)-zp)&255u)<<16u)|((((b3>>4u)-zp)&255u)<<24u);
                let sp=unpack2x16float(Scales[si/2u]);
                let ws=select(sp.x,sp.y,(si&1u)!=0u);
                for(var r=0u;r<ROWS;r++){
                    let base=r*64u+lane*2u;
                    let dot=dot4I8Packed(xq[base],wq0)+dot4I8Packed(xq[base+1u],wq1);
                    acc[c*ROWS+r]+=f32(dot)*ws*xs[r*8u+xblock];
                }
            }
        }
        workgroupBarrier();
    }
    for(var c=0u;c<COLS_PER_WARP;c++){for(var r=0u;r<ROWS;r++){
        let sum=subgroupAdd(acc[c*ROWS+r]);
        if(lane==0u&&valid[c]&&row0+r<M){Y[(row0+r)*N+cols[c]]=sum;}
    }}
}
)WGSL";


struct ShaderInfo {
    const char* source;
    uint32_t numBindings;
    bool isGenerated;  // true = generated WGSL, false = hand-written WGSL
};

// ─── [attention] ───────────────────────────────────────────────────

// [attention] bidirectional_attn — f32 instantiated
static const char* WGSL_BIDIRECTIONAL_ATTN = R"WGSL(
// Bidirectional multi-head attention (no causal mask)
// For DiT transformer and VAE self-attention.
// Uses online softmax (single-pass, no shared memory barriers).
//
// Q/K/V layout: [batch, seq_len, num_heads * head_dim]
// Each thread handles one output element (token, head, dim).
//
// Dispatch: (ceil(head_dim/128), num_heads, T_q)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let T_q = _params_[0];
    let num_heads = _params_[1];
    let head_dim = _params_[2];
    let T_kv = _params_[3];
    let scale = bitcast<f32>(_params_[4]);
    let kv_heads = _params_[5]; // 0 = same as num_heads (standard MHA)

    let d = gid.x;
    let head = gid.y;
    let q_tok = gid.z;
    if (d >= head_dim || head >= num_heads || q_tok >= T_q) { return; }

    let kv_head = select(head, head / (num_heads / kv_heads), kv_heads > 0u && kv_heads < num_heads);
    let kv_hd_stride = select(num_heads, kv_heads, kv_heads > 0u && kv_heads < num_heads);

    let q_base = q_tok * num_heads * head_dim + head * head_dim;

    // Online softmax: single pass over KV tokens
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;
    var acc: f32 = 0.0;

    for (var kv = 0u; kv < T_kv; kv++) {
        // Compute Q·K score
        var score: f32 = 0.0;
        let k_base = kv * kv_hd_stride * head_dim + kv_head * head_dim;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += t_read(&Q, q_base + dd) * t_read(&K, k_base + dd);
        }
        score *= scale;

        // Online softmax update
        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;

        // Rescale accumulator and add new V contribution
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);
        acc = acc * rescale + w * t_read(&V, k_base + d);

        m_prev = m_new;
        l_prev = l_new;
    }

    t_write(&Out, q_base + d, acc);
}
)WGSL";

// [attention] bidirectional_attn — dtype template
static const char* WGSL_BIDIRECTIONAL_ATTN_T = R"WGSL(
// Bidirectional multi-head attention (no causal mask)
// For DiT transformer and VAE self-attention.
// Uses online softmax (single-pass, no shared memory barriers).
//
// Q/K/V layout: [batch, seq_len, num_heads * head_dim]
// Each thread handles one output element (token, head, dim).
//
// Dispatch: (ceil(head_dim/128), num_heads, T_q)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> Q: array<${T}>;
@group(0) @binding(1) var<storage, read> K: array<${T}>;
@group(0) @binding(2) var<storage, read> V: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Out: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let T_q = _params_[0];
    let num_heads = _params_[1];
    let head_dim = _params_[2];
    let T_kv = _params_[3];
    let scale = bitcast<f32>(_params_[4]);
    let kv_heads = _params_[5]; // 0 = same as num_heads (standard MHA)

    let d = gid.x;
    let head = gid.y;
    let q_tok = gid.z;
    if (d >= head_dim || head >= num_heads || q_tok >= T_q) { return; }

    let kv_head = select(head, head / (num_heads / kv_heads), kv_heads > 0u && kv_heads < num_heads);
    let kv_hd_stride = select(num_heads, kv_heads, kv_heads > 0u && kv_heads < num_heads);

    let q_base = q_tok * num_heads * head_dim + head * head_dim;

    // Online softmax: single pass over KV tokens
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;
    var acc: f32 = 0.0;

    for (var kv = 0u; kv < T_kv; kv++) {
        // Compute Q·K score
        var score: f32 = 0.0;
        let k_base = kv * kv_hd_stride * head_dim + kv_head * head_dim;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += t_read(&Q, q_base + dd) * t_read(&K, k_base + dd);
        }
        score *= scale;

        // Online softmax update
        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;

        // Rescale accumulator and add new V contribution
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);
        acc = acc * rescale + w * t_read(&V, k_base + d);

        m_prev = m_new;
        l_prev = l_new;
    }

    t_write(&Out, q_base + d, acc);
}
)WGSL";

// [attention] causal_attn
static const char* WGSL_CAUSAL_ATTN = R"WGSL(
enable f16;
enable subgroups;

// Multi-query causal attention with fp16 KV + early exit.
// 4 query positions per WG. Uniform params for data-dependent loop bound.
// Used on D3D12 (no MMA) and as Vulkan fallback.
//
// Grid: (n_head, ceil(T/4), 1)
// WG: 128 threads (4 warps × 32 threads)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

struct AttnParams {
    kv_stride: u32,
    n_rep: u32,
    T_total: u32,
    cache_offset: u32,
    T_prefill: u32,
    scale_bits: u32,
    neg_inf_bits: u32,
    kv_start: u32,
};
@group(0) @binding(4) var<uniform> params: AttnParams;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;
const QUERIES_PER_WG: u32 = 4u;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let q_block = wid.y;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let kv_stride = params.kv_stride;
    let n_rep = params.n_rep;
    let T_total = params.T_total;
    let cache_offset = params.cache_offset;
    let scale = bitcast<f32>(params.scale_bits);
    let neg_inf = bitcast<f32>(params.neg_inf_bits);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let n_head_total = kv_stride / HD * n_rep;

    let q_idx = q_block * QUERIES_PER_WG + warp_id;
    let q_abs_pos = cache_offset + q_idx;
    let q_valid = q_idx < params.T_prefill;

    let q_base = q_idx * n_head_total * HD + head * HD;
    var q: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        q[e] = 0.0;
        if (q_valid && d < HD) {
            q[e] = Q[q_base + d];
        }
    }

    var acc: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        acc[e] = 0.0;
    }
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    // Early exit: uniform loop bound from var<uniform> params
    let max_causal = min(cache_offset + q_block * QUERIES_PER_WG + QUERIES_PER_WG, T_total);

    for (var t = 0u; t < max_causal; t = t + 1u) {
        let causal_valid = q_valid && t >= params.kv_start && t <= q_abs_pos;

        let k_base = t * kv_stride + kv_off;
        let k_off = k_base + lane * HD_PER_THREAD;

        var partial = 0.0;
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var k = 0.0;
            if (causal_valid && d < HD) {
                k = f32(K_cache[k_off + e]);
            }
            partial += q[e] * k;
        }
        let dot_qk = subgroupAdd(partial);
        let score = select(neg_inf, dot_qk * scale, causal_valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_off = k_base + lane * HD_PER_THREAD;
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var v = 0.0;
            if (causal_valid && d < HD) {
                v = f32(V_cache[v_off + e]);
            }
            acc[e] = acc[e] * rescale + v * w;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    let out_base = q_idx * n_head_total * HD + head * HD;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        if (q_valid && d < HD) {
            Out[out_base + d] = acc[e];
        }
    }
}
)WGSL";

// [attention] flash_attn_vulkan
static const char* WGSL_FLASH_ATTN_VULKAN = R"WGSL(
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
    kv_start: u32,
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
    let kv_window_start = params.kv_start;
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
    let max_kv = min(cache_offset + min(q_base + BQ, params.T_prefill), T_total);
    let n_kv_blocks = (max_kv + BK - 1u) / BK;

    for (var kb = kv_window_start / BK; kb < n_kv_blocks; kb++) {
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
                    gq < params.T_prefill);
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
            if (gk < kv_window_start || gk >= T_total || gk > cache_offset + gq) {
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
        if (gq < params.T_prefill) {
            let l = max(row_l[qr], 1e-10);
            OutBuf[gq * n_head_total * HD + head * HD + hd] = out_acc[i] / l;
        }
    }
}
)WGSL";

// [attention] gemma_rope_batched
static const char* WGSL_GEMMA_ROPE_BATCHED = R"WGSL(
enable f16;

// Gemma batched Q/K normalization, RoPE and cache write. Params:
// [n_head, q_dim, kv_dim, pos_offset, half_dim, cache_len, n_kv, flags]
// flags bit 0: Q-only/shared-KV layer, bit 1: normalize V.

@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> KCache: array<f16>;
@group(0) @binding(3) var<storage, read_write> VCache: array<f16>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QW: array<f32>;
@group(0) @binding(7) var<storage, read> KW: array<f32>;
@group(0) @binding(8) var<storage, read> P: array<u32>;

const HD: u32 = 128u;
var<workgroup> sums: array<f32, 128>;

fn row_rms(base: u32, tid: u32) -> f32 {
    var ss=0.0;
    for(var i=tid;i<HD;i+=128u){let v=QKV[base+i];ss+=v*v;}
    sums[tid]=ss;
    workgroupBarrier();
    for(var stride=64u;stride>0u;stride>>=1u){
        if(tid<stride){sums[tid]+=sums[tid+stride];}
        workgroupBarrier();
    }
    return inverseSqrt(sums[0]/f32(HD)+1e-6);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let h=wid.x;let t=wid.y;let tid=lid.x;
    let nh=P[0];let qdim=P[1];let kvdim=P[2];let pos=P[3]+t;
    let half=P[4];let cache_pos=P[5]+t;let nkv=P[6];let flags=P[7];
    let qonly=(flags&1u)!=0u;let vnorm=(flags&2u)!=0u;
    let stride=select(qdim+2u*kvdim,qdim,qonly);
    if(h<nh){
        let src=t*stride+h*HD;let dst=t*qdim+h*HD;
        let r=row_rms(src,tid);let cb=pos*half;
        for(var i=tid;i<half;i+=128u){let j=i+half;
            let a=QKV[src+i]*r*QW[i];let b=QKV[src+j]*r*QW[j];
            let c=Cos[cb+i];let s=Sin[cb+i];QRot[dst+i]=a*c-b*s;QRot[dst+j]=b*c+a*s;
        }
        for(var i=tid+2u*half;i<HD;i+=128u){QRot[dst+i]=QKV[src+i]*r*QW[i];}
    }else if(!qonly){
        let kh=h-nh;let ks=t*stride+qdim+kh*HD;let vs=ks+kvdim;
        let dst=cache_pos*nkv*HD+kh*HD;let kr=row_rms(ks,tid);let cb=pos*half;
        for(var i=tid;i<half;i+=128u){let j=i+half;
            let a=QKV[ks+i]*kr*KW[i];let b=QKV[ks+j]*kr*KW[j];let c=Cos[cb+i];let s=Sin[cb+i];
            KCache[dst+i]=f16(a*c-b*s);KCache[dst+j]=f16(b*c+a*s);
        }
        for(var i=tid+2u*half;i<HD;i+=128u){KCache[dst+i]=f16(QKV[ks+i]*kr*KW[i]);}
        var vr=1.0;
        if(vnorm){vr=row_rms(vs,tid);}
        for(var i=tid;i<HD;i+=128u){VCache[dst+i]=f16(QKV[vs+i]*vr);}
    }
}
)WGSL";

// [attention] gqa_chunked_pass1
static const char* WGSL_GQA_CHUNKED_PASS1 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;
var<workgroup> dot_scratch: array<f32, 32>;
var<workgroup> active_chunk_count: u32;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let chunk_id = wid.y;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let kv_start = _params_[3];
    let n_chunks = _params_[4];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    if (lane == 0u) { active_chunk_count = n_chunks; }
    workgroupBarrier();
    let chunk_active = chunk_id < workgroupUniformLoad(&active_chunk_count);
    if (!chunk_active) { return; }

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    // Load Q into local array. HD_PER_THREAD may be ceil(HD / 32), so the
    // final lanes are padding for head_dim values like 80.
    var q: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        q[e] = 0.0;
        if (d < HD) {
            q[e] = Q[q_base + d];
        }
    }

    let t_start = chunk_id * CHUNK;

    var acc: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) { acc[e] = 0.0; }
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var i = 0u; i < CHUNK; i = i + 1u) {
        let logical_t = t_start + i;
        let valid = chunk_active && logical_t < T_total;
        let t = kv_start + logical_t;

        let k_base = select(0u, t * kv_stride + kv_off, valid);
        var dot_partial: f32 = 0.0;
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var k = 0.0;
            if (valid && d < HD) {
                k = f32(K_cache[k_base + d]);
            }
            dot_partial += q[e] * k;
        }
        dot_scratch[lane] = dot_partial;
        workgroupBarrier();
        for (var stride = 16u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                dot_scratch[lane] += dot_scratch[lane + stride];
            }
            workgroupBarrier();
        }
        let dot = dot_scratch[0];
        let raw_score = dot * scale;
        let score = select(neg_inf, raw_score, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_base = select(0u, t * kv_stride + kv_off, valid);
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var v = 0.0;
            if (valid && d < HD) {
                v = f32(V_cache[v_base + d]);
            }
            acc[e] = acc[e] * rescale + v * w;
        }

        m_prev = m_new;
        l_prev = l_new;
        workgroupBarrier();
    }

    // Use max_chunks for the stride to prevent overlapping writes across heads.
    let partial_stride = HD + 2u;
    let base = head * max_chunks * partial_stride + chunk_id * partial_stride;
    if (lane == 0u) {
        Partials[base] = m_prev;
        Partials[base + 1u] = l_prev;
    }
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        if (d < HD) {
            Partials[base + 2u + d] = acc[e];
        }
    }
}

const CHUNK: u32 = 64u;
)WGSL";

// [attention] gqa_chunked_pass2
static const char* WGSL_GQA_CHUNKED_PASS2 = R"WGSL(
@group(0) @binding(0) var<storage, read_write> Partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;

    let n_chunks = _params_[4];
    let neg_inf = bitcast<f32>(_params_[6]);
    let max_chunks = _params_[7];

    let partial_stride = HD + 2u;
    let head_base = head * max_chunks * partial_stride;

    var acc: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) { acc[e] = 0.0; }
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var c = 0u; c < n_chunks; c = c + 1u) {
        let base = head_base + c * partial_stride;
        let m_chunk = Partials[base];
        let l_chunk = Partials[base + 1u];

        if (l_chunk > 0.0) {
            let m_new = max(m_prev, m_chunk);
            let exp_prev = exp(m_prev - m_new);
            let exp_chunk = exp(m_chunk - m_new);
            let l_new = l_prev * exp_prev + l_chunk * exp_chunk;
            let rescale = l_prev * exp_prev / max(l_new, 1e-10);
            let w = l_chunk * exp_chunk / max(l_new, 1e-10);

            for (var e = 0u; e < HD_PER_THREAD; e++) {
                let d = lane * HD_PER_THREAD + e;
                var v = 0.0;
                if (d < HD) {
                    v = Partials[base + 2u + d];
                }
                acc[e] = acc[e] * rescale + v * w;
            }

            m_prev = m_new;
            l_prev = l_new;
        }
    }

    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        if (d < HD) {
            Out[head * HD + d] = acc[e];
        }
    }
}
)WGSL";

// [attention] gqa_decode — f32 instantiated
static const char* WGSL_GQA_DECODE = R"WGSL(
// GQA decode — single query attending to all KV cache entries.
// Q: [num_heads * head_dim]  (f32, single token)
// K: [kv_heads, kv_stride, head_dim]  (f32, kv_stride >= total_seq)
// V: [kv_heads, kv_stride, head_dim]  (f32, kv_stride >= total_seq)
// Out: [num_heads * head_dim]  (f32)
// Params: [0]=num_heads, [1]=head_dim, [2]=total_seq, [3]=kv_heads, [4]=scale_u32,
//         [5]=kv_stride (0 = use total_seq, for static KV cache with max_seq slots)
// One workgroup per Q head. Each thread handles dims d, d+64, d+128, ...
// Dispatch: (1, num_heads, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let total_seq = _params_[2];
    let kv_heads = _params_[3];
    let scale = bitcast<f32>(_params_[4]);
    let kv_stride_raw = _params_[5];
    let kv_stride = select(kv_stride_raw, total_seq, kv_stride_raw == 0u);

    let h = wid.y;  // Q head index
    if (h >= num_heads) { return; }

    let kv_h = h / (num_heads / kv_heads);  // map Q head to KV head
    let q_base = h * head_dim;
    let d0 = lid.x;  // base dimension for this thread

    // Each thread accumulates up to 4 V dimensions (supports head_dim up to 256)
    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;

    // Online softmax state (shared across all dims — only depends on Q·K scores)
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;

    for (var s = 0u; s < total_seq; s++) {
        let k_base = (kv_h * kv_stride + s) * head_dim;

        // Compute Q·K score once per position (same for all dims)
        var score: f32 = 0.0;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += t_read(&Q, q_base + dd) * t_read(&K, k_base + dd);
        }
        score *= scale;

        // Online softmax
        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;

        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        // Accumulate V for all dims this thread handles
        let v_base = (kv_h * kv_stride + s) * head_dim;
        acc0 = acc0 * rescale + w * t_read(&V, v_base + d0);
        if (d0 + 64u < head_dim) { acc1 = acc1 * rescale + w * t_read(&V, v_base + d0 + 64u); }
        if (d0 + 128u < head_dim) { acc2 = acc2 * rescale + w * t_read(&V, v_base + d0 + 128u); }
        if (d0 + 192u < head_dim) { acc3 = acc3 * rescale + w * t_read(&V, v_base + d0 + 192u); }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write results
    if (d0 < head_dim) { t_write(&Out, q_base + d0, acc0); }
    if (d0 + 64u < head_dim) { t_write(&Out, q_base + d0 + 64u, acc1); }
    if (d0 + 128u < head_dim) { t_write(&Out, q_base + d0 + 128u, acc2); }
    if (d0 + 192u < head_dim) { t_write(&Out, q_base + d0 + 192u, acc3); }
}
)WGSL";

// [attention] gqa_decode — dtype template
static const char* WGSL_GQA_DECODE_T = R"WGSL(
enable subgroups;
// GQA decode — single query attending to all KV cache entries.
// Q: [num_heads * head_dim]  (f32, single token)
// K: [kv_heads, kv_stride, head_dim]  (f32, kv_stride >= total_seq)
// V: [kv_heads, kv_stride, head_dim]  (f32, kv_stride >= total_seq)
// Out: [num_heads * head_dim]  (f32)
// Params: [0]=num_heads, [1]=head_dim, [2]=total_seq, [3]=kv_heads, [4]=scale_u32,
//         [5]=kv_stride (0 = use total_seq, for static KV cache with max_seq slots)
// One workgroup per Q head. Each thread handles dims d, d+64, d+128, ...
// Dispatch: (1, num_heads, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> Q: array<${T}>;
@group(0) @binding(1) var<storage, read> K: array<${T}>;
@group(0) @binding(2) var<storage, read> V: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Out: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> score_scratch: array<f32, 8>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(subgroup_id) sg_id: u32,
        @builtin(num_subgroups) num_sg: u32) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let total_seq = _params_[2];
    let kv_heads = _params_[3];
    let scale = bitcast<f32>(_params_[4]);
    let kv_stride_raw = _params_[5];
    let kv_stride = select(kv_stride_raw, total_seq, kv_stride_raw == 0u);

    let h = wid.y;  // Q head index
    if (h >= num_heads) { return; }

    let kv_h = h / (num_heads / kv_heads);  // map Q head to KV head
    let q_base = h * head_dim;
    let d0 = lid.x;  // base dimension for this thread

    // Each thread accumulates up to 4 V dimensions (supports head_dim up to 256)
    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;

    // Online softmax state (shared across all dims — only depends on Q·K scores)
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;

    for (var s = 0u; s < total_seq; s++) {
        let k_base = (kv_h * kv_stride + s) * head_dim;

        // Compute Q·K once cooperatively instead of repeating the full dot
        // product in every output-dimension thread.
        var score: f32 = 0.0;
        for (var dd = d0; dd < head_dim; dd += 64u) {
            score += t_read(&Q, q_base + dd) * t_read(&K, k_base + dd);
        }
        score = subgroupAdd(score);
        if (sg_lane == 0u) { score_scratch[sg_id] = score; }
        workgroupBarrier();
        if (d0 == 0u) {
            var total = 0.0;
            for (var sg = 0u; sg < num_sg; sg++) { total += score_scratch[sg]; }
            score_scratch[0] = total;
        }
        workgroupBarrier();
        score = score_scratch[0] * scale;

        // Online softmax
        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;

        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        // Accumulate V for all dims this thread handles
        let v_base = (kv_h * kv_stride + s) * head_dim;
        acc0 = acc0 * rescale + w * t_read(&V, v_base + d0);
        if (d0 + 64u < head_dim) { acc1 = acc1 * rescale + w * t_read(&V, v_base + d0 + 64u); }
        if (d0 + 128u < head_dim) { acc2 = acc2 * rescale + w * t_read(&V, v_base + d0 + 128u); }
        if (d0 + 192u < head_dim) { acc3 = acc3 * rescale + w * t_read(&V, v_base + d0 + 192u); }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write results
    if (d0 < head_dim) { t_write(&Out, q_base + d0, acc0); }
    if (d0 + 64u < head_dim) { t_write(&Out, q_base + d0 + 64u, acc1); }
    if (d0 + 128u < head_dim) { t_write(&Out, q_base + d0 + 128u, acc2); }
    if (d0 + 192u < head_dim) { t_write(&Out, q_base + d0 + 192u, acc3); }
}
)WGSL";

// [attention] gqa_decode_attn_sink
static const char* WGSL_GQA_DECODE_ATTN_SINK = R"WGSL(
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> Sinks: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let T_win = _params_[2];
    let kv_heads = _params_[3];
    let scale = bitcast<f32>(_params_[4]);
    let kv_stride = _params_[5];
    let kv_start = _params_[6];

    let h = wid.y;
    if (h >= num_heads) { return; }

    let kv_h = h / (num_heads / kv_heads);
    let q_base = h * head_dim;
    let d0 = lid.x;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;

    // Sliding window: attend to [kv_start, kv_start + T_win)
    for (var t = 0u; t < T_win; t++) {
        let s = kv_start + t;
        let k_base = (kv_h * kv_stride + s) * head_dim;

        var score: f32 = 0.0;
        for (var dd = 0u; dd < head_dim; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score *= scale;

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;

        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_base = (kv_h * kv_stride + s) * head_dim;
        acc0 = acc0 * rescale + w * V[v_base + d0];
        if (d0 + 64u < head_dim) { acc1 = acc1 * rescale + w * V[v_base + d0 + 64u]; }
        if (d0 + 128u < head_dim) { acc2 = acc2 * rescale + w * V[v_base + d0 + 128u]; }
        if (d0 + 192u < head_dim) { acc3 = acc3 * rescale + w * V[v_base + d0 + 192u]; }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Attention sink: competes in softmax but contributes no V
    let sink_logit = Sinks[h];
    let m_new2 = max(m_prev, sink_logit);
    let exp_prev2 = exp(m_prev - m_new2);
    let exp_sink = exp(sink_logit - m_new2);
    let l_new2 = l_prev * exp_prev2 + exp_sink;
    let sink_rescale = l_prev * exp_prev2 / max(l_new2, 1e-10);

    acc0 *= sink_rescale;
    acc1 *= sink_rescale;
    acc2 *= sink_rescale;
    acc3 *= sink_rescale;

    if (d0 < head_dim) { Out[q_base + d0] = acc0; }
    if (d0 + 64u < head_dim) { Out[q_base + d0 + 64u] = acc1; }
    if (d0 + 128u < head_dim) { Out[q_base + d0 + 128u] = acc2; }
    if (d0 + 192u < head_dim) { Out[q_base + d0 + 192u] = acc3; }
}
)WGSL";

// [attention] gqa_fused_attn
static const char* WGSL_GQA_FUSED_ATTN = R"WGSL(
enable f16;
enable subgroups;
diagnostic(off, subgroup_uniformity);

// Fused single-dispatch GQA decode attention with fp16 KV cache.
// Replaces the 2-dispatch chunked approach (attn_p1 + attn_p2).
//
// Each workgroup handles one Q head for T=1 decode.
// Grid: (n_head, 1, 1)
// WG: 32 threads (1 warp)

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(2) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const HD: u32 = 128u;
const HD_PER_THREAD: u32 = 4u;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;

    let kv_stride = _params_[0];
    let n_rep = _params_[1];
    let T_total = _params_[2];
    let max_seq = _params_[3];
    let scale = bitcast<f32>(_params_[5]);
    let neg_inf = bitcast<f32>(_params_[6]);

    let kv_head = head / n_rep;
    let kv_off = kv_head * HD;
    let q_base = head * HD;

    var q: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        q[e] = 0.0;
        if (d < HD) {
            q[e] = Q[q_base + d];
        }
    }

    var acc: array<f32, HD_PER_THREAD>;
    for (var e = 0u; e < HD_PER_THREAD; e++) { acc[e] = 0.0; }
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    for (var t = 0u; t < max_seq; t = t + 1u) {
        let valid = t < T_total;
        let k_base = select(0u, t * kv_stride + kv_off, valid);
        var dot_partial: f32 = 0.0;
        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var k = 0.0;
            if (valid && d < HD) {
                k = f32(K_cache[k_base + d]);
            }
            dot_partial += q[e] * k;
        }
        let dot_qk = subgroupAdd(dot_partial);
        let score = select(neg_inf, dot_qk * scale, valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        for (var e = 0u; e < HD_PER_THREAD; e++) {
            let d = lane * HD_PER_THREAD + e;
            var v = 0.0;
            if (valid && d < HD) {
                v = f32(V_cache[k_base + d]);
            }
            acc[e] = acc[e] * rescale + v * w;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    for (var e = 0u; e < HD_PER_THREAD; e++) {
        let d = lane * HD_PER_THREAD + e;
        if (d < HD) {
            Out[head * HD + d] = acc[e];
        }
    }
}
)WGSL";

// [attention] gqa_prefill
static const char* WGSL_GQA_PREFILL = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const QUERIES_PER_WG: u32 = 4u;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let pastSeq = _params_[2];
    let kv_heads = _params_[3];
    let scale = bitcast<f32>(_params_[4]);
    let max_seq = _params_[5];
    let T = _params_[6];

    let head = wid.x;
    let q_block = wid.y;
    let tid = lid.x;
    let warp_id = tid / 32u;      // which of 4 query positions
    let lane = tid % 32u;

    let q_idx = q_block * QUERIES_PER_WG + warp_id;
    // Don't early return — all threads must participate in subgroupAdd.
    // Use valid flag to mask output writes.
    let valid = q_idx < T;

    let kv_h = head / (num_heads / kv_heads);

    // For head_dim=64, each thread handles 2 dims. For head_dim=128, 4 dims.
    let hd_per_thread = (head_dim + 31u) / 32u;

    // Load Q into registers
    let q_base = select(0u, (q_idx * num_heads + head) * head_dim, valid);
    var q_reg: array<f32, 4>;
    for (var i = 0u; i < hd_per_thread; i = i + 1u) {
        let d = lane * hd_per_thread + i;
        if (d < head_dim && valid) {
            q_reg[i] = Q[q_base + d];
        }
    }

    // Causal bound: this query can attend to positions [0, pastSeq + q_idx + 1)
    let causal_bound = select(0u, pastSeq + q_idx + 1u, valid);

    // Uniform loop bound across all warps in this workgroup: max causal bound
    let max_causal = pastSeq + min(q_block * QUERIES_PER_WG + QUERIES_PER_WG, T);

    var acc: array<f32, 4>;
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;

    for (var s = 0u; s < max_causal; s = s + 1u) {
        let k_base = (kv_h * max_seq + s) * head_dim;
        let causal_valid = s < causal_bound;

        // QK dot product via subgroup reduction
        var partial: f32 = 0.0;
        for (var i = 0u; i < hd_per_thread; i = i + 1u) {
            let d = lane * hd_per_thread + i;
            if (d < head_dim && causal_valid) {
                partial += q_reg[i] * K[k_base + d];
            }
        }
        let dot_qk = subgroupAdd(partial) * scale;
        let score = select(-1e30, dot_qk, causal_valid);

        // Online softmax
        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        // Accumulate V
        let v_base = (kv_h * max_seq + s) * head_dim;
        for (var i = 0u; i < hd_per_thread; i = i + 1u) {
            let d = lane * hd_per_thread + i;
            if (d < head_dim) {
                let v_val = select(0.0, V[v_base + d], causal_valid);
                acc[i] = acc[i] * rescale + w * v_val;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write output (only for valid query positions)
    if (valid) {
        let out_base = (q_idx * num_heads + head) * head_dim;
        for (var i = 0u; i < hd_per_thread; i = i + 1u) {
            let d = lane * hd_per_thread + i;
            if (d < head_dim) {
                Out[out_base + d] = acc[i];
            }
        }
    }
}
)WGSL";

// [attention] kv_cache_append — f32 instantiated
static const char* WGSL_KV_CACHE_APPEND = R"WGSL(
// KV cache append — Copy new K or V into present_key/value at position offset.
// new_kv: [batch, 1, kv_heads * head_dim]  (from current step)
// present: [batch, kv_heads, total_seq, head_dim]  (output = past + new)
// past: [batch, kv_heads, past_seq, head_dim]  (input)
// Params: [0]=kv_heads, [1]=head_dim, [2]=past_seq, [3]=total_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> new_kv: array<f32>;
@group(0) @binding(1) var<storage, read> past: array<f32>;
@group(0) @binding(2) var<storage, read_write> present: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let past_seq = _params_[2];
    let total_seq = _params_[3];

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;

    // Copy past values: present[h, 0..past_seq-1, d] = past[h, 0..past_seq-1, d]
    for (var s = 0u; s < past_seq; s++) {
        t_write(&present, (h * total_seq + s) * head_dim + d, t_read(&past, (h * past_seq + s) * head_dim + d));
    }

    // Append new value: present[h, past_seq, d] = new_kv[h * head_dim + d]
    t_write(&present, (h * total_seq + past_seq) * head_dim + d, t_read(&new_kv, h * head_dim + d));
}
)WGSL";

// [attention] kv_cache_append — dtype template
static const char* WGSL_KV_CACHE_APPEND_T = R"WGSL(
// KV cache append — Copy new K or V into present_key/value at position offset.
// new_kv: [batch, 1, kv_heads * head_dim]  (from current step)
// present: [batch, kv_heads, total_seq, head_dim]  (output = past + new)
// past: [batch, kv_heads, past_seq, head_dim]  (input)
// Params: [0]=kv_heads, [1]=head_dim, [2]=past_seq, [3]=total_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> new_kv: array<${T}>;
@group(0) @binding(1) var<storage, read> past: array<${T}>;
@group(0) @binding(2) var<storage, read_write> present: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let past_seq = _params_[2];
    let total_seq = _params_[3];

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;

    // Copy past values: present[h, 0..past_seq-1, d] = past[h, 0..past_seq-1, d]
    for (var s = 0u; s < past_seq; s++) {
        t_write(&present, (h * total_seq + s) * head_dim + d, t_read(&past, (h * past_seq + s) * head_dim + d));
    }

    // Append new value: present[h, past_seq, d] = new_kv[h * head_dim + d]
    t_write(&present, (h * total_seq + past_seq) * head_dim + d, t_read(&new_kv, h * head_dim + d));
}
)WGSL";

// [attention] kv_cache_write — f32 instantiated
static const char* WGSL_KV_CACHE_WRITE = R"WGSL(
// KV cache write — Write new K or V token into static cache at position offset.
// Unlike kv_cache_append, this does NOT copy past data — the cache buffer is reused.
// new_kv: [kv_heads * head_dim]  (from current step, after RoPE)
// cache:  [kv_heads, max_seq, head_dim]  (static buffer, read_write)
// Params: [0]=kv_heads, [1]=head_dim, [2]=write_pos, [3]=max_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> new_kv: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let write_pos = _params_[2];
    let max_seq = _params_[3];

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;

    // Write new value at cache[h, write_pos, d]
    t_write(&cache, (h * max_seq + write_pos) * head_dim + d, t_read(&new_kv, h * head_dim + d));
}
)WGSL";

// [attention] kv_cache_write — dtype template
static const char* WGSL_KV_CACHE_WRITE_T = R"WGSL(
// KV cache write — Write new K or V token into static cache at position offset.
// Unlike kv_cache_append, this does NOT copy past data — the cache buffer is reused.
// new_kv: [kv_heads * head_dim]  (from current step, after RoPE)
// cache:  [kv_heads, max_seq, head_dim]  (static buffer, read_write)
// Params: [0]=kv_heads, [1]=head_dim, [2]=write_pos, [3]=max_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> new_kv: array<${T}>;
@group(0) @binding(1) var<storage, read_write> cache: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let write_pos = _params_[2];
    let max_seq = _params_[3];

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;

    // Write new value at cache[h, write_pos, d]
    t_write(&cache, (h * max_seq + write_pos) * head_dim + d, t_read(&new_kv, h * head_dim + d));
}
)WGSL";

// [attention] kv_cache_write_batched
static const char* WGSL_KV_CACHE_WRITE_BATCHED = R"WGSL(
// Batched KV cache write — T tokens via workgroup_id.y.
// new_kv: [T, kv_heads * head_dim], cache: [kv_heads, max_seq, head_dim]
// Writes at positions [pastSeq, pastSeq+T)
// Params: [0]=kv_heads, [1]=head_dim, [2]=pastSeq, [3]=max_seq
// Dispatch: (ceil(kv_heads * head_dim / 256), T, 1)

@group(0) @binding(0) var<storage, read> new_kv: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let kv_heads = _params_[0];
    let head_dim = _params_[1];
    let pastSeq = _params_[2];
    let max_seq = _params_[3];
    let tok = wid.y;

    let flat = gid.x;
    let total_elems = kv_heads * head_dim;
    if (flat >= total_elems) { return; }

    let h = flat / head_dim;
    let d = flat % head_dim;
    let write_pos = pastSeq + tok;

    cache[(h * max_seq + write_pos) * head_dim + d] = new_kv[tok * total_elems + h * head_dim + d];
}
)WGSL";

// [attention] qwen35_kv_cache_write
static const char* WGSL_QWEN35_KV_CACHE_WRITE = R"WGSL(
enable f16;

// Write already-normalized/rotated Qwen3.5 K and raw V into fp16 KV cache.
//
// Bindings:
//   0: K        [kv_dim] f32
//   1: V        [kv_dim] f32
//   2: K_cache  [seq * kv_dim] f16
//   3: V_cache  [seq * kv_dim] f16
//   4: params   [kv_dim, cache_offset_words]

@group(0) @binding(0) var<storage, read>       K:       array<f32>;
@group(0) @binding(1) var<storage, read>       V:       array<f32>;
@group(0) @binding(2) var<storage, read_write> KCache:  array<f16>;
@group(0) @binding(3) var<storage, read_write> VCache:  array<f16>;
@group(0) @binding(4) var<storage, read>       params:  array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_dim = params[0];
    let off = params[1];
    let i = gid.x;
    if (i >= kv_dim) { return; }
    KCache[off + i] = f16(K[i]);
    VCache[off + i] = f16(V[i]);
}
)WGSL";

// [attention] qwen35_kv_cache_write_rope
static const char* WGSL_QWEN35_KV_CACHE_WRITE_ROPE = R"WGSL(
enable f16;

// Qwen3.5 K RoPE decode path fused with fp16 KV cache write.
//
// Bindings:
//   0: K        [kv_dim] f32, normalized K input
//   1: V        [kv_dim] f32
//   2: K_cache  [seq * kv_dim] f16
//   3: V_cache  [seq * kv_dim] f16
//   4: cos      [max_seq * rope_half] f32, standard RoPE cos table
//   5: sin      [max_seq * rope_half] f32, standard RoPE sin table
//   6: rope_p   [n_kv_head, head_dim, s0, s1, s2, s3, pos, rope_half]
//   7: kv_p     [kv_dim, cache_offset_words]

@group(0) @binding(0) var<storage, read>       K:       array<f32>;
@group(0) @binding(1) var<storage, read>       V:       array<f32>;
@group(0) @binding(2) var<storage, read_write> KCache:  array<f16>;
@group(0) @binding(3) var<storage, read_write> VCache:  array<f16>;
@group(0) @binding(4) var<storage, read>       Cos:     array<f32>;
@group(0) @binding(5) var<storage, read>       Sin:     array<f32>;
@group(0) @binding(6) var<storage, read>       rope_p:  array<u32>;
@group(0) @binding(7) var<storage, read>       kv_p:    array<u32>;

fn section_local_idx(pair_idx: u32, s0: u32, s1: u32, s2: u32) -> u32 {
    var local_idx = pair_idx;
    var section = 0u;
    if (local_idx >= s0) {
        local_idx = local_idx - s0;
        section = 1u;
    }
    if (section == 1u && local_idx >= s1) {
        local_idx = local_idx - s1;
        section = 2u;
    }
    if (section == 2u && local_idx >= s2) {
        local_idx = local_idx - s2;
    }
    return local_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_dim = kv_p[0];
    let off = kv_p[1];
    let n_kv_head = rope_p[0];
    let head_dim  = rope_p[1];
    let head_idx = gid.y;
    let elem_idx = gid.x;
    if (head_idx >= n_kv_head || elem_idx >= head_dim) { return; }

    let i = head_idx * head_dim + elem_idx;
    if (i >= kv_dim) { return; }

    let s0        = rope_p[2];
    let s1        = rope_p[3];
    let s2        = rope_p[4];
    let s3        = rope_p[5];
    let pos       = rope_p[6];
    let rope_half = rope_p[7];

    let head_base = head_idx * head_dim;
    let total_pairs = s0 + s1 + s2 + s3;

    var k_val = K[i];
    if (elem_idx < total_pairs) {
        let pair_idx = elem_idx;
        let local_idx = section_local_idx(pair_idx, s0, s1, s2);
        let table_idx = pos * rope_half + local_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = K[head_base + pair_idx];
        let x1 = K[head_base + pair_idx + total_pairs];
        k_val = x0 * c - x1 * s;
    } else if (elem_idx < total_pairs * 2u) {
        let pair_idx = elem_idx - total_pairs;
        let local_idx = section_local_idx(pair_idx, s0, s1, s2);
        let table_idx = pos * rope_half + local_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = K[head_base + pair_idx];
        let x1 = K[head_base + pair_idx + total_pairs];
        k_val = x0 * s + x1 * c;
    }

    KCache[off + i] = f16(k_val);
    VCache[off + i] = f16(V[i]);
}
)WGSL";

// [attention] qwen35_rope_kv_batched
static const char* WGSL_QWEN35_ROPE_KV_BATCHED = R"WGSL(
enable f16;
// Batched Q RoPE and K RoPE + KV write. Params:
// [T,n_head,n_kv,head_dim,pos0,cache0,s0,s1] followed by [s2,s3,rope_half].
@group(0) @binding(0) var<storage,read> Q:array<f32>;
@group(0) @binding(1) var<storage,read> K:array<f32>;
@group(0) @binding(2) var<storage,read> V:array<f32>;
@group(0) @binding(3) var<storage,read_write> QO:array<f32>;
@group(0) @binding(4) var<storage,read_write> KC:array<f16>;
@group(0) @binding(5) var<storage,read_write> VC:array<f16>;
@group(0) @binding(6) var<storage,read> Cos:array<f32>;
@group(0) @binding(7) var<storage,read> Sin:array<f32>;
@group(0) @binding(8) var<storage,read> P:array<u32>;
fn local_pair(p:u32,s0:u32,s1:u32,s2:u32)->u32{if(p<s0){return p;}if(p<s0+s1){return p-s0;}if(p<s0+s1+s2){return p-s0-s1;}return p-s0-s1-s2;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
 let T=P[0];let nh=P[1];let nk=P[2];let hd=P[3];let pos0=P[4];let cache0=P[5];
 let s0=P[6];let s1=P[7];let s2=P[8];let s3=P[9];let rh=P[10];let pairs=s0+s1+s2+s3;
 let h=wid.x;let t=wid.y;let d=lid.x;if(t>=T||d>=hd){return;}let pos=pos0+t;
 if(h<nh){let base=(t*nh+h)*hd;var x=Q[base+d];if(d<2u*pairs){let p=d%pairs;let ti=pos*rh+p;let a=Q[base+p];let b=Q[base+p+pairs];x=select(a*Cos[ti]-b*Sin[ti],a*Sin[ti]+b*Cos[ti],d>=pairs);}QO[base+d]=x;}
 if(h<nk){let base=(t*nk+h)*hd;var x=K[base+d];if(d<2u*pairs){let p=d%pairs;let ti=pos*rh+local_pair(p,s0,s1,s2);let a=K[base+p];let b=K[base+p+pairs];x=select(a*Cos[ti]-b*Sin[ti],a*Sin[ti]+b*Cos[ti],d>=pairs);}let dst=((cache0+t)*nk+h)*hd+d;KC[dst]=f16(x);VC[dst]=f16(V[base+d]);}
}
)WGSL";

// [attention] qwen35_split_qg_batched
static const char* WGSL_QWEN35_SPLIT_QG_BATCHED = R"WGSL(
// Params [T,n_head,head_dim].
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read_write> Q:array<f32>;
@group(0) @binding(2) var<storage,read_write> G:array<f32>;
@group(0) @binding(3) var<storage,read> P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let T=P[0];let nh=P[1];let hd=P[2];let D=nh*hd;let i=gid.x;if(i>=T*D){return;}
 let t=i/D;let j=i%D;let h=j/hd;let d=j%hd;let src=t*2u*D+h*2u*hd+d;
 Q[i]=X[src];G[i]=X[src+hd];
}
)WGSL";

// [attention] rope_batched
static const char* WGSL_ROPE_BATCHED = R"WGSL(
enable f16;
enable subgroups;

// Batched RoPE + KV cache scatter for prefill (NO QK-norm).
// Each workgroup handles ONE head for ONE token.
// Grid: (n_heads_total, T, 1) where n_heads_total = n_head + n_kv
//   WG (head < n_head, t): processes Q head
//   WG (head >= n_head, t): processes K head + V scatter
//
// QKV layout: [T × (qDim + 2*kvDim)] row-major
// Output Q: QRot[T × qDim] row-major (T × n_head × HD)
// Output K/V: scattered into KV cache at positions pos_offset..pos_offset+T-1
//
// Params: [n_head, qDim, kvDim, pos_offset, half_dim, cache_len, eps_u32, n_kv]
// Bindings 6,7 (NormQ/NormK) are unused but bound for compatibility.

@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> _unused_normQ: array<f32>;
@group(0) @binding(7) var<storage, read> _unused_normK: array<f32>;
@group(0) @binding(8) var<storage, read> _params_: array<u32>;

const HD: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = wid.x;
    let t = wid.y;
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let cache_len = _params_[5];
    let n_kv = _params_[7];

    let qkv_stride = qDim + 2u * kvDim;
    let pos = pos_offset + t;

    if (head_idx < n_head) {
        // ── Q head: RoPE only, no norm ──────────────────────────────
        let src_base = t * qkv_stride + head_idx * HD;
        let dst_base = t * n_head * HD + head_idx * HD;
        let cos_base = pos * half_dim;

        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let qi = QKV[src_base + i];
            let qj = QKV[src_base + j];
            let c = Cos[cos_base + i];
            let s = Sin[cos_base + i];
            QRot[dst_base + i] = qi * c - qj * s;
            QRot[dst_base + j] = qj * c + qi * s;
        }
        // Non-rotary dimensions: pass through unchanged
        let rot_dim = 2u * half_dim;
        for (var i = rot_dim + tid; i < HD; i += 128u) {
            QRot[dst_base + i] = QKV[src_base + i];
        }
    } else {
        // ── KV head: RoPE + cache scatter, no norm ──────────────────
        let kv_head = head_idx - n_head;
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        let cache_pos = cache_len + t;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;

        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let ki = QKV[k_src + i];
            let kj = QKV[k_src + j];
            let c = Cos[pos * half_dim + i];
            let s = Sin[pos * half_dim + i];
            K_cache[k_dst + i] = f16(ki * c - kj * s);
            K_cache[k_dst + j] = f16(kj * c + ki * s);
        }
        // Non-rotary dimensions: pass through unchanged
        let rot_dim_k = 2u * half_dim;
        for (var i = rot_dim_k + tid; i < HD; i += 128u) {
            K_cache[k_dst + i] = f16(QKV[k_src + i]);
        }

        // V: just copy to cache
        for (var i = tid; i < HD; i += 128u) {
            V_cache[v_dst + i] = f16(QKV[v_src + i]);
        }
    }
}
)WGSL";

// [attention] rope_batched_simple
static const char* WGSL_ROPE_BATCHED_SIMPLE = R"WGSL(
enable f16;
enable subgroups;

// Simple batched RoPE + QK-norm + KV cache scatter for prefill.
// Each workgroup handles ONE head for ONE token.
// Grid: (n_heads_total, T, 1) where n_heads_total = n_head + n_kv
//   WG (head < n_head, t): processes Q head
//   WG (head >= n_head, t): processes K head + V scatter
//
// QKV layout: [T × (qDim + 2*kvDim)] row-major
// Output Q: QRot[T × qDim] row-major (T × n_head × HD)
// Output K/V: scattered into KV cache at positions pos_offset..pos_offset+T-1
//
// Params: [n_head, qDim, kvDim, pos_offset, half_dim, cache_len, eps_u32, n_kv]

@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f16>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f16>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QNormW: array<f32>;
@group(0) @binding(7) var<storage, read> KNormW: array<f32>;
@group(0) @binding(8) var<storage, read> _params_: array<u32>;

const HD: u32 = 128u;
const HALF: u32 = 64u;

var<workgroup> smem_q: array<f32, 4>;
var<workgroup> smem_k: array<f32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = wid.x;   // which head (0..n_head-1 for Q, n_head..n_head+n_kv-1 for K)
    let t = wid.y;           // which token in the batch
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let cache_len = _params_[5];
    let eps = bitcast<f32>(_params_[6]);
    let n_kv = _params_[7];

    let qkv_stride = qDim + 2u * kvDim;
    let pos = pos_offset + t;

    if (head_idx < n_head) {
        // ── Q head processing ────────────────────────────────────────
        let src_base = t * qkv_stride + head_idx * HD;

        // Load Q values and compute RMS norm
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            let v = QKV[src_base + i];
            sum_sq += v * v;
        }
        // Warp reduce
        let warp_sum = subgroupAdd(sum_sq);
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { smem_q[warp_id] = warp_sum; }
        workgroupBarrier();
        let final_sq = smem_q[0] + smem_q[1] + smem_q[2] + smem_q[3];
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply norm + RoPE, write to QRot
        let dst_base = t * n_head * HD + head_idx * HD;
        let cos_base = pos * half_dim;
        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let qi = QKV[src_base + i] * rstd * QNormW[i];
            let qj = QKV[src_base + j] * rstd * QNormW[j];
            let c = Cos[cos_base + i];
            let s = Sin[cos_base + i];
            QRot[dst_base + i] = qi * c - qj * s;
            QRot[dst_base + j] = qj * c + qi * s;
        }
        // Non-rotary dimensions: apply norm only (no rotation)
        let rot_dim = 2u * half_dim;
        for (var i = rot_dim + tid; i < HD; i += 128u) {
            QRot[dst_base + i] = QKV[src_base + i] * rstd * QNormW[i];
        }
    } else {
        // ── KV head processing ───────────────────────────────────────
        let kv_head = head_idx - n_head;
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        // K: load, norm, RoPE, scatter to cache
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            let v = QKV[k_src + i];
            sum_sq += v * v;
        }
        let warp_sum = subgroupAdd(sum_sq);
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { smem_k[warp_id] = warp_sum; }
        workgroupBarrier();
        let final_sq = smem_k[0] + smem_k[1] + smem_k[2] + smem_k[3];
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // KV cache layout: [MAX_SEQ × n_kv × HD]
        let cache_pos = cache_len + t;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;

        for (var i = tid; i < half_dim; i += 128u) {
            let j = i + half_dim;
            let ki = QKV[k_src + i] * rstd * KNormW[i];
            let kj = QKV[k_src + j] * rstd * KNormW[j];
            let c = Cos[pos * half_dim + i];
            let s = Sin[pos * half_dim + i];
            K_cache[k_dst + i] = f16(ki * c - kj * s);
            K_cache[k_dst + j] = f16(kj * c + ki * s);
        }
        // Non-rotary dimensions: apply norm only (no rotation)
        let rot_dim_k = 2u * half_dim;
        for (var i = rot_dim_k + tid; i < HD; i += 128u) {
            K_cache[k_dst + i] = f16(QKV[k_src + i] * rstd * KNormW[i]);
        }

        // V: just copy to cache (no RoPE, no norm)
        for (var i = tid; i < HD; i += 128u) {
            V_cache[v_dst + i] = f16(QKV[v_src + i]);
        }
    }
}
)WGSL";

// [attention] rope_inplace — f32 instantiated
static const char* WGSL_ROPE_INPLACE = R"WGSL(
// RoPE in-place — Rotary positional embedding applied in-place.
// One workgroup per head. Sequential loop over rotary dimensions.
// Params: [0]=num_heads, [1]=head_dim, [2]=rotary_dim, [3]=pos
// Dispatch: (ceil(num_heads/64), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_read_rw(buf: ptr<storage, array<f32>, read_write>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let rotary_dim = _params_[2];
    let pos = _params_[3];

    let h = gid.x;
    if (h >= num_heads) { return; }

    let half_rot = rotary_dim / 2u;
    let head_base = h * head_dim;
    let cache_base = pos * half_rot;

    for (var d = 0u; d < half_rot; d++) {
        let cos_val = t_read(&cos_cache, cache_base + d);
        let sin_val = t_read(&sin_cache, cache_base + d);
        let x0 = t_read_rw(&data, head_base + d);
        let x1 = t_read_rw(&data, head_base + half_rot + d);
        t_write(&data, head_base + d, x0 * cos_val - x1 * sin_val);
        t_write(&data, head_base + half_rot + d, x1 * cos_val + x0 * sin_val);
    }
}
)WGSL";

// [attention] rope_inplace — dtype template
static const char* WGSL_ROPE_INPLACE_T = R"WGSL(
// RoPE in-place — Rotary positional embedding applied in-place.
// One workgroup per head. Sequential loop over rotary dimensions.
// Params: [0]=num_heads, [1]=head_dim, [2]=rotary_dim, [3]=pos
// Dispatch: (ceil(num_heads/64), 1, 1)

${T_READ}
${T_READ_RW}
${T_WRITE}

@group(0) @binding(0) var<storage, read_write> data: array<${T}>;
@group(0) @binding(1) var<storage, read> cos_cache: array<${T}>;
@group(0) @binding(2) var<storage, read> sin_cache: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let rotary_dim = _params_[2];
    let pos = _params_[3];

    let h = gid.x;
    if (h >= num_heads) { return; }

    let half_rot = rotary_dim / 2u;
    let head_base = h * head_dim;
    let cache_base = pos * half_rot;

    for (var d = 0u; d < half_rot; d++) {
        let cos_val = t_read(&cos_cache, cache_base + d);
        let sin_val = t_read(&sin_cache, cache_base + d);
        let x0 = t_read_rw(&data, head_base + d);
        let x1 = t_read_rw(&data, head_base + half_rot + d);
        t_write(&data, head_base + d, x0 * cos_val - x1 * sin_val);
        t_write(&data, head_base + half_rot + d, x1 * cos_val + x0 * sin_val);
    }
}
)WGSL";

// [attention] rope_inplace_batched
static const char* WGSL_ROPE_INPLACE_BATCHED = R"WGSL(
// Batched RoPE — T tokens via workgroup_id.y.
// data: [T, num_heads, head_dim], positions [posOffset, posOffset+T)
// Params: [0]=num_heads, [1]=head_dim, [2]=rotary_dim, [3]=posOffset
// Dispatch: (ceil(num_heads/64), T, 1)

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let num_heads = _params_[0];
    let head_dim = _params_[1];
    let rotary_dim = _params_[2];
    let posOffset = _params_[3];
    let tok = wid.y;

    let h = gid.x;
    if (h >= num_heads) { return; }

    let pos = posOffset + tok;
    let half_rot = rotary_dim / 2u;
    let head_base = tok * num_heads * head_dim + h * head_dim;
    let cache_base = pos * half_rot;

    for (var d = 0u; d < half_rot; d++) {
        let cos_val = cos_cache[cache_base + d];
        let sin_val = sin_cache[cache_base + d];
        let x0 = data[head_base + d];
        let x1 = data[head_base + half_rot + d];
        data[head_base + d] = x0 * cos_val - x1 * sin_val;
        data[head_base + half_rot + d] = x1 * cos_val + x0 * sin_val;
    }
}
)WGSL";

// [attention] rotary_embedding — f32 instantiated
static const char* WGSL_ROTARY_EMBEDDING = R"WGSL(
// RotaryEmbedding — apply rotary position embeddings
// Supports both interleaved and non-interleaved modes.
//
// Dispatch: (ceil(total/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> PosIds: array<i32>;
@group(0) @binding(2) var<storage, read> CosCache: array<f32>;
@group(0) @binding(3) var<storage, read> SinCache: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let head_dim = _params_[1];
    let interleaved = _params_[2];

    let idx = gid.x;
    if (idx >= total) { return; }

    let half = head_dim / 2u;
    let d = idx % head_dim;
    let pos_idx = idx / head_dim;
    let nPosIds = _params_[3];
    let pos = select(0, PosIds[pos_idx % nPosIds], nPosIds > 0u);

    if (interleaved != 0u) {
        let pair = d / 2u;
        let cos_val = t_read(&CosCache, u32(pos) * half + pair);
        let sin_val = t_read(&SinCache, u32(pos) * half + pair);
        let base = idx - d;
        if (d % 2u == 0u) {
            t_write(&Y, idx, t_read(&X, idx) * cos_val - t_read(&X, base + d + 1u) * sin_val);
        } else {
            t_write(&Y, idx, t_read(&X, base + d - 1u) * sin_val + t_read(&X, idx) * cos_val);
        }
    } else {
        if (d < half) {
            let cos_val = t_read(&CosCache, u32(pos) * half + d);
            let sin_val = t_read(&SinCache, u32(pos) * half + d);
            let base = idx - d;
            t_write(&Y, idx, t_read(&X, idx) * cos_val - t_read(&X, base + d + half) * sin_val);
        } else {
            let d2 = d - half;
            let cos_val = t_read(&CosCache, u32(pos) * half + d2);
            let sin_val = t_read(&SinCache, u32(pos) * half + d2);
            let base = idx - d;
            t_write(&Y, idx, t_read(&X, base + d2) * sin_val + t_read(&X, idx) * cos_val);
        }
    }
}
)WGSL";

// [attention] rotary_embedding — dtype template
static const char* WGSL_ROTARY_EMBEDDING_T = R"WGSL(
// RotaryEmbedding — apply rotary position embeddings
// Supports both interleaved and non-interleaved modes.
//
// Dispatch: (ceil(total/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> PosIds: array<i32>;
@group(0) @binding(2) var<storage, read> CosCache: array<${T}>;
@group(0) @binding(3) var<storage, read> SinCache: array<${T}>;
@group(0) @binding(4) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let head_dim = _params_[1];
    let interleaved = _params_[2];

    let idx = gid.x;
    if (idx >= total) { return; }

    let half = head_dim / 2u;
    let d = idx % head_dim;
    let pos_idx = idx / head_dim;
    let nPosIds = _params_[3];
    let pos = select(0, PosIds[pos_idx % nPosIds], nPosIds > 0u);

    if (interleaved != 0u) {
        let pair = d / 2u;
        let cos_val = t_read(&CosCache, u32(pos) * half + pair);
        let sin_val = t_read(&SinCache, u32(pos) * half + pair);
        let base = idx - d;
        if (d % 2u == 0u) {
            t_write(&Y, idx, t_read(&X, idx) * cos_val - t_read(&X, base + d + 1u) * sin_val);
        } else {
            t_write(&Y, idx, t_read(&X, base + d - 1u) * sin_val + t_read(&X, idx) * cos_val);
        }
    } else {
        if (d < half) {
            let cos_val = t_read(&CosCache, u32(pos) * half + d);
            let sin_val = t_read(&SinCache, u32(pos) * half + d);
            let base = idx - d;
            t_write(&Y, idx, t_read(&X, idx) * cos_val - t_read(&X, base + d + half) * sin_val);
        } else {
            let d2 = d - half;
            let cos_val = t_read(&CosCache, u32(pos) * half + d2);
            let sin_val = t_read(&SinCache, u32(pos) * half + d2);
            let base = idx - d;
            t_write(&Y, idx, t_read(&X, base + d2) * sin_val + t_read(&X, idx) * cos_val);
        }
    }
}
)WGSL";

// ─── [attn] ────────────────────────────────────────────────────────

// [attn] gated_output
static const char* WGSL_GATED_OUTPUT = R"WGSL(
// Gated attention output — qwen35moe-style modulation
//
// qwen35moe replaces the standard `attn_output @ W_O` with a learned gate:
//   gated_out[c] = attn_out[c] * sigmoid(gate_proj[c])
// where gate_proj = x @ attn_gate.weight (computed separately).
//
// Bindings:
//   0: attn_out  [d]  — attention output (will be modulated)
//   1: gate_in   [d]  — pre-sigmoid gate
//   2: result    [d]  — output
//   3: _params_      — [d]

@group(0) @binding(0) var<storage, read>       attn_out: array<f32>;
@group(0) @binding(1) var<storage, read>       gate_in:  array<f32>;
@group(0) @binding(2) var<storage, read_write> result:   array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = _params_[0];
    let c = gid.x;
    if (c >= d) { return; }
    let g = gate_in[c];
    let sig = 1.0 / (1.0 + exp(-g));
    result[c] = attn_out[c] * sig;
}
)WGSL";

// [attn] head_rmsnorm
static const char* WGSL_HEAD_RMSNORM = R"WGSL(
// Apply RMSNorm in-place (per-head if needed)
// Used for qwen35moe attn_q_norm + attn_k_norm dispatch
//
// Bindings:
//   0: X         array<f32>  in-place [n_head * head_dim]
//   1: W         array<f32>  norm weights [head_dim]
//   2: _params_  [n_head, head_dim, eps_as_u32_bits]

@group(0) @binding(0) var<storage, read_write> X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W:        array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

var<workgroup> sm: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_head   = _params_[0];
    let head_dim = _params_[1];
    let eps      = bitcast<f32>(_params_[2]);
    let head_idx = wid.x;
    if (head_idx >= n_head) { return; }

    let base = head_idx * head_dim;
    let tid  = lid.x;

    var sum_sq: f32 = 0.0;
    for (var j = tid; j < head_dim; j = j + 64u) {
        let v = X[base + j];
        sum_sq = sum_sq + v * v;
    }
    sm[tid] = sum_sq;
    workgroupBarrier();
    if (tid < 32u) { sm[tid] = sm[tid] + sm[tid + 32u]; } workgroupBarrier();
    if (tid < 16u) { sm[tid] = sm[tid] + sm[tid + 16u]; } workgroupBarrier();
    if (tid < 8u)  { sm[tid] = sm[tid] + sm[tid + 8u];  } workgroupBarrier();
    if (tid < 4u)  { sm[tid] = sm[tid] + sm[tid + 4u];  } workgroupBarrier();
    if (tid < 2u)  { sm[tid] = sm[tid] + sm[tid + 2u];  } workgroupBarrier();
    if (tid < 1u)  { sm[tid] = sm[tid] + sm[tid + 1u];  } workgroupBarrier();

    let inv_rms = 1.0 / sqrt(sm[0] / f32(head_dim) + eps);
    for (var j = tid; j < head_dim; j = j + 64u) {
        X[base + j] = X[base + j] * inv_rms * W[j];
    }
}
)WGSL";

// [attn] qwen35_rope_multi_partial
static const char* WGSL_QWEN35_ROPE_MULTI_PARTIAL = R"WGSL(
// Qwen3.5 partial multi-section RoPE for decode.
//
// Text-only Qwen3.5 uses MRoPE sections over the rotated prefix. For the
// current GGUFs the 64 rotary dimensions are split as 32 pairs: [11, 11, 10, 0].
// Sections select independent position channels, but the RoPE frequency index
// remains continuous across them (matching ggml_rope_multi for MRoPE).
//
// Bindings:
//   0: X        [n_head * head_dim] f32, in-place
//   1: cos      [max_seq * rope_half] f32, standard RoPE cos table
//   2: sin      [max_seq * rope_half] f32, standard RoPE sin table
//   3: params   [n_head, head_dim, s0, s1, s2, s3, pos, rope_half]

@group(0) @binding(0) var<storage, read_write> X:      array<f32>;
@group(0) @binding(1) var<storage, read>       Cos:    array<f32>;
@group(0) @binding(2) var<storage, read>       Sin:    array<f32>;
@group(0) @binding(3) var<storage, read>       params: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head    = params[0];
    let head_dim  = params[1];
    let s0        = params[2];
    let s1        = params[3];
    let s2        = params[4];
    let s3        = params[5];
    let pos       = params[6];
    let rope_half = params[7];

    let total_pairs = s0 + s1 + s2 + s3;
    let head_idx = gid.y;
    let pair_idx = gid.x;
    if (head_idx >= n_head || pair_idx >= total_pairs) { return; }

    let table_idx = pos * rope_half + pair_idx;
    let c = Cos[table_idx];
    let s = Sin[table_idx];

    let head_base = head_idx * head_dim;
    let x0_idx = head_base + pair_idx;
    let x1_idx = head_base + pair_idx + total_pairs;
    let x0 = X[x0_idx];
    let x1 = X[x1_idx];

    X[x0_idx] = x0 * c - x1 * s;
    X[x1_idx] = x0 * s + x1 * c;
}
)WGSL";

// [attn] qwen35_rope_q_to_qrot
static const char* WGSL_QWEN35_ROPE_Q_TO_QROT = R"WGSL(
// Qwen3.5 Q RoPE decode path, writing the attention-ready Q buffer.
//
// Bindings:
//   0: X        [n_head * head_dim] f32, normalized Q input
//   1: Out      [n_head * head_dim] f32, rotated/copy output for attention
//   2: cos      [max_seq * rope_half] f32, standard RoPE cos table
//   3: sin      [max_seq * rope_half] f32, standard RoPE sin table
//   4: params   [n_head, head_dim, s0, s1, s2, s3, pos, rope_half]

@group(0) @binding(0) var<storage, read>       X:      array<f32>;
@group(0) @binding(1) var<storage, read_write> Out:    array<f32>;
@group(0) @binding(2) var<storage, read>       Cos:    array<f32>;
@group(0) @binding(3) var<storage, read>       Sin:    array<f32>;
@group(0) @binding(4) var<storage, read>       params: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head    = params[0];
    let head_dim  = params[1];
    let s0        = params[2];
    let s1        = params[3];
    let s2        = params[4];
    let s3        = params[5];
    let pos       = params[6];
    let rope_half = params[7];

    let head_idx = gid.y;
    let elem_idx = gid.x;
    if (head_idx >= n_head || elem_idx >= head_dim) { return; }

    let total_pairs = s0 + s1 + s2 + s3;
    let head_base = head_idx * head_dim;
    let out_idx = head_base + elem_idx;

    if (elem_idx < total_pairs) {
        let pair_idx = elem_idx;
        let table_idx = pos * rope_half + pair_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = X[head_base + pair_idx];
        let x1 = X[head_base + pair_idx + total_pairs];
        Out[out_idx] = x0 * c - x1 * s;
    } else if (elem_idx < total_pairs * 2u) {
        let pair_idx = elem_idx - total_pairs;
        let table_idx = pos * rope_half + pair_idx;
        let c = Cos[table_idx];
        let s = Sin[table_idx];
        let x0 = X[head_base + pair_idx];
        let x1 = X[head_base + pair_idx + total_pairs];
        Out[out_idx] = x0 * s + x1 * c;
    } else {
        Out[out_idx] = X[out_idx];
    }
}
)WGSL";

// [attn] rope_multi
static const char* WGSL_ROPE_MULTI = R"WGSL(
// Multi-axis RoPE (MRoPE) — used by qwen35moe attention layers for multimodal
//
// Standard RoPE rotates pairs (x_i, x_{i+nrot/2}) by angle theta_i = pos * base^(-2i/d).
// MRoPE splits the rotation dimensions into multiple sections (e.g. T/H/W axes
// for vision tokens) with each section getting position from a different axis.
//
// Sections: 4 ints [s0, s1, s2, s3] from rope.dimension_sections (qwen35moe = [11,11,10,0]).
// Each section uses its own position; for text decode all sections see the same pos.
//
// Bindings:
//   0: Q          [n_head * head_dim]       in-place (read+write)
//   1: cos_sin    [4 sections, each (rot_dim/2) cos + sin] precomputed per position
//   2: _params_  — [n_head, head_dim, s0, s1, s2, s3, pos]

@group(0) @binding(0) var<storage, read_write> X:        array<f32>;
@group(0) @binding(1) var<storage, read>       cos_sin:  array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head    = _params_[0];
    let head_dim  = _params_[1];
    let s0        = _params_[2];
    let s1        = _params_[3];
    let s2        = _params_[4];
    let s3        = _params_[5];
    // s0+s1+s2+s3 = rot_dim/2 (total pairs to rotate per head)

    let head_idx = gid.y;
    let pair_idx = gid.x;  // 0..(rot_dim/2)-1
    if (head_idx >= n_head) { return; }
    let total_pairs = s0 + s1 + s2 + s3;
    if (pair_idx >= total_pairs) { return; }

    // Find which axis-section this pair belongs to.
    var section: u32 = 0u;
    var local_idx: u32 = pair_idx;
    if (local_idx >= s0) { local_idx = local_idx - s0; section = 1u; }
    if (section == 1u && local_idx >= s1) { local_idx = local_idx - s1; section = 2u; }
    if (section == 2u && local_idx >= s2) { local_idx = local_idx - s2; section = 3u; }

    // cos/sin table is laid out per section: section i has 2*s_i floats (cos+sin alternating).
    var sec_off: u32 = 0u;
    if (section >= 1u) { sec_off = sec_off + 2u * s0; }
    if (section >= 2u) { sec_off = sec_off + 2u * s1; }
    if (section >= 3u) { sec_off = sec_off + 2u * s2; }
    let c = cos_sin[sec_off + 2u * local_idx + 0u];
    let s = cos_sin[sec_off + 2u * local_idx + 1u];

    // Apply rotation to the pair (x_i, x_{i+total_pairs}) — neox-style
    let base = head_idx * head_dim + pair_idx;
    let x0 = X[base];
    let x1 = X[base + total_pairs];
    X[base]               = x0 * c - x1 * s;
    X[base + total_pairs] = x0 * s + x1 * c;
}
)WGSL";

// [attn] rope_multi_partial
static const char* WGSL_ROPE_MULTI_PARTIAL = R"WGSL(
// Partial-rotation Multi-axis RoPE for qwen35moe
//
// qwen35moe has head_dim=256 but rope_dim=64 — only the first 64 elements
// per head rotate (across 4 axis sections [11,11,10,0]). The remaining
// 192 elements pass through unchanged.
//
// Bindings:
//   0: X         [n_head * head_dim]  in-place
//   1: cos_sin   [2 * rope_dim/2 * 4_sections, but only s0+s1+s2+s3 pairs used]
//   2: _params_  [n_head, head_dim, s0, s1, s2, s3, pos]
//
// rot_dim total pairs = s0 + s1 + s2 + s3 (each pair = 2 elements; total
// rotated = 2 * total_pairs which must equal rope_dim).

@group(0) @binding(0) var<storage, read_write> X:        array<f32>;
@group(0) @binding(1) var<storage, read>       cos_sin:  array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head    = _params_[0];
    let head_dim  = _params_[1];
    let s0        = _params_[2];
    let s1        = _params_[3];
    let s2        = _params_[4];
    let s3        = _params_[5];
    let total_pairs = s0 + s1 + s2 + s3;
    let rope_dim    = 2u * total_pairs;

    let head_idx = gid.y;
    let pair_idx = gid.x;
    if (head_idx >= n_head) { return; }
    if (pair_idx >= total_pairs) { return; }

    // Determine axis-section
    var section: u32 = 0u;
    var local_idx: u32 = pair_idx;
    if (local_idx >= s0) { local_idx = local_idx - s0; section = 1u; }
    if (section == 1u && local_idx >= s1) { local_idx = local_idx - s1; section = 2u; }
    if (section == 2u && local_idx >= s2) { local_idx = local_idx - s2; section = 3u; }

    var sec_off: u32 = 0u;
    if (section >= 1u) { sec_off = sec_off + 2u * s0; }
    if (section >= 2u) { sec_off = sec_off + 2u * s1; }
    if (section >= 3u) { sec_off = sec_off + 2u * s2; }
    let c = cos_sin[sec_off + 2u * local_idx + 0u];
    let s = cos_sin[sec_off + 2u * local_idx + 1u];

    // Apply rotation to first rope_dim elements of this head only.
    // pair (x_i, x_{i + total_pairs}) — neox-style — within first rope_dim of head.
    let head_base = head_idx * head_dim;
    let x0 = X[head_base + pair_idx];
    let x1 = X[head_base + pair_idx + total_pairs];
    X[head_base + pair_idx]               = x0 * c - x1 * s;
    X[head_base + pair_idx + total_pairs] = x0 * s + x1 * c;
    // Elements [rope_dim .. head_dim) per head are untouched.
}
)WGSL";

// [attn] split_qg
static const char* WGSL_SPLIT_QG = R"WGSL(
// Joint Q+gate output splitter for qwen35moe attention layers
//
// qwen35moe's wq output shape: [(head_dim * 2) * n_head] per token
//   first head_dim elements per head = Q
//   next head_dim elements per head = gate (pre-sigmoid)
//
// This kernel writes Q to one buffer and gate to another, contiguous per-head.
//
// Bindings:
//   0: Qcur_full   [(2 * head_dim) * n_head] — joint output of wq matmul
//   1: Q_out       [head_dim * n_head]
//   2: gate_out    [head_dim * n_head]
//   3: _params_   — [n_head, head_dim]

@group(0) @binding(0) var<storage, read>       Qfull:    array<f32>;
@group(0) @binding(1) var<storage, read_write> Q_out:    array<f32>;
@group(0) @binding(2) var<storage, read_write> gate_out: array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_head   = _params_[0];
    let head_dim = _params_[1];
    let total = n_head * head_dim;
    let i = gid.x;
    if (i >= total) { return; }
    let head_idx  = i / head_dim;
    let inner_idx = i % head_dim;
    let src_base = head_idx * (head_dim * 2u);
    Q_out[i]    = Qfull[src_base + inner_idx];
    gate_out[i] = Qfull[src_base + head_dim + inner_idx];
}
)WGSL";

// ─── [conv] ────────────────────────────────────────────────────────

// [conv] conv2d — f32 instantiated
static const char* WGSL_CONV2D = R"WGSL(
// Conv2D — 2D convolution with bias
// Y[n,co,oh,ow] = sum_{ci,kh,kw} X[n,ci,ih,iw] * W[co,ci,kh,kw] + bias[co]
//
// Supports padding, stride, dilation, and groups.
// Dispatch: (ceil(total_output/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];
    let dil_h = _params_[14];
    let dil_w = _params_[15];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                let ih = i32(oh * stride_h + kh * dil_h) - i32(pad_h);
                let iw = i32(ow * stride_w + kw * dil_w) - i32(pad_w);
                if (ih >= 0 && ih < i32(H_in) && iw >= 0 && iw < i32(W_in)) {
                    let x_val = t_read(&X, n * C_in * H_in * W_in + ci * H_in * W_in + u32(ih) * W_in + u32(iw));
                    let w_val = t_read(&W, co * ci_per_group * KH * KW + ci_local * KH * KW + kh * KW + kw);
                    acc += x_val * w_val;
                }
            }
        }
    }

    t_write(&Y, idx, acc + t_read(&Bias, co));
}
)WGSL";

// [conv] conv2d — dtype template
static const char* WGSL_CONV2D_T = R"WGSL(
// Conv2D — 2D convolution with bias
// Y[n,co,oh,ow] = sum_{ci,kh,kw} X[n,ci,ih,iw] * W[co,ci,kh,kw] + bias[co]
//
// Supports padding, stride, dilation, and groups.
// Dispatch: (ceil(total_output/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<${T}>;
@group(0) @binding(2) var<storage, read> Bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];
    let dil_h = _params_[14];
    let dil_w = _params_[15];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                let ih = i32(oh * stride_h + kh * dil_h) - i32(pad_h);
                let iw = i32(ow * stride_w + kw * dil_w) - i32(pad_w);
                if (ih >= 0 && ih < i32(H_in) && iw >= 0 && iw < i32(W_in)) {
                    let x_val = t_read(&X, n * C_in * H_in * W_in + ci * H_in * W_in + u32(ih) * W_in + u32(iw));
                    let w_val = t_read(&W, co * ci_per_group * KH * KW + ci_local * KH * KW + kh * KW + kw);
                    acc += x_val * w_val;
                }
            }
        }
    }

    t_write(&Y, idx, acc + t_read(&Bias, co));
}
)WGSL";

// [conv] conv2d_f16
static const char* WGSL_CONV2D_F16 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<f16>;
@group(0) @binding(1) var<storage, read> W: array<f16>;
@group(0) @binding(2) var<storage, read> Bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];
    let dil_h = _params_[14];
    let dil_w = _params_[15];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                let ih = i32(oh * stride_h + kh * dil_h) - i32(pad_h);
                let iw = i32(ow * stride_w + kw * dil_w) - i32(pad_w);
                if (ih >= 0 && ih < i32(H_in) && iw >= 0 && iw < i32(W_in)) {
                    let x_val = f32(X[n * C_in * H_in * W_in + ci * H_in * W_in + u32(ih) * W_in + u32(iw)]);
                    let w_val = f32(W[co * ci_per_group * KH * KW + ci_local * KH * KW + kh * KW + kw]);
                    acc += x_val * w_val;
                }
            }
        }
    }

    Y[idx] = acc + f32(Bias[co]);
}
)WGSL";

// [conv] conv_transpose2d — f32 instantiated
static const char* WGSL_CONV_TRANSPOSE2D = R"WGSL(
// ConvTranspose2D — 2D transposed convolution with bias
// Y[n,co,oh,ow] = sum_{ci,kh,kw} X[n,ci,ih,iw] * W[ci,co,kh,kw] + bias[co]
//
// Note: weight layout is [C_in, C_out, KH, KW] (transposed from Conv)
//
// Supports padding, stride, output_padding, and groups.
// Dispatch: (ceil(total_output/256), 1, 1)
//
// Params: [batch, C_in, H_in, W_in, C_out, KH, KW, pad_h,
//          pad_w, stride_h, stride_w, H_out, W_out, group, out_pad_h, out_pad_w]


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;
    let co_local = co % co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                // In transposed conv, output position (oh, ow) is related to
                // input position by: oh = ih * stride_h - pad_h + kh
                // So: ih = (oh + pad_h - kh) / stride_h
                let oh_off = i32(oh) + i32(pad_h) - i32(kh);
                let ow_off = i32(ow) + i32(pad_w) - i32(kw);
                if (oh_off >= 0 && ow_off >= 0 &&
                    oh_off % i32(stride_h) == 0 && ow_off % i32(stride_w) == 0) {
                    let ih = u32(oh_off) / stride_h;
                    let iw = u32(ow_off) / stride_w;
                    if (ih < H_in && iw < W_in) {
                        let x_val = t_read(&X, n * C_in * H_in * W_in + ci * H_in * W_in + ih * W_in + iw);
                        // Weight layout: [C_in, C_out_per_group, KH, KW]
                        let w_val = t_read(&W, ci * co_per_group * KH * KW + co_local * KH * KW + kh * KW + kw);
                        acc += x_val * w_val;
                    }
                }
            }
        }
    }

    t_write(&Y, idx, acc + t_read(&Bias, co));
}
)WGSL";

// [conv] conv_transpose2d — dtype template
static const char* WGSL_CONV_TRANSPOSE2D_T = R"WGSL(
// ConvTranspose2D — 2D transposed convolution with bias
// Y[n,co,oh,ow] = sum_{ci,kh,kw} X[n,ci,ih,iw] * W[ci,co,kh,kw] + bias[co]
//
// Note: weight layout is [C_in, C_out, KH, KW] (transposed from Conv)
//
// Supports padding, stride, output_padding, and groups.
// Dispatch: (ceil(total_output/256), 1, 1)
//
// Params: [batch, C_in, H_in, W_in, C_out, KH, KW, pad_h,
//          pad_w, stride_h, stride_w, H_out, W_out, group, out_pad_h, out_pad_w]

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<${T}>;
@group(0) @binding(2) var<storage, read> Bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;
    let co_local = co % co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                // In transposed conv, output position (oh, ow) is related to
                // input position by: oh = ih * stride_h - pad_h + kh
                // So: ih = (oh + pad_h - kh) / stride_h
                let oh_off = i32(oh) + i32(pad_h) - i32(kh);
                let ow_off = i32(ow) + i32(pad_w) - i32(kw);
                if (oh_off >= 0 && ow_off >= 0 &&
                    oh_off % i32(stride_h) == 0 && ow_off % i32(stride_w) == 0) {
                    let ih = u32(oh_off) / stride_h;
                    let iw = u32(ow_off) / stride_w;
                    if (ih < H_in && iw < W_in) {
                        let x_val = t_read(&X, n * C_in * H_in * W_in + ci * H_in * W_in + ih * W_in + iw);
                        // Weight layout: [C_in, C_out_per_group, KH, KW]
                        let w_val = t_read(&W, ci * co_per_group * KH * KW + co_local * KH * KW + kh * KW + kw);
                        acc += x_val * w_val;
                    }
                }
            }
        }
    }

    t_write(&Y, idx, acc + t_read(&Bias, co));
}
)WGSL";

// [conv] conv_transpose2d_f16
static const char* WGSL_CONV_TRANSPOSE2D_F16 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<f16>;
@group(0) @binding(1) var<storage, read> W: array<f16>;
@group(0) @binding(2) var<storage, read> Bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;
    let co_local = co % co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                let oh_off = i32(oh) + i32(pad_h) - i32(kh);
                let ow_off = i32(ow) + i32(pad_w) - i32(kw);
                if (oh_off >= 0 && ow_off >= 0 &&
                    oh_off % i32(stride_h) == 0 && ow_off % i32(stride_w) == 0) {
                    let ih = u32(oh_off) / stride_h;
                    let iw = u32(ow_off) / stride_w;
                    if (ih < H_in && iw < W_in) {
                        let x_val = f32(X[n * C_in * H_in * W_in + ci * H_in * W_in + ih * W_in + iw]);
                        let w_val = f32(W[ci * co_per_group * KH * KW + co_local * KH * KW + kh * KW + kw]);
                        acc += x_val * w_val;
                    }
                }
            }
        }
    }

    Y[idx] = acc + f32(Bias[co]);
}
)WGSL";

// [conv] resize_nearest — f32 instantiated
static const char* WGSL_RESIZE_NEAREST = R"WGSL(
// Nearest-neighbor resize (upsampling)
// Dispatch: (ceil(N*C*H_out*W_out/512), 1, 1) — each thread handles 2 elements


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_resize(idx: u32, N: u32, C: u32, H_in: u32, W_in: u32, H_out: u32, W_out: u32) -> f32 {
    let n = idx / (C * H_out * W_out);
    let c = (idx / (H_out * W_out)) % C;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;
    let ih = min(oh * H_in / H_out, H_in - 1u);
    let iw = min(ow * W_in / W_out, W_in - 1u);
    return t_read(&X, n * C * H_in * W_in + c * H_in * W_in + ih * W_in + iw);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let C = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let H_out = _params_[4];
    let W_out = _params_[5];

    let total = N * C * H_out * W_out;
    let base = gid.x * 2u;
    if (base >= total) { return; }

    let v0 = compute_resize(base, N, C, H_in, W_in, H_out, W_out);
    var v1: f32 = 0.0;
    if (base + 1u < total) {
        v1 = compute_resize(base + 1u, N, C, H_in, W_in, H_out, W_out);
    }
    t_write2(&Y, base, v0, v1);
}
)WGSL";

// [conv] resize_nearest — dtype template
static const char* WGSL_RESIZE_NEAREST_T = R"WGSL(
// Nearest-neighbor resize (upsampling)
// Dispatch: (ceil(N*C*H_out*W_out/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_resize(idx: u32, N: u32, C: u32, H_in: u32, W_in: u32, H_out: u32, W_out: u32) -> f32 {
    let n = idx / (C * H_out * W_out);
    let c = (idx / (H_out * W_out)) % C;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;
    let ih = min(oh * H_in / H_out, H_in - 1u);
    let iw = min(ow * W_in / W_out, W_in - 1u);
    return t_read(&X, n * C * H_in * W_in + c * H_in * W_in + ih * W_in + iw);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let C = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let H_out = _params_[4];
    let W_out = _params_[5];

    let total = N * C * H_out * W_out;
    let base = gid.x * 2u;
    if (base >= total) { return; }

    let v0 = compute_resize(base, N, C, H_in, W_in, H_out, W_out);
    var v1: f32 = 0.0;
    if (base + 1u < total) {
        v1 = compute_resize(base + 1u, N, C, H_in, W_in, H_out, W_out);
    }
    t_write2(&Y, base, v0, v1);
}
)WGSL";

// ─── [elementwise] ─────────────────────────────────────────────────

// [elementwise] add_inplace_batched
static const char* WGSL_ADD_INPLACE_BATCHED = R"WGSL(
@group(0) @binding(0) var<storage,read_write> X:array<f32>;
@group(0) @binding(1) var<storage,read> A:array<f32>;
@group(0) @binding(2) var<storage,read> P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let n=P[0];let i=gid.x;if(i<n){X[i]+=A[i];}}
)WGSL";

// [elementwise] binary_elementwise — f32 instantiated
static const char* WGSL_BINARY_ELEMENTWISE = R"WGSL(
// Binary elementwise ops: Add(0), Sub(1), Mul(2), Div(3)
// With broadcasting support.
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

fn compute_op(a: f32, b: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        default: { return a + b; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let A_N = _params_[2];
    let B_N = _params_[3];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let i0 = base;
    let a0_idx = select(i0, i0 % A_N, A_N < N && A_N > 0u);
    let b0_idx = select(i0, i0 % B_N, B_N < N && B_N > 0u);
    let r0 = compute_op(t_read(&A, a0_idx), t_read(&B, b0_idx), op);

    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i1 = base + 1u;
        let a1_idx = select(i1, i1 % A_N, A_N < N && A_N > 0u);
        let b1_idx = select(i1, i1 % B_N, B_N < N && B_N > 0u);
        r1 = compute_op(t_read(&A, a1_idx), t_read(&B, b1_idx), op);
    }

    t_write2(&C, base, r0, r1);
}
)WGSL";

// [elementwise] binary_elementwise — dtype template
static const char* WGSL_BINARY_ELEMENTWISE_BROADCAST_T = R"WGSL(
// Binary elementwise ops: Add(0), Sub(1), Mul(2), Div(3)
// With broadcasting support.
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> C: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

fn compute_op(a: f32, b: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        default: { return a + b; }
    }
}

fn broadcast_index(flat: u32, dims_offset: u32) -> u32 {
    let rank = _params_[0];
    if (rank == 0u) { return flat % _params_[dims_offset]; }
    var input_idx = 0u;
    var input_stride = 1u;
    var output_stride = 1u;
    var axis = rank;
    loop {
        if (axis == 0u) { break; }
        axis -= 1u;
        let output_dim = _params_[12u + axis];
        let coord = (flat / output_stride) % output_dim;
        let input_dim = _params_[dims_offset + axis];
        input_idx += select(coord * input_stride, 0u, input_dim == 1u);
        input_stride *= input_dim;
        output_stride *= output_dim;
    }
    return input_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let op = _params_[2];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let i0 = base;
    let a0_idx = broadcast_index(i0, 4u);
    let b0_idx = broadcast_index(i0, 8u);
    let r0 = compute_op(t_read(&A, a0_idx), t_read(&B, b0_idx), op);

    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i1 = base + 1u;
        let a1_idx = broadcast_index(i1, 4u);
        let b1_idx = broadcast_index(i1, 8u);
        r1 = compute_op(t_read(&A, a1_idx), t_read(&B, b1_idx), op);
    }

    t_write2(&C, base, r0, r1);
}
)WGSL";

// Fast path for equal-shaped tensors and scalar broadcasting.
static const char* WGSL_BINARY_ELEMENTWISE_T = R"WGSL(
${T_READ}
${T_WRITE2}
@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> C: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;
fn compute_op(a: f32, b: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return a + b; } case 1u: { return a - b; }
        case 2u: { return a * b; } case 3u: { return a / b; }
        default: { return a + b; }
    }
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0]; let op = _params_[1];
    let A_N = _params_[2]; let B_N = _params_[3];
    let base = gid.x * 2u;
    if (base >= N) { return; }
    let a0 = select(base, base % A_N, A_N < N);
    let b0 = select(base, base % B_N, B_N < N);
    let r0 = compute_op(t_read(&A, a0), t_read(&B, b0), op);
    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i = base + 1u;
        let a1 = select(i, i % A_N, A_N < N);
        let b1 = select(i, i % B_N, B_N < N);
        r1 = compute_op(t_read(&A, a1), t_read(&B, b1), op);
    }
    t_write2(&C, base, r0, r1);
}
)WGSL";

// [elementwise] binary_elementwise_f16
static const char* WGSL_BINARY_ELEMENTWISE_F16 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f16>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let N_A = _params_[2];
    let N_B = _params_[3];
    let idx = gid.x;
    if (idx >= N) { return; }
    let a_idx = select(idx, idx % N_A, N_A < N && N_A > 0u);
    let b_idx = select(idx, idx % N_B, N_B < N && N_B > 0u);
    let a = f32(A[a_idx]);
    let b = f32(B[b_idx]);
    var result: f32;
    switch (op) {
        case 0u: { result = a + b; }
        case 1u: { result = a - b; }
        case 2u: { result = a * b; }
        case 3u: { result = a / b; }
        default: { result = a; }
    }
    C[idx] = f16(result);
}
)WGSL";

// [elementwise] cast_f16_to_f32
static const char* WGSL_CAST_F16_TO_F32 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    B[idx] = f32(A[idx]);
}
)WGSL";

// [elementwise] cast_f32_to_f16
static const char* WGSL_CAST_F32_TO_F16 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f16>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    B[idx] = f16(A[idx]);
}
)WGSL";

// [elementwise] equal_op — f32 instantiated
static const char* WGSL_EQUAL_OP = R"WGSL(
// Equal — compare two tensors, output packed bool (u32)
// Each thread at idx % 4 == 0 packs 4 consecutive bool results into one u32.
// Dispatch: (ceil(N/256), 1, 1)
// Params: [N, N_B]


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let N_B = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let b_idx = select(idx, idx % N_B, N_B < N && N_B > 0u);
    let eq = select(0u, 1u, t_read(&A, idx) == t_read(&B, b_idx));
    // Pack bool into bytes within u32
    let u32_idx = idx / 4u;
    let byte_pos = (idx % 4u) * 8u;
    // Note: atomicOr would be ideal but not available. Write full u32.
    // This is a simplification that works when N is a multiple of 4.
    if (idx % 4u == 0u) {
        var packed: u32 = 0u;
        for (var i = 0u; i < 4u && (idx + i) < N; i++) {
            let bi = select(idx + i, (idx + i) % N_B, N_B < N && N_B > 0u);
            let e = select(0u, 1u, t_read(&A, idx + i) == t_read(&B, bi));
            packed |= e << (i * 8u);
        }
        Out[u32_idx] = packed;
    }
}
)WGSL";

// [elementwise] equal_op — dtype template
static const char* WGSL_EQUAL_OP_T = R"WGSL(
// Equal — compare two tensors, output packed bool (u32)
// Each thread at idx % 4 == 0 packs 4 consecutive bool results into one u32.
// Dispatch: (ceil(N/256), 1, 1)
// Params: [N, N_B]

${T_READ}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let N_B = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let b_idx = select(idx, idx % N_B, N_B < N && N_B > 0u);
    let eq = select(0u, 1u, t_read(&A, idx) == t_read(&B, b_idx));
    // Pack bool into bytes within u32
    let u32_idx = idx / 4u;
    let byte_pos = (idx % 4u) * 8u;
    // Note: atomicOr would be ideal but not available. Write full u32.
    // This is a simplification that works when N is a multiple of 4.
    if (idx % 4u == 0u) {
        var packed: u32 = 0u;
        for (var i = 0u; i < 4u && (idx + i) < N; i++) {
            let bi = select(idx + i, (idx + i) % N_B, N_B < N && N_B > 0u);
            let e = select(0u, 1u, t_read(&A, idx + i) == t_read(&B, bi));
            packed |= e << (i * 8u);
        }
        Out[u32_idx] = packed;
    }
}
)WGSL";

// [elementwise] gated_output_batched
static const char* WGSL_GATED_OUTPUT_BATCHED = R"WGSL(
@group(0) @binding(0) var<storage,read>A:array<f32>;
@group(0) @binding(1) var<storage,read>G:array<f32>;
@group(0) @binding(2) var<storage,read_write>O:array<f32>;
@group(0) @binding(3) var<storage,read>P:array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id)gid:vec3<u32>){let i=gid.x;if(i<P[0]){O[i]=A[i]/(1.0+exp(-G[i]));}}
)WGSL";

// [elementwise] scale — f32 instantiated
static const char* WGSL_SCALE = R"WGSL(
// Scale — in-place element-wise multiply by scalar
// data[i] *= scale
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements
// Params: [N, bitcast<u32>(scale)]


fn t_read_rw(buf: ptr<storage, array<f32>, read_write>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params[0];
    let scale = bitcast<f32>(params[1]);

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let v0 = t_read_rw(&data, base) * scale;
    var v1: f32 = 0.0;
    if (base + 1u < N) {
        v1 = t_read_rw(&data, base + 1u) * scale;
    }
    t_write2(&data, base, v0, v1);
}
)WGSL";

// [elementwise] scale — dtype template
static const char* WGSL_SCALE_T = R"WGSL(
// Scale — in-place element-wise multiply by scalar
// data[i] *= scale
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements
// Params: [N, bitcast<u32>(scale)]

${T_READ_RW}
${T_WRITE2}

@group(0) @binding(0) var<storage, read_write> data: array<${T}>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params[0];
    let scale = bitcast<f32>(params[1]);

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let v0 = t_read_rw(&data, base) * scale;
    var v1: f32 = 0.0;
    if (base + 1u < N) {
        v1 = t_read_rw(&data, base + 1u) * scale;
    }
    t_write2(&data, base, v0, v1);
}
)WGSL";

// [elementwise] scale_by_buffer
static const char* WGSL_SCALE_BY_BUFFER = R"WGSL(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x;if(i<P[0]){X[i]*=Scale[0];}
}
)WGSL";

// [elementwise] unary_elementwise — f32 instantiated
static const char* WGSL_UNARY_ELEMENTWISE = R"WGSL(
// Unary elementwise ops:
//   Sigmoid(0), Tanh(1), Neg(2), Sqrt(3), Sin(4), Cos(5), Identity(6),
//   Gelu(7), Silu(8), Erf(9), Relu(10), Exp(11), Log(12), Abs(13),
//   Floor(14), Ceil(15), Round(16), Softplus(18)
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_unary(x: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return 1.0 / (1.0 + exp(-x)); }                               // Sigmoid
        case 1u: { return tanh(x); }                                              // Tanh
        case 2u: { return -x; }                                                   // Neg
        case 3u: { return sqrt(max(x, 0.0)); }                                   // Sqrt
        case 4u: { return sin(x); }                                               // Sin
        case 5u: { return cos(x); }                                               // Cos
        case 6u: { return x; }                                                    // Identity
        case 7u: { return x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); } // GELU
        case 8u: { return x / (1.0 + exp(-x)); }                                 // SiLU (Swish)
        case 10u: { return max(x, 0.0); }                                        // ReLU
        case 11u: { return exp(x); }                                              // Exp
        case 12u: { return log(max(x, 1e-10)); }                                 // Log
        case 13u: { return abs(x); }                                              // Abs
        case 14u: { return floor(x); }                                            // Floor
        case 15u: { return ceil(x); }                                             // Ceil
        case 16u: { return round(x); }                                            // Round
        case 18u: { if (x > 20.0) { return x; } return log(exp(x) + 1.0); }     // Softplus
        default: { return x; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let r0 = compute_unary(t_read(&A, base), op);
    var r1: f32 = 0.0;
    if (base + 1u < N) {
        r1 = compute_unary(t_read(&A, base + 1u), op);
    }
    t_write2(&C, base, r0, r1);
}
)WGSL";

// [elementwise] unary_elementwise — dtype template
static const char* WGSL_UNARY_ELEMENTWISE_T = R"WGSL(
// Unary elementwise ops:
//   Sigmoid(0), Tanh(1), Neg(2), Sqrt(3), Sin(4), Cos(5), Identity(6),
//   Gelu(7), Silu(8), Erf(9), Relu(10), Exp(11), Log(12), Abs(13),
//   Floor(14), Ceil(15), Round(16), Softplus(18)
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read_write> C: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_unary(x: f32, op: u32) -> f32 {
    switch (op) {
        case 0u: { return 1.0 / (1.0 + exp(-x)); }                               // Sigmoid
        case 1u: { return tanh(x); }                                              // Tanh
        case 2u: { return -x; }                                                   // Neg
        case 3u: { return sqrt(max(x, 0.0)); }                                   // Sqrt
        case 4u: { return sin(x); }                                               // Sin
        case 5u: { return cos(x); }                                               // Cos
        case 6u: { return x; }                                                    // Identity
        case 7u: { return x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); } // GELU
        case 8u: { return x / (1.0 + exp(-x)); }                                 // SiLU (Swish)
        case 10u: { return max(x, 0.0); }                                        // ReLU
        case 11u: { return exp(x); }                                              // Exp
        case 12u: { return log(max(x, 1e-10)); }                                 // Log
        case 13u: { return abs(x); }                                              // Abs
        case 14u: { return floor(x); }                                            // Floor
        case 15u: { return ceil(x); }                                             // Ceil
        case 16u: { return round(x); }                                            // Round
        case 18u: { if (x > 20.0) { return x; } return log(exp(x) + 1.0); }     // Softplus
        default: { return x; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let r0 = compute_unary(t_read(&A, base), op);
    var r1: f32 = 0.0;
    if (base + 1u < N) {
        r1 = compute_unary(t_read(&A, base + 1u), op);
    }
    t_write2(&C, base, r0, r1);
}
)WGSL";

// [elementwise] unary_elementwise_f16
static const char* WGSL_UNARY_ELEMENTWISE_F16 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read_write> C: array<f16>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let e = 1.0 - p * exp(-a * a);
    return select(-e, e, x >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let op = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let a = f32(A[idx]);
    var result: f32;
    switch (op) {
        case 0u: { result = 1.0 / (1.0 + exp(-a)); }
        case 1u: { result = tanh(a); }
        case 2u: { result = -a; }
        case 3u: { result = sqrt(a); }
        case 4u: { result = sin(a); }
        case 5u: { result = cos(a); }
        case 6u: { result = a; }
        case 7u: { result = 0.5 * a * (1.0 + erf_approx(a * 0.7071067811865476)); }
        case 8u: { result = a / (1.0 + exp(-a)); }
        case 9u: { result = erf_approx(a); }
        case 10u: { result = max(a, 0.0); }
        case 11u: { result = exp(a); }
        case 12u: { result = log(a); }
        case 13u: { result = abs(a); }
        case 14u: { result = floor(a); }
        case 15u: { result = ceil(a); }
        case 16u: { result = round(a); }
        case 17u: { result = log(1.0 + exp(a)); }
        default: { result = a; }
    }
    C[idx] = f16(result);
}
)WGSL";

// [elementwise] where_select — f32 instantiated
static const char* WGSL_WHERE_SELECT = R"WGSL(
// Where — conditional select: out = cond ? X : Y
// Cond is packed as bytes in u32 (bool array).
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> Cond: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

fn read_cond(idx: u32) -> bool {
    let byte_idx = idx / 4u;
    let bit_pos = (idx % 4u) * 8u;
    return ((Cond[byte_idx] >> bit_pos) & 0xFFu) != 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let N_cond = _params_[1];
    let N_x = _params_[2];
    let N_y = _params_[3];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let c0_idx = select(base, base % N_cond, N_cond < N && N_cond > 0u);
    let x0_idx = select(base, base % N_x, N_x < N && N_x > 0u);
    let y0_idx = select(base, base % N_y, N_y < N && N_y > 0u);
    let r0 = select(t_read(&Y, y0_idx), t_read(&X, x0_idx), read_cond(c0_idx));

    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i1 = base + 1u;
        let c1_idx = select(i1, i1 % N_cond, N_cond < N && N_cond > 0u);
        let x1_idx = select(i1, i1 % N_x, N_x < N && N_x > 0u);
        let y1_idx = select(i1, i1 % N_y, N_y < N && N_y > 0u);
        r1 = select(t_read(&Y, y1_idx), t_read(&X, x1_idx), read_cond(c1_idx));
    }

    t_write2(&Out, base, r0, r1);
}
)WGSL";

// [elementwise] where_select — dtype template
static const char* WGSL_WHERE_SELECT_T = R"WGSL(
// Where — conditional select: out = cond ? X : Y
// Cond is packed as bytes in u32 (bool array).
// Dispatch: (ceil(N/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> Cond: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<${T}>;
@group(0) @binding(2) var<storage, read> Y: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Out: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

fn read_cond(idx: u32) -> bool {
    let byte_idx = idx / 4u;
    let bit_pos = (idx % 4u) * 8u;
    return ((Cond[byte_idx] >> bit_pos) & 0xFFu) != 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let N_cond = _params_[1];
    let N_x = _params_[2];
    let N_y = _params_[3];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    let c0_idx = select(base, base % N_cond, N_cond < N && N_cond > 0u);
    let x0_idx = select(base, base % N_x, N_x < N && N_x > 0u);
    let y0_idx = select(base, base % N_y, N_y < N && N_y > 0u);
    let r0 = select(t_read(&Y, y0_idx), t_read(&X, x0_idx), read_cond(c0_idx));

    var r1: f32 = 0.0;
    if (base + 1u < N) {
        let i1 = base + 1u;
        let c1_idx = select(i1, i1 % N_cond, N_cond < N && N_cond > 0u);
        let x1_idx = select(i1, i1 % N_x, N_x < N && N_x > 0u);
        let y1_idx = select(i1, i1 % N_y, N_y < N && N_y > 0u);
        r1 = select(t_read(&Y, y1_idx), t_read(&X, x1_idx), read_cond(c1_idx));
    }

    t_write2(&Out, base, r0, r1);
}
)WGSL";

// ─── [fused] ───────────────────────────────────────────────────────

// [fused] add_scaled — f32 instantiated
static const char* WGSL_ADD_SCALED = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let alpha = bitcast<f32>(_params_[1]);
    let a = t_read(&acc, idx);
    let v = t_read(&x, idx);
    t_write(&acc, idx, a + alpha * v);
}
)WGSL";

// [fused] add_scaled — dtype template
static const char* WGSL_ADD_SCALED_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read_write> acc: array<${T}>;
@group(0) @binding(1) var<storage, read> x: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let alpha = bitcast<f32>(_params_[1]);
    let a = t_read(&acc, idx);
    let v = t_read(&x, idx);
    t_write(&acc, idx, a + alpha * v);
}
)WGSL";

// [fused] concat_2d — f32 instantiated
static const char* WGSL_CONCAT_2D = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_total = _params_[1];
    let idx = gid.x;
    if (idx >= N_total) { return; }
    let N_a = _params_[0];
    if (idx < N_a) {
        t_write(&output, idx, t_read(&a, idx));
    } else {
        t_write(&output, idx, t_read(&b, idx - N_a));
    }
}
)WGSL";

// [fused] concat_2d — dtype template
static const char* WGSL_CONCAT_2D_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> a: array<${T}>;
@group(0) @binding(1) var<storage, read> b: array<${T}>;
@group(0) @binding(2) var<storage, read_write> output: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N_total = _params_[1];
    let idx = gid.x;
    if (idx >= N_total) { return; }
    let N_a = _params_[0];
    if (idx < N_a) {
        t_write(&output, idx, t_read(&a, idx));
    } else {
        t_write(&output, idx, t_read(&b, idx - N_a));
    }
}
)WGSL";

// [fused] gate_residual_add — f32 instantiated
static const char* WGSL_GATE_RESIDUAL_ADD = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let D = _params_[0];
    let mod_idx = idx % D;
    let r = t_read(&residual, idx);
    let g = t_read(&gate, mod_idx);
    let v = t_read(&x, idx);
    t_write(&residual, idx, r + g * v);
}
)WGSL";

// [fused] gate_residual_add — dtype template
static const char* WGSL_GATE_RESIDUAL_ADD_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read_write> residual: array<${T}>;
@group(0) @binding(1) var<storage, read> gate: array<${T}>;
@group(0) @binding(2) var<storage, read> x: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let D = _params_[0];
    let mod_idx = idx % D;
    let r = t_read(&residual, idx);
    let g = t_read(&gate, mod_idx);
    let v = t_read(&x, idx);
    t_write(&residual, idx, r + g * v);
}
)WGSL";

// [fused] gelu_mul — f32 instantiated
static const char* WGSL_GELU_MUL = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let g = t_read(&gate, idx);
    let u = t_read(&up, idx);
    let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
    let tanh_val = 1.0 - 2.0 / (exp(2.0 * inner) + 1.0);
    let gelu = 0.5 * g * (1.0 + tanh_val);
    t_write(&output, idx, gelu * u);
}
)WGSL";

// [fused] gelu_mul — dtype template
static const char* WGSL_GELU_MUL_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> gate: array<${T}>;
@group(0) @binding(1) var<storage, read> up: array<${T}>;
@group(0) @binding(2) var<storage, read_write> output: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let g = t_read(&gate, idx);
    let u = t_read(&up, idx);
    let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
    let tanh_val = 1.0 - 2.0 / (exp(2.0 * inner) + 1.0);
    let gelu = 0.5 * g * (1.0 + tanh_val);
    t_write(&output, idx, gelu * u);
}
)WGSL";

// [fused] gelu_mul_batched
static const char* WGSL_GELU_MUL_BATCHED = R"WGSL(
@group(0) @binding(0) var<storage, read> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M=P[0]; let N=P[1]; let flat=gid.x;
    if (flat>=M*N) { return; }
    let row=flat/N; let i=flat%N; let base=row*2u*N;
    let g=GateUp[base+i]; let u=GateUp[base+N+i];
    let gelu=0.5*g*(1.0+tanh(0.7978845608*(g+0.044715*g*g*g)));
    Out[flat]=gelu*u;
}
)WGSL";

// [fused] gelu_mul_fused
static const char* WGSL_GELU_MUL_FUSED = R"WGSL(
// GELU-mul activation for fused GateUp buffer.
// Reads gate[i] from buf[i], up[i] from buf[N + i].
// Computes: out[i] = GELU(gate[i]) * up[i]
// Uses GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Dispatch: (ceil(N/256), 1, 1) where N = intermediate_size
//
// Bindings:
//   0: GateUp (read) — 2*N floats (gate || up concatenated)
//   1: Out (write) — N floats
//   2: _params_ — [N, 0, 0, 0]

@group(0) @binding(0) var<storage, read> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i >= N) { return; }
    let g = GateUp[i];
    let u = GateUp[N + i];
    let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
    // GELU is already saturated outside this interval. Clamping avoids an
    // Intel D3D12 tanh edge case that returns NaN for large cubic arguments.
    let tanh_val = tanh(clamp(inner, -10.0, 10.0));
    let gelu = 0.5 * g * (1.0 + tanh_val);
    Out[i] = gelu * u;
}
)WGSL";

// [fused] gptoss_gate — f32 instantiated
static const char* WGSL_GPTOSS_GATE = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let gate = t_read(&gate_up, idx * 2u);
    let up = t_read(&gate_up, idx * 2u + 1u);
    let x = gate * 1.702;
    t_write(&output, idx, (up + 1.0) * gate * (1.0 / (1.0 + exp(-x))));
}
)WGSL";

// [fused] gptoss_gate — dtype template
static const char* WGSL_GPTOSS_GATE_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> gate_up: array<${T}>;
@group(0) @binding(1) var<storage, read_write> output: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let gate = t_read(&gate_up, idx * 2u);
    let up = t_read(&gate_up, idx * 2u + 1u);
    let x = gate * 1.702;
    t_write(&output, idx, (up + 1.0) * gate * (1.0 / (1.0 + exp(-x))));
}
)WGSL";

// [fused] gptoss_gate_batched
static const char* WGSL_GPTOSS_GATE_BATCHED = R"WGSL(
// Batched GPT-OSS gated activation: T tokens via workgroup_id.y.
// y = (up + 1.0) * gate * sigmoid(gate * 1.702)
// Input: gate_up[T * 2*N], output[T * N], interleaved gate/up per token.
// Params: [0]=N
// Dispatch: (ceil(N/256), T, 1)

@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let tok = wid.y;
    let in_base = tok * N * 2u;
    let out_base = tok * N;
    let gate = gate_up[in_base + idx * 2u];
    let up = gate_up[in_base + idx * 2u + 1u];
    let x = gate * 1.702;
    output[out_base + idx] = (up + 1.0) * gate * (1.0 / (1.0 + exp(-x)));
}
)WGSL";

// [fused] mod_scale_shift — f32 instantiated
static const char* WGSL_MOD_SCALE_SHIFT = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> shift: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let D = _params_[0];
    let mod_idx = idx % D;
    let xv = t_read(&x, idx);
    let s = t_read(&scale, mod_idx);
    let sh = t_read(&shift, mod_idx);
    t_write(&output, idx, (1.0 + s) * xv + sh);
}
)WGSL";

// [fused] mod_scale_shift — dtype template
static const char* WGSL_MOD_SCALE_SHIFT_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> x: array<${T}>;
@group(0) @binding(1) var<storage, read> scale: array<${T}>;
@group(0) @binding(2) var<storage, read> shift: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let D = _params_[0];
    let mod_idx = idx % D;
    let xv = t_read(&x, idx);
    let s = t_read(&scale, mod_idx);
    let sh = t_read(&shift, mod_idx);
    t_write(&output, idx, (1.0 + s) * xv + sh);
}
)WGSL";

// [fused] ple_combine
static const char* WGSL_PLE_COMBINE = R"WGSL(
// PLE: In-place combine — adds per-layer token embedding and scales.
// Computes: Y[i] = (Y[i] + B[i]) * scale
// Used after ple_slice_rms_norm to merge normalized projection (already in Y)
// with per-layer token embeddings (B), with scaling by pleInputScale.
//
// Dispatch: (ceil(count/256), 1, 1)
//
// Bindings:
//   0: Y (read_write) — [count] fp32, normalized projection (modified in-place)
//   1: B (read) — [count] fp32, per-layer token embedding (CPU-uploaded)
//   2: _params_ — [count, scale_bits, 0, 0]

@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = _params_[0];
    let scale = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= count) { return; }
    Y[i] = (Y[i] + B[i]) * scale;
}
)WGSL";

// [fused] ple_combine_batched
static const char* WGSL_PLE_COMBINE_BATCHED = R"WGSL(
enable subgroups;

// Normalize each projected PLE slice and combine it with its token signal.
// Grid: (M, n_layers, 1), WG=256.

@group(0) @binding(0) var<storage, read> Proj: array<f32>;
@group(0) @binding(1) var<storage, read> Norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> Signal: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;
var<workgroup> sums: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = P[0]; let D = P[1]; let L = P[2];
    let eps = bitcast<f32>(P[3]);
    let row = wid.x; let layer = wid.y; let tid = lid.x;
    if (row >= M || layer >= L) { return; }
    let base = (row * L + layer) * D;
    var ss = 0.0;
    for (var i = tid; i < D; i += 256u) { let v=Proj[base+i]; ss += v*v; }
    let lane=tid&31u; let warp=tid/32u; let ws=subgroupAdd(ss);
    if (lane==0u) { sums[warp]=ws; }
    workgroupBarrier();
    var total=0.0; for (var w=0u; w<8u; w++) { total += sums[w]; }
    let r=inverseSqrt(total/f32(D)+eps);
    for (var i=tid; i<D; i+=256u) {
        let p=base+i;
        Signal[p]=(Proj[p]*r*Norm[i]+Signal[p])*0.7071067811865476;
    }
}
)WGSL";

// [fused] ple_gelu_mul
static const char* WGSL_PLE_GELU_MUL = R"WGSL(
// PLE: GELU activation + elementwise multiply with per-layer input.
// Computes: Gate[i] = GELU(Gate[i]) * PleInput[offset + i]
// Used after the gate projection (E → ple_dim) in Per-Layer Embeddings.
// The offset allows indexing into a concatenated per-layer input buffer.
//
// Dispatch: (ceil(N/256), 1, 1) where N = ple_dim
//
// Bindings:
//   0: Gate (read_write) — [N] fp32, gate projection output (modified in-place)
//   1: PleInput (read) — [nLayer*N] fp32, concatenated per-layer inputs
//   2: _params_ — [N, offset, 0, 0]

@group(0) @binding(0) var<storage, read_write> Gate: array<f32>;
@group(0) @binding(1) var<storage, read> PleInput: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let offset = _params_[1];
    let i = gid.x;
    if (i >= N) { return; }
    // The tanh approximation's cubic term and the subsequent product can
    // overflow on some native WebGPU drivers even when the normalized result
    // is representable. GELU is already saturated outside this interval.
    let g = clamp(Gate[i], -20.0, 20.0);
    let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
    let gelu = select(select(0.0, g, g > 10.0),
                      0.5 * g * (1.0 + tanh(inner)), abs(g) <= 10.0);
    let ple = clamp(PleInput[offset + i], -65504.0, 65504.0);
    Gate[i] = gelu * ple;
}
)WGSL";

// [fused] ple_gelu_mul_batched
static const char* WGSL_PLE_GELU_MUL_BATCHED = R"WGSL(
@group(0) @binding(0) var<storage, read_write> Gate: array<f32>;
@group(0) @binding(1) var<storage, read> PleInput: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M=P[0]; let D=P[1]; let layer=P[2]; let L=P[3];
    let flat=gid.x;
    if (flat>=M*D) { return; }
    let row=flat/D; let i=flat%D; let g=Gate[flat];
    let gelu=0.5*g*(1.0+tanh(0.7978845608*(g+0.044715*g*g*g)));
    Gate[flat]=gelu*PleInput[(row*L+layer)*D+i];
}
)WGSL";

// [fused] ple_slice_rms_norm
static const char* WGSL_PLE_SLICE_RMS_NORM = R"WGSL(
// PLE: Per-slice RMSNorm — normalizes N slices of D elements each.
// Used to normalize the model projection output before combining with
// per-layer token embeddings.
//
// Each workgroup handles one slice of D elements.
// Dispatch: (N, 1, 1) where N = number of slices (nLayer)
//
// Bindings:
//   0: X (read) — [N*D] fp32, input (model projection output)
//   1: Y (read_write) — [N*D] fp32, output (normalized)
//   2: W (read) — [D] fp32, learned norm weights (shared across slices)
//   3: Rstd (read_write) — [N] fp32, reciprocal std per slice (unused but needed for bind group compat)
//   4: _params_ — {stride=D, N=D, eps_bits, 0}

enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> wg_sum: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let D = _params_[1];  // slice dimension
    let eps = bitcast<f32>(_params_[2]);
    let slice = wid.x;
    let base = slice * D;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Pass 1: sum of squares (strided for coalesced access)
    var ss: f32 = 0.0;
    var i = tid;
    for (; i < D; i = i + 256u) {
        let v = X[base + i];
        ss = ss + v * v;
    }

    // Warp reduce
    ss = ss + subgroupShuffleXor(ss, 16u);
    ss = ss + subgroupShuffleXor(ss, 8u);
    ss = ss + subgroupShuffleXor(ss, 4u);
    ss = ss + subgroupShuffleXor(ss, 2u);
    ss = ss + subgroupShuffleXor(ss, 1u);

    if (lane == 0u) { wg_sum[warp_id] = ss; }
    workgroupBarrier();

    // Reduce across 8 warps
    if (tid == 0u) {
        var total: f32 = 0.0;
        for (var w = 0u; w < 8u; w = w + 1u) { total = total + wg_sum[w]; }
        let rms = 1.0 / sqrt(total / f32(D) + eps);
        wg_sum[0] = rms;
    }
    workgroupBarrier();
    let rms = wg_sum[0];

    // Pass 2: normalize and apply weights
    i = tid;
    for (; i < D; i = i + 256u) {
        Y[base + i] = X[base + i] * rms * W[i];
    }
}
)WGSL";

// [fused] sigmoid_gate_interleaved — f32 instantiated
static const char* WGSL_SIGMOID_GATE_INTERLEAVED = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let HD = _params_[1];
    let h = idx / HD;
    let d = idx % HD;
    let gate_idx = h * HD * 2u + d;
    let val_idx = gate_idx + HD;
    let g = t_read(&input, gate_idx);
    let v = t_read(&input, val_idx);
    t_write(&output, idx, (1.0 / (1.0 + exp(-g))) * v);
}
)WGSL";

// [fused] sigmoid_gate_interleaved — dtype template
static const char* WGSL_SIGMOID_GATE_INTERLEAVED_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read_write> output: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let HD = _params_[1];
    let h = idx / HD;
    let d = idx % HD;
    let gate_idx = h * HD * 2u + d;
    let val_idx = gate_idx + HD;
    let g = t_read(&input, gate_idx);
    let v = t_read(&input, val_idx);
    t_write(&output, idx, (1.0 / (1.0 + exp(-g))) * v);
}
)WGSL";

// [fused] silu_mul_batched
static const char* WGSL_SILU_MUL_BATCHED = R"WGSL(
@group(0) @binding(0) var<storage,read> GU:array<f32>;
@group(0) @binding(1) var<storage,read_write> O:array<f32>;
@group(0) @binding(2) var<storage,read> P:array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id)gid:vec3<u32>){let M=P[0];let N=P[1];let i=gid.x;if(i>=M*N){return;}let r=i/N;let d=i%N;let b=r*2u*N;let g=GU[b+d];O[i]=g/(1.0+exp(-g))*GU[b+N+d];}
)WGSL";

// [fused] silu_mul_fused
static const char* WGSL_SILU_MUL_FUSED = R"WGSL(
// Generated WGSL kernel
// Kernel: silu_mul_fused_kernel
// Workgroup size: 128 (4 warps x 32 threads)

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // GateUp
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Out

struct Params {
    N: i32,
};
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v6: i32 = i32(_wg_id.x);
    let v7: i32 = v6 * 128;
    let v8: i32 = i32(_lid.x);
    let v9: i32 = v8 & 127;
    let v10: i32 = i32(u32(v9) % u32(32));
    let v11: i32 = i32(_lid.x);
    let v12: i32 = i32(u32(v11) / u32(32));
    let v13: i32 = v10 << u32(0);
    let v14: i32 = 0 | v13;
    let v15: i32 = v12 << u32(5);
    let v16: i32 = v14 | v15;
    let v17: i32 = v16 & 127;
    let v18: i32 = i32(u32(v17) >> u32(0));
    let v19: i32 = v18 | 0;
    let v20: i32 = 0 ^ v19;
    let v21: i32 = v20 ^ 0;
    let v22: i32 = v21 + 0;
    let v23: i32 = v7 + v22;
    let v24: bool = v23 < params.N;
    let v26: f32 = buf0[u32(v23)];
    let v27: f32 = select(f32(0.0), v26, v24);
    let v30: f32 = buf0[u32((params.N + v23))];
    let v31: f32 = select(f32(0.0), v30, v24);
    let v32: f32 = f32(0.0) - v27;
    let v33: f32 = exp(v32);
    let v34: f32 = v33 + f32(1.0);
    let v35: f32 = v27 / v34;
    let v37: f32 = v35 * v31;
    let v38: f32 = select(f32(0), v37, v24);
    if v24 { buf1[u32(v23)] = v38; }
}
)WGSL";

// [fused] split_copy — f32 instantiated
static const char* WGSL_SPLIT_COPY = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let src_offset = _params_[0];
    t_write(&dst, idx, t_read(&src, src_offset + idx));
}
)WGSL";

// [fused] split_copy — dtype template
static const char* WGSL_SPLIT_COPY_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> src: array<${T}>;
@group(0) @binding(1) var<storage, read_write> dst: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let src_offset = _params_[0];
    t_write(&dst, idx, t_read(&src, src_offset + idx));
}
)WGSL";

// ─── [matmul] ──────────────────────────────────────────────────────

// [matmul] fp16_gemm
static const char* WGSL_FP16_GEMM = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // Each lane handles K/32 elements, strided by 4 (vec4 processing)
    // lane processes elements: lane*4, lane*4 + 32*4, lane*4 + 64*4, ...
    var acc: f32 = 0.0;

    if (col < N) {
        // W layout: (N, K) row-major, stored as u32 (2 fp16 per u32)
        // w_base = col * (K/2) — index into u32 array
        let w_base = col * (K / 2u);

        // Each lane processes 4 consecutive elements starting at lane*4
        // Stride between lane iterations: 32 lanes * 4 elements = 128
        let stride = 128u;
        var k = lane * 4u;

        for (; k + 3u < K; k = k + stride) {
            // Load 4 fp32 activations
            let x0 = X[x_base + k];
            let x1 = X[x_base + k + 1u];
            let x2 = X[x_base + k + 2u];
            let x3 = X[x_base + k + 3u];

            // Load 2 u32 = 4 fp16 weights, unpack to fp32
            let w01 = unpack2x16float(W[w_base + k / 2u]);
            let w23 = unpack2x16float(W[w_base + k / 2u + 1u]);

            // dot(vec4<f32>, vec4<f32>) — 4 FMAs
            acc += dot(vec4<f32>(x0, x1, x2, x3),
                       vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Reduce across 32 lanes in warp
    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [matmul] fp16_gemm_wide
static const char* WGSL_FP16_GEMM_WIDE = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const COLS_PER_WARP: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each warp handles COLS_PER_WARP=4 columns
    let base_col = tile_col * TILE_N + warp_id * COLS_PER_WARP;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    let half_K = K / 2u;
    let stride = 128u;  // 32 lanes * 4 elements = 128
    var k = lane * 4u;

    for (; k + 3u < K; k = k + stride) {
        // Load X once, reuse for all 4 columns
        let x0 = X[x_base + k];
        let x1 = X[x_base + k + 1u];
        let x2 = X[x_base + k + 2u];
        let x3 = X[x_base + k + 3u];
        let xv = vec4<f32>(x0, x1, x2, x3);

        // Column 0
        if (base_col < N) {
            let w_base0 = base_col * half_K;
            let w01 = unpack2x16float(W[w_base0 + k / 2u]);
            let w23 = unpack2x16float(W[w_base0 + k / 2u + 1u]);
            acc0 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 1
        if (base_col + 1u < N) {
            let w_base1 = (base_col + 1u) * half_K;
            let w01 = unpack2x16float(W[w_base1 + k / 2u]);
            let w23 = unpack2x16float(W[w_base1 + k / 2u + 1u]);
            acc1 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 2
        if (base_col + 2u < N) {
            let w_base2 = (base_col + 2u) * half_K;
            let w01 = unpack2x16float(W[w_base2 + k / 2u]);
            let w23 = unpack2x16float(W[w_base2 + k / 2u + 1u]);
            acc2 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 3
        if (base_col + 3u < N) {
            let w_base3 = (base_col + 3u) * half_K;
            let w01 = unpack2x16float(W[w_base3 + k / 2u]);
            let w23 = unpack2x16float(W[w_base3 + k / 2u + 1u]);
            acc3 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Warp reduce
    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);
    let sum2 = subgroupAdd(acc2);
    let sum3 = subgroupAdd(acc3);

    if (lane == 0u) {
        if (base_col      < N) { Y[row * N + base_col]      = sum0 + Bias[base_col]; }
        if (base_col + 1u < N) { Y[row * N + base_col + 1u] = sum1 + Bias[base_col + 1u]; }
        if (base_col + 2u < N) { Y[row * N + base_col + 2u] = sum2 + Bias[base_col + 2u]; }
        if (base_col + 3u < N) { Y[row * N + base_col + 3u] = sum3 + Bias[base_col + 3u]; }
    }
}
)WGSL";

// [matmul] gemm — f32 instantiated
static const char* WGSL_GEMM = R"WGSL(
// Gemm — Y = A * B^T + Bias (or A * B + Bias)
// Dispatch: (ceil(N/16), ceil(M/16), 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }
    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        let b_val = select(t_read(&B, k * N + col), t_read(&B, col * K + k), transB != 0u);
        acc += t_read(&A, row * K + k) * b_val;
    }
    t_write(&Y, row * N + col, acc + t_read(&Bias, col));
}
)WGSL";

// [matmul] gemm — dtype template
static const char* WGSL_GEMM_T = R"WGSL(
// Gemm — Y = A * B^T + Bias (or A * B + Bias)
// Dispatch: (ceil(N/16), ceil(M/16), 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read> Bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }
    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        let b_val = select(t_read(&B, k * N + col), t_read(&B, col * K + k), transB != 0u);
        acc += t_read(&A, row * K + k) * b_val;
    }
    t_write(&Y, row * N + col, acc + t_read(&Bias, col));
}
)WGSL";

// [matmul] gemm_fp16_packed
static const char* WGSL_GEMM_FP16_PACKED = R"WGSL(
enable subgroups;

// Gemm with FP16-packed weights: Y = A * B^T + Bias
// A: (M, K) f32 activations
// B: (N, K/2) u32 — each u32 holds 2 fp16 weights via unpack2x16float
// Bias: (N) f32, added to output (pass zeros for no bias)
// Y: (M, N) f32 output
// Dispatch: (M, ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];

    if (row >= M) { return; }

    let a_base = row * K;
    let half_K = K / 2u;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    var acc: f32 = 0.0;

    if (col < N) {
        // transB=1: B is (N, K), row-major → b_base = col * half_K
        // transB=0: B is (K, N), row-major → b[k,col] = B[(k/2)*N + col] (interleaved)
        // The packed layout only makes sense for transB=1 (contiguous K dim per row).
        // For transB=0, we still support it but stride differently.

        let stride = 128u; // 32 lanes × 4 elements
        var k = lane * 4u;

        if (transB != 0u) {
            let b_base = col * half_K;
            for (; k + 3u < K; k = k + stride) {
                let a0 = A[a_base + k];
                let a1 = A[a_base + k + 1u];
                let a2 = A[a_base + k + 2u];
                let a3 = A[a_base + k + 3u];

                let w01 = unpack2x16float(B[b_base + k / 2u]);
                let w23 = unpack2x16float(B[b_base + k / 2u + 1u]);

                acc += dot(vec4<f32>(a0, a1, a2, a3),
                           vec4<f32>(w01.x, w01.y, w23.x, w23.y));
            }
            for (; k < K; k = k + stride) {
                let pair = unpack2x16float(B[b_base + k / 2u]);
                let w_val = select(pair.x, pair.y, (k & 1u) == 1u);
                acc += A[a_base + k] * w_val;
            }
        } else {
            // transB=0: B is (K, N) row-major, packed along N dimension
            // B[k][col] → u32 at index (k * N + col) / 2, but col pairs share a u32
            // For simplicity, iterate per-element
            for (; k < K; k = k + stride) {
                let a_val = A[a_base + k];
                let b_idx = k * N + col;
                let pair = unpack2x16float(B[b_idx / 2u]);
                let w_val = select(pair.x, pair.y, (b_idx & 1u) == 1u);
                acc += a_val * w_val;
            }
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [matmul] gemm_fp16_packed_nosub
static const char* WGSL_GEMM_FP16_PACKED_NOSUB = R"WGSL(
// Gemm fp16-packed weights, no subgroup — shared memory reduction
// Dispatch: (M, ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const WG_SIZE: u32 = 256u;
const WARP_SIZE: u32 = 32u;

var<workgroup> shared_reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let transB = _params_[3];

    if (row >= M) { return; }

    let a_base = row * K;
    let half_K = K / 2u;
    let half_N = N / 2u;

    let warp_id = tid / WARP_SIZE;
    let lane = tid % WARP_SIZE;
    let col = tile_col * TILE_N + warp_id;

    var acc: f32 = 0.0;

    if (col < N) {
        let stride = 128u; // 32 lanes × 4 elements

        if (transB != 0u) {
            // B layout: (N, K) packed as (N, K/2) u32. K is even.
            let b_base = col * half_K;
            var k = lane * 4u;
            for (; k + 3u < K; k = k + stride) {
                let a0 = A[a_base + k];
                let a1 = A[a_base + k + 1u];
                let a2 = A[a_base + k + 2u];
                let a3 = A[a_base + k + 3u];

                let w01 = unpack2x16float(B[b_base + k / 2u]);
                let w23 = unpack2x16float(B[b_base + k / 2u + 1u]);

                acc += dot(vec4<f32>(a0, a1, a2, a3),
                           vec4<f32>(w01.x, w01.y, w23.x, w23.y));
            }
            // Handle remaining elements (K not divisible by 4)
            for (; k < K; k = k + 1u) {
                let pair_idx = k / 2u;
                let w = unpack2x16float(B[b_base + pair_idx]);
                let v = select(w.x, w.y, (k & 1u) == 1u);
                acc += A[a_base + k] * v;
            }
        } else {
            // B layout: (K, N) packed as (K, N/2) u32. N is even (enforced by host).
            let col_pair = col / 2u;
            let col_odd = col & 1u;
            var k = lane * 4u;
            for (; k + 3u < K; k = k + stride) {
                let a0 = A[a_base + k];
                let a1 = A[a_base + k + 1u];
                let a2 = A[a_base + k + 2u];
                let a3 = A[a_base + k + 3u];

                let w0 = unpack2x16float(B[(k) * half_N + col_pair]);
                let w1 = unpack2x16float(B[(k + 1u) * half_N + col_pair]);
                let w2 = unpack2x16float(B[(k + 2u) * half_N + col_pair]);
                let w3 = unpack2x16float(B[(k + 3u) * half_N + col_pair]);

                let v0 = select(w0.x, w0.y, col_odd == 1u);
                let v1 = select(w1.x, w1.y, col_odd == 1u);
                let v2 = select(w2.x, w2.y, col_odd == 1u);
                let v3 = select(w3.x, w3.y, col_odd == 1u);

                acc += dot(vec4<f32>(a0, a1, a2, a3),
                           vec4<f32>(v0, v1, v2, v3));
            }
            // Handle remaining elements (K not divisible by 4)
            for (; k < K; k = k + 1u) {
                let w = unpack2x16float(B[k * half_N + col_pair]);
                let v = select(w.x, w.y, col_odd == 1u);
                acc += A[a_base + k] * v;
            }
        }
    }

    // Shared memory reduction within each warp (replace subgroupAdd)
    shared_reduce[tid] = acc;
    workgroupBarrier();

    let warp_base = warp_id * WARP_SIZE;
    // Tree reduction within warp
    if (lane < 16u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 2u]; }
    workgroupBarrier();
    if (lane < 1u) { shared_reduce[warp_base + lane] += shared_reduce[warp_base + lane + 1u]; }
    workgroupBarrier();

    if (lane == 0u && col < N) {
        Y[row * N + col] = shared_reduce[warp_base] + Bias[col];
    }
}
)WGSL";

// [matmul] matmul — f32 instantiated
static const char* WGSL_MATMUL = R"WGSL(
// MatMul — matrix multiplication
// C[m,n] = sum_k A[m,k] * B[k,n]
// Dispatch: (ceil(N/32), ceil(M/16), 1) — each thread handles 2 adjacent columns


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x * 2u;
    if (row >= M || col >= N) { return; }

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        let a_val = t_read(&A, row * K + k);
        acc0 += a_val * t_read(&B, k * N + col);
        if (col + 1u < N) {
            acc1 += a_val * t_read(&B, k * N + col + 1u);
        }
    }
    t_write2(&C, row * N + col, acc0, acc1);
}
)WGSL";

// [matmul] matmul — dtype template
static const char* WGSL_MATMUL_T = R"WGSL(
// MatMul — matrix multiplication
// C[m,n] = sum_k A[m,k] * B[k,n]
// Dispatch: (ceil(N/32), ceil(M/16), 1) — each thread handles 2 adjacent columns

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> C: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x * 2u;
    if (row >= M || col >= N) { return; }

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        let a_val = t_read(&A, row * K + k);
        acc0 += a_val * t_read(&B, k * N + col);
        if (col + 1u < N) {
            acc1 += a_val * t_read(&B, k * N + col + 1u);
        }
    }
    t_write2(&C, row * N + col, acc0, acc1);
}
)WGSL";

// [matmul] matmul_f16
static const char* WGSL_MATMUL_F16 = R"WGSL(
enable f16;

// MatMul — fp32 activations by fp16 weights
// C[m,n] = sum_k A[m,k] * B[k,n]

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }
    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        acc += A[row * K + k] * f32(B[k * N + col]);
    }
    C[row * N + col] = acc;
}
)WGSL";

// [matmul] matmul_f32
static const char* WGSL_MATMUL_F32 = R"WGSL(
// MatMul — fp32 matrix multiplication
// C[m,n] = sum_k A[m,k] * B[k,n]
// Dispatch: (ceil(N/16), ceil(M/16), 1)

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }
    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
)WGSL";

// [matmul] matmul_fp16_packed
static const char* WGSL_MATMUL_FP16_PACKED = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

// Tiled FP16-packed matmul: Y = X * W^T + Bias
// X: (M, K) f32 activations
// W: (N, K/2) u32 — each u32 holds 2 fp16 weights, unpacked via unpack2x16float
// Y: (M, N) f32 output
// No 'enable f16' required.

const TILE_M: u32 = 4u;
const TILE_N: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let M = _params_[2];

    let half_K = K / 2u;

    // 256 threads = 8 warps of 32
    // Each warp handles one (row, col) pair from the TILE_M × TILE_N tile
    // 8 warps → 4 rows × 2 col-pairs (each warp does 2 columns)
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let local_row = warp_id / 2u;   // 0..3
    let local_col2 = warp_id % 2u;  // 0..1, each handles 2 cols

    let row = tile_row * TILE_M + local_row;
    let col0 = tile_col * TILE_N + local_col2 * 2u;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;

    if (row < M) {
        let x_base = row * K;
        let stride = 128u;  // 32 lanes * 4 elements
        var k = lane * 4u;

        for (; k + 3u < K; k = k + stride) {
            let x0 = X[x_base + k];
            let x1 = X[x_base + k + 1u];
            let x2 = X[x_base + k + 2u];
            let x3 = X[x_base + k + 3u];
            let xv = vec4<f32>(x0, x1, x2, x3);

            if (col0 < N) {
                let w_base0 = col0 * half_K;
                let w01 = unpack2x16float(W[w_base0 + k / 2u]);
                let w23 = unpack2x16float(W[w_base0 + k / 2u + 1u]);
                acc0 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
            }

            if (col0 + 1u < N) {
                let w_base1 = (col0 + 1u) * half_K;
                let w01 = unpack2x16float(W[w_base1 + k / 2u]);
                let w23 = unpack2x16float(W[w_base1 + k / 2u + 1u]);
                acc1 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
            }
        }

        // Handle remaining elements when K is not a multiple of 4
        for (; k < K; k = k + stride) {
            let xv = X[x_base + k];
            if (col0 < N) {
                let w_base0 = col0 * half_K;
                let pair = unpack2x16float(W[w_base0 + k / 2u]);
                let w_val = select(pair.x, pair.y, (k & 1u) == 1u);
                acc0 += xv * w_val;
            }
            if (col0 + 1u < N) {
                let w_base1 = (col0 + 1u) * half_K;
                let pair = unpack2x16float(W[w_base1 + k / 2u]);
                let w_val = select(pair.x, pair.y, (k & 1u) == 1u);
                acc1 += xv * w_val;
            }
        }
    }

    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);

    if (lane == 0u) {
        if (row < M && col0 < N) {
            Y[row * N + col0] = sum0 + Bias[col0];
        }
        if (row < M && col0 + 1u < N) {
            Y[row * N + col0 + 1u] = sum1 + Bias[col0 + 1u];
        }
    }
}
)WGSL";

// [matmul] matmul_fp16_packed_nt
static const char* WGSL_MATMUL_FP16_PACKED_NT = R"WGSL(
// MatMul - fp32 activations by packed fp16 weights without requiring shader-f16.
// C[m,n] = sum_k A[m,k] * B[k,n]
// B is stored as row-major fp16 and viewed as array<u32>, two fp16 values per word.

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

fn read_b(k: u32, n: u32, N: u32) -> f32 {
    let linear = k * N + n;
    let pair = unpack2x16float(B[linear / 2u]);
    return select(pair.x, pair.y, (linear & 1u) == 1u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let row = gid.y;
    let col = gid.x;
    if (row >= M || col >= N) { return; }

    var acc: f32 = 0.0;
    for (var k = 0u; k < K; k++) {
        acc += A[row * K + k] * read_b(k, col, N);
    }
    C[row * N + col] = acc;
}
)WGSL";

// ─── [moe] ─────────────────────────────────────────────────────────

// [moe] compute_offsets
static const char* WGSL_COMPUTE_OFFSETS = R"WGSL(
// MoE compute expert offsets — converts indices to per-direction row offsets
//
// Reads moeIndicesBuf[k] (top-k expert indices, uint32) and writes per-slot
// row offsets for the 3 indirect matmul directions:
//   gate_offsets[k] = idx * IM_e   (offset into ffn_gate_exps as row index)
//   up_offsets[k]   = idx * IM_e   (offset into ffn_up_exps)
//   down_offsets[k] = idx * E      (offset into ffn_down_exps)
//
// These offsets are then read by the indirect IQ matmul kernels via a
// separate binding, indexed by slot_idx (a per-dispatch uniform).
//
// Bindings:
//   0: indices       array<u32>  read  — [k] expert indices
//   1: gate_offsets  array<u32>  write — [k]
//   2: up_offsets    array<u32>  write — [k]
//   3: down_offsets  array<u32>  write — [k]
//   4: _params_                       — [k, IM_e, E]

@group(0) @binding(0) var<storage, read>       indices:     array<u32>;
@group(0) @binding(1) var<storage, read_write> gate_off:    array<u32>;
@group(0) @binding(2) var<storage, read_write> up_off:      array<u32>;
@group(0) @binding(3) var<storage, read_write> down_off:    array<u32>;
@group(0) @binding(4) var<storage, read>       _params_:    array<u32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k    = _params_[0];
    let IM_e = _params_[1];
    let E    = _params_[2];
    let s = gid.x;
    if (s >= k) { return; }
    let idx = indices[s];
    gate_off[s] = idx * IM_e;
    up_off[s]   = idx * IM_e;
    down_off[s] = idx * E;
}
)WGSL";

// [moe] gather_elements — f32 instantiated
static const char* WGSL_GATHER_ELEMENTS = R"WGSL(
// GatherElements on data along last axis (axis=-1).
// Params: [0]=N (total elements in indices), [1]=dimSize (data last dim), [2]=outDim
// Dispatch: (ceil(N/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let dataDim = _params_[1];
    let outDim = _params_[2];
    let idx = gid.x;
    if (idx >= N) { return; }
    let slice = idx / outDim;
    var gi = indices[idx];
    if (gi < 0) { gi = gi + i32(dataDim); }
    t_write(&output, idx, t_read(&data, slice * dataDim + u32(gi)));
}
)WGSL";

// [moe] gather_elements — dtype template
static const char* WGSL_GATHER_ELEMENTS_T = R"WGSL(
// GatherElements on data along last axis (axis=-1).
// Params: [0]=N (total elements in indices), [1]=dimSize (data last dim), [2]=outDim
// Dispatch: (ceil(N/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> data: array<${T}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let dataDim = _params_[1];
    let outDim = _params_[2];
    let idx = gid.x;
    if (idx >= N) { return; }
    let slice = idx / outDim;
    var gi = indices[idx];
    if (gi < 0) { gi = gi + i32(dataDim); }
    t_write(&output, idx, t_read(&data, slice * dataDim + u32(gi)));
}
)WGSL";

// [moe] matmul_q4_indirect — f32 instantiated
static const char* WGSL_MATMUL_Q4_INDIRECT = R"WGSL(
// Q4 matmul with expert index from GPU buffer.
// Like MATMUL_Q4 but reads expert index from binding 5 to compute weight offset.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
// Weights: W[num_experts, N, K/2] packed uint8
// Scales:  S[num_experts, N, blocks_per_col] packed fp16
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot
// Dispatch: (ceil(N/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];

    let n = gid.x;
    if (n >= N) { return; }

    let expert = expert_indices[slot];

    // Byte offsets into the [num_experts, N, K/2] weight buffer
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;

    var acc: f32 = 0.0;
    let w_base = expert_w_offset + n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = expert_s_offset + n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        for (var j = 0u; j < 16u; j++) {
            let byte_idx = w_blk_base + j;
            let byte_u32 = W[byte_idx / 4u];
            let byte_val = (byte_u32 >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            let lo = f32(byte_val & 0xFu) - 8.0;
            let hi = f32((byte_val >> 4u) & 0xFu) - 8.0;
            acc += t_read(&X, k_base + j * 2u) * lo * scale;
            acc += t_read(&X, k_base + j * 2u + 1u) * hi * scale;
        }
    }

    t_write(&Y, n, acc);
}
)WGSL";

// [moe] matmul_q4_indirect — dtype template
static const char* WGSL_MATMUL_Q4_INDIRECT_T = R"WGSL(
// Q4 matmul with expert index from GPU buffer.
// Like MATMUL_Q4 but reads expert index from binding 5 to compute weight offset.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
// Weights: W[num_experts, N, K/2] packed uint8
// Scales:  S[num_experts, N, blocks_per_col] packed fp16
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot
// Dispatch: (ceil(N/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];

    let n = gid.x;
    if (n >= N) { return; }

    let expert = expert_indices[slot];

    // Byte offsets into the [num_experts, N, K/2] weight buffer
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;

    var acc: f32 = 0.0;
    let w_base = expert_w_offset + n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = expert_s_offset + n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        for (var j = 0u; j < 16u; j++) {
            let byte_idx = w_blk_base + j;
            let byte_u32 = W[byte_idx / 4u];
            let byte_val = (byte_u32 >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            let lo = f32(byte_val & 0xFu) - 8.0;
            let hi = f32((byte_val >> 4u) & 0xFu) - 8.0;
            acc += t_read(&X, k_base + j * 2u) * lo * scale;
            acc += t_read(&X, k_base + j * 2u + 1u) * hi * scale;
        }
    }

    t_write(&Y, n, acc);
}
)WGSL";

// [moe] matmul_q4_indirect_sub — f32 instantiated
static const char* WGSL_MATMUL_Q4_INDIRECT_SUB = R"WGSL(
// Q4 matmul with K-parallel subgroup reduction and TILE_N=8.
// 256 threads = 8 warps × 32 lanes. Each warp computes one output element.
// 32 lanes split blocks_per_col, then reduce via subgroupAdd.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
// Weights: W[num_experts, N, K/2] packed uint8
// Scales:  S[num_experts, N, blocks_per_col] packed fp16
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot
// Dispatch: (ceil(N/8), 1, 1)

enable subgroups;


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];

    let warp_id = lid.x / 32u;
    let lane = lid.x % 32u;
    let n = wid.x * 8u + warp_id;
    let valid = n < N;

    let expert = select(0u, expert_indices[slot], valid);
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;
    let w_base = expert_w_offset + n * (K / 2u);

    var acc: f32 = 0.0;

    if (valid) {
        // Each lane processes blocks: lane, lane+32, lane+64, ...
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = expert_s_offset + n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = W[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                acc += t_read(&X, k0)      * (f32(b0 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 1u) * (f32(b0 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 2u) * (f32(b1 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 3u) * (f32(b1 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 4u) * (f32(b2 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 5u) * (f32(b2 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 6u) * (f32(b3 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 7u) * (f32(b3 >> 4u)  - 8.0) * scale;
            }
        }
    }

    // subgroupAdd requires subgroup-uniform control flow
    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        t_write(&Y, n, warp_sum);
    }
}
)WGSL";

// [moe] matmul_q4_indirect_sub — dtype template
static const char* WGSL_MATMUL_Q4_INDIRECT_SUB_T = R"WGSL(
// Q4 matmul with K-parallel subgroup reduction and TILE_N=8.
// 256 threads = 8 warps × 32 lanes. Each warp computes one output element.
// 32 lanes split blocks_per_col, then reduce via subgroupAdd.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
// Weights: W[num_experts, N, K/2] packed uint8
// Scales:  S[num_experts, N, blocks_per_col] packed fp16
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot
// Dispatch: (ceil(N/8), 1, 1)

enable subgroups;

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];

    let warp_id = lid.x / 32u;
    let lane = lid.x % 32u;
    let n = wid.x * 8u + warp_id;
    let valid = n < N;

    let expert = select(0u, expert_indices[slot], valid);
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;
    let w_base = expert_w_offset + n * (K / 2u);

    var acc: f32 = 0.0;

    if (valid) {
        // Each lane processes blocks: lane, lane+32, lane+64, ...
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = expert_s_offset + n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = W[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                acc += t_read(&X, k0)      * (f32(b0 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 1u) * (f32(b0 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 2u) * (f32(b1 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 3u) * (f32(b1 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 4u) * (f32(b2 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 5u) * (f32(b2 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 6u) * (f32(b3 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 7u) * (f32(b3 >> 4u)  - 8.0) * scale;
            }
        }
    }

    // subgroupAdd requires subgroup-uniform control flow
    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        t_write(&Y, n, warp_sum);
    }
}
)WGSL";

// [moe] matmul_q4_indirect_sub_batched
static const char* WGSL_MATMUL_Q4_INDIRECT_SUB_BATCHED = R"WGSL(
// Batched Q4 matmul with K-parallel subgroup reduction. T tokens via workgroup_id.y.
// Each token reads its own expert index from expert_indices[tok * k + slot].
// X[tok * K + k], Y[tok * N + n]
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot, [4]=k_val (topk k)
// Dispatch: (ceil(N/8), nTokens, 1)

enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];
    let k_val = _params_[4];
    let tok = wid.y;

    let warp_id = lid.x / 32u;
    let lane = lid.x % 32u;
    let n = wid.x * 8u + warp_id;
    let valid = n < N;

    let expert = select(0u, expert_indices[tok * k_val + slot], valid);
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;
    let w_base = expert_w_offset + n * (K / 2u);
    let x_base = tok * K;

    var acc: f32 = 0.0;

    if (valid) {
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = expert_s_offset + n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;
            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;
            for (var j = 0u; j < 4u; j++) {
                let packed = W[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                acc += X[x_base + k0]      * (f32(b0 & 0xFu) - 8.0) * scale;
                acc += X[x_base + k0 + 1u] * (f32(b0 >> 4u)  - 8.0) * scale;
                acc += X[x_base + k0 + 2u] * (f32(b1 & 0xFu) - 8.0) * scale;
                acc += X[x_base + k0 + 3u] * (f32(b1 >> 4u)  - 8.0) * scale;
                acc += X[x_base + k0 + 4u] * (f32(b2 & 0xFu) - 8.0) * scale;
                acc += X[x_base + k0 + 5u] * (f32(b2 >> 4u)  - 8.0) * scale;
                acc += X[x_base + k0 + 6u] * (f32(b3 & 0xFu) - 8.0) * scale;
                acc += X[x_base + k0 + 7u] * (f32(b3 >> 4u)  - 8.0) * scale;
            }
        }
    }
    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        Y[tok * N + n] = warp_sum;
    }
}
)WGSL";

// [moe] matmul_q4_indirect_wide_batched
static const char* WGSL_MATMUL_Q4_INDIRECT_WIDE_BATCHED = R"WGSL(
// Wide-tile batched Q4 matmul with expert indirection and shared-memory X reuse.
// 128 threads, TILE_N=128. Each thread owns one N-column.
// X tile [1 x 32] is cooperatively loaded into shared memory per Q4 block,
// then reused by all 128 threads for different N-column weight-activation dot products.
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot, [4]=k_val (topk k)
// Dispatch: (ceil(N/128), nTokens, 1)

enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

var<workgroup> smem_x: array<f32, 32>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];
    let k_val = _params_[4];
    let tok = wid.y;

    let tid = lid.x;
    let n = wid.x * 128u + tid;
    let valid = n < N;

    let expert = expert_indices[tok * k_val + slot];
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;
    let w_base = select(0u, expert_w_offset + n * (K / 2u), valid);
    let x_base = tok * K;

    var acc: f32 = 0.0;

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let k_off = blk * 32u;

        // Cooperative X load: 128 threads load 32 elements
        if (tid < 32u) {
            smem_x[tid] = X[x_base + k_off + tid];
        }
        workgroupBarrier();

        if (valid) {
            let scale_flat = expert_s_offset + n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let w_blk_base = w_base + k_off / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = W[w_blk_base / 4u + j];
                let kl = j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                acc += smem_x[kl]      * (f32(b0 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 1u] * (f32(b0 >> 4u)  - 8.0) * scale
                     + smem_x[kl + 2u] * (f32(b1 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 3u] * (f32(b1 >> 4u)  - 8.0) * scale
                     + smem_x[kl + 4u] * (f32(b2 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 5u] * (f32(b2 >> 4u)  - 8.0) * scale
                     + smem_x[kl + 6u] * (f32(b3 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 7u] * (f32(b3 >> 4u)  - 8.0) * scale;
            }
        }

        workgroupBarrier();
    }

    if (valid) {
        Y[tok * N + n] = acc;
    }
}
)WGSL";

// [moe] moe_gate — f32 instantiated
static const char* WGSL_MOE_GATE = R"WGSL(
// MoE gate — GPU expert selection for MoE.
// Scans router weights, finds active experts (value > -60000), applies exp+normalize.
// Input: router[num_experts] (ln(sigmoid) for active, -65504 for inactive)
// Output: expert_indices[k] u32, expert_weights[k] f32
// Params: [0]=num_experts, [1]=k, [2]=normalize
// Dispatch: (1, 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


@group(0) @binding(0) var<storage, read> router: array<f32>;
@group(0) @binding(1) var<storage, read_write> expert_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let num_experts = _params_[0];
    let k = _params_[1];
    let normalize = _params_[2];

    var count = 0u;
    for (var e = 0u; e < num_experts; e++) {
        let v = t_read(&router, e);
        if (v > -60000.0 && count < k) {
            expert_indices[count] = e;
            expert_weights[count] = exp(v);
            count++;
        }
    }

    // Zero out unused slots
    for (var i = count; i < k; i++) {
        expert_indices[i] = 0u;
        expert_weights[i] = 0.0;
    }

    // Normalize weights
    if (normalize != 0u && count > 0u) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < count; i++) {
            sum += expert_weights[i];
        }
        if (sum > 0.0) {
            for (var i = 0u; i < count; i++) {
                expert_weights[i] /= sum;
            }
        }
    }
}
)WGSL";

// [moe] moe_gate — dtype template
static const char* WGSL_MOE_GATE_T = R"WGSL(
// MoE gate — GPU expert selection for MoE.
// Scans router weights, finds active experts (value > -60000), applies exp+normalize.
// Input: router[num_experts] (ln(sigmoid) for active, -65504 for inactive)
// Output: expert_indices[k] u32, expert_weights[k] f32
// Params: [0]=num_experts, [1]=k, [2]=normalize
// Dispatch: (1, 1, 1)

${T_READ}

@group(0) @binding(0) var<storage, read> router: array<${T}>;
@group(0) @binding(1) var<storage, read_write> expert_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let num_experts = _params_[0];
    let k = _params_[1];
    let normalize = _params_[2];

    var count = 0u;
    for (var e = 0u; e < num_experts; e++) {
        let v = t_read(&router, e);
        if (v > -60000.0 && count < k) {
            expert_indices[count] = e;
            expert_weights[count] = exp(v);
            count++;
        }
    }

    // Zero out unused slots
    for (var i = count; i < k; i++) {
        expert_indices[i] = 0u;
        expert_weights[i] = 0.0;
    }

    // Normalize weights
    if (normalize != 0u && count > 0u) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < count; i++) {
            sum += expert_weights[i];
        }
        if (sum > 0.0) {
            for (var i = 0u; i < count; i++) {
                expert_weights[i] /= sum;
            }
        }
    }
}
)WGSL";

// [moe] moe_gate_batched
static const char* WGSL_MOE_GATE_BATCHED = R"WGSL(
// Batched MoE gate — select top-k experts for T tokens in parallel.
// Input: router[T * num_experts] f32 (token-major)
// Output: expert_indices[T * k] u32, expert_weights[T * k] f32
// Params: [0]=num_experts, [1]=k, [2]=normalize, [3]=nTokens
// Dispatch: (1, nTokens, 1)

@group(0) @binding(0) var<storage, read> router: array<f32>;
@group(0) @binding(1) var<storage, read_write> expert_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let num_experts = _params_[0];
    let k = _params_[1];
    let normalize = _params_[2];
    let tok = wid.y;

    let r_base = tok * num_experts;
    let o_base = tok * k;

    var count = 0u;
    for (var e = 0u; e < num_experts; e++) {
        let v = router[r_base + e];
        if (v > -60000.0 && count < k) {
            expert_indices[o_base + count] = e;
            expert_weights[o_base + count] = exp(v);
            count++;
        }
    }
    for (var i = count; i < k; i++) {
        expert_indices[o_base + i] = 0u;
        expert_weights[o_base + i] = 0.0;
    }
    if (normalize != 0u && count > 0u) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < count; i++) { sum += expert_weights[o_base + i]; }
        if (sum > 0.0) {
            for (var i = 0u; i < count; i++) { expert_weights[o_base + i] /= sum; }
        }
    }
}
)WGSL";

// [moe] qmoe_matmul_q4
static const char* WGSL_QMOE_MATMUL_Q4 = R"WGSL(
// QMoE gate_up Q4 matmul — Q4 matmul for one MoE expert's gate_up projection.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
//
// Weight layout: W_q4[num_experts, N, K/2] uint8
// Scale layout:  S[num_experts, N, K/block_size] fp16
// Params: [0]=N (output dim = 2*intermediate), [1]=K (input dim = hidden),
//         [2]=expertIdx, [3]=blocks_per_col (K/32)
//
// Dispatch: (ceil(N/256), 1, 1) — one thread per output element

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> _params2_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let expert = _params_[2];
    let blocks_per_col = _params_[3];

    let n = gid.x;
    if (n >= N) { return; }

    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;

    var acc: f32 = 0.0;
    let w_base = expert_w_offset + n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = expert_s_offset + n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        for (var j = 0u; j < 16u; j++) {
            let byte_idx = w_blk_base + j;
            let byte_u32 = W[byte_idx / 4u];
            let byte_val = (byte_u32 >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            let lo = f32(byte_val & 0xFu) - 8.0;
            let hi = f32((byte_val >> 4u) & 0xFu) - 8.0;
            acc += X[k_base + j * 2u] * lo * scale;
            acc += X[k_base + j * 2u + 1u] * hi * scale;
        }
    }

    Y[n] = acc;
}
)WGSL";

// [moe] scatter_elements — f32 instantiated
static const char* WGSL_SCATTER_ELEMENTS = R"WGSL(
// ScatterElements on data along last axis.
// Copies data → output, then scatters updates at index positions.
// Params: [0]=dataN, [1]=dataDim, [2]=idxN, [3]=idxDim, [4]=mode (0=copy, 1=scatter)
// Dispatch: (ceil(dataN/256), 1, 1) for copy, (ceil(idxN/256), 1, 1) for scatter


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> updates: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let mode = _params_[4];
    if (mode == 0u) {
        // Copy pass: data → output
        let dataN = _params_[0];
        let idx = gid.x;
        if (idx >= dataN) { return; }
        t_write(&output, idx, t_read(&data, idx));
    } else {
        // Scatter pass: write updates at indexed positions
        let dataDim = _params_[1];
        let idxN = _params_[2];
        let idxDim = _params_[3];
        let idx = gid.x;
        if (idx >= idxN) { return; }
        let slice = idx / idxDim;
        var gi = indices[idx];
        if (gi < 0) { gi = gi + i32(dataDim); }
        let dst = slice * dataDim + u32(gi);
        t_write(&output, dst, t_read(&updates, idx));
    }
}
)WGSL";

// [moe] scatter_elements — dtype template
static const char* WGSL_SCATTER_ELEMENTS_T = R"WGSL(
// ScatterElements on data along last axis.
// Copies data → output, then scatters updates at index positions.
// Params: [0]=dataN, [1]=dataDim, [2]=idxN, [3]=idxDim, [4]=mode (0=copy, 1=scatter)
// Dispatch: (ceil(dataN/256), 1, 1) for copy, (ceil(idxN/256), 1, 1) for scatter

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> data: array<${T}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> updates: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let mode = _params_[4];
    if (mode == 0u) {
        // Copy pass: data → output
        let dataN = _params_[0];
        let idx = gid.x;
        if (idx >= dataN) { return; }
        t_write(&output, idx, t_read(&data, idx));
    } else {
        // Scatter pass: write updates at indexed positions
        let dataDim = _params_[1];
        let idxN = _params_[2];
        let idxDim = _params_[3];
        let idx = gid.x;
        if (idx >= idxN) { return; }
        let slice = idx / idxDim;
        var gi = indices[idx];
        if (gi < 0) { gi = gi + i32(dataDim); }
        let dst = slice * dataDim + u32(gi);
        t_write(&output, dst, t_read(&updates, idx));
    }
}
)WGSL";

// [moe] swiglu — f32 instantiated
static const char* WGSL_SWIGLU = R"WGSL(
// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
// Input is [N*2] with interleaved layout: [gate[0], up[0], gate[1], up[1], ...]
// Output is [N].
// Params: [0]=N (half_size = moe_intermediate_size)
// Dispatch: (ceil(N/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let gate = t_read(&gate_up, idx * 2u);
    let up = t_read(&gate_up, idx * 2u + 1u);
    let silu = gate / (1.0 + exp(-gate));
    t_write(&output, idx, silu * up);
}
)WGSL";

// [moe] swiglu — dtype template
static const char* WGSL_SWIGLU_T = R"WGSL(
// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
// Input is [N*2] with interleaved layout: [gate[0], up[0], gate[1], up[1], ...]
// Output is [N].
// Params: [0]=N (half_size = moe_intermediate_size)
// Dispatch: (ceil(N/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> gate_up: array<${T}>;
@group(0) @binding(1) var<storage, read_write> output: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let gate = t_read(&gate_up, idx * 2u);
    let up = t_read(&gate_up, idx * 2u + 1u);
    let silu = gate / (1.0 + exp(-gate));
    t_write(&output, idx, silu * up);
}
)WGSL";

// [moe] swiglu_batched
static const char* WGSL_SWIGLU_BATCHED = R"WGSL(
// Batched SwiGLU: T tokens via workgroup_id.y.
// gate_up[tok * N*2 + i], output[tok * N + i]
// Params: [0]=N (half_size)
// Dispatch: (ceil(N/256), nTokens, 1)

@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let tok = wid.y;
    let gate = gate_up[tok * N * 2u + idx * 2u];
    let up = gate_up[tok * N * 2u + idx * 2u + 1u];
    let silu = gate / (1.0 + exp(-gate));
    output[tok * N + idx] = silu * up;
}
)WGSL";

// [moe] topk_softmax
static const char* WGSL_TOPK_SOFTMAX = R"WGSL(
// MoE top-k routing — find top-k of N router logits, then softmax-normalize
//
// For 256 experts top-8, this scans the router logits, finds the 8 highest,
// applies softmax over only those, and outputs indices + weights.
//
// Bindings:
//   0: router_logits [num_experts] f32
//   1: out_indices   [k] u32
//   2: out_weights   [k] f32
//   3: _params_     — [num_experts, k, normalize_flag]
//
// Single-workgroup design: 256 threads, parallel argmax with iterated removal.
// For top-8 of 256, that's 8 sequential argmax passes (each O(N) = 256 muls).

@group(0) @binding(0) var<storage, read>       logits:  array<f32>;
@group(0) @binding(1) var<storage, read_write> idx_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> wt_out:  array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

const MAX_K: u32 = 16u;
const NEG_INF: f32 = -3.4e38;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let num_experts = _params_[0];
    let k           = _params_[1];
    let normalize   = _params_[2];
    let tid = lid.x;
    if (tid != 0u) { return; }  // serial implementation — k is small, N=256

    // Iterated argmax with masking (k passes, each O(N))
    var selected: array<u32, MAX_K>;
    var values:   array<f32, MAX_K>;
    var mask:     array<bool, 1024>;  // assumes num_experts <= 1024 (fine for qwen35moe 256)
    for (var i: u32 = 0u; i < num_experts; i = i + 1u) {
        mask[i] = false;
    }
    for (var ki: u32 = 0u; ki < k; ki = ki + 1u) {
        var best_v: f32 = NEG_INF;
        var best_i: u32 = 0u;
        for (var e: u32 = 0u; e < num_experts; e = e + 1u) {
            if (!mask[e]) {
                let v = logits[e];
                if (v > best_v) { best_v = v; best_i = e; }
            }
        }
        selected[ki] = best_i;
        values[ki]   = best_v;
        mask[best_i] = true;
    }

    // Softmax over the k selected
    var max_v: f32 = values[0];
    for (var ki: u32 = 1u; ki < k; ki = ki + 1u) {
        if (values[ki] > max_v) { max_v = values[ki]; }
    }
    var sum_e: f32 = 0.0;
    for (var ki: u32 = 0u; ki < k; ki = ki + 1u) {
        values[ki] = exp(values[ki] - max_v);
        sum_e = sum_e + values[ki];
    }
    let inv = select(1.0, 1.0 / sum_e, normalize != 0u && sum_e > 0.0);
    for (var ki: u32 = 0u; ki < k; ki = ki + 1u) {
        idx_out[ki] = selected[ki];
        wt_out[ki]  = values[ki] * inv;
    }
}
)WGSL";

// [moe] weighted_accumulate_decode
static const char* WGSL_WEIGHTED_ACCUMULATE_DECODE = R"WGSL(
// MoE per-expert weighted accumulate
//
// After computing one expert's down-projection output, accumulate into the
// per-token output with weight from the softmax routing:
//   out[c] += weights[k_slot] * src[c]   for c in 0..d_model
//
// k_slot is the slot index (0..numExpertsPerTok-1) — host code does one
// dispatch per active expert.
//
// Bindings:
//   0: out      [d_model]                  (read+write)
//   1: src      [d_model]
//   2: weights  [numExpertsPerTok]
//   3: _params_                           — [d_model, k_slot]

@group(0) @binding(0) var<storage, read_write> out:      array<f32>;
@group(0) @binding(1) var<storage, read>       src:      array<f32>;
@group(0) @binding(2) var<storage, read>       weights:  array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = _params_[0];
    let k_slot = _params_[1];
    let c = gid.x;
    if (c >= d) { return; }
    let w = weights[k_slot];
    out[c] = out[c] + w * src[c];
}
)WGSL";

// [moe] weighted_add
static const char* WGSL_WEIGHTED_ADD = R"WGSL(
// Weighted accumulate: out[i] += weight * src[i]
// Params: [0]=N, [1]=weight_as_u32
// Dispatch: (ceil(N/256), 1, 1)

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let idx = gid.x;
    if (idx >= N) { return; }
    let weight = bitcast<f32>(_params_[1]);
    dst[idx] = dst[idx] + weight * src[idx];
}
)WGSL";

// [moe] weighted_add_indirect — f32 instantiated
static const char* WGSL_WEIGHTED_ADD_INDIRECT = R"WGSL(
// Weighted add indirect — accumulate with weight from GPU buffer.
// dst[i] += expert_weights[slot] * src[i]
// Params: [0]=N, [1]=slot
// Dispatch: (ceil(N/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_read_rw(buf: ptr<storage, array<f32>, read_write>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;
@group(0) @binding(3) var<storage, read> expert_weights: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let slot = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let weight = expert_weights[slot];
    // slot 0: write (clears stale data); slot > 0: accumulate
    if (slot == 0u) {
        t_write(&dst, idx, weight * t_read(&src, idx));
    } else {
        t_write(&dst, idx, t_read_rw(&dst, idx) + weight * t_read(&src, idx));
    }
}
)WGSL";

// [moe] weighted_add_indirect — dtype template
static const char* WGSL_WEIGHTED_ADD_INDIRECT_T = R"WGSL(
// Weighted add indirect — accumulate with weight from GPU buffer.
// dst[i] += expert_weights[slot] * src[i]
// Params: [0]=N, [1]=slot
// Dispatch: (ceil(N/256), 1, 1)

${T_READ}
${T_READ_RW}
${T_WRITE}

@group(0) @binding(0) var<storage, read> src: array<${T}>;
@group(0) @binding(1) var<storage, read_write> dst: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;
@group(0) @binding(3) var<storage, read> expert_weights: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let slot = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let weight = expert_weights[slot];
    // slot 0: write (clears stale data); slot > 0: accumulate
    if (slot == 0u) {
        t_write(&dst, idx, weight * t_read(&src, idx));
    } else {
        t_write(&dst, idx, t_read_rw(&dst, idx) + weight * t_read(&src, idx));
    }
}
)WGSL";

// [moe] weighted_add_indirect_batched
static const char* WGSL_WEIGHTED_ADD_INDIRECT_BATCHED = R"WGSL(
// Batched weighted add: T tokens via workgroup_id.y.
// src[tok * N + i], dst[tok * N + i], expert_weights[tok * k + slot]
// Params: [0]=N, [1]=slot, [2]=k_val
// Dispatch: (ceil(N/256), nTokens, 1)

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;
@group(0) @binding(3) var<storage, read> expert_weights: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let slot = _params_[1];
    let k_val = _params_[2];
    let idx = gid.x;
    if (idx >= N) { return; }
    let tok = wid.y;
    let weight = expert_weights[tok * k_val + slot];
    let src_idx = tok * N + idx;
    let dst_idx = tok * N + idx;
    if (slot == 0u) {
        dst[dst_idx] = weight * src[src_idx];
    } else {
        dst[dst_idx] = dst[dst_idx] + weight * src[src_idx];
    }
}
)WGSL";

// ─── [norm] ────────────────────────────────────────────────────────

// [norm] add_rms_norm
static const char* WGSL_ADD_RMS_NORM = R"WGSL(
enable subgroups;

// Generated WGSL kernel
// Kernel: add_rms_norm_loop_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read_write> buf0: array<f32>;  // X
@group(0) @binding(1) var<storage, read> buf1: array<f32>;  // Residual
@group(0) @binding(2) var<storage, read_write> buf2: array<f32>;  // Y
@group(0) @binding(3) var<storage, read> buf3: array<f32>;  // W
@group(0) @binding(4) var<storage, read_write> buf4: array<f32>;  // Rstd

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(5) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v11: i32 = i32(_wg_id.x);
    let v12: i32 = params.N + 127;
    let v13: i32 = v12 / 128;
    var v15: i32 = 0;
    var v16_0: f32 = f32(0);
    loop {
        let v17: bool = v15 < v13;
        if !v17 { break; }
        let v19: i32 = v15 * 128;
        let v20: i32 = i32(_lid.x);
        let v21: i32 = v20 & 127;
        let v22: i32 = i32(u32(v21) % u32(32));
        let v23: i32 = i32(_lid.x);
        let v24: i32 = i32(u32(v23) / u32(32));
        let v25: i32 = v22 << u32(0);
        let v26: i32 = 0 | v25;
        let v27: i32 = v24 << u32(5);
        let v28: i32 = v26 | v27;
        let v29: i32 = v28 & 127;
        let v30: i32 = i32(u32(v29) >> u32(0));
        let v31: i32 = v30 | 0;
        let v32: i32 = 0 ^ v31;
        let v33: i32 = v32 ^ 0;
        let v34: i32 = v33 + 0;
        let v35: i32 = v19 + v34;
        let v36: bool = v35 < params.N;
        let v37: i32 = v11 * params.stride;
        let v40: f32 = buf0[u32((v37 + v35))];
        let v41: f32 = select(f32(0.0), v40, v36);
        let v44: f32 = buf1[u32((v37 + v35))];
        let v45: f32 = select(f32(0.0), v44, v36);
        let v46: f32 = v41 + v45;
        let v47: f32 = select(f32(0), v46, v36);
        if v36 { buf0[u32((v37 + v35))] = v47; }
        let v48: f32 = v46 * v46;
        let v50: f32 = v16_0 + v48;
        let v52: i32 = v15 + 1;
        v15 = v52;
        v16_0 = v50;
    }
    let v55: i32 = bitcast<i32>(v16_0);
    let v56: i32 = subgroupShuffleXor(v55, u32(16));
    let v57: f32 = bitcast<f32>(v56);
    let v58: f32 = v16_0 + v57;
    let v59: i32 = bitcast<i32>(v58);
    let v60: i32 = subgroupShuffleXor(v59, u32(8));
    let v61: f32 = bitcast<f32>(v60);
    let v62: f32 = v58 + v61;
    let v63: i32 = bitcast<i32>(v62);
    let v64: i32 = subgroupShuffleXor(v63, u32(4));
    let v65: f32 = bitcast<f32>(v64);
    let v66: f32 = v62 + v65;
    let v67: i32 = bitcast<i32>(v66);
    let v68: i32 = subgroupShuffleXor(v67, u32(2));
    let v69: f32 = bitcast<f32>(v68);
    let v70: f32 = v66 + v69;
    let v71: i32 = bitcast<i32>(v70);
    let v72: i32 = subgroupShuffleXor(v71, u32(1));
    let v73: f32 = bitcast<f32>(v72);
    let v74: f32 = v70 + v73;
    let v75: i32 = i32(_lid.x);
    let v76: i32 = v75 & 127;
    let v77: i32 = i32(u32(v76) % u32(32));
    let v78: i32 = i32(_lid.x);
    let v79: i32 = i32(u32(v78) / u32(32));
    let v80: i32 = v77 << u32(0);
    let v81: i32 = 0 | v80;
    let v82: i32 = v79 << u32(5);
    let v83: i32 = v81 | v82;
    let v84: i32 = v83 & 96;
    let v85: i32 = i32(u32(v84) >> u32(3));
    let v86: i32 = 0 | v85;
    let v87: i32 = 0 ^ v86;
    let v88: i32 = v87 ^ 0;
    let v89: i32 = v88 ^ 0;
    let v90: i32 = v89 + 0;
    _smem[u32(v90) >> 2u] = bitcast<i32>(v74);
    workgroupBarrier();
    let v94: i32 = i32(_lid.x);
    let v95: i32 = v94 & 127;
    let v96: i32 = i32(u32(v95) % u32(32));
    let v97: i32 = i32(_lid.x);
    let v98: i32 = i32(u32(v97) / u32(32));
    let v99: i32 = v96 << u32(0);
    let v100: i32 = 0 | v99;
    let v101: i32 = v98 << u32(5);
    let v102: i32 = v100 | v101;
    let v103: i32 = v102 & 3;
    let v104: i32 = v103 << u32(2);
    let v105: i32 = v104 | 0;
    let v106: i32 = 0 ^ v105;
    let v107: i32 = v106 ^ 0;
    let v108: i32 = v107 ^ 0;
    let v109: i32 = v108 + 0;
    let v111: f32 = bitcast<f32>(_smem[u32(v109) >> 2u]);
    let v114: i32 = bitcast<i32>(v111);
    let v115: i32 = subgroupShuffleXor(v114, u32(2));
    let v116: f32 = bitcast<f32>(v115);
    let v117: f32 = v111 + v116;
    let v118: i32 = bitcast<i32>(v117);
    let v119: i32 = subgroupShuffleXor(v118, u32(1));
    let v120: f32 = bitcast<f32>(v119);
    let v121: f32 = v117 + v120;
    let v122: f32 = f32(params.N);
    let v123: f32 = v121 / v122;
    let v124: f32 = v123 + params.eps;
    let v125: f32 = sqrt(v124);
    let v126: f32 = f32(1.0) / v125;
    var v128: i32 = 0;
    loop {
        let v129: bool = v128 < v13;
        if !v129 { break; }
        let v131: i32 = v128 * 128;
        let v132: i32 = i32(_lid.x);
        let v133: i32 = v132 & 127;
        let v134: i32 = i32(u32(v133) % u32(32));
        let v135: i32 = i32(_lid.x);
        let v136: i32 = i32(u32(v135) / u32(32));
        let v137: i32 = v134 << u32(0);
        let v138: i32 = 0 | v137;
        let v139: i32 = v136 << u32(5);
        let v140: i32 = v138 | v139;
        let v141: i32 = v140 & 127;
        let v142: i32 = i32(u32(v141) >> u32(0));
        let v143: i32 = v142 | 0;
        let v144: i32 = 0 ^ v143;
        let v145: i32 = v144 ^ 0;
        let v146: i32 = v145 + 0;
        let v147: i32 = v131 + v146;
        let v148: bool = v147 < params.N;
        let v149: i32 = v11 * params.stride;
        let v152: f32 = buf0[u32((v149 + v147))];
        let v153: f32 = select(f32(0.0), v152, v148);
        let v155: f32 = buf3[u32(v147)];
        let v156: f32 = select(f32(1.0), v155, v148);
        let v159: f32 = v153 * v126;
        let v160: f32 = v159 * v156;
        let v161: f32 = select(f32(0), v160, v148);
        if v148 { buf2[u32((v149 + v147))] = v161; }
        let v162: i32 = v128 + 1;
        v128 = v162;
    }
    if _lid.x == 0u { buf4[u32(v11)] = v126; }
}
)WGSL";

// [norm] add_rms_norm_batched
static const char* WGSL_ADD_RMS_NORM_BATCHED = R"WGSL(
enable subgroups;

// Batched add + RMSNorm for T rows:
//   X[t] = X[t] + A[t]  (residual add, in-place)
//   Y[t] = X[t] * W / rms(X[t])  (RMSNorm for next op)
// Grid: (T, 1, 1)
// WG: 128 threads

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;  // residual [T × N], updated in-place
@group(0) @binding(1) var<storage, read> A: array<f32>;  // addend [T × N]
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;  // normed output [T × N]
@group(0) @binding(3) var<storage, read> W: array<f32>;  // norm weight [N]
@group(0) @binding(4) var<storage, read_write> Rstd: array<f32>;  // [T]

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(5) var<storage, read> params: Params;

const MAX_N: u32 = 8192u;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
) {
    let row = i32(_wg_id.x);
    let tid = i32(_lid.x);
    let N = params.N;
    let stride = params.stride;
    let base = row * stride;

    // Pass 1: Add residual and compute sum of squares
    var sum_sq: f32 = 0.0;
    var idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let pos = u32(base + idx);
            let v = X[pos] + A[pos];
            X[pos] = v;
            sum_sq += v * v;
        }
    }

    let warp_sum = subgroupAdd(sum_sq);
    let warp_id = tid / 32;
    let lane_id = tid % 32;
    if (lane_id == 0) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    var final_sum: f32 = 0.0;
    for (var w = 0; w < 4; w++) {
        final_sum += bitcast<f32>(_smem[w]);
    }
    let rstd = 1.0 / sqrt(final_sum / f32(N) + params.eps);
    if (tid == 0) {
        Rstd[u32(row)] = rstd;
    }

    // Pass 2: Apply norm + weight
    idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let pos = u32(base + idx);
            Y[pos] = X[pos] * rstd * W[u32(idx)];
        }
    }
}
)WGSL";

// [norm] add_rms_norm_qwen_vulkan
static const char* WGSL_ADD_RMS_NORM_QWEN_VULKAN = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read_write> Rstd: array<f32>;

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(5) var<storage, read> params: Params;

var<workgroup> warp_sums: array<f32, 8>;
var<workgroup> rstd_shared: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let N = u32(params.N);
    let base = row * u32(params.stride);

    var sum_sq = 0.0;
    for (var j = tid; j < N; j = j + 256u) {
        let idx = base + j;
        let v = X[idx] + Residual[idx];
        X[idx] = v;
        sum_sq += v * v;
    }

    let warp_sum = subgroupAdd(sum_sq);
    if (lane == 0u) {
        warp_sums[warp] = warp_sum;
    }
    workgroupBarrier();

    let cross = select(0.0, warp_sums[min(lane, 7u)], warp == 0u && lane < 8u);
    let total = subgroupAdd(cross);
    if (tid == 0u) {
        rstd_shared = 1.0 / sqrt(total / f32(N) + params.eps);
        Rstd[row] = rstd_shared;
    }
    workgroupBarrier();

    let rstd = rstd_shared;
    for (var j = tid; j < N; j = j + 256u) {
        Y[base + j] = X[base + j] * rstd * W[j];
    }
}
)WGSL";

// [norm] fused_qknorm_rope
static const char* WGSL_FUSED_QKNORM_ROPE = R"WGSL(
enable f16;

// Fused QK-norm + RoPE kernel for single-token decode.
// Applies RMSNorm (with learned weights) to Q/K, then RoPE, then scatters K/V to cache.
// Optionally applies V-norm (RMSNorm without learned scale) to V before cache write.
//
// Grid: (n_head + n_kv, 1, 1)
//   WG x < n_head  → Q head
//   WG x >= n_head → K head + V scatter
//
// Workgroup: 128 threads, each handles HD/WG_SIZE elements
// For HD=128: 1 element/thread. HD=256: 2. HD=512: 4.

const WG_SIZE: u32 = 128u;

var<workgroup> _q_sums: array<f32, 128>;
var<workgroup> _v_sums: array<f32, 128>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // QKV
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Q_out
@group(0) @binding(2) var<storage, read_write> buf2: array<f16>;  // K_cache
@group(0) @binding(3) var<storage, read_write> buf3: array<f16>;  // V_cache
@group(0) @binding(4) var<storage, read> buf4: array<f32>;  // CosTable
@group(0) @binding(5) var<storage, read> buf5: array<f32>;  // SinTable
@group(0) @binding(6) var<storage, read> buf6: array<f32>;  // NormQ
@group(0) @binding(7) var<storage, read> buf7: array<f32>;  // NormK

struct Params {
    n_head: i32,
    q_size: i32,
    kv_size: i32,
    pos: i32,
    half_rot: i32,
    cache_offset: i32,
    eps: f32,
    v_norm: i32,   // 1 = apply RMSNorm (no scale) to V before cache write
};
@group(0) @binding(8) var<storage, read> params: Params;

const HD: u32 = 128u;
const EPT: u32 = (HD + WG_SIZE - 1u) / WG_SIZE;  // elements per thread

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head_idx = i32(wid.x);
    let tid = lid.x;
    let is_q = head_idx < params.n_head;
    let kv_head = head_idx - params.n_head;

    // Source offset into QKV buffer
    let q_off = head_idx * i32(HD);
    let k_off = params.q_size + kv_head * i32(HD);
    let src_off = select(k_off, q_off, is_q);

    // V offset (only used by KV heads)
    let v_off_base = params.q_size + params.kv_size + kv_head * i32(HD);
    let do_vnorm = params.v_norm != 0 && !is_q;

    // Step 1: Compute sum of squares for RMSNorm (Q/K) and optionally V
    var sum_sq: f32 = 0.0;
    var v_sum_sq: f32 = 0.0;
    for (var e: u32 = 0u; e < EPT; e++) {
        let d = tid + e * WG_SIZE;
        if (d < HD) {
            let val = buf0[u32(src_off + i32(d))];
            sum_sq += val * val;
            if (do_vnorm) {
                let vval = buf0[u32(v_off_base + i32(d))];
                v_sum_sq += vval * vval;
            }
        }
    }

    // Workgroup reduction. Do not assume a particular hardware subgroup width:
    // AMD commonly exposes 64-wide subgroups while other adapters use 32.
    _q_sums[tid] = sum_sq;
    _v_sums[tid] = v_sum_sq;
    workgroupBarrier();
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            _q_sums[tid] += _q_sums[tid + stride];
            _v_sums[tid] += _v_sums[tid + stride];
        }
        workgroupBarrier();
    }

    // RMSNorm scale: 1 / sqrt(mean_sq + eps)
    let rms_scale = 1.0 / sqrt(_q_sums[0] / f32(HD) + params.eps);

    // V RMSNorm scale (only for KV heads with v_norm enabled)
    var v_rms_scale: f32 = 1.0;
    if (do_vnorm) {
        v_rms_scale = 1.0 / sqrt(_v_sums[0] / f32(HD) + params.eps);
    }

    // Step 2: Apply RMSNorm + RoPE + scatter
    for (var e: u32 = 0u; e < EPT; e++) {
        let d = i32(tid + e * WG_SIZE);
        if (u32(d) >= HD) { continue; }

        // Load and apply RMSNorm
        let val = buf0[u32(src_off + d)];
        let norm_w = select(buf7[u32(d)], buf6[u32(d)], is_q);
        let normed = val * rms_scale * norm_w;

        // RoPE
        let half_rot = params.half_rot;
        let rot_dim = half_rot * 2;
        let in_rot = d < rot_dim;

        let d_mod = d % half_rot;
        let cos_sin_idx = params.pos * half_rot + d_mod;
        let c = select(f32(1.0), buf4[u32(cos_sin_idx)], in_rot);
        let s = select(f32(0.0), buf5[u32(cos_sin_idx)], in_rot);

        // Pair element for rotation
        let in_first_half = d < half_rot;
        let pair_d = select(d - half_rot, d + half_rot, in_first_half);
        let pair_d_final = select(d, pair_d, in_rot);
        let pair_val = buf0[u32(src_off + pair_d_final)];
        let pair_norm_w = select(buf7[u32(pair_d_final)], buf6[u32(pair_d_final)], is_q);
        let pair_normed = pair_val * rms_scale * pair_norm_w;

        let sign = select(f32(1.0), f32(-1.0), in_first_half);
        let rotated = select(normed, normed * c + sign * pair_normed * s, in_rot);

        // Write Q output
        if (is_q) {
            buf1[u32(head_idx * i32(HD) + d)] = rotated;
        }

        // Write K/V to cache
        if (!is_q) {
            buf2[u32(params.cache_offset + kv_head * i32(HD) + d)] = f16(rotated);
            let v_val = buf0[u32(v_off_base + d)];
            let v_normed = select(v_val, v_val * v_rms_scale, do_vnorm);
            buf3[u32(params.cache_offset + kv_head * i32(HD) + d)] = f16(v_normed);
        }
    }
}
)WGSL";

// [norm] fused_qknorm_rope_batched
static const char* WGSL_FUSED_QKNORM_ROPE_BATCHED = R"WGSL(
enable subgroups;

// Batched fused QKnorm + RoPE + KV cache scatter for prefill.
// Processes T tokens: applies QK-norm, rotary embeddings, and writes K/V to cache.
//
// QKV layout: [T × (qDim + 2*kvDim)]
//   Q: [T × qDim], K: [T × kvDim], V: [T × kvDim]
//
// Grid: ((n_head + n_kv) * T, 1, 1)
//   First n_head*T WGs handle Q heads (norm + RoPE → qRotBuf)
//   Next n_kv*T WGs handle KV heads (norm + RoPE + scatter to cache)
//
// Params: [n_head, qDim, kvDim, pos_offset, half_dim, 0, eps, T]

@group(0) @binding(0) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> QRot: array<f32>;
@group(0) @binding(2) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(4) var<storage, read> Cos: array<f32>;
@group(0) @binding(5) var<storage, read> Sin: array<f32>;
@group(0) @binding(6) var<storage, read> QNormW: array<f32>;
@group(0) @binding(7) var<storage, read> KNormW: array<f32>;
@group(0) @binding(8) var<storage, read> _params_: array<u32>;

const HD: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let flat_id = wid.x;
    let tid = lid.x;

    let n_head = _params_[0];
    let qDim   = _params_[1];
    let kvDim  = _params_[2];
    let pos_offset = _params_[3];
    let half_dim = _params_[4];
    let eps = bitcast<f32>(_params_[6]);
    let T = _params_[7];
    let n_kv = kvDim / HD;

    let total_q_wgs = n_head * T;

    if (flat_id < total_q_wgs) {
        // ── Q head processing ────────────────────────────────────────
        let t = flat_id / n_head;
        let head = flat_id % n_head;
        let qkv_stride = qDim + 2u * kvDim;

        // Source: QKV[t, head*HD .. (head+1)*HD]
        let src_base = t * qkv_stride + head * HD;

        // Load Q values
        var q: array<f32, 128>;
        for (var i = tid; i < HD; i += 128u) {
            q[i] = QKV[src_base + i];
        }

        // QK-norm: compute rstd across HD
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            sum_sq += q[i] * q[i];
        }
        let total_sq = subgroupAdd(sum_sq);
        // Cross-warp (128 threads = 4 warps)
        var _smem: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        // Use subgroup to collect
        if (lane == 0u) { _smem[warp_id] = total_sq; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = _smem[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply norm weight + RoPE
        let pos = pos_offset + t;
        let cos_base = pos * half_dim;
        let sin_base = pos * half_dim;

        let dst_base = t * n_head * HD + head * HD;
        for (var i = tid; i < HD; i += 128u) {
            let normed = q[i] * rstd * QNormW[i];
            if (i < half_dim) {
                let j = i + half_dim;
                let nj = q[j] * rstd * QNormW[j];
                let c = Cos[cos_base + i];
                let s = Sin[sin_base + i];
                QRot[dst_base + i] = normed * c - nj * s;
                QRot[dst_base + j] = nj * c + normed * s;
            } else if (i >= 2u * half_dim) {
                // Non-rotary dimensions: apply norm only
                QRot[dst_base + i] = normed;
            }
        }
    } else {
        // ── KV head processing ────────────────────────────────────────
        let kv_flat = flat_id - total_q_wgs;
        let t = kv_flat / n_kv;
        let kv_head = kv_flat % n_kv;
        let qkv_stride = qDim + 2u * kvDim;

        // K source: QKV[t, qDim + kv_head*HD ..]
        let k_src = t * qkv_stride + qDim + kv_head * HD;
        // V source: QKV[t, qDim + kvDim + kv_head*HD ..]
        let v_src = t * qkv_stride + qDim + kvDim + kv_head * HD;

        // Load K
        var k: array<f32, 128>;
        for (var i = tid; i < HD; i += 128u) {
            k[i] = QKV[k_src + i];
        }

        // K-norm
        var sum_sq: f32 = 0.0;
        for (var i = tid; i < HD; i += 128u) {
            sum_sq += k[i] * k[i];
        }
        let total_sq = subgroupAdd(sum_sq);
        var _smem2: array<f32, 4>;
        let warp_id = tid / 32u;
        let lane = tid % 32u;
        if (lane == 0u) { _smem2[warp_id] = total_sq; }
        workgroupBarrier();
        var ss: f32 = 0.0;
        if (tid < 4u) { ss = _smem2[tid]; }
        let final_sq = subgroupAdd(ss);
        let rstd = 1.0 / sqrt(final_sq / f32(HD) + eps);

        // Apply K-norm + RoPE + scatter to cache
        let pos = pos_offset + t;
        let cos_base = pos * half_dim;
        let sin_base = pos * half_dim;
        let kv_stride = _params_[5];   // cache_offset * n_kv * HD (not used, derive from pos)
        // KV cache layout: [T_total × n_kv × HD]
        let cache_pos = pos;
        let k_dst = cache_pos * n_kv * HD + kv_head * HD;
        let v_dst = cache_pos * n_kv * HD + kv_head * HD;

        for (var i = tid; i < HD; i += 128u) {
            let normed = k[i] * rstd * KNormW[i];
            if (i < half_dim) {
                let j = i + half_dim;
                let nj = k[j] * rstd * KNormW[j];
                let c = Cos[cos_base + i];
                let s = Sin[sin_base + i];
                K_cache[k_dst + i] = normed * c - nj * s;
                K_cache[k_dst + j] = nj * c + normed * s;
            } else if (i >= 2u * half_dim) {
                // Non-rotary dimensions: apply norm only
                K_cache[k_dst + i] = normed;
            }
            // V: just copy to cache (no RoPE)
            V_cache[v_dst + i] = QKV[v_src + i];
        }
        }
    }
}
)WGSL";

// [norm] fused_rope
static const char* WGSL_FUSED_ROPE = R"WGSL(
enable f16;
enable subgroups;

// Hand-written fused RoPE kernel for single-token decode.
// Applies RoPE to Q and K heads, scatters K/V into cache.
// No QK-norm is applied (norm weight buffers are identity=1.0).
//
// Grid: (n_head + n_kv, 1, 1)
//   WG x < n_head  → Q head
//   WG x >= n_head  → K head + V scatter
//
// Must declare all 9 bindings to match bind group layout (auto-layout).

@group(0) @binding(0) var<storage, read> buf0: array<f32>;       // QKV
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>; // Q_out
@group(0) @binding(2) var<storage, read_write> buf2: array<f16>; // K_cache
@group(0) @binding(3) var<storage, read_write> buf3: array<f16>; // V_cache
@group(0) @binding(4) var<storage, read> buf4: array<f32>;       // CosTable
@group(0) @binding(5) var<storage, read> buf5: array<f32>;       // SinTable
@group(0) @binding(6) var<storage, read> buf6: array<f32>;       // NormQ (identity)
@group(0) @binding(7) var<storage, read> buf7: array<f32>;       // NormK (identity)

struct Params {
    n_head: i32,
    q_size: i32,
    kv_size: i32,
    pos: i32,
    half_rot: i32,
    cache_offset: i32,
    eps: f32,
};
@group(0) @binding(8) var<storage, read> params: Params;

const HD: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head_idx = i32(wid.x);
    let d = i32(lid.x);  // 0..127 = element within head
    if (u32(d) >= HD) {
        return;
    }

    let is_q = head_idx < params.n_head;
    let kv_head = head_idx - params.n_head;

    // Source offset into QKV buffer
    let q_off = head_idx * i32(HD);
    let k_off = params.q_size + kv_head * i32(HD);
    let src_off = select(k_off, q_off, is_q);

    // Load value
    let val = buf0[u32(src_off + d)];

    // Read norm weight (identity=1.0 for non-QK-norm models)
    let norm_w = select(buf7[u32(d)], buf6[u32(d)], is_q);
    // Apply identity norm weight (no-op multiply by 1.0)
    let normed = val * norm_w;

    // RoPE parameters
    let half_rot = params.half_rot;
    let rot_dim = half_rot * 2;
    let in_rot = d < rot_dim;

    // Cos/sin lookup
    let d_mod = d % half_rot;
    let cos_sin_idx = params.pos * half_rot + d_mod;
    let c = select(f32(1.0), buf4[u32(cos_sin_idx)], in_rot);
    let s = select(f32(0.0), buf5[u32(cos_sin_idx)], in_rot);

    // Pair element for rotation
    let in_first_half = d < half_rot;
    let pair_d = select(d - half_rot, d + half_rot, in_first_half);
    let pair_d_final = select(d, pair_d, in_rot);
    let pair_val = buf0[u32(src_off + pair_d_final)];
    let pair_norm_w = select(buf7[u32(pair_d_final)], buf6[u32(pair_d_final)], is_q);
    let pair_normed = pair_val * pair_norm_w;

    // Apply rotation: first half uses -sin, second half uses +sin
    let sign = select(f32(1.0), f32(-1.0), in_first_half);
    let rotated = select(normed, normed * c + sign * pair_normed * s, in_rot);

    // Write Q output
    if (is_q) {
        buf1[u32(head_idx * i32(HD) + d)] = rotated;
    }

    // Write K/V to cache
    if (!is_q) {
        buf2[u32(params.cache_offset + kv_head * i32(HD) + d)] = f16(rotated);
        // V: straight copy, no RoPE
        let v_off = params.q_size + params.kv_size + kv_head * i32(HD) + d;
        buf3[u32(params.cache_offset + kv_head * i32(HD) + d)] = f16(buf0[u32(v_off)]);
    }
}
)WGSL";

// [norm] gemma_norm_add_batched
static const char* WGSL_GEMMA_NORM_ADD_BATCHED = R"WGSL(
enable subgroups;

// Normalize each addend row, then add it to the residual in place:
//   X += RMSNorm(A, W)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x; let tid = lid.x;
    let N = P[0]; let stride = P[1];
    let eps = bitcast<f32>(P[2]);
    let base = row * stride;
    var ss = 0.0;
    for (var i = tid; i < N; i += 256u) {
        let v = A[base + i]; ss += v * v;
    }
    let lane = tid & 31u; let warp = tid / 32u;
    let ws = subgroupAdd(ss);
    if (lane == 0u) { sums[warp] = ws; }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < 8u; i++) { total += sums[i]; }
    let r = inverseSqrt(total / f32(N) + eps);
    if (tid == 0u) { Rstd[row] = r; }
    for (var i = tid; i < N; i += 256u) {
        let p = base + i;
        X[p] += A[p] * r * W[i];
    }
}
)WGSL";

// [norm] gemma_sandwich_attn_batched
static const char* WGSL_GEMMA_SANDWICH_ATTN_BATCHED = R"WGSL(
enable subgroups;

// Gemma sandwich attention epilogue, one workgroup per prompt row:
//   A = RMSNorm(A, post_attn_weight)
//   X += A
//   Y = RMSNorm(X, ffn_weight)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<storage, read> PostW: array<f32>;
@group(0) @binding(3) var<storage, read> NextW: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> Rstd: array<f32>;
@group(0) @binding(6) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 8>;

fn reduce_sum(v: f32, tid: u32) -> f32 {
    let lane = tid & 31u;
    let warp = tid / 32u;
    let ws = subgroupAdd(v);
    if (lane == 0u) { sums[warp] = ws; }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < 8u; i++) { total += sums[i]; }
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x; let tid = lid.x;
    let N = P[0]; let stride = P[1];
    let eps = bitcast<f32>(P[2]);
    let base = row * stride;

    var ss = 0.0;
    for (var i = tid; i < N; i += 256u) {
        let v = A[base + i]; ss += v * v;
    }
    let ar = inverseSqrt(reduce_sum(ss, tid) / f32(N) + eps);

    ss = 0.0;
    for (var i = tid; i < N; i += 256u) {
        let p = base + i;
        let v = X[p] + A[p] * ar * PostW[i];
        X[p] = v; ss += v * v;
    }
    let xr = inverseSqrt(reduce_sum(ss, tid) / f32(N) + eps);
    if (tid == 0u) { Rstd[row] = xr; }
    for (var i = tid; i < N; i += 256u) {
        let p = base + i;
        Y[p] = X[p] * xr * NextW[i];
    }
}
)WGSL";

// [norm] group_norm — f32 instantiated
static const char* WGSL_GROUP_NORM = R"WGSL(
// GroupNorm — Group Normalization (commonly used in VAE decoders)
// Y[n,c,h,w] = scale[c] * ((X[n,c,h,w] - mean[n,g]) / sqrt(var[n,g] + eps)) + bias[c]
// where g = c / (C / num_groups)
//
// One workgroup per (batch, group) pair. Each workgroup computes mean/var
// over all (channels_per_group * H * W) elements, then normalizes.
// Dispatch: (N * num_groups, 1, 1) where N = batch size
//
// Params: [C, HW, num_groups, eps_as_u32]


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> smem_sum: f32;
var<workgroup> smem_sq: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let num_groups = _params_[2];
    let eps = bitcast<f32>(_params_[3]);

    let group_idx = wid.x;
    let batch = group_idx / num_groups;
    let g = group_idx % num_groups;
    let cpg = C / num_groups;  // channels per group
    let group_size = cpg * HW;

    let tid = lid.x;

    // Parallel reduction for mean
    var local_sum: f32 = 0.0;
    for (var i = tid; i < group_size; i += 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        local_sum += t_read(&X, batch * C * HW + c * HW + hw);
    }

    // Workgroup reduction (simple sequential for now)
    if (tid == 0u) { smem_sum = 0.0; smem_sq = 0.0; }
    workgroupBarrier();
    // Atomic add emulation via loop (WGSL doesn't have atomicAdd on f32)
    // Use sequential fallback: thread 0 does the full reduction
    if (tid == 0u) {
        var s: f32 = 0.0;
        var sq: f32 = 0.0;
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = t_read(&X, batch * C * HW + c * HW + hw);
            s += v;
        }
        let mean = s / f32(group_size);
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = t_read(&X, batch * C * HW + c * HW + hw) - mean;
            sq += v * v;
        }
        let inv_std = 1.0 / sqrt(sq / f32(group_size) + eps);
        smem_sum = mean;
        smem_sq = inv_std;
    }
    workgroupBarrier();

    let mean = smem_sum;
    let inv_std = smem_sq;

    // Normalize + scale + bias
    for (var i = tid; i < group_size; i += 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        let offset = batch * C * HW + c * HW + hw;
        let normed = (t_read(&X, offset) - mean) * inv_std;
        t_write(&Y, offset, normed * t_read(&Scale, c) + t_read(&Bias, c));
    }
}
)WGSL";

// [norm] group_norm — dtype template
static const char* WGSL_GROUP_NORM_T = R"WGSL(
// GroupNorm — Group Normalization (commonly used in VAE decoders)
// Y[n,c,h,w] = scale[c] * ((X[n,c,h,w] - mean[n,g]) / sqrt(var[n,g] + eps)) + bias[c]
// where g = c / (C / num_groups)
//
// One workgroup per (batch, group) pair. Each workgroup computes mean/var
// over all (channels_per_group * H * W) elements, then normalizes.
// Dispatch: (N * num_groups, 1, 1) where N = batch size
//
// Params: [C, HW, num_groups, eps_as_u32]

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> Scale: array<${T}>;
@group(0) @binding(2) var<storage, read> Bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> smem_sum: f32;
var<workgroup> smem_sq: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let num_groups = _params_[2];
    let eps = bitcast<f32>(_params_[3]);

    let group_idx = wid.x;
    let batch = group_idx / num_groups;
    let g = group_idx % num_groups;
    let cpg = C / num_groups;  // channels per group
    let group_size = cpg * HW;

    let tid = lid.x;

    // Parallel reduction for mean
    var local_sum: f32 = 0.0;
    for (var i = tid; i < group_size; i += 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        local_sum += t_read(&X, batch * C * HW + c * HW + hw);
    }

    // Workgroup reduction (simple sequential for now)
    if (tid == 0u) { smem_sum = 0.0; smem_sq = 0.0; }
    workgroupBarrier();
    // Atomic add emulation via loop (WGSL doesn't have atomicAdd on f32)
    // Use sequential fallback: thread 0 does the full reduction
    if (tid == 0u) {
        var s: f32 = 0.0;
        var sq: f32 = 0.0;
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = t_read(&X, batch * C * HW + c * HW + hw);
            s += v;
        }
        let mean = s / f32(group_size);
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = t_read(&X, batch * C * HW + c * HW + hw) - mean;
            sq += v * v;
        }
        let inv_std = 1.0 / sqrt(sq / f32(group_size) + eps);
        smem_sum = mean;
        smem_sq = inv_std;
    }
    workgroupBarrier();

    let mean = smem_sum;
    let inv_std = smem_sq;

    // Normalize + scale + bias
    for (var i = tid; i < group_size; i += 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        let offset = batch * C * HW + c * HW + hw;
        let normed = (t_read(&X, offset) - mean) * inv_std;
        t_write(&Y, offset, normed * t_read(&Scale, c) + t_read(&Bias, c));
    }
}
)WGSL";

// [norm] group_norm_f16wb
static const char* WGSL_GROUP_NORM_F16WB = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f16>;
@group(0) @binding(2) var<storage, read> Bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> smem_sum: f32;
var<workgroup> smem_sq: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let num_groups = _params_[2];
    let eps = bitcast<f32>(_params_[3]);
    let group_idx = wid.x;
    let batch = group_idx / num_groups;
    let g = group_idx % num_groups;
    let cpg = C / num_groups;
    let group_size = cpg * HW;
    let tid = lid.x;
    if (tid == 0u) { smem_sum = 0.0; smem_sq = 0.0; }
    workgroupBarrier();
    if (tid == 0u) {
        var s: f32 = 0.0;
        var sq: f32 = 0.0;
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            s += X[batch * C * HW + c * HW + hw];
        }
        let mean = s / f32(group_size);
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = X[batch * C * HW + c * HW + hw] - mean;
            sq += v * v;
        }
        smem_sum = mean;
        smem_sq = 1.0 / sqrt(sq / f32(group_size) + eps);
    }
    workgroupBarrier();
    let mean = smem_sum;
    let inv_std = smem_sq;
    for (var i = tid; i < group_size; i = i + 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        let offset = batch * C * HW + c * HW + hw;
        let normed = (X[offset] - mean) * inv_std;
        Y[offset] = normed * f32(Scale[c]) + f32(Bias[c]);
    }
}
)WGSL";

// [norm] head_rmsnorm_batched
static const char* WGSL_HEAD_RMSNORM_BATCHED = R"WGSL(
enable subgroups;
// Params [T,n_head,head_dim,eps]. Grid (n_head,T).
@group(0) @binding(0) var<storage,read_write> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<f32>;
@group(0) @binding(2) var<storage,read> P:array<u32>;
var<workgroup>sums:array<f32,8>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
 let T=P[0];let nh=P[1];let hd=P[2];let eps=bitcast<f32>(P[3]);let h=wid.x;let t=wid.y;let d=lid.x;
 if(h>=nh||t>=T){return;}let idx=(t*nh+h)*hd+d;let v=select(0.0,X[idx],d<hd);
 let s=subgroupAdd(v*v);if((d&31u)==0u){sums[d/32u]=s;}workgroupBarrier();
 var total=0.0;for(var j=0u;j<(hd+31u)/32u;j++){total+=sums[j];}let r=inverseSqrt(total/f32(hd)+eps);
 if(d<hd){X[idx]=v*r*W[d];}
}
)WGSL";

// [norm] instance_norm — f32 instantiated
static const char* WGSL_INSTANCE_NORM = R"WGSL(
// Instance Normalization — per-channel per-sample normalization
// X layout: [N, C, H, W]
// Each thread handles one (batch, channel) pair.
//
// Dispatch: (ceil(N*C/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let N = _params_[2];
    let eps = bitcast<f32>(_params_[3]);

    let idx = gid.x;
    if (idx >= N * C) { return; }

    let n = idx / C;
    let c = idx % C;
    let base = n * C * HW + c * HW;

    var mean: f32 = 0.0;
    for (var i = 0u; i < HW; i++) { mean += t_read(&X, base + i); }
    mean /= f32(HW);

    var var_sum: f32 = 0.0;
    for (var i = 0u; i < HW; i++) {
        let d = t_read(&X, base + i) - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(HW) + eps);

    let s = t_read(&Scale, c);
    let b = t_read(&Bias, c);
    for (var i = 0u; i < HW; i++) {
        t_write(&Y, base + i, (t_read(&X, base + i) - mean) * inv_std * s + b);
    }
}
)WGSL";

// [norm] instance_norm — dtype template
static const char* WGSL_INSTANCE_NORM_T = R"WGSL(
// Instance Normalization — per-channel per-sample normalization
// X layout: [N, C, H, W]
// Each thread handles one (batch, channel) pair.
//
// Dispatch: (ceil(N*C/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> Scale: array<${T}>;
@group(0) @binding(2) var<storage, read> Bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let N = _params_[2];
    let eps = bitcast<f32>(_params_[3]);

    let idx = gid.x;
    if (idx >= N * C) { return; }

    let n = idx / C;
    let c = idx % C;
    let base = n * C * HW + c * HW;

    var mean: f32 = 0.0;
    for (var i = 0u; i < HW; i++) { mean += t_read(&X, base + i); }
    mean /= f32(HW);

    var var_sum: f32 = 0.0;
    for (var i = 0u; i < HW; i++) {
        let d = t_read(&X, base + i) - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(HW) + eps);

    let s = t_read(&Scale, c);
    let b = t_read(&Bias, c);
    for (var i = 0u; i < HW; i++) {
        t_write(&Y, base + i, (t_read(&X, base + i) - mean) * inv_std * s + b);
    }
}
)WGSL";

// [norm] instance_norm_f16wb
static const char* WGSL_INSTANCE_NORM_F16WB = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f16>;
@group(0) @binding(2) var<storage, read> Bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let N = _params_[2];
    let eps = bitcast<f32>(_params_[3]);
    let lane = lid.x;
    let idx = wid.x;
    if (idx >= N * C) { return; }
    let n = idx / C;
    let c = idx % C;
    let base = n * C * HW + c * HW;
    var sum: f32 = 0.0;
    for (var i = lane; i < HW; i = i + 256u) { sum += X[base + i]; }
    partial[lane] = sum;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }
    let mean = partial[0] / f32(HW);
    var var_sum: f32 = 0.0;
    for (var i = lane; i < HW; i = i + 256u) {
        let d = X[base + i] - mean;
        var_sum += d * d;
    }
    partial[lane] = var_sum;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }
    let inv_std = 1.0 / sqrt(partial[0] / f32(HW) + eps);
    let s = f32(Scale[c]);
    let b = f32(Bias[c]);
    for (var i = lane; i < HW; i = i + 256u) {
        Y[base + i] = (X[base + i] - mean) * inv_std * s + b;
    }
}
)WGSL";

// [norm] layer_norm — f32 instantiated
static const char* WGSL_LAYER_NORM = R"WGSL(
// LayerNormalization — mean + variance normalization
// Y = (X - mean) / sqrt(var + eps) * W + B
// Dispatch: (ceil(nRows/256), 1, 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);

    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;

    var mean: f32 = 0.0;
    for (var i = 0u; i < N; i++) { mean += t_read(&X, base + i); }
    mean = mean / f32(N);

    var var_sum: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let d = t_read(&X, base + i) - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(N) + eps);

    for (var i = 0u; i < N; i++) {
        t_write(&Y, base + i, (t_read(&X, base + i) - mean) * inv_std * t_read(&W, i) + t_read(&B, i));
    }
}
)WGSL";

// [norm] layer_norm — dtype template
static const char* WGSL_LAYER_NORM_T = R"WGSL(
// LayerNormalization — mean + variance normalization
// Y = (X - mean) / sqrt(var + eps) * W + B
// Dispatch: (ceil(nRows/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<${T}>;
@group(0) @binding(2) var<storage, read> B: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);

    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;

    var mean: f32 = 0.0;
    for (var i = 0u; i < N; i++) { mean += t_read(&X, base + i); }
    mean = mean / f32(N);

    var var_sum: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let d = t_read(&X, base + i) - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(N) + eps);

    for (var i = 0u; i < N; i++) {
        t_write(&Y, base + i, (t_read(&X, base + i) - mean) * inv_std * t_read(&W, i) + t_read(&B, i));
    }
}
)WGSL";

// [norm] layer_norm_f16wb
static const char* WGSL_LAYER_NORM_F16WB = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f16>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;
    var mean: f32 = 0.0;
    for (var i = 0u; i < N; i++) { mean += X[base + i]; }
    mean = mean / f32(N);
    var var_sum: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let d = X[base + i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(N) + eps);
    for (var i = 0u; i < N; i++) {
        Y[base + i] = (X[base + i] - mean) * inv_std * f32(W[i]) + f32(B[i]);
    }
}
)WGSL";

// [norm] norm_then_add
static const char* WGSL_NORM_THEN_ADD = R"WGSL(
enable subgroups;

// Fused norm-then-add: RMSNorm(A, W) → X += result
// Replaces 2 dispatches: post_ffn_norm + ffn_residual_add
// Grid: (1, 1, 1) — single workgroup for single-row decode
// WG: 128 threads (4 warps)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> A: array<f32>;       // projOutBuf (FFN output)
@group(0) @binding(1) var<storage, read_write> X: array<f32>; // xBuf (residual, updated in-place)
@group(0) @binding(2) var<storage, read> W: array<f32>;       // post_ffn_norm weight
@group(0) @binding(3) var<storage, read> params: array<u32>;  // [N, 0, eps_as_u32, 0]

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let N = params[0];
    let eps = bitcast<f32>(params[2]);
    let warp_id = tid / 32u;
    let lane_id = tid % 32u;

    // Pass 1: Compute RMSNorm of A
    var sum_sq: f32 = 0.0;
    for (var idx = tid; idx < N; idx += 128u) {
        let v = A[idx];
        sum_sq += v * v;
    }

    let warp_sum = subgroupAdd(sum_sq);
    if (lane_id == 0u) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    let final_sum = bitcast<f32>(_smem[0]) + bitcast<f32>(_smem[1])
                  + bitcast<f32>(_smem[2]) + bitcast<f32>(_smem[3]);
    let rstd = 1.0 / sqrt(final_sum / f32(N) + eps);

    // Pass 2: Apply norm weight and add to X
    for (var idx = tid; idx < N; idx += 128u) {
        X[idx] = X[idx] + A[idx] * rstd * W[idx];
    }
}
)WGSL";

// [norm] ple_norm_add_scale
static const char* WGSL_PLE_NORM_ADD_SCALE = R"WGSL(
enable subgroups;

// Fused PLE norm + add + optional scale:
//   X = (X + RMSNorm(A, W)) * scale
// Replaces 2-3 dispatches: ple_norm + ple_add + layer_scalar
// Grid: (1, 1, 1) — single workgroup for single-row decode
// WG: 128 threads (4 warps)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> A: array<f32>;       // projOutBuf (PLE projection output)
@group(0) @binding(1) var<storage, read_write> X: array<f32>; // xBuf (residual, updated in-place)
@group(0) @binding(2) var<storage, read> W: array<f32>;       // pleNormW
@group(0) @binding(3) var<storage, read> params: array<u32>;  // [N, scale_as_u32, eps_as_u32, 0]

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let N = params[0];
    let scale = bitcast<f32>(params[1]);
    let eps = bitcast<f32>(params[2]);
    let warp_id = tid / 32u;
    let lane_id = tid % 32u;

    // Pass 1: Compute RMSNorm of A
    var sum_sq: f32 = 0.0;
    for (var idx = tid; idx < N; idx += 128u) {
        let v = A[idx];
        sum_sq += v * v;
    }

    let warp_sum = subgroupAdd(sum_sq);
    if (lane_id == 0u) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    let final_sum = bitcast<f32>(_smem[0]) + bitcast<f32>(_smem[1])
                  + bitcast<f32>(_smem[2]) + bitcast<f32>(_smem[3]);
    let rstd = 1.0 / sqrt(final_sum / f32(N) + eps);

    // Pass 2: norm + add + scale
    for (var idx = tid; idx < N; idx += 128u) {
        let normed = A[idx] * rstd * W[idx];
        X[idx] = (X[idx] + normed) * scale;
    }
}
)WGSL";

// [norm] rms_norm
static const char* WGSL_RMS_NORM = R"WGSL(
enable subgroups;

// Generated WGSL kernel
// Kernel: rms_norm_loop_kernel
// Workgroup size: 128 (4 warps x 32 threads)

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // X
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Y
@group(0) @binding(2) var<storage, read> buf2: array<f32>;  // W
@group(0) @binding(3) var<storage, read_write> buf3: array<f32>;  // Rstd

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
    @builtin(num_workgroups) _num_wg: vec3<u32>,
) {
    let v10: i32 = i32(_wg_id.x);
    let v11: i32 = params.N + 127;
    let v12: i32 = v11 / 128;
    var v14: i32 = 0;
    var v15_0: f32 = f32(0);
    loop {
        let v16: bool = v14 < v12;
        if !v16 { break; }
        let v18: i32 = v14 * 128;
        let v19: i32 = i32(_lid.x);
        let v20: i32 = v19 & 127;
        let v21: i32 = i32(u32(v20) % u32(32));
        let v22: i32 = i32(_lid.x);
        let v23: i32 = i32(u32(v22) / u32(32));
        let v24: i32 = v21 << u32(0);
        let v25: i32 = 0 | v24;
        let v26: i32 = v23 << u32(5);
        let v27: i32 = v25 | v26;
        let v28: i32 = v27 & 127;
        let v29: i32 = i32(u32(v28) >> u32(0));
        let v30: i32 = v29 | 0;
        let v31: i32 = 0 ^ v30;
        let v32: i32 = v31 ^ 0;
        let v33: i32 = v32 + 0;
        let v34: i32 = v18 + v33;
        let v35: bool = v34 < params.N;
        let v36: i32 = v10 * params.stride;
        let v39: f32 = buf0[u32((v36 + v34))];
        let v40: f32 = select(f32(0.0), v39, v35);
        let v41: f32 = v40 * v40;
        let v43: f32 = v15_0 + v41;
        let v45: i32 = v14 + 1;
        v14 = v45;
        v15_0 = v43;
    }
    let v48: i32 = bitcast<i32>(v15_0);
    let v49: i32 = subgroupShuffleXor(v48, u32(16));
    let v50: f32 = bitcast<f32>(v49);
    let v51: f32 = v15_0 + v50;
    let v52: i32 = bitcast<i32>(v51);
    let v53: i32 = subgroupShuffleXor(v52, u32(8));
    let v54: f32 = bitcast<f32>(v53);
    let v55: f32 = v51 + v54;
    let v56: i32 = bitcast<i32>(v55);
    let v57: i32 = subgroupShuffleXor(v56, u32(4));
    let v58: f32 = bitcast<f32>(v57);
    let v59: f32 = v55 + v58;
    let v60: i32 = bitcast<i32>(v59);
    let v61: i32 = subgroupShuffleXor(v60, u32(2));
    let v62: f32 = bitcast<f32>(v61);
    let v63: f32 = v59 + v62;
    let v64: i32 = bitcast<i32>(v63);
    let v65: i32 = subgroupShuffleXor(v64, u32(1));
    let v66: f32 = bitcast<f32>(v65);
    let v67: f32 = v63 + v66;
    let v68: i32 = i32(_lid.x);
    let v69: i32 = v68 & 127;
    let v70: i32 = i32(u32(v69) % u32(32));
    let v71: i32 = i32(_lid.x);
    let v72: i32 = i32(u32(v71) / u32(32));
    let v73: i32 = v70 << u32(0);
    let v74: i32 = 0 | v73;
    let v75: i32 = v72 << u32(5);
    let v76: i32 = v74 | v75;
    let v77: i32 = v76 & 96;
    let v78: i32 = i32(u32(v77) >> u32(3));
    let v79: i32 = 0 | v78;
    let v80: i32 = 0 ^ v79;
    let v81: i32 = v80 ^ 0;
    let v82: i32 = v81 ^ 0;
    let v83: i32 = v82 + 0;
    _smem[u32(v83) >> 2u] = bitcast<i32>(v67);
    workgroupBarrier();
    let v87: i32 = i32(_lid.x);
    let v88: i32 = v87 & 127;
    let v89: i32 = i32(u32(v88) % u32(32));
    let v90: i32 = i32(_lid.x);
    let v91: i32 = i32(u32(v90) / u32(32));
    let v92: i32 = v89 << u32(0);
    let v93: i32 = 0 | v92;
    let v94: i32 = v91 << u32(5);
    let v95: i32 = v93 | v94;
    let v96: i32 = v95 & 3;
    let v97: i32 = v96 << u32(2);
    let v98: i32 = v97 | 0;
    let v99: i32 = 0 ^ v98;
    let v100: i32 = v99 ^ 0;
    let v101: i32 = v100 ^ 0;
    let v102: i32 = v101 + 0;
    let v104: f32 = bitcast<f32>(_smem[u32(v102) >> 2u]);
    let v107: i32 = bitcast<i32>(v104);
    let v108: i32 = subgroupShuffleXor(v107, u32(2));
    let v109: f32 = bitcast<f32>(v108);
    let v110: f32 = v104 + v109;
    let v111: i32 = bitcast<i32>(v110);
    let v112: i32 = subgroupShuffleXor(v111, u32(1));
    let v113: f32 = bitcast<f32>(v112);
    let v114: f32 = v110 + v113;
    let v115: f32 = f32(params.N);
    let v116: f32 = v114 / v115;
    let v117: f32 = v116 + params.eps;
    let v118: f32 = sqrt(v117);
    let v119: f32 = f32(1.0) / v118;
    var v121: i32 = 0;
    loop {
        let v122: bool = v121 < v12;
        if !v122 { break; }
        let v124: i32 = v121 * 128;
        let v125: i32 = i32(_lid.x);
        let v126: i32 = v125 & 127;
        let v127: i32 = i32(u32(v126) % u32(32));
        let v128: i32 = i32(_lid.x);
        let v129: i32 = i32(u32(v128) / u32(32));
        let v130: i32 = v127 << u32(0);
        let v131: i32 = 0 | v130;
        let v132: i32 = v129 << u32(5);
        let v133: i32 = v131 | v132;
        let v134: i32 = v133 & 127;
        let v135: i32 = i32(u32(v134) >> u32(0));
        let v136: i32 = v135 | 0;
        let v137: i32 = 0 ^ v136;
        let v138: i32 = v137 ^ 0;
        let v139: i32 = v138 + 0;
        let v140: i32 = v124 + v139;
        let v141: bool = v140 < params.N;
        let v142: i32 = v10 * params.stride;
        let v145: f32 = buf0[u32((v142 + v140))];
        let v146: f32 = select(f32(0.0), v145, v141);
        let v148: f32 = buf2[u32(v140)];
        let v149: f32 = select(f32(1.0), v148, v141);
        let v152: f32 = v146 * v119;
        let v153: f32 = v152 * v149;
        let v154: f32 = select(f32(0), v153, v141);
        if v141 { buf1[u32((v142 + v140))] = v154; }
        let v155: i32 = v121 + 1;
        v121 = v155;
    }
    if _lid.x == 0u { buf3[u32(v10)] = v119; }
}
)WGSL";

// [norm] rms_norm_batched
static const char* WGSL_RMS_NORM_BATCHED = R"WGSL(
enable subgroups;

// Batched RMSNorm for T rows: Y[t] = X[t] * W / rms(X[t]) for each row t.
// Grid: (T, 1, 1) — one workgroup per row.
// WG: 128 threads.

var<workgroup> _smem: array<i32, 4>;

@group(0) @binding(0) var<storage, read> buf0: array<f32>;  // X [T × N]
@group(0) @binding(1) var<storage, read_write> buf1: array<f32>;  // Y [T × N]
@group(0) @binding(2) var<storage, read> buf2: array<f32>;  // W [N]
@group(0) @binding(3) var<storage, read_write> buf3: array<f32>;  // Rstd [T]

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

const MAX_N: u32 = 8192u;  // supports up to E=8192

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) _wg_id: vec3<u32>,
    @builtin(local_invocation_id) _lid: vec3<u32>,
) {
    let row = i32(_wg_id.x);
    let tid = i32(_lid.x);
    let N = params.N;
    let stride = params.stride;
    let base = row * stride;

    var sum_sq: f32 = 0.0;
    var idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let v = buf0[u32(base + idx)];
            sum_sq += v * v;
        }
    }

    // Warp reduce
    let warp_sum = subgroupAdd(sum_sq);

    // Cross-warp reduce via shared memory
    let warp_id = tid / 32;
    let lane_id = tid % 32;
    if (lane_id == 0) {
        _smem[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    var final_sum: f32 = 0.0;
    for (var w = 0; w < 4; w++) {
        final_sum += bitcast<f32>(_smem[w]);
    }

    let rstd = 1.0 / sqrt(final_sum / f32(N) + params.eps);
    if (tid == 0) {
        buf3[u32(row)] = rstd;
    }

    // Apply norm + weight
    idx = tid;
    for (; idx < i32(MAX_N); idx += 128) {
        if (idx < N) {
            let v = buf0[u32(base + idx)];
            buf1[u32(base + idx)] = v * rstd * buf2[u32(idx)];
        }
    }
}
)WGSL";

// [norm] rms_norm_qwen_vulkan
static const char* WGSL_RMS_NORM_QWEN_VULKAN = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

var<workgroup> warp_sums: array<f32, 8>;
var<workgroup> rstd_shared: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let N = u32(params.N);
    let base = row * u32(params.stride);

    var sum_sq = 0.0;
    for (var j = tid; j < N; j = j + 256u) {
        let v = X[base + j];
        sum_sq += v * v;
    }

    let warp_sum = subgroupAdd(sum_sq);
    if (lane == 0u) {
        warp_sums[warp] = warp_sum;
    }
    workgroupBarrier();

    let cross = select(0.0, warp_sums[min(lane, 7u)], warp == 0u && lane < 8u);
    let total = subgroupAdd(cross);
    if (tid == 0u) {
        rstd_shared = 1.0 / sqrt(total / f32(N) + params.eps);
        Rstd[row] = rstd_shared;
    }
    workgroupBarrier();

    let rstd = rstd_shared;
    for (var j = tid; j < N; j = j + 256u) {
        Y[base + j] = X[base + j] * rstd * W[j];
    }
}
)WGSL";

// [norm] rmsnorm — f32 instantiated
static const char* WGSL_RMSNORM = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params { stride: i32, N: i32, eps: f32, };
@group(0) @binding(4) var<storage, read> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let N = u32(params.N);
    if (row >= u32(params.stride) / N) { return; }
    let base = row * N;
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) { let v = t_read(&X, base + i); sum_sq += v * v; }
    let rms = sqrt(sum_sq / f32(N) + params.eps);
    let inv_rms = 1.0 / rms;
    // Write pairs to avoid fp16 u32 write races
    let pairs = N / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) * inv_rms * t_read(&W, i0);
        let v1 = t_read(&X, base + i0 + 1u) * inv_rms * t_read(&W, i0 + 1u);
        t_write2(&Y, base + i0, v0, v1);
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        t_write2(&Y, base + last, t_read(&X, base + last) * inv_rms * t_read(&W, last), 0.0);
    }
    if (gid.x == 0u) { Rstd[row] = inv_rms; }
}
)WGSL";

// [norm] rmsnorm — dtype template
static const char* WGSL_RMSNORM_T = R"WGSL(
${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> W: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params { stride: i32, N: i32, eps: f32, };
@group(0) @binding(4) var<storage, read> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let N = u32(params.N);
    if (row >= u32(params.stride) / N) { return; }
    let base = row * N;
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) { let v = t_read(&X, base + i); sum_sq += v * v; }
    let rms = sqrt(sum_sq / f32(N) + params.eps);
    let inv_rms = 1.0 / rms;
    // Write pairs to avoid fp16 u32 write races
    let pairs = N / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) * inv_rms * t_read(&W, i0);
        let v1 = t_read(&X, base + i0 + 1u) * inv_rms * t_read(&W, i0 + 1u);
        t_write2(&Y, base + i0, v0, v1);
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        t_write2(&Y, base + last, t_read(&X, base + last) * inv_rms * t_read(&W, last), 0.0);
    }
    if (gid.x == 0u) { Rstd[row] = inv_rms; }
}
)WGSL";

// [norm] rmsnorm_simple_f16w
static const char* WGSL_RMSNORM_SIMPLE_F16W = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params { stride: i32, N: i32, eps: f32, };
@group(0) @binding(4) var<storage, read> params: Params;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.x;
    let N = u32(params.N);
    let nRows = u32(params.stride) / N;
    let base = row * N;
    let tid = lid.x;

    // Phase 1: parallel sum_sq reduction
    var local_sum: f32 = 0.0;
    if (row < nRows) {
        for (var i = tid; i < N; i += 256u) {
            let v = X[base + i];
            local_sum += v * v;
        }
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    if (row >= nRows) { return; }

    let inv_rms = 1.0 / sqrt(shared_sum[0] / f32(N) + params.eps);

    // Phase 2: parallel normalize + scale
    for (var i = tid; i < N; i += 256u) {
        Y[base + i] = X[base + i] * inv_rms * f32(W[i]);
    }
    if (tid == 0u) { Rstd[row] = inv_rms; }
}
)WGSL";

// [norm] sandwich_norm_add_norm
static const char* WGSL_SANDWICH_NORM_ADD_NORM = R"WGSL(
enable subgroups;

// Fused sandwich norm: RMSNorm(A, W1) → add to X → RMSNorm(X, W2) → Y
// Replaces 3 dispatches: post_attn_norm + attn_residual_add + pre_ffn_norm
// Grid: (1, 1, 1) — single workgroup for single-row decode
// WG: 128 threads (4 warps)
//
// Uses shared memory to cache updated X for pass 2, avoiding
// storage buffer re-reads on D3D12/Dawn.
// Pass 1 uses subgroupAdd; Pass 2 uses tree reduction (avoids
// D3D12 issue with multiple subgroupAdd calls in one kernel).

const STRIDE: u32 = 128u;

var<workgroup> _smem: array<f32, 4096>; // dual-use: X cache + tree reduction

@group(0) @binding(0) var<storage, read> A: array<f32>;       // projOutBuf
@group(0) @binding(1) var<storage, read_write> X: array<f32>; // xBuf
@group(0) @binding(2) var<storage, read_write> Y: array<f32>; // normOutBuf
@group(0) @binding(3) var<storage, read> W1: array<f32>;      // post_attn_norm weight
@group(0) @binding(4) var<storage, read> W2: array<f32>;      // pre_ffn_norm weight
@group(0) @binding(5) var<storage, read> params: array<u32>;  // [N, 0, eps_as_u32, 0]

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let N = params[0];
    let eps = bitcast<f32>(params[2]);
    let warp_id = tid / 32u;
    let lane_id = tid % 32u;

    // ── Pass 1: Compute rstd1 from A ──
    var sum_sq1: f32 = 0.0;
    for (var idx = tid; idx < N; idx += STRIDE) {
        let v = A[idx];
        sum_sq1 += v * v;
    }
    let ws1 = subgroupAdd(sum_sq1);
    if (lane_id == 0u) { _smem[tid] = ws1; }
    workgroupBarrier();
    let fs1 = _smem[0] + _smem[32] + _smem[64] + _smem[96];
    let rstd1 = 1.0 / sqrt(fs1 / f32(N) + eps);

    // ── Fused: norm(A)*W1 + X → write to X and cache in smem, accumulate pass-2 sum ──
    var sum_sq2: f32 = 0.0;
    for (var idx = tid; idx < N; idx += STRIDE) {
        let new_x = X[idx] + A[idx] * rstd1 * W1[idx];
        X[idx] = new_x;       // update storage
        _smem[idx] = new_x;   // cache in shared memory
        sum_sq2 += new_x * new_x;
    }
    workgroupBarrier();

    // ── Pass 2: tree reduction for rstd2 ──
    // Use smem slots [4080..4095+] for tree reduction (above X cache)
    let red_base = 3968u; // 4096 - 128
    _smem[red_base + tid] = sum_sq2;
    workgroupBarrier();
    if (tid < 64u) { _smem[red_base + tid] += _smem[red_base + tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { _smem[red_base + tid] += _smem[red_base + tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { _smem[red_base + tid] += _smem[red_base + tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { _smem[red_base + tid] += _smem[red_base + tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { _smem[red_base + tid] += _smem[red_base + tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { _smem[red_base + tid] += _smem[red_base + tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { _smem[red_base] += _smem[red_base + 1u]; }
    workgroupBarrier();

    let rstd2 = 1.0 / sqrt(_smem[red_base] / f32(N) + eps);

    // ── Write Y from cached smem ──
    for (var idx = tid; idx < N; idx += STRIDE) {
        Y[idx] = _smem[idx] * rstd2 * W2[idx];
    }
}
)WGSL";

// [norm] skip_rmsnorm — f32 instantiated
static const char* WGSL_SKIP_RMSNORM = R"WGSL(
// SkipSimplifiedLayerNormalization — residual add + RMSNorm
// Computes: SkipOut = X + Skip, Y = RMSNorm(SkipOut) * W
// Dispatch: (ceil(nRows/256), 1, 1)
// Params: [0]=N (hidden dim), [1]=nRows, [2]=eps (bitcast<f32>)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Skip: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;

    // Pass 1: Residual add + sum of squares, write SkipOut in pairs
    var sum_sq: f32 = 0.0;
    let pairs = N / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) + t_read(&Skip, base + i0);
        let v1 = t_read(&X, base + i0 + 1u) + t_read(&Skip, base + i0 + 1u);
        sum_sq += v0 * v0 + v1 * v1;
        t_write2(&SkipOut, base + i0, v0, v1);
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        let v = t_read(&X, base + last) + t_read(&Skip, base + last);
        sum_sq += v * v;
        t_write2(&SkipOut, base + last, v, 0.0);
    }
    let inv_rms = 1.0 / sqrt(sum_sq / f32(N) + eps);

    // Pass 2: Recompute residual + apply norm weight, paired writes
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) + t_read(&Skip, base + i0);
        let v1 = t_read(&X, base + i0 + 1u) + t_read(&Skip, base + i0 + 1u);
        t_write2(&Y, base + i0, v0 * inv_rms * t_read(&W, i0), v1 * inv_rms * t_read(&W, i0 + 1u));
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        let v = t_read(&X, base + last) + t_read(&Skip, base + last);
        t_write2(&Y, base + last, v * inv_rms * t_read(&W, last), 0.0);
    }
}
)WGSL";

// [norm] skip_rmsnorm — dtype template
static const char* WGSL_SKIP_RMSNORM_T = R"WGSL(
// SkipSimplifiedLayerNormalization — residual add + RMSNorm
// Computes: SkipOut = X + Skip, Y = RMSNorm(SkipOut) * W
// Dispatch: (ceil(nRows/256), 1, 1)
// Params: [0]=N (hidden dim), [1]=nRows, [2]=eps (bitcast<f32>)

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> Skip: array<${T}>;
@group(0) @binding(2) var<storage, read> W: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<${T}>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;

    // Pass 1: Residual add + sum of squares, write SkipOut in pairs
    var sum_sq: f32 = 0.0;
    let pairs = N / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) + t_read(&Skip, base + i0);
        let v1 = t_read(&X, base + i0 + 1u) + t_read(&Skip, base + i0 + 1u);
        sum_sq += v0 * v0 + v1 * v1;
        t_write2(&SkipOut, base + i0, v0, v1);
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        let v = t_read(&X, base + last) + t_read(&Skip, base + last);
        sum_sq += v * v;
        t_write2(&SkipOut, base + last, v, 0.0);
    }
    let inv_rms = 1.0 / sqrt(sum_sq / f32(N) + eps);

    // Pass 2: Recompute residual + apply norm weight, paired writes
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) + t_read(&Skip, base + i0);
        let v1 = t_read(&X, base + i0 + 1u) + t_read(&Skip, base + i0 + 1u);
        t_write2(&Y, base + i0, v0 * inv_rms * t_read(&W, i0), v1 * inv_rms * t_read(&W, i0 + 1u));
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        let v = t_read(&X, base + last) + t_read(&Skip, base + last);
        t_write2(&Y, base + last, v * inv_rms * t_read(&W, last), 0.0);
    }
}
)WGSL";

// [norm] skip_rmsnorm_f16w
static const char* WGSL_SKIP_RMSNORM_F16W_VEC4 = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Skip: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;
var<workgroup> sum_sq_shared: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = wid.x;
    if (row >= nRows) { return; }
    let N4 = N / 4u;
    let base = row * N4;
    var sum_sq: f32 = 0.0;
    for (var i = lid.x; i < N4; i += 128u) {
        let v = X[base + i] + Skip[base + i];
        SkipOut[base + i] = v;
        sum_sq += dot(v, v);
    }
    sum_sq_shared[lid.x] = sum_sq;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            sum_sq_shared[lid.x] += sum_sq_shared[lid.x + stride];
        }
        workgroupBarrier();
    }
    let inv_rms = inverseSqrt(sum_sq_shared[0] / f32(N) + eps);
    for (var i = lid.x; i < N4; i += 128u) {
        Y[base + i] = SkipOut[base + i] * inv_rms * vec4<f32>(W[i]);
    }
}
)WGSL";

static const char* WGSL_SKIP_RMSNORM_F16W = R"WGSL(
enable f16;
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Skip: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;
var<workgroup> sum_sq_shared: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let N = _params_[0]; let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]); let row = wid.x;
    if (row >= nRows) { return; }
    let base = row * N; var sum_sq: f32 = 0.0;
    for (var i = lid.x; i < N; i += 128u) {
        let v = X[base + i] + Skip[base + i];
        SkipOut[base + i] = v; sum_sq += v * v;
    }
    sum_sq_shared[lid.x] = sum_sq; workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) { sum_sq_shared[lid.x] += sum_sq_shared[lid.x + stride]; }
        workgroupBarrier();
    }
    let inv_rms = inverseSqrt(sum_sq_shared[0] / f32(N) + eps);
    for (var i = lid.x; i < N; i += 128u) {
        Y[base + i] = SkipOut[base + i] * inv_rms * f32(W[i]);
    }
}
)WGSL";

// [norm] skip_rmsnorm_f16w_serial
static const char* WGSL_SKIP_RMSNORM_F16W_SERIAL = R"WGSL(
enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Skip: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let v = X[base + i] + Skip[base + i];
        SkipOut[base + i] = v;
        sum_sq += v * v;
    }
    let inv_rms = inverseSqrt(sum_sq / f32(N) + eps);
    for (var i = 0u; i < N; i++) {
        Y[base + i] = SkipOut[base + i] * inv_rms * f32(W[i]);
    }
}
)WGSL";

// ─── [onnx_q4] ─────────────────────────────────────────────────────

// [onnx_q4] gather_bq_q4
static const char* WGSL_GATHER_BQ_Q4 = R"WGSL(
// GatherBlockQuantized Q4 — ONNX Q4 embedding lookup
// Dequantizes Q4 packed weights at gathered indices.
// Weight: [V, K] where each element is 4 bits (2 per byte)
// Scale: [V, n_groups] fp16 packed
//
// Dispatch: (ceil(K/256), nIndices, 1)

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let K = _params_[1];
    let n_groups = _params_[2];
    let bs = _params_[3];
    let idx_i = gid.y;
    let k = gid.x;
    if (idx_i >= nIdx || k >= K) { return; }
    let vocab_idx = u32(Indices[idx_i]);
    let group = k / bs;
    let scale_flat = vocab_idx * n_groups + group;
    let scale_u32 = Scales[scale_flat / 2u];
    let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
    let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;
    let byte_flat = vocab_idx * (K / 2u) + k / 2u;
    let byte_u32 = W[byte_flat / 4u];
    let byte_val = (byte_u32 >> ((byte_flat % 4u) * 8u)) & 0xFFu;
    let nibble = select(byte_val & 0x0Fu, (byte_val >> 4u) & 0x0Fu, (k & 1u) != 0u);
    // UINT4 with default zero_point=8: dequant = (nibble - 8) * scale
    let centered = f32(i32(nibble)) - 8.0;
    Y[idx_i * K + k] = centered * scale;
}
)WGSL";

// [onnx_q4] gather_bq_q4_zp
static const char* WGSL_GATHER_BQ_Q4_ZP = R"WGSL(
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let K = _params_[1];
    let n_groups = _params_[2];
    let bs = _params_[3];
    let idx_i = gid.y;
    let k = gid.x;
    if (idx_i >= nIdx || k >= K) { return; }
    let vocab_idx = u32(Indices[idx_i]);
    let group = k / bs;
    // Scale
    let scale_flat = vocab_idx * n_groups + group;
    let scale_u32 = Scales[scale_flat / 2u];
    let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
    let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;
    // Zero point (packed nibbles like scales)
    let zp_byte_idx = scale_flat / 2u;
    let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
    let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));
    // Weight nibble
    let byte_flat = vocab_idx * (K / 2u) + k / 2u;
    let byte_u32 = W[byte_flat / 4u];
    let byte_val = (byte_u32 >> ((byte_flat % 4u) * 8u)) & 0xFFu;
    let nibble = select(byte_val & 0x0Fu, (byte_val >> 4u) & 0x0Fu, (k & 1u) != 0u);
    let centered = f32(i32(nibble)) - zp;
    Y[idx_i * K + k] = centered * scale;
}
)WGSL";

// [onnx_q4] gather_bq_q8
static const char* WGSL_GATHER_BQ_Q8 = R"WGSL(
// GatherBlockQuantized Q8 — ONNX Q8 embedding lookup
// Weight: [V, n_groups, bs] uint8 with zero_point=128
// Scale: [V, n_groups] fp16 packed
//
// Dispatch: (ceil(K/256), nIndices, 1)

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let K = _params_[1];
    let n_groups = _params_[2];
    let bs = _params_[3];
    let idx_i = gid.y;
    let k = gid.x;
    if (idx_i >= nIdx || k >= K) { return; }
    let vocab_idx = u32(Indices[idx_i]);
    let group = k / bs;
    let scale_flat = vocab_idx * n_groups + group;
    let scale_u32 = Scales[scale_flat / 2u];
    let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
    let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;
    let byte_flat = vocab_idx * n_groups * bs + k;
    let byte_u32 = W[byte_flat / 4u];
    let byte_val = (byte_u32 >> ((byte_flat % 4u) * 8u)) & 0xFFu;
    let centered = f32(i32(byte_val) - 128);
    Y[idx_i * K + k] = centered * scale;
}
)WGSL";

// [onnx_q4] matmul_q4
static const char* WGSL_MATMUL_Q4 = R"WGSL(
// MatMulNBits Q4 — simple per-element kernel
// Y[m,n] = sum_k X[m,k] * dequant(W[n,k])
// Weight: W[N, K/2] packed uint8 (2 Q4 values per byte)
// Scale: [N * blocks_per_col] fp16 packed into u32
// block_size=32, blocks_per_col = K/32
// Dispatch: (ceil(N/8), M, 1)

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let n = gid.x;
    let m = gid.y;
    if (n >= N || m >= M) { return; }

    var acc: f32 = 0.0;
    let a_base = m * K;
    let w_base = n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        for (var j = 0u; j < 16u; j++) {
            let byte_idx = w_blk_base + j;
            let byte_u32 = B[byte_idx / 4u];
            let byte_val = (byte_u32 >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            let lo = f32(byte_val & 0xFu) - 8.0;
            let hi = f32((byte_val >> 4u) & 0xFu) - 8.0;
            acc += A[a_base + k_base + j * 2u] * lo * scale;
            acc += A[a_base + k_base + j * 2u + 1u] * hi * scale;
        }
    }

    Y[m * N + n] = acc;
}
)WGSL";

// [onnx_q4] matmul_q4_batched
static const char* WGSL_ORT_DP4A_QUANTIZE_EXACT = R"WGSL(
requires packed_4x8_integer_dot_product;
enable f16;
enable subgroups;
const workgroup_size_x: u32 = 64;
const workgroup_size_y: u32 = 1;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f16>;
struct Uniforms {
  output_size: u32
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

alias input_a_value_t = vec4<f16>;
alias input_a_indices_t = vec3<u32>;
alias input_a_element_t = f16;

var<workgroup> a_values : array<array<input_a_value_t, 32>, 2>;
var<workgroup> max_values : array<input_a_value_t, 4>;

fn readInput(offset: u32) -> input_a_value_t
{
  if (offset >= uniforms.output_size) {
    return input_a_value_t(0);
  }
  return input_a[offset];
}


@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;

  if (sg_size == 32) {
    let local_a = readInput(global_idx);
    let max_val = subgroupMax(abs(local_a));
    if (global_idx >= uniforms.output_size) {
      return;
    }
    let max_temp = max(max_val.xy, max_val.zw);
    let scale = max(max_temp[0], max_temp[1]);
    let norm_a = local_a/scale;
    output[global_idx]=pack4x8snorm(vec4<f32>(norm_a));;
    if (local_idx % 32 == 0)
    {

      scales[workgroup_idx * 2 + local_idx / 32]=scale/127;;
    }
  } else if (sg_size == 16) {
    let local_a = readInput(global_idx);
    let sub_max_value = subgroupMax(abs(local_a));
    if (local_idx % 16 == 0) {
      max_values[local_idx / 16] = sub_max_value;
    }
    workgroupBarrier();

    if (global_idx >= uniforms.output_size) {
      return;
    }

    var max_val = input_a_value_t(0);
    if (local_idx < 32) {
      max_val = max(max_values[0], max_values[1]);
    } else {
      max_val = max(max_values[2], max_values[3]);
    }
    let max_temp = max(max_val.xy, max_val.zw);
    let scale = max(max_temp[0], max_temp[1]);
    let norm_a = local_a/scale;
    output[global_idx]=pack4x8snorm(vec4<f32>(norm_a));;
    if (local_idx % 32 == 0)
    {

      scales[workgroup_idx * 2 + local_idx / 32]=scale/127;;
    }
  } else {
    let local_row = local_idx / 32u;
    let local_col = local_idx % 32u;
    a_values[local_row][local_col] = readInput(global_idx);
    workgroupBarrier();

    if (global_idx >= uniforms.output_size) {
      return;
    }

    var max_val = input_a_value_t(0);

    for (var i = 0u; i < 32u; i++)
    {
      max_val = max(max_val, abs(a_values[local_row][i]));
    }
    let max_temp = max(max_val.xy, max_val.zw);
    let scale = max(max_temp[0], max_temp[1]);
    let norm_a = a_values[local_row][local_col]/scale;
    output[global_idx]=pack4x8snorm(vec4<f32>(norm_a));;
    if (local_col == 0u)
    {

      scales[workgroup_idx * 2 + local_row]=scale/127;;
    }
  }

}
)WGSL";

static const char* WGSL_ORT_DP4A_MATMUL_EXACT = R"WGSL(
requires packed_4x8_integer_dot_product;
enable f16;
enable subgroups;
const workgroup_size_x: u32 = 256;
const workgroup_size_y: u32 = 1;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> input_a: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales_a: array<f16>;
@group(0) @binding(2) var<storage, read> input_b: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> scales_b: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f16>>;
struct Uniforms {
  batch_count: u32,
  M: u32,
  N: u32,
  K: u32,
  K8: u32,
  K16: u32,
  num_M_tile: u32,
  num_N_tile: u32,
  zero_blocks_per_col: u32,
  weight_idx: u32
};
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

alias input_a_value_t = vec4<u32>;
alias input_a_indices_t = vec3<u32>;
alias output_element_t = f16;

  alias output_type = i32;
  const default_zero_point = 8;
  const bit_mask = 0xFu;
  fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> output_type {
    return output_type(default_zero_point);
  }
    alias mul_precision = output_element_t;
    fn DequantizedFrom4BitsTo8Bits(in: vec2<u32>, zero: i32) -> vec4<u32>
    {
        var out = vec4<u32>(0);
        var value_lower = vec4<i32>(unpack4xU8(in[0] & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        var value_upper = vec4<i32>(unpack4xU8((in[0] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        out[0] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
        out[1] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
        value_lower = vec4<i32>(unpack4xU8(in[1] & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        value_upper = vec4<i32>(unpack4xU8((in[1] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(zero);
        out[2] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
        out[3] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
        return out;
    }
    fn SDP8AI(a1:vec4<u32>, b1:vec4<u32>, a2:vec4<u32>, b2:vec4<u32>, scale:output_element_t) -> output_element_t
    {
        var local_sum = dot4I8Packed(a1[0], b1[0]);
        local_sum += dot4I8Packed(a1[1], b1[1]);
        local_sum += dot4I8Packed(a1[2], b1[2]);
        local_sum += dot4I8Packed(a1[3], b1[3]);
        local_sum += dot4I8Packed(a2[0], b2[0]);
        local_sum += dot4I8Packed(a2[1], b2[1]);
        local_sum += dot4I8Packed(a2[2], b2[2]);
        local_sum += dot4I8Packed(a2[3], b2[3]);
        return output_element_t(mul_precision(local_sum) * mul_precision(scale));
    }
const tile_size = 64;
const subtile_size = 16;
const tile_size_k =  32;
const vec_factor = 4;
const u32_factor = 4;
const tile_size_k_vec = 2;

var<workgroup> tile_A : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;
var<workgroup> scale_A : array<output_element_t, tile_size>;
var<workgroup> tile_B : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;
var<workgroup> scale_B : array<output_element_t, tile_size>;

fn loadSHMA(batch:u32, a_global_base:u32, kidx_v:u32, row: u32, col: u32)
{
    let a_global = a_global_base + row;
    if (a_global >= uniforms.M)
    {
        return;
    }
    tile_A[col][row] = input_a[batch*uniforms.M*uniforms.K16+a_global*uniforms.K16+kidx_v+col];
    if (col == 0)
    {

        scale_A[row] = scales_a[batch*uniforms.M*(uniforms.K/128) + a_global*(uniforms.K/128) + kidx_v/8];
    }
}

    fn loadSHMB(b_global_base:u32, kidx_v:u32, row: u32, col: u32)
    {
        let b_global = b_global_base + row;
        if (b_global >= uniforms.N)
        {
            return;
        }
        const actual_weight_idx : u32 = 0;
        let b_value = input_b[b_global * uniforms.K16+kidx_v + col];
        let block_idx = kidx_v/(32/16);
        let zero = mm_read_zero(b_global, block_idx, uniforms.N, uniforms.zero_blocks_per_col);
        tile_B[col][row] = DequantizedFrom4BitsTo8Bits(b_value, zero);
        if (col == 0)
        {

            let b_scale_offset = actual_weight_idx * uniforms.N * (uniforms.K/32);
            scale_B[row] = scales_b[b_scale_offset + b_global*(uniforms.K/32) + block_idx];
        }
    }

@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;

    let batch = workgroup_idx / (uniforms.num_M_tile * uniforms.num_N_tile);
    if (batch >= uniforms.batch_count) {
        return;
    }
    let a_global_base = u32((workgroup_idx / uniforms.num_N_tile) % uniforms.num_M_tile) * tile_size;
    let b_global_base = (workgroup_idx % uniforms.num_N_tile) * tile_size;
    let load_AorB = u32(local_idx/128);
    let load_row = u32((local_idx%128)/2);
    let load_col = u32(local_idx%2);

    var subtile_id = u32(local_idx / subtile_size);
    var subtile_idx = u32(subtile_id / 4);
    var subtile_idy = u32(subtile_id % 4);
    var base_A = subtile_idx * 16;
    var base_B = subtile_idy * 16;

    var a_idx = u32(local_idx % subtile_size);

    var lane_output1: vec4<output_element_t>;
    var lane_output2: vec4<output_element_t>;
    var lane_output3: vec4<output_element_t>;
    var lane_output4: vec4<output_element_t>;
    for (var kidx_v:u32 = 0; kidx_v < uniforms.K16; kidx_v+=tile_size_k_vec)
    {

        if (load_AorB == 0)
        {
            loadSHMA(batch, a_global_base, kidx_v, load_row, load_col);
        }
        else
        {
            loadSHMB(b_global_base, kidx_v, load_row, load_col);
        }
        workgroupBarrier();

        var own_a0: vec4<u32> = tile_A[0][base_A + a_idx];
        var own_a1: vec4<u32> = tile_A[1][base_A + a_idx];
        var own_scale_a: output_element_t = scale_A[base_A + a_idx];

        if (sg_size == 16)
        {
            var own_b0: vec4<u32> = tile_B[0][base_B + sg_id];
            var own_b1: vec4<u32> = tile_B[1][base_B + sg_id];
            var own_scale_b: output_element_t  = scale_B[base_B + sg_id];

            lane_output1[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 0), own_a1, subgroupShuffle(own_b1, 0), subgroupShuffle(own_scale_b, 0) * own_scale_a);
            lane_output1[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 1), own_a1, subgroupShuffle(own_b1, 1), subgroupShuffle(own_scale_b, 1) * own_scale_a);
            lane_output1[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 2), own_a1, subgroupShuffle(own_b1, 2), subgroupShuffle(own_scale_b, 2) * own_scale_a);
            lane_output1[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 3), own_a1, subgroupShuffle(own_b1, 3), subgroupShuffle(own_scale_b, 3) * own_scale_a);

            lane_output2[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 4), own_a1, subgroupShuffle(own_b1, 4), subgroupShuffle(own_scale_b, 4) * own_scale_a);
            lane_output2[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 5), own_a1, subgroupShuffle(own_b1, 5), subgroupShuffle(own_scale_b, 5) * own_scale_a);
            lane_output2[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 6), own_a1, subgroupShuffle(own_b1, 6), subgroupShuffle(own_scale_b, 6) * own_scale_a);
            lane_output2[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 7), own_a1, subgroupShuffle(own_b1, 7), subgroupShuffle(own_scale_b, 7) * own_scale_a);

            lane_output3[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 8), own_a1, subgroupShuffle(own_b1, 8), subgroupShuffle(own_scale_b, 8) * own_scale_a);
            lane_output3[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 9), own_a1, subgroupShuffle(own_b1, 9), subgroupShuffle(own_scale_b, 9) * own_scale_a);
            lane_output3[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 10), own_a1, subgroupShuffle(own_b1, 10), subgroupShuffle(own_scale_b, 10) * own_scale_a);
            lane_output3[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 11), own_a1, subgroupShuffle(own_b1, 11), subgroupShuffle(own_scale_b, 11) * own_scale_a);

            lane_output4[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 12), own_a1, subgroupShuffle(own_b1, 12), subgroupShuffle(own_scale_b, 12) * own_scale_a);
            lane_output4[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 13), own_a1, subgroupShuffle(own_b1, 13), subgroupShuffle(own_scale_b, 13) * own_scale_a);
            lane_output4[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 14), own_a1, subgroupShuffle(own_b1, 14), subgroupShuffle(own_scale_b, 14) * own_scale_a);
            lane_output4[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 15), own_a1, subgroupShuffle(own_b1, 15), subgroupShuffle(own_scale_b, 15) * own_scale_a);
        }
        else
        {

            lane_output1[0] += SDP8AI(own_a0, tile_B[0][base_B + 0], own_a1, tile_B[1][base_B + 0],  own_scale_a * scale_B[base_B + 0]);
            lane_output1[1] += SDP8AI(own_a0, tile_B[0][base_B + 1], own_a1, tile_B[1][base_B + 1],  own_scale_a * scale_B[base_B + 1]);
            lane_output1[2] += SDP8AI(own_a0, tile_B[0][base_B + 2], own_a1, tile_B[1][base_B + 2],  own_scale_a * scale_B[base_B + 2]);
            lane_output1[3] += SDP8AI(own_a0, tile_B[0][base_B + 3], own_a1, tile_B[1][base_B + 3],  own_scale_a * scale_B[base_B + 3]);

            lane_output2[0] += SDP8AI(own_a0, tile_B[0][base_B + 4], own_a1, tile_B[1][base_B + 4],  own_scale_a * scale_B[base_B + 4]);
            lane_output2[1] += SDP8AI(own_a0, tile_B[0][base_B + 5], own_a1, tile_B[1][base_B + 5],  own_scale_a * scale_B[base_B + 5]);
            lane_output2[2] += SDP8AI(own_a0, tile_B[0][base_B + 6], own_a1, tile_B[1][base_B + 6],  own_scale_a * scale_B[base_B + 6]);
            lane_output2[3] += SDP8AI(own_a0, tile_B[0][base_B + 7], own_a1, tile_B[1][base_B + 7],  own_scale_a * scale_B[base_B + 7]);

            lane_output3[0] += SDP8AI(own_a0, tile_B[0][base_B + 8], own_a1, tile_B[1][base_B + 8],  own_scale_a * scale_B[base_B + 8]);
            lane_output3[1] += SDP8AI(own_a0, tile_B[0][base_B + 9], own_a1, tile_B[1][base_B + 9],  own_scale_a * scale_B[base_B + 9]);
            lane_output3[2] += SDP8AI(own_a0, tile_B[0][base_B + 10], own_a1, tile_B[1][base_B + 10],  own_scale_a * scale_B[base_B + 10]);
            lane_output3[3] += SDP8AI(own_a0, tile_B[0][base_B + 11], own_a1, tile_B[1][base_B + 11],  own_scale_a * scale_B[base_B + 11]);

            lane_output4[0] += SDP8AI(own_a0, tile_B[0][base_B + 12], own_a1, tile_B[1][base_B + 12],  own_scale_a * scale_B[base_B + 12]);
            lane_output4[1] += SDP8AI(own_a0, tile_B[0][base_B + 13], own_a1, tile_B[1][base_B + 13],  own_scale_a * scale_B[base_B + 13]);
            lane_output4[2] += SDP8AI(own_a0, tile_B[0][base_B + 14], own_a1, tile_B[1][base_B + 14],  own_scale_a * scale_B[base_B + 14]);
            lane_output4[3] += SDP8AI(own_a0, tile_B[0][base_B + 15], own_a1, tile_B[1][base_B + 15],  own_scale_a * scale_B[base_B + 15]);
        }
        workgroupBarrier();
    }

    let a_global = a_global_base + base_A + a_idx;
    let b_global = b_global_base + base_B;
    let output_idx = (batch * uniforms.M * uniforms.N + a_global * uniforms.N + b_global)/4;
    if (a_global < uniforms.M && b_global < uniforms.N)
    {
        output[output_idx]=lane_output1;;
        output[output_idx+1]=lane_output2;;
        output[output_idx+2]=lane_output3;;
        output[output_idx+3]=lane_output4;;
    }

}
)WGSL";

static const char* WGSL_MATMUL_Q4_BATCHED = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Native Q4_0 matmul for prefill. A workgroup computes 32 output columns for
// four prompt rows. Each warp owns four columns, while the activation tile and
// its Q8 quantization are shared by all 32 columns.
// Dispatch: (ceil(N / 32), ceil(M / 4), 1).

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

const ROWS: u32 = 4u;
const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;

var<workgroup> xq: array<u32, ROWS * 64u>;
var<workgroup> xs: array<f32, ROWS * 8u>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = P[0]; let N = P[1]; let K = P[2];
    let tid = lid.x;
    let warp = tid / 32u;
    let lane = tid & 31u;
    let row0 = wid.y * ROWS;
    let n_blocks = K / 32u;
    let words_per_row = K / 8u;

    var cols: array<u32, COLS_PER_WARP>;
    var valid: array<bool, COLS_PER_WARP>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 32u + warp * COLS_PER_WARP + c;
        valid[c] = cols[c] < N;
    }
    var acc: array<f32, ROWS * COLS_PER_WARP>;
    for (var i = 0u; i < ROWS * COLS_PER_WARP; i++) { acc[i] = 0.0; }

    for (var kb = 0u; kb < K; kb += BK) {
        let block = tid / 32u;
        let elem = tid & 31u;
        let pack_lane = elem & 3u;
        let pack_group = elem / 4u;

        for (var r = 0u; r < ROWS; r++) {
            let row = row0 + r;
            let xv = select(0.0, X[row * K + kb + tid], row < M);
            var amax = abs(xv);
            amax = max(amax, subgroupShuffleXor(amax, 16u));
            amax = max(amax, subgroupShuffleXor(amax, 8u));
            amax = max(amax, subgroupShuffleXor(amax, 4u));
            amax = max(amax, subgroupShuffleXor(amax, 2u));
            amax = max(amax, subgroupShuffleXor(amax, 1u));
            let scale = amax / 127.0;
            if (elem == 0u) { xs[r * 8u + block] = scale; }
            let safe = select(1.0, scale, scale != 0.0);
            let q = u32(clamp(i32(round(xv / safe)), -127, 127)) & 0xffu;
            var packed = q << (pack_lane * 8u);
            packed |= subgroupShuffleXor(packed, 1u);
            packed |= subgroupShuffleXor(packed, 2u);
            if (pack_lane == 0u) {
                xq[r * 64u + block * 8u + pack_group] = packed;
            }
        }
        workgroupBarrier();

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (valid[c]) {
                let col = cols[c];
                let q4 = B[col * words_per_row + kb / 8u + lane];
                let b0 = q4 & 0xffu; let b1 = (q4 >> 8u) & 0xffu;
                let b2 = (q4 >> 16u) & 0xffu; let b3 = (q4 >> 24u) & 0xffu;
                let wq0 = (((b0 & 15u) - 8u) & 255u) |
                    ((((b0 >> 4u) - 8u) & 255u) << 8u) |
                    ((((b1 & 15u) - 8u) & 255u) << 16u) |
                    ((((b1 >> 4u) - 8u) & 255u) << 24u);
                let wq1 = (((b2 & 15u) - 8u) & 255u) |
                    ((((b2 >> 4u) - 8u) & 255u) << 8u) |
                    ((((b3 & 15u) - 8u) & 255u) << 16u) |
                    ((((b3 >> 4u) - 8u) & 255u) << 24u);
                let xblock = lane / 4u;
                let wb = kb / 32u + xblock;
                let si = col * n_blocks + wb;
                let sp = unpack2x16float(Scales[si / 2u]);
                let ws = select(sp.x, sp.y, (si & 1u) != 0u);
                for (var r = 0u; r < ROWS; r++) {
                    let base = r * 64u + lane * 2u;
                    let dot = dot4I8Packed(xq[base], wq0) +
                              dot4I8Packed(xq[base + 1u], wq1);
                    acc[c * ROWS + r] += f32(dot) * ws * xs[r * 8u + xblock];
                }
            }
        }
        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var r = 0u; r < ROWS; r++) {
            let sum = subgroupAdd(acc[c * ROWS + r]);
            if (lane == 0u && valid[c]) {
                let row = row0 + r;
                if (row < M) { Y[row * N + cols[c]] = sum; }
            }
        }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_decode
static const char* WGSL_MATMUL_Q4_DECODE = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Q4 matmul for decode (M=1), DP4A-accelerated, hardcoded ZP=8.
// N-parallel: 256 threads = 8 warps × 32 lanes.
// Each warp computes 4 output columns; each lane processes K/256 strides.
// Quantizes X to int8, unpacks Q4 nibbles to int8, uses dot4I8Packed.
//
// Dispatch: (ceil(N/32), 1, 1)
//
// Bindings:
//   0: X (read) — input activations [K] fp32
//   1: B (read) — packed Q4 weights [N × K/8] as u32 (8 nibbles per u32)
//   2: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   3: Y (write) — output [N] fp32
//   4: _params_ — [0, N, K, 0]

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const BK: u32 = 256u;  // K elements processed per outer iteration
const COLS_PER_WARP: u32 = 4u;

// Quantized X in shared memory: 64 packed u32 (256 int8 values)
var<workgroup> smem_xq: array<u32, 64>;
// X scales: 8 blocks of 32 elements each
var<workgroup> smem_xs: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[1];
    let K = _params_[2];
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;  // 8 nibbles per u32

    // Pre-compute column info for this warp
    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * (8u * COLS_PER_WARP) + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    // Quantization lane mapping: tid 0..255 → K element 0..255
    let block_id = tid / 32u;           // 0..7 — which Q8 block within BK
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;

        // ── Quantize X[k_base..k_base+256] to int8 in shared memory ──
        let x_val = X[k_base + tid];

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul: each lane processes 8 K-elements (2 x dot4I8Packed) ──
        let x_block = lane / 4u;  // which Q8 block (of 32 elems) this lane's elements are in

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                // Read 1 Q4 u32 = 8 nibbles = 8 elements
                // Lane processes 8 K-elements starting at k_base + lane * 8
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                // Unpack 8 nibbles into 2 u32s of 4 int8s each (with -8 bias)
                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                // Pack nibble pairs as int8s: (nib - 8) fits in [-8, +7]
                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                // Weight scale
                let w_block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                // X quantized values from shared memory
                let xq0 = smem_xq[lane * 2u];
                let xq1 = smem_xq[lane * 2u + 1u];
                let x_scale = smem_xs[x_block];

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_scale;
            }
        }

        workgroupBarrier();
    }

    // Write output
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        // Reduce a logical 32-lane warp explicitly. subgroupAdd combines all
        // 64 lanes on RDNA, mixing the two independent column groups. XOR
        // shuffles with masks <= 16 stay within each 32-lane half on both
        // wave32 and wave64 hardware.
        var warp_sum = acc[c];
        warp_sum += subgroupShuffleXor(warp_sum, 16u);
        warp_sum += subgroupShuffleXor(warp_sum, 8u);
        warp_sum += subgroupShuffleXor(warp_sum, 4u);
        warp_sum += subgroupShuffleXor(warp_sum, 2u);
        warp_sum += subgroupShuffleXor(warp_sum, 1u);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] = warp_sum;
        }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_decode_norm
static const char* WGSL_MATMUL_Q4_DECODE_NORM = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Fused RMSNorm + Q4 matmul for decode (M=1), DP4A-accelerated.
// N-parallel: 256 threads = 8 warps × 32 lanes.
// Pass 1: compute sum_sq over X using all 256 threads
// Pass 2: quantize normalized X to int8, DP4A with Q4 weights
//
// Dispatch: (ceil(N/32), 1, 1)
//
// Bindings:
//   0: X (read) — raw input vector (pre-norm), K floats
//   1: B (read) — packed Q4 weights [N × K/8] as u32
//   2: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   3: Y (write) — output [N] fp32
//   4: _params_ — [K, N, 0, eps_as_u32]
//   5: NormW (read) — norm weight vector, K floats
//   6: Bias (read) — per-output bias

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> NormW: array<f32>;
@group(0) @binding(6) var<storage, read> Bias: array<f32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;

var<workgroup> smem_xq: array<u32, 64>;
var<workgroup> smem_xs: array<f32, 8>;
var<workgroup> smem_rstd: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];
    let N = _params_[1];
    let eps = bitcast<f32>(_params_[3]);
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;

    // ── Pass 1: compute RMSNorm rstd ──────────────────────────────
    // Each thread handles K/256 elements
    var sum_sq: f32 = 0.0;
    for (var k = tid; k < K; k += 256u) {
        let v = X[k];
        sum_sq += v * v;
    }
    // Reduce within warp
    sum_sq = subgroupAdd(sum_sq);
    // Reduce across warps using shared memory
    if (lane == 0u) {
        smem_xs[warp_id] = sum_sq;
    }
    workgroupBarrier();

    var total_sq: f32 = 0.0;
    if (tid < 8u) {
        total_sq = smem_xs[tid];
    }
    total_sq = subgroupAdd(total_sq);
    if (tid == 0u) {
        smem_rstd = 1.0 / sqrt(total_sq / f32(K) + eps);
    }
    workgroupBarrier();
    let rstd = smem_rstd;

    // Pre-compute column info
    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 32u + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;

        // ── Quantize normalized X to int8 ──
        let k = k_base + tid;
        let x_val = X[k] * NormW[k] * rstd;

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul ──
        let x_block = lane / 4u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                let w_block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                let xq0 = smem_xq[lane * 2u];
                let xq1 = smem_xq[lane * 2u + 1u];
                let x_scale = smem_xs[x_block];

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_scale;
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let warp_sum = subgroupAdd(acc[c]);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] = warp_sum + Bias[cols[c]];
        }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_decode_norm_wide
static const char* WGSL_MATMUL_Q4_DECODE_NORM_WIDE = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Fused RMSNorm + Q4 matmul for decode (M=1), DP4A-accelerated.
// Wide variant: COLS_PER_WARP=8 (vs 4 in base), 64 output cols per WG.
//
// Dispatch: (ceil(N/64), 1, 1)
//
// Bindings:
//   0: X (read) — raw input vector (pre-norm), K floats
//   1: B (read) — packed Q4 weights [N × K/8] as u32
//   2: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   3: Y (write) — output [N] fp32
//   4: _params_ — [K, N, 0, eps_as_u32]
//   5: NormW (read) — norm weight vector, K floats
//   6: Bias (read) — per-output bias

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> NormW: array<f32>;
@group(0) @binding(6) var<storage, read> Bias: array<f32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 8u;

var<workgroup> smem_xq: array<u32, 64>;
var<workgroup> smem_xs: array<f32, 8>;
var<workgroup> smem_rstd: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];
    let N = _params_[1];
    let eps = bitcast<f32>(_params_[3]);
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;

    // ── Pass 1: compute RMSNorm rstd ──
    var sum_sq: f32 = 0.0;
    for (var k = tid; k < K; k += 256u) {
        let v = X[k];
        sum_sq += v * v;
    }
    sum_sq = subgroupAdd(sum_sq);
    if (lane == 0u) {
        smem_xs[warp_id] = sum_sq;
    }
    workgroupBarrier();

    var total_sq: f32 = 0.0;
    if (tid < 8u) {
        total_sq = smem_xs[tid];
    }
    total_sq = subgroupAdd(total_sq);
    if (tid == 0u) {
        smem_rstd = 1.0 / sqrt(total_sq / f32(K) + eps);
    }
    workgroupBarrier();
    let rstd = smem_rstd;

    var cols: array<u32, 8>;
    var col_valid: array<bool, 8>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 64u + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 8>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;

        // ── Quantize normalized X to int8 ──
        let k = k_base + tid;
        let x_val = X[k] * NormW[k] * rstd;

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul ──
        let x_block = lane / 4u;
        let xq0 = smem_xq[lane * 2u];
        let xq1 = smem_xq[lane * 2u + 1u];
        let x_sc = smem_xs[x_block];
        let w_block = g * 8u + x_block;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_sc;
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let warp_sum = subgroupAdd(acc[c]);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] = warp_sum + Bias[cols[c]];
        }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_decode_pair
static const char* WGSL_MATMUL_Q4_DECODE_PAIR = R"WGSL(
// @meta bindings=8
requires packed_4x8_integer_dot_product;
enable subgroups;

// Paired Q4 decode projection for gate/up matrices sharing one activation.
// Quantizes X once and computes matching columns from both matrices.
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B0: array<u32>;
@group(0) @binding(2) var<storage, read> S0: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y0: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
@group(0) @binding(5) var<storage, read> B1: array<u32>;
@group(0) @binding(6) var<storage, read> S1: array<u32>;
@group(0) @binding(7) var<storage, read_write> Y1: array<f32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 2u;
var<workgroup> xq: array<u32, 64>;
var<workgroup> xs: array<f32, 8>;

fn scale_at(s: ptr<storage, array<u32>, read>, index: u32) -> f32 {
    let pair = unpack2x16float((*s)[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) != 0u);
}

fn pack4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    return ((a - 8u) & 255u) | (((b - 8u) & 255u) << 8u) |
           (((c - 8u) & 255u) << 16u) | (((d - 8u) & 255u) << 24u);
}
fn q4_lo(p: u32) -> u32 {
    return pack4(p & 15u, (p >> 4u) & 15u,
                 (p >> 8u) & 15u, (p >> 12u) & 15u);
}
fn q4_hi(p: u32) -> u32 {
    return pack4((p >> 16u) & 15u, (p >> 20u) & 15u,
                 (p >> 24u) & 15u, (p >> 28u) & 15u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = P[1]; let K = P[2]; let tid = lid.x;
    let warp = tid / 32u; let lane = tid & 31u;
    let blocks = K / 32u; let words = K / 8u;
    var cols: array<u32, COLS_PER_WARP>;
    var valid: array<bool, COLS_PER_WARP>;
    var a0: array<f32, COLS_PER_WARP>;
    var a1: array<f32, COLS_PER_WARP>;
    for (var c=0u; c<COLS_PER_WARP; c++) {
        cols[c] = wid.x * (8u * COLS_PER_WARP) + warp * COLS_PER_WARP + c;
        valid[c] = cols[c] < N; a0[c] = 0.0; a1[c] = 0.0;
    }
    let block = tid / 32u; let e = tid & 31u;
    let byte = e & 3u; let pack = e / 4u;

    for (var g=0u; g<K/BK; g++) {
        let xv = X[g*BK + tid];
        var mx = abs(xv);
        mx=max(mx,subgroupShuffleXor(mx,16u)); mx=max(mx,subgroupShuffleXor(mx,8u));
        mx=max(mx,subgroupShuffleXor(mx,4u)); mx=max(mx,subgroupShuffleXor(mx,2u));
        mx=max(mx,subgroupShuffleXor(mx,1u));
        let sx=mx/127.0; if(e==0u){xs[block]=sx;}
        let safe=select(1.0,sx,sx!=0.0);
        let qi=clamp(i32(round(xv/safe)),-127,127);
        var pq=u32(qi&255)<<(byte*8u);
        pq|=subgroupShuffleXor(pq,1u); pq|=subgroupShuffleXor(pq,2u);
        if(byte==0u){xq[block*8u+pack]=pq;}
        workgroupBarrier();

        let xb=lane/4u; let x0=xq[lane*2u]; let x1=xq[lane*2u+1u];
        for(var c=0u;c<COLS_PER_WARP;c++){
            if(valid[c]){
                let n=cols[c]; let off=n*words+g*32u+lane;
                let w0=B0[off]; let w1=B1[off];
                let d0=dot4I8Packed(x0,q4_lo(w0))+dot4I8Packed(x1,q4_hi(w0));
                let d1=dot4I8Packed(x0,q4_lo(w1))+dot4I8Packed(x1,q4_hi(w1));
                let si=n*blocks+g*8u+xb;
                a0[c]+=f32(d0)*xs[xb]*scale_at(&S0,si);
                a1[c]+=f32(d1)*xs[xb]*scale_at(&S1,si);
            }
        }
        workgroupBarrier();
    }
    for(var c=0u;c<COLS_PER_WARP;c++){
        var v0=a0[c]; var v1=a1[c];
        v0+=subgroupShuffleXor(v0,16u);v1+=subgroupShuffleXor(v1,16u);
        v0+=subgroupShuffleXor(v0,8u);v1+=subgroupShuffleXor(v1,8u);
        v0+=subgroupShuffleXor(v0,4u);v1+=subgroupShuffleXor(v1,4u);
        v0+=subgroupShuffleXor(v0,2u);v1+=subgroupShuffleXor(v1,2u);
        v0+=subgroupShuffleXor(v0,1u);v1+=subgroupShuffleXor(v1,1u);
        if(lane==0u&&valid[c]){Y0[cols[c]]=v0;Y1[cols[c]]=v1;}
    }
}
)WGSL";

// [onnx_q4] matmul_q4_decode_wide
static const char* WGSL_MATMUL_Q4_DECODE_WIDE = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Q4 matmul for decode (M=1), DP4A-accelerated, hardcoded ZP=8.
// Wide variant: COLS_PER_WARP=8 (vs 4 in base), 64 output cols per WG.
// Amortizes X quantization + barrier cost over more columns.
//
// Dispatch: (ceil(N/64), 1, 1)
//
// Bindings:
//   0: X (read) — input activations [K] fp32
//   1: B (read) — packed Q4 weights [N × K/8] as u32 (8 nibbles per u32)
//   2: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   3: Y (write) — output [N] fp32
//   4: _params_ — [0, N, K, 0]

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 8u;

var<workgroup> smem_xq: array<u32, 64>;
var<workgroup> smem_xs: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[1];
    let K = _params_[2];
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;

    var cols: array<u32, 8>;
    var col_valid: array<bool, 8>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 64u + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 8>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;

        // ── Quantize X to int8 ──
        let x_val = X[k_base + tid];

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul ──
        let x_block = lane / 4u;
        let xq0 = smem_xq[lane * 2u];
        let xq1 = smem_xq[lane * 2u + 1u];
        let x_sc = smem_xs[x_block];
        let w_block = g * 8u + x_block;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_sc;
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let warp_sum = subgroupAdd(acc[c]);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] = warp_sum;
        }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_zp — f32 instantiated
static const char* WGSL_MATMUL_Q4_ZP = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let n = gid.x;
    let m = gid.y;
    if (n >= N || m >= M) { return; }

    var acc: f32 = 0.0;
    let a_base = m * K;
    let w_base = n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let zp_byte_idx = scale_flat / 2u;
        let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
        let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        // Process 4 bytes (8 Q4 values) per iteration for better ILP
        for (var j = 0u; j < 4u; j++) {
            let packed = B[w_blk_base / 4u + j];
            let k0 = k_base + j * 8u;
            let b0 = packed & 0xFFu;
            let b1 = (packed >> 8u) & 0xFFu;
            let b2 = (packed >> 16u) & 0xFFu;
            let b3 = (packed >> 24u) & 0xFFu;
            acc += t_read(&A, a_base + k0)      * (f32(b0 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 1u) * (f32(b0 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 2u) * (f32(b1 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 3u) * (f32(b1 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 4u) * (f32(b2 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 5u) * (f32(b2 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 6u) * (f32(b3 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 7u) * (f32(b3 >> 4u)  - zp) * scale;
        }
    }

    t_write(&Y, m * N + n, acc);
}
)WGSL";

// [onnx_q4] matmul_q4_zp — dtype template
static const char* WGSL_MATMUL_Q4_ZP_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let n = gid.x;
    let m = gid.y;
    if (n >= N || m >= M) { return; }

    var acc: f32 = 0.0;
    let a_base = m * K;
    let w_base = n * (K / 2u);

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let scale_flat = n * blocks_per_col + blk;
        let scale_u32 = Scales[scale_flat / 2u];
        let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
        let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

        let zp_byte_idx = scale_flat / 2u;
        let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
        let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

        let k_base = blk * 32u;
        let w_blk_base = w_base + k_base / 2u;

        // Process 4 bytes (8 Q4 values) per iteration for better ILP
        for (var j = 0u; j < 4u; j++) {
            let packed = B[w_blk_base / 4u + j];
            let k0 = k_base + j * 8u;
            let b0 = packed & 0xFFu;
            let b1 = (packed >> 8u) & 0xFFu;
            let b2 = (packed >> 16u) & 0xFFu;
            let b3 = (packed >> 24u) & 0xFFu;
            acc += t_read(&A, a_base + k0)      * (f32(b0 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 1u) * (f32(b0 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 2u) * (f32(b1 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 3u) * (f32(b1 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 4u) * (f32(b2 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 5u) * (f32(b2 >> 4u)  - zp) * scale;
            acc += t_read(&A, a_base + k0 + 6u) * (f32(b3 & 0xFu) - zp) * scale;
            acc += t_read(&A, a_base + k0 + 7u) * (f32(b3 >> 4u)  - zp) * scale;
        }
    }

    t_write(&Y, m * N + n, acc);
}
)WGSL";

// [onnx_q4] matmul_q4_zp_sub — f32 instantiated
static const char* WGSL_MATMUL_Q4_ZP_SUB = R"WGSL(
// Q4 matmul with zero points + K-parallel subgroup reduction, TILE_N=8.
// Optimized for decode (M=1): 256 threads = 8 warps × 32 lanes.
// Each warp computes one output N, 32 lanes split blocks_per_col.
// Y[n] = sum_k A[k] * (dequant(B[n,k]) - zp) * scale
// Dispatch: (ceil(N/8), 1, 1) — M must be 1

enable subgroups;


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let warp_id = lid.x / 32u;
    let lane = lid.x % 32u;
    let n = wid.x * 8u + warp_id;
    let valid = n < N;

    var acc: f32 = 0.0;
    let w_base = n * (K / 2u);

    if (valid) {
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                acc += t_read(&A, k0)      * (f32(b0 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 1u) * (f32(b0 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 2u) * (f32(b1 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 3u) * (f32(b1 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 4u) * (f32(b2 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 5u) * (f32(b2 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 6u) * (f32(b3 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 7u) * (f32(b3 >> 4u)  - zp) * scale;
            }
        }
    }

    // subgroupAdd requires subgroup-uniform control flow
    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        t_write(&Y, n, warp_sum);
    }
}
)WGSL";

// [onnx_q4] matmul_q4_zp_sub — dtype template
static const char* WGSL_MATMUL_Q4_ZP_SUB_T = R"WGSL(
// Q4 matmul with zero points + K-parallel subgroup reduction, TILE_N=8.
// Optimized for decode (M=1): 256 threads = 8 warps × 32 lanes.
// Each warp computes one output N, 32 lanes split blocks_per_col.
// Y[n] = sum_k A[k] * (dequant(B[n,k]) - zp) * scale
// Dispatch: (ceil(N/8), 1, 1) — M must be 1

enable subgroups;

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let warp_id = lid.x / 32u;
    let lane = lid.x % 32u;
    let n = wid.x * 8u + warp_id;
    let valid = n < N;

    var acc: f32 = 0.0;
    let w_base = n * (K / 2u);

    if (valid) {
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                acc += t_read(&A, k0)      * (f32(b0 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 1u) * (f32(b0 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 2u) * (f32(b1 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 3u) * (f32(b1 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 4u) * (f32(b2 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 5u) * (f32(b2 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 6u) * (f32(b3 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 7u) * (f32(b3 >> 4u)  - zp) * scale;
            }
        }
    }

    // subgroupAdd requires subgroup-uniform control flow
    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        t_write(&Y, n, warp_sum);
    }
}
)WGSL";

// [onnx_q4] matmul_q4_zp_sub_prefill — f32 instantiated
static const char* WGSL_MATMUL_Q4_ZP_SUB_PREFILL = R"WGSL(
// K-parallel subgroup Q4+ZP matmul for prefill with weight reuse.
// 256 threads = 8 warps x 32 lanes. TILE_N=8, TILE_M=4.
// Each warp computes 1 output N × TILE_M rows, reusing weights across rows.
// 32 lanes split blocks_per_col for K-parallel reduction.
// Dispatch: (ceil(N/8), ceil(M/TILE_M), 1)

enable subgroups;


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const TILE_M: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let warp_id = lid.x / 32u;     // 0..7
    let lane = lid.x % 32u;        // 0..31
    let n = wid.x * 8u + warp_id;
    let m_base = wid.y * TILE_M;
    let n_valid = n < N;

    let w_base = n * (K / 2u);

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    if (n_valid) {
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                let w0 = (f32(b0 & 0xFu) - zp) * scale;
                let w1 = (f32(b0 >> 4u)  - zp) * scale;
                let w2 = (f32(b1 & 0xFu) - zp) * scale;
                let w3 = (f32(b1 >> 4u)  - zp) * scale;
                let w4 = (f32(b2 & 0xFu) - zp) * scale;
                let w5 = (f32(b2 >> 4u)  - zp) * scale;
                let w6 = (f32(b3 & 0xFu) - zp) * scale;
                let w7 = (f32(b3 >> 4u)  - zp) * scale;

                // Row 0
                if (m_base < M) {
                    let ab = m_base * K;
                    acc0 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 1
                if (m_base + 1u < M) {
                    let ab = (m_base + 1u) * K;
                    acc1 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 2
                if (m_base + 2u < M) {
                    let ab = (m_base + 2u) * K;
                    acc2 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 3
                if (m_base + 3u < M) {
                    let ab = (m_base + 3u) * K;
                    acc3 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
            }
        }
    }

    // Subgroup reduction across 32 K-parallel lanes
    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);
    let sum2 = subgroupAdd(acc2);
    let sum3 = subgroupAdd(acc3);

    if (lane == 0u && n_valid) {
        if (m_base < M)      { t_write(&Y, m_base * N + n, sum0); }
        if (m_base + 1u < M) { t_write(&Y, (m_base + 1u) * N + n, sum1); }
        if (m_base + 2u < M) { t_write(&Y, (m_base + 2u) * N + n, sum2); }
        if (m_base + 3u < M) { t_write(&Y, (m_base + 3u) * N + n, sum3); }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_zp_sub_prefill — dtype template
static const char* WGSL_MATMUL_Q4_ZP_SUB_PREFILL_T = R"WGSL(
// K-parallel subgroup Q4+ZP matmul for prefill with weight reuse.
// 256 threads = 8 warps x 32 lanes. TILE_N=8, TILE_M=4.
// Each warp computes 1 output N × TILE_M rows, reusing weights across rows.
// 32 lanes split blocks_per_col for K-parallel reduction.
// Dispatch: (ceil(N/8), ceil(M/TILE_M), 1)

enable subgroups;

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const TILE_M: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let warp_id = lid.x / 32u;     // 0..7
    let lane = lid.x % 32u;        // 0..31
    let n = wid.x * 8u + warp_id;
    let m_base = wid.y * TILE_M;
    let n_valid = n < N;

    let w_base = n * (K / 2u);

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    if (n_valid) {
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                let w0 = (f32(b0 & 0xFu) - zp) * scale;
                let w1 = (f32(b0 >> 4u)  - zp) * scale;
                let w2 = (f32(b1 & 0xFu) - zp) * scale;
                let w3 = (f32(b1 >> 4u)  - zp) * scale;
                let w4 = (f32(b2 & 0xFu) - zp) * scale;
                let w5 = (f32(b2 >> 4u)  - zp) * scale;
                let w6 = (f32(b3 & 0xFu) - zp) * scale;
                let w7 = (f32(b3 >> 4u)  - zp) * scale;

                // Row 0
                if (m_base < M) {
                    let ab = m_base * K;
                    acc0 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 1
                if (m_base + 1u < M) {
                    let ab = (m_base + 1u) * K;
                    acc1 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 2
                if (m_base + 2u < M) {
                    let ab = (m_base + 2u) * K;
                    acc2 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
                // Row 3
                if (m_base + 3u < M) {
                    let ab = (m_base + 3u) * K;
                    acc3 += t_read(&A, ab + k0)      * w0 + t_read(&A, ab + k0 + 1u) * w1
                          + t_read(&A, ab + k0 + 2u) * w2 + t_read(&A, ab + k0 + 3u) * w3
                          + t_read(&A, ab + k0 + 4u) * w4 + t_read(&A, ab + k0 + 5u) * w5
                          + t_read(&A, ab + k0 + 6u) * w6 + t_read(&A, ab + k0 + 7u) * w7;
                }
            }
        }
    }

    // Subgroup reduction across 32 K-parallel lanes
    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);
    let sum2 = subgroupAdd(acc2);
    let sum3 = subgroupAdd(acc3);

    if (lane == 0u && n_valid) {
        if (m_base < M)      { t_write(&Y, m_base * N + n, sum0); }
        if (m_base + 1u < M) { t_write(&Y, (m_base + 1u) * N + n, sum1); }
        if (m_base + 2u < M) { t_write(&Y, (m_base + 2u) * N + n, sum2); }
        if (m_base + 3u < M) { t_write(&Y, (m_base + 3u) * N + n, sum3); }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_zp_wide — f32 instantiated
static const char* WGSL_MATMUL_Q4_ZP_WIDE = R"WGSL(
// Wide-tile Q4+ZP matmul for prefill with shared-memory A reuse.
// 128 threads, TILE_M=8, TILE_N=128. Each thread owns one N-column.
// A tile [8 x 32] is cooperatively loaded into shared memory per Q4 block,
// then reused by all 128 threads for weight-activation dot products.
// Dispatch: (ceil(N/128), ceil(M/8), 1)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const TILE_M: u32 = 8u;

var<workgroup> smem_a: array<f32, 256>;  // TILE_M(8) * 32

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let tid = lid.x;
    let n = wid.x * 128u + tid;
    let m_base = wid.y * TILE_M;
    let n_valid = n < N;

    let w_base = select(0u, n * (K / 2u), n_valid);

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    var acc4: f32 = 0.0;
    var acc5: f32 = 0.0;
    var acc6: f32 = 0.0;
    var acc7: f32 = 0.0;

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let k_off = blk * 32u;

        // Cooperative A load: 128 threads load 256 elements (8 rows x 32 cols)
        {
            let idx0 = tid;           // 0..127
            let row0 = idx0 / 32u;    // 0..3
            let col0 = idx0 % 32u;
            let g_row0 = m_base + row0;
            smem_a[idx0] = select(0.0, t_read(&A, g_row0 * K + k_off + col0), g_row0 < M);
        }
        {
            let idx1 = tid + 128u;    // 128..255
            let row1 = idx1 / 32u;    // 4..7
            let col1 = idx1 % 32u;
            let g_row1 = m_base + row1;
            smem_a[idx1] = select(0.0, t_read(&A, g_row1 * K + k_off + col1), g_row1 < M);
        }

        workgroupBarrier();

        if (n_valid) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let w_blk_base = w_base + k_off / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let kl = j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                let w0 = (f32(b0 & 0xFu) - zp) * scale;
                let w1 = (f32(b0 >> 4u)  - zp) * scale;
                let w2 = (f32(b1 & 0xFu) - zp) * scale;
                let w3 = (f32(b1 >> 4u)  - zp) * scale;
                let w4 = (f32(b2 & 0xFu) - zp) * scale;
                let w5 = (f32(b2 >> 4u)  - zp) * scale;
                let w6 = (f32(b3 & 0xFu) - zp) * scale;
                let w7 = (f32(b3 >> 4u)  - zp) * scale;

                // Row 0
                acc0 += smem_a[kl]      * w0 + smem_a[kl + 1u] * w1
                      + smem_a[kl + 2u] * w2 + smem_a[kl + 3u] * w3
                      + smem_a[kl + 4u] * w4 + smem_a[kl + 5u] * w5
                      + smem_a[kl + 6u] * w6 + smem_a[kl + 7u] * w7;
                // Row 1
                acc1 += smem_a[32u + kl]      * w0 + smem_a[32u + kl + 1u] * w1
                      + smem_a[32u + kl + 2u] * w2 + smem_a[32u + kl + 3u] * w3
                      + smem_a[32u + kl + 4u] * w4 + smem_a[32u + kl + 5u] * w5
                      + smem_a[32u + kl + 6u] * w6 + smem_a[32u + kl + 7u] * w7;
                // Row 2
                acc2 += smem_a[64u + kl]      * w0 + smem_a[64u + kl + 1u] * w1
                      + smem_a[64u + kl + 2u] * w2 + smem_a[64u + kl + 3u] * w3
                      + smem_a[64u + kl + 4u] * w4 + smem_a[64u + kl + 5u] * w5
                      + smem_a[64u + kl + 6u] * w6 + smem_a[64u + kl + 7u] * w7;
                // Row 3
                acc3 += smem_a[96u + kl]      * w0 + smem_a[96u + kl + 1u] * w1
                      + smem_a[96u + kl + 2u] * w2 + smem_a[96u + kl + 3u] * w3
                      + smem_a[96u + kl + 4u] * w4 + smem_a[96u + kl + 5u] * w5
                      + smem_a[96u + kl + 6u] * w6 + smem_a[96u + kl + 7u] * w7;
                // Row 4
                acc4 += smem_a[128u + kl]      * w0 + smem_a[128u + kl + 1u] * w1
                      + smem_a[128u + kl + 2u] * w2 + smem_a[128u + kl + 3u] * w3
                      + smem_a[128u + kl + 4u] * w4 + smem_a[128u + kl + 5u] * w5
                      + smem_a[128u + kl + 6u] * w6 + smem_a[128u + kl + 7u] * w7;
                // Row 5
                acc5 += smem_a[160u + kl]      * w0 + smem_a[160u + kl + 1u] * w1
                      + smem_a[160u + kl + 2u] * w2 + smem_a[160u + kl + 3u] * w3
                      + smem_a[160u + kl + 4u] * w4 + smem_a[160u + kl + 5u] * w5
                      + smem_a[160u + kl + 6u] * w6 + smem_a[160u + kl + 7u] * w7;
                // Row 6
                acc6 += smem_a[192u + kl]      * w0 + smem_a[192u + kl + 1u] * w1
                      + smem_a[192u + kl + 2u] * w2 + smem_a[192u + kl + 3u] * w3
                      + smem_a[192u + kl + 4u] * w4 + smem_a[192u + kl + 5u] * w5
                      + smem_a[192u + kl + 6u] * w6 + smem_a[192u + kl + 7u] * w7;
                // Row 7
                acc7 += smem_a[224u + kl]      * w0 + smem_a[224u + kl + 1u] * w1
                      + smem_a[224u + kl + 2u] * w2 + smem_a[224u + kl + 3u] * w3
                      + smem_a[224u + kl + 4u] * w4 + smem_a[224u + kl + 5u] * w5
                      + smem_a[224u + kl + 6u] * w6 + smem_a[224u + kl + 7u] * w7;
            }
        }

        workgroupBarrier();
    }

    if (n_valid) {
        if (m_base < M)      { t_write(&Y, m_base * N + n, acc0); }
        if (m_base + 1u < M) { t_write(&Y, (m_base + 1u) * N + n, acc1); }
        if (m_base + 2u < M) { t_write(&Y, (m_base + 2u) * N + n, acc2); }
        if (m_base + 3u < M) { t_write(&Y, (m_base + 3u) * N + n, acc3); }
        if (m_base + 4u < M) { t_write(&Y, (m_base + 4u) * N + n, acc4); }
        if (m_base + 5u < M) { t_write(&Y, (m_base + 5u) * N + n, acc5); }
        if (m_base + 6u < M) { t_write(&Y, (m_base + 6u) * N + n, acc6); }
        if (m_base + 7u < M) { t_write(&Y, (m_base + 7u) * N + n, acc7); }
    }
}
)WGSL";

// [onnx_q4] matmul_q4_zp_wide — dtype template
static const char* WGSL_MATMUL_Q4_ZP_WIDE_T = R"WGSL(
// Wide-tile Q4+ZP matmul for prefill with shared-memory A reuse.
// 128 threads, TILE_M=8, TILE_N=128. Each thread owns one N-column.
// A tile [8 x 32] is cooperatively loaded into shared memory per Q4 block,
// then reused by all 128 threads for weight-activation dot products.
// Dispatch: (ceil(N/128), ceil(M/8), 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

const TILE_M: u32 = 8u;

var<workgroup> smem_a: array<f32, 256>;  // TILE_M(8) * 32

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let M = _params_[0];
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let tid = lid.x;
    let n = wid.x * 128u + tid;
    let m_base = wid.y * TILE_M;
    let n_valid = n < N;

    let w_base = select(0u, n * (K / 2u), n_valid);

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    var acc4: f32 = 0.0;
    var acc5: f32 = 0.0;
    var acc6: f32 = 0.0;
    var acc7: f32 = 0.0;

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let k_off = blk * 32u;

        // Cooperative A load: 128 threads load 256 elements (8 rows x 32 cols)
        {
            let idx0 = tid;           // 0..127
            let row0 = idx0 / 32u;    // 0..3
            let col0 = idx0 % 32u;
            let g_row0 = m_base + row0;
            smem_a[idx0] = select(0.0, t_read(&A, g_row0 * K + k_off + col0), g_row0 < M);
        }
        {
            let idx1 = tid + 128u;    // 128..255
            let row1 = idx1 / 32u;    // 4..7
            let col1 = idx1 % 32u;
            let g_row1 = m_base + row1;
            smem_a[idx1] = select(0.0, t_read(&A, g_row1 * K + k_off + col1), g_row1 < M);
        }

        workgroupBarrier();

        if (n_valid) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let w_blk_base = w_base + k_off / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let kl = j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                let w0 = (f32(b0 & 0xFu) - zp) * scale;
                let w1 = (f32(b0 >> 4u)  - zp) * scale;
                let w2 = (f32(b1 & 0xFu) - zp) * scale;
                let w3 = (f32(b1 >> 4u)  - zp) * scale;
                let w4 = (f32(b2 & 0xFu) - zp) * scale;
                let w5 = (f32(b2 >> 4u)  - zp) * scale;
                let w6 = (f32(b3 & 0xFu) - zp) * scale;
                let w7 = (f32(b3 >> 4u)  - zp) * scale;

                // Row 0
                acc0 += smem_a[kl]      * w0 + smem_a[kl + 1u] * w1
                      + smem_a[kl + 2u] * w2 + smem_a[kl + 3u] * w3
                      + smem_a[kl + 4u] * w4 + smem_a[kl + 5u] * w5
                      + smem_a[kl + 6u] * w6 + smem_a[kl + 7u] * w7;
                // Row 1
                acc1 += smem_a[32u + kl]      * w0 + smem_a[32u + kl + 1u] * w1
                      + smem_a[32u + kl + 2u] * w2 + smem_a[32u + kl + 3u] * w3
                      + smem_a[32u + kl + 4u] * w4 + smem_a[32u + kl + 5u] * w5
                      + smem_a[32u + kl + 6u] * w6 + smem_a[32u + kl + 7u] * w7;
                // Row 2
                acc2 += smem_a[64u + kl]      * w0 + smem_a[64u + kl + 1u] * w1
                      + smem_a[64u + kl + 2u] * w2 + smem_a[64u + kl + 3u] * w3
                      + smem_a[64u + kl + 4u] * w4 + smem_a[64u + kl + 5u] * w5
                      + smem_a[64u + kl + 6u] * w6 + smem_a[64u + kl + 7u] * w7;
                // Row 3
                acc3 += smem_a[96u + kl]      * w0 + smem_a[96u + kl + 1u] * w1
                      + smem_a[96u + kl + 2u] * w2 + smem_a[96u + kl + 3u] * w3
                      + smem_a[96u + kl + 4u] * w4 + smem_a[96u + kl + 5u] * w5
                      + smem_a[96u + kl + 6u] * w6 + smem_a[96u + kl + 7u] * w7;
                // Row 4
                acc4 += smem_a[128u + kl]      * w0 + smem_a[128u + kl + 1u] * w1
                      + smem_a[128u + kl + 2u] * w2 + smem_a[128u + kl + 3u] * w3
                      + smem_a[128u + kl + 4u] * w4 + smem_a[128u + kl + 5u] * w5
                      + smem_a[128u + kl + 6u] * w6 + smem_a[128u + kl + 7u] * w7;
                // Row 5
                acc5 += smem_a[160u + kl]      * w0 + smem_a[160u + kl + 1u] * w1
                      + smem_a[160u + kl + 2u] * w2 + smem_a[160u + kl + 3u] * w3
                      + smem_a[160u + kl + 4u] * w4 + smem_a[160u + kl + 5u] * w5
                      + smem_a[160u + kl + 6u] * w6 + smem_a[160u + kl + 7u] * w7;
                // Row 6
                acc6 += smem_a[192u + kl]      * w0 + smem_a[192u + kl + 1u] * w1
                      + smem_a[192u + kl + 2u] * w2 + smem_a[192u + kl + 3u] * w3
                      + smem_a[192u + kl + 4u] * w4 + smem_a[192u + kl + 5u] * w5
                      + smem_a[192u + kl + 6u] * w6 + smem_a[192u + kl + 7u] * w7;
                // Row 7
                acc7 += smem_a[224u + kl]      * w0 + smem_a[224u + kl + 1u] * w1
                      + smem_a[224u + kl + 2u] * w2 + smem_a[224u + kl + 3u] * w3
                      + smem_a[224u + kl + 4u] * w4 + smem_a[224u + kl + 5u] * w5
                      + smem_a[224u + kl + 6u] * w6 + smem_a[224u + kl + 7u] * w7;
            }
        }

        workgroupBarrier();
    }

    if (n_valid) {
        if (m_base < M)      { t_write(&Y, m_base * N + n, acc0); }
        if (m_base + 1u < M) { t_write(&Y, (m_base + 1u) * N + n, acc1); }
        if (m_base + 2u < M) { t_write(&Y, (m_base + 2u) * N + n, acc2); }
        if (m_base + 3u < M) { t_write(&Y, (m_base + 3u) * N + n, acc3); }
        if (m_base + 4u < M) { t_write(&Y, (m_base + 4u) * N + n, acc4); }
        if (m_base + 5u < M) { t_write(&Y, (m_base + 5u) * N + n, acc5); }
        if (m_base + 6u < M) { t_write(&Y, (m_base + 6u) * N + n, acc6); }
        if (m_base + 7u < M) { t_write(&Y, (m_base + 7u) * N + n, acc7); }
    }
}
)WGSL";

// [onnx_q4] matmul_q8_block32_subgroup
static const char* WGSL_MATMUL_Q8_BLOCK32_SUBGROUP = R"WGSL(
enable subgroups;

// Exact f32 decode path retained for devices where activation quantization
// changes model output. One logical 32-lane warp computes one output row.
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
fn scale_at(index: u32) -> f32 {
    let pair = unpack2x16float(scales[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) != 0u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let logical_warp = lid.x / 32u;
    let lane = lid.x & 31u;
    let n = wid.x * 8u + logical_warp;
    let valid = n < p.N;
    let words_per_row = p.K / 4u;
    let groups_per_row = p.K / 32u;
    var acc = 0.0;
    if (valid) {
        for (var word = lane; word < words_per_row; word += 32u) {
            let packed = W[n * words_per_row + word];
            let q = vec4<f32>(
                f32(i32(packed & 255u) - 128),
                f32(i32((packed >> 8u) & 255u) - 128),
                f32(i32((packed >> 16u) & 255u) - 128),
                f32(i32((packed >> 24u) & 255u) - 128));
            let k = word * 4u;
            acc += dot(vec4<f32>(X[k], X[k + 1u], X[k + 2u], X[k + 3u]), q) *
                   scale_at(n * groups_per_row + word / 8u);
        }
    }
    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
    if (lane == 0u && valid) { Y[n] = acc; }
}
)WGSL";

// [onnx_q4] matmul_q8_block32_dp4a
static const char* WGSL_MATMUL_Q8_BLOCK32_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Decode-specialized ORT block-Q8 projection. Quantize each 32-value
// activation block once per workgroup, then share it across eight output rows
// and use DP4A for the dot products. ORT weights are unsigned bytes with an
// implicit zero point of 128; XOR 0x80 converts them to signed packed bytes.
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> xq: array<u32, 64>;
var<workgroup> xs: array<f32, 8>;

fn scale_at(index: u32) -> f32 {
    let pair = unpack2x16float(scales[index >> 1u]);
    return select(pair.x, pair.y, (index & 1u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let logical_warp = lid.x / 32u;
    let lane = lid.x & 31u;
    let n = wid.x * 8u + logical_warp;
    let valid = n < p.N;

    let words_per_row = p.K / 4u;
    let groups_per_row = p.K / 32u;
    var acc = 0.0;
    let block = lid.x / 32u;
    let elem = lid.x & 31u;
    let pack_lane = elem & 3u;
    let pack = elem / 4u;

    for (var k0 = 0u; k0 < p.K; k0 += 256u) {
        let xv = X[k0 + lid.x];
        var xmax = abs(xv);
        xmax = max(xmax, subgroupShuffleXor(xmax, 16u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 8u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 4u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 2u));
        xmax = max(xmax, subgroupShuffleXor(xmax, 1u));
        let scale_x = xmax / 127.0;
        if (elem == 0u) { xs[block] = scale_x; }
        let safe_scale = select(1.0, scale_x, scale_x != 0.0);
        let qx = u32(clamp(i32(round(xv / safe_scale)), -127, 127)) & 255u;
        var packed_x = qx << (pack_lane * 8u);
        packed_x |= subgroupShuffleXor(packed_x, 1u);
        packed_x |= subgroupShuffleXor(packed_x, 2u);
        if (pack_lane == 0u) { xq[block * 8u + pack] = packed_x; }
        workgroupBarrier();

        if (valid) {
            let word = k0 / 4u + lane * 2u;
            let w0 = W[n * words_per_row + word] ^ 0x80808080u;
            let w1 = W[n * words_per_row + word + 1u] ^ 0x80808080u;
            let idot = dot4I8Packed(xq[lane * 2u], w0) +
                       dot4I8Packed(xq[lane * 2u + 1u], w1);
            let group = k0 / 32u + lane / 4u;
            acc += f32(idot) * xs[lane / 4u] *
                   scale_at(n * groups_per_row + group);
        }
        workgroupBarrier();
    }

    // Explicit logical-warp reduction is valid for both wave32 and wave64.
    acc += subgroupShuffleXor(acc, 16u);
    acc += subgroupShuffleXor(acc, 8u);
    acc += subgroupShuffleXor(acc, 4u);
    acc += subgroupShuffleXor(acc, 2u);
    acc += subgroupShuffleXor(acc, 1u);
    if (lane == 0u && valid) { Y[n] = acc; }
}
)WGSL";

// [onnx_q4] q4_add_norm_decode
static const char* WGSL_Q4_ADD_NORM_DECODE = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Fused: residual add + RMSNorm + Q4 matmul for decode (M=1).
// DP4A-accelerated. Computes X += Residual, then RMSNorm(X), then Q4 matmul.
// Eliminates the separate add_rms_norm dispatch.
//
// Dispatch: (ceil(N/32), 1, 1)
//
// Bindings:
//   0: X (read_write) — hidden state, K floats. Updated in-place: X += Residual
//   1: Residual (read) — residual vector (e.g. oproj output), K floats
//   2: B (read) — packed Q4 weights [N × K/8] as u32
//   3: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   4: Y (write) — output [N] fp32
//   5: _params_ — [K, N, 0, eps_as_u32]
//   6: NormW (read) — norm weight vector, K floats
//   7: Bias (read) — per-output bias

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Residual: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<u32>;
@group(0) @binding(3) var<storage, read> Scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;
@group(0) @binding(6) var<storage, read> NormW: array<f32>;
@group(0) @binding(7) var<storage, read> Bias: array<f32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;

var<workgroup> smem_xq: array<u32, 64>;
var<workgroup> smem_xs: array<f32, 8>;
var<workgroup> smem_rstd: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];
    let N = _params_[1];
    let eps = bitcast<f32>(_params_[3]);
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;

    // ── Pass 1: residual add + compute RMSNorm rstd ──────────────
    // Each thread handles K/256 elements. Add residual and accumulate sum_sq.
    var sum_sq: f32 = 0.0;
    for (var k = tid; k < K; k += 256u) {
        let v = X[k] + Residual[k];
        X[k] = v;  // write back the sum
        sum_sq += v * v;
    }
    // Reduce within warp
    sum_sq = subgroupAdd(sum_sq);
    // Reduce across warps using shared memory
    if (lane == 0u) {
        smem_xs[warp_id] = sum_sq;
    }
    workgroupBarrier();

    var total_sq: f32 = 0.0;
    if (tid < 8u) {
        total_sq = smem_xs[tid];
    }
    total_sq = subgroupAdd(total_sq);
    if (tid == 0u) {
        smem_rstd = 1.0 / sqrt(total_sq / f32(K) + eps);
    }
    workgroupBarrier();
    let rstd = smem_rstd;

    // Pre-compute column info
    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 32u + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;

        // ── Quantize normalized X to int8 ──
        let k = k_base + tid;
        let x_val = X[k] * NormW[k] * rstd;

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul ──
        let x_block = lane / 4u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                let w_block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                let xq0 = smem_xq[lane * 2u];
                let xq1 = smem_xq[lane * 2u + 1u];
                let x_scale = smem_xs[x_block];

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_scale;
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let warp_sum = subgroupAdd(acc[c]);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] = warp_sum + Bias[cols[c]];
        }
    }
}
)WGSL";

// [onnx_q4] q4_down_gelu_add_decode
static const char* WGSL_Q4_DOWN_GELU_ADD_DECODE = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Fused: GELU·mul + Q4 matmul + residual add for decode (M=1).
// DP4A-accelerated. Reads gateUpBuf (2*IM elements), applies gelu(gate)*up,
// quantizes to int8, then DP4A with Q4 W_down and adds to residual Y.
// GELU variant of q4_down_silu_add_decode for Gemma-4 and similar models.
//
// Dispatch: (ceil(N/32), 1, 1) where N = E (n_embd)
//
// Bindings:
//   0: GateUp (read) — 2*IM floats (gate || up concatenated)
//   1: B (read) — packed Q4 weights [N × K/8] as u32
//   2: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   3: Bias (read) — per-output bias
//   4: Y (read_write) — output += matmul result (residual add)
//   5: _params_ — [K=IM, N=E, IM, 0]

@group(0) @binding(0) var<storage, read> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;

var<workgroup> smem_xq: array<u32, 64>;
var<workgroup> smem_xs: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];   // IM (intermediate_size)
    let N = _params_[1];   // E (n_embd)
    let IM = _params_[2];  // same as K
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;

    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 32u + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;
        let k = k_base + tid;

        // ── Compute gelu(gate) * up and quantize to int8 ──
        let gate = GateUp[k];
        let up = GateUp[IM + k];
        let inner = 0.7978845608 * (gate + 0.044715 * gate * gate * gate);
        let gelu = 0.5 * gate * (1.0 + tanh(inner));
        let x_val = gelu * up;

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul ──
        let x_block = lane / 4u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                let w_block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                let xq0 = smem_xq[lane * 2u];
                let xq1 = smem_xq[lane * 2u + 1u];
                let x_scale = smem_xs[x_block];

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_scale;
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let warp_sum = subgroupAdd(acc[c]);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] += warp_sum + Bias[cols[c]];
        }
    }
}
)WGSL";

// [onnx_q4] q4_down_silu_add_decode
static const char* WGSL_Q4_DOWN_SILU_ADD_DECODE = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Fused: SiLU·mul + Q4 matmul + residual add for decode (M=1).
// DP4A-accelerated. Reads gateUpBuf (2*IM elements), applies silu(gate)*up,
// quantizes to int8, then DP4A with Q4 W_down and adds to residual Y.
//
// Dispatch: (ceil(N/32), 1, 1) where N = E (n_embd)
//
// Bindings:
//   0: GateUp (read) — 2*IM floats (gate || up concatenated)
//   1: B (read) — packed Q4 weights [N × K/8] as u32
//   2: Scales (read) — fp16 scales [N × nGroups] packed as u32
//   3: Bias (read) — per-output bias
//   4: Y (read_write) — output += matmul result (residual add)
//   5: _params_ — [K=IM, N=E, IM, 0]

@group(0) @binding(0) var<storage, read> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

const BK: u32 = 256u;
const COLS_PER_WARP: u32 = 4u;

var<workgroup> smem_xq: array<u32, 64>;
var<workgroup> smem_xs: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];   // IM (intermediate_size)
    let N = _params_[1];   // E (n_embd)
    let IM = _params_[2];  // same as K
    let tid = lid.x;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let n_blocks = K / 32u;
    let q4_words_per_row = K / 8u;

    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.x * 32u + warp_id * COLS_PER_WARP + c;
        col_valid[c] = cols[c] < N;
    }

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    var acc: array<f32, 4>;
    for (var c = 0u; c < COLS_PER_WARP; c++) { acc[c] = 0.0; }

    let nk = K / BK;
    for (var g = 0u; g < nk; g++) {
        let k_base = g * BK;
        let k = k_base + tid;

        // ── Compute silu(gate) * up and quantize to int8 ──
        let gate = GateUp[k];
        let up = GateUp[IM + k];
        let x_val = gate / (1.0 + exp(-gate)) * up;

        var max_val = abs(x_val);
        max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
        max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

        let x_scale = max_val / 127.0;
        if (elem_in_block == 0u) {
            smem_xs[block_id] = x_scale;
        }

        let safe_scale = select(1.0, x_scale, x_scale != 0.0);
        let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

        let byte_val = u32(q_val & 0xFF);
        let shifted = byte_val << (pack_lane * 8u);
        var packed = shifted;
        packed = packed | subgroupShuffleXor(packed, 1u);
        packed = packed | subgroupShuffleXor(packed, 2u);

        if (pack_lane == 0u) {
            smem_xq[block_id * 8u + pack_group] = packed;
        }

        workgroupBarrier();

        // ── DP4A matmul ──
        let x_block = lane / 4u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let n = cols[c];
                let q4_off = n * q4_words_per_row + (g * BK + lane * 8u) / 8u;
                let q4_packed = B[q4_off];

                let b0 = q4_packed & 0xFFu;
                let b1 = (q4_packed >> 8u) & 0xFFu;
                let b2 = (q4_packed >> 16u) & 0xFFu;
                let b3 = (q4_packed >> 24u) & 0xFFu;

                let w0 = (u32(b0 & 0xFu) - 8u) & 0xFFu;
                let w1 = (u32(b0 >> 4u) - 8u) & 0xFFu;
                let w2 = (u32(b1 & 0xFu) - 8u) & 0xFFu;
                let w3 = (u32(b1 >> 4u) - 8u) & 0xFFu;
                let w4 = (u32(b2 & 0xFu) - 8u) & 0xFFu;
                let w5 = (u32(b2 >> 4u) - 8u) & 0xFFu;
                let w6 = (u32(b3 & 0xFu) - 8u) & 0xFFu;
                let w7 = (u32(b3 >> 4u) - 8u) & 0xFFu;

                let wq0 = w0 | (w1 << 8u) | (w2 << 16u) | (w3 << 24u);
                let wq1 = w4 | (w5 << 8u) | (w6 << 16u) | (w7 << 24u);

                let w_block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(n * n_blocks + w_block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((n * n_blocks + w_block) & 1u) != 0u);

                let xq0 = smem_xq[lane * 2u];
                let xq1 = smem_xq[lane * 2u + 1u];
                let x_scale = smem_xs[x_block];

                let idot = dot4I8Packed(xq0, wq0) + dot4I8Packed(xq1, wq1);
                acc[c] += f32(idot) * w_scale * x_scale;
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let warp_sum = subgroupAdd(acc[c]);
        if (lane == 0u && col_valid[c]) {
            Y[cols[c]] += warp_sum + Bias[cols[c]];
        }
    }
}
)WGSL";

// [onnx_q4] q4_gather_batched
static const char* WGSL_Q4_GATHER_BATCHED = R"WGSL(
// Gather M rows from a repacked native-Q4 table and dequantize to fp32.
// Dispatch: (ceil(M*D/256), 1, 1).

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Tokens: array<i32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = P[0]; let D = P[1]; let vocab = P[2];
    let scale_out = bitcast<f32>(P[3]);
    let flat = gid.x;
    if (flat >= M * D) { return; }
    let row = flat / D; let i = flat % D;
    let token_i = Tokens[row];
    let token = select(0u, u32(token_i), token_i >= 0 && u32(token_i) < vocab);
    let wi = token * (D / 8u) + i / 8u;
    let q = (W[wi] >> (4u * (i & 7u))) & 15u;
    let block = token * (D / 32u) + i / 32u;
    let sp = unpack2x16float(Scales[block / 2u]);
    let s = select(sp.x, sp.y, (block & 1u) != 0u);
    Out[flat] = (f32(q) - 8.0) * s * scale_out;
}
)WGSL";

// ─── [quant_kq] ────────────────────────────────────────────────────

// [quant_kq] iq2s_matmul
static const char* WGSL_IQ2S_MATMUL = R"WGSL(
// IQ2_S matmul — Phase 4 (WIP, not yet validated)
//
// Block: 82 bytes / 256 elements
//   [0..1]   fp16 d
//   [2..33]  qs[32]      — 2-bit grid index low byte (4 per ib32)
//   [34..65] signs[32]   — 1 sign bit per element (overlapping qs region)
//   [66..73] qh[8]       — high bits (2 bits per ib32)
//   [74..81] scales[8]   — 4-bit sub-block scales (2 packed per byte = 16 sub-blocks? actually 8)
//
// Per sub-block ib32 in 0..7:
//   db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25
//   db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25
//   For l in 0..3:
//     grid_idx = qs[ib32*4 + l] | ((qh[ib32] << (8 - 2*l)) & 0x300)   // 10-bit, 0..1023
//     grid64 = iq2s_grid[grid_idx]   // 8 packed uint8 magnitudes
//     For j in 0..7:
//       mag = (grid64 >> (8*j)) & 0xff
//       sign_bit = (signs[ib32*4 + l] >> j) & 1
//       val = db[l/2] * f32(mag) * (sign_bit ? -1 : +1)
//
// Codebook: 1024 entries of uint64. Stored as 2048 uint32 (low/high pairs).
//
// NOTE: not yet registered or dispatched.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ2S:   array<u32>;
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>;  // iq2s_grid as 2048 u32s (2 per entry)
@group(0) @binding(3) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(5) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 21u;  // host packs 82-byte IQ2_S blocks padded to 84 bytes

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ2S[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) {
        f = (sign << 31u) | (mant << 13u);
    } else if (exp == 31u) {
        f = (sign << 31u) | 0x7F800000u | (mant << 13u);
    } else {
        f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];

    let col = col_base + warp_id;
    if (col >= N) {
        return;
    }

    let x_base = row * K;
    var acc: f32 = 0.0;

    let row_w_base = col * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        // Cooperative X load
        let x_idx = b * QK_K + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx];
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        let blk_word_base = row_w_base + b * BLOCK_WORDS;

        // Super-block scale d (fp16 at bytes 0..1)
        let d_u32 = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let d     = fp16_to_f32(d_u32);

        // One lane per ib32 (8 used, 24 idle)
        let ib32 = lane;
        if (ib32 < 8u) {
            let scales_byte = get_u8(blk_word_base, 74u + ib32);
            let db0 = d * (0.5 + f32(scales_byte & 0xfu)) * 0.25;
            let db1 = d * (0.5 + f32(scales_byte >> 4u)) * 0.25;
            let qh_byte = get_u8(blk_word_base, 66u + ib32);

            var partial: f32 = 0.0;
            for (var l = 0u; l < 4u; l = l + 1u) {
                let qs_byte = get_u8(blk_word_base, 2u + ib32 * 4u + l);
                let grid_idx = qs_byte | (((qh_byte << (8u - 2u * l)) & 0x300u));
                // Read 64-bit codebook entry (2 u32s)
                let cb_lo = Codebook[grid_idx * 2u + 0u];
                let cb_hi = Codebook[grid_idx * 2u + 1u];
                let signs_byte = get_u8(blk_word_base, 34u + ib32 * 4u + l);
                let dl = select(db1, db0, l < 2u);  // l in 0..3, db0 covers l/2==0, db1 covers l/2==1

                for (var j = 0u; j < 4u; j = j + 1u) {
                    let mag_lo = (cb_lo >> (8u * j)) & 0xFFu;
                    let mag_hi = (cb_hi >> (8u * j)) & 0xFFu;
                    let sign_lo = (signs_byte >> j) & 1u;
                    let sign_hi = (signs_byte >> (j + 4u)) & 1u;
                    let v_lo = dl * f32(mag_lo) * select(1.0, -1.0, sign_lo == 1u);
                    let v_hi = dl * f32(mag_hi) * select(1.0, -1.0, sign_hi == 1u);
                    let k_off = ib32 * 32u + l * 8u + j;
                    partial = partial + v_lo * smem_x[k_off];
                    partial = partial + v_hi * smem_x[k_off + 4u];
                }
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] iq2s_matmul_moe
static const char* WGSL_IQ2S_MATMUL_MOE = R"WGSL(
// IQ2_S MoE matmul — indirect-via-buffer expert offset
// Same pattern as iq3s_matmul_moe but for IQ2_S blocks.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ2S:   array<u32>;
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>;  // iq2s_grid as 2 u32 per entry
@group(0) @binding(3) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(5) var<storage, read>       offsets:  array<u32>;
@group(0) @binding(6) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 21u;

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ2S[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) { f = (sign << 31u) | (mant << 13u); }
    else if (exp == 31u) { f = (sign << 31u) | 0x7F800000u | (mant << 13u); }
    else { f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u); }
    return bitcast<f32>(f);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];
    let slot_idx       = _params_[5];
    let expert_row_off = offsets[slot_idx];

    let col = col_base + warp_id;

    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_word_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let d     = fp16_to_f32(d_u32);

        let ib32 = lane;
        if (ib32 < 8u) {
            let scales_byte = get_u8(blk_word_base, 74u + ib32);
            let db0 = d * (0.5 + f32(scales_byte & 0xfu)) * 0.25;
            let db1 = d * (0.5 + f32(scales_byte >> 4u)) * 0.25;
            let qh_byte = get_u8(blk_word_base, 66u + ib32);

            var partial: f32 = 0.0;
            for (var l = 0u; l < 4u; l = l + 1u) {
                let qs_byte = get_u8(blk_word_base, 2u + ib32 * 4u + l);
                let grid_idx = qs_byte | (((qh_byte << (8u - 2u * l)) & 0x300u));
                let cb_lo = Codebook[grid_idx * 2u + 0u];
                let cb_hi = Codebook[grid_idx * 2u + 1u];
                let signs_byte = get_u8(blk_word_base, 34u + ib32 * 4u + l);
                let dl = select(db1, db0, l < 2u);

                for (var j = 0u; j < 4u; j = j + 1u) {
                    let mag_lo = (cb_lo >> (8u * j)) & 0xFFu;
                    let mag_hi = (cb_hi >> (8u * j)) & 0xFFu;
                    let sign_lo = (signs_byte >> j) & 1u;
                    let sign_hi = (signs_byte >> (j + 4u)) & 1u;
                    let v_lo = dl * f32(mag_lo) * select(1.0, -1.0, sign_lo == 1u);
                    let v_hi = dl * f32(mag_hi) * select(1.0, -1.0, sign_hi == 1u);
                    let k_off = ib32 * 32u + l * 8u + j;
                    partial = partial + v_lo * smem_x[k_off];
                    partial = partial + v_hi * smem_x[k_off + 4u];
                }
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] iq3s_matmul
static const char* WGSL_IQ3S_MATMUL = R"WGSL(
// IQ3_S matmul — Phase 4 (WIP, untested at runtime)
//
// Per-element formula (port of dequantize_row_iq3_s from llama.cpp):
//   block: 110 bytes for QK_K=256 elements
//     [0..1]    fp16 d
//     [2..65]   qs[64]
//     [66..73]  qh[8]
//     [74..105] signs[32]
//     [106..109] scales[4]   (4-bit sub-block scales, 2 per byte)
//   Per sub-block of 32 elements (ib32 in 0..7):
//     sub_scale_4bit = (scales[ib32/2] >> (4*(ib32 & 1))) & 0xf
//     db             = d * (1 + 2*sub_scale_4bit)
//     For l in 0..3, p in 0..1:
//       qs_byte = qs[ib32*8 + 2*l + p]
//       qh_bit  = (qh[ib32] >> (2*l + p)) & 1
//       grid_idx = qs_byte | (qh_bit << 8)              (9-bit, 0..511)
//       grid32  = Codebook[grid_idx]                    (4 packed uint8)
//       For j in 0..3:
//         mag = (grid32 >> (8*j)) & 0xff                (uint8 magnitude)
//         sign_bit = (signs[ib32*4 + l] >> (4*p + j)) & 1
//         val = db * f32(mag) * (sign_bit ? -1.0 : 1.0)
//
// NOTE: still wiring up — not yet registered or dispatched by model_runner.cpp.
// Use op_test_runner to validate against the CPU dq_iq3_s reference.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:       array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ3S:  array<u32>;  // raw 110-byte blocks packed as u32 words
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>; // iq3s_grid[512]
@group(0) @binding(3) var<storage, read>       Bias:    array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:       array<f32>;
@group(0) @binding(5) var<storage, read>       _params_: array<u32>; // [K, N, n_blocks, row_stride_words, y_offset, expert_row_offset]

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ3S[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) {
        f = (sign << 31u) | (mant << 13u);
    } else if (exp == 31u) {
        f = (sign << 31u) | 0x7F800000u | (mant << 13u);
    } else {
        f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f);
}

// Decode one element at (ib32 in 0..7, inner in 0..31). Returns dequantized
// fp32 value (without the X multiply).
fn dq_one_iq3s(base_word: u32, d_scale: f32, ib32: u32, inner: u32) -> f32 {
    // inner = 4*l + j  with l in 0..3 splitting into a pair (p=0 first 4 elts,
    // p=1 next 4 elts of the 8-elt mini-group; ib32 groups of 8 cover 32).
    // Actually llama.cpp's loop is l=0..3, p=0..1, j=0..3 → 32 elts.
    // Here inner = (2*l + p)*4 + j  for canonical addressing.
    let lp = inner / 4u;          // 0..7  → splits into l (0..3) and p (0..1)
    let l  = lp / 2u;
    let p  = lp & 1u;
    let j  = inner & 3u;

    // qs region starts at byte 2; one qs byte per (2*l+p) within this ib32.
    let qs_byte = get_u8(base_word, 2u + ib32 * 8u + 2u * l + p);
    // qh region starts at byte 66; one byte per ib32.
    let qh_byte = get_u8(base_word, 66u + ib32);
    let qh_bit  = (qh_byte >> (2u * l + p)) & 1u;
    let grid_idx = qs_byte | (qh_bit << 8u);              // 0..511
    let grid32   = Codebook[grid_idx];
    let mag      = (grid32 >> (8u * j)) & 0xFFu;          // uint8 magnitude

    // signs region starts at byte 74; 4 bytes per ib32, one byte per l.
    let signs_byte = get_u8(base_word, 74u + ib32 * 4u + l);
    let sign_bit = (signs_byte >> (4u * p + j)) & 1u;

    // sub-block scale: 8 sub-blocks → 4 bytes, 2 sub-blocks per byte (low / high nibble)
    let scales_byte = get_u8(base_word, 106u + ib32 / 2u);
    let sub_4bit    = (scales_byte >> (4u * (ib32 & 1u))) & 0xFu;
    let db          = d_scale * (1.0 + 2.0 * f32(sub_4bit));

    let val_unsigned = db * f32(mag);
    return select(val_unsigned, -val_unsigned, sign_bit == 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];
    let expert_row_off = _params_[5];  // 0 for non-MoE; expert_idx * N for indirect MoE expert dispatch

    // One row of X (size K) is shared across all TILE_N output columns; load once.
    let x_base = row * K;
    var acc: f32 = 0.0;

    let col = col_base + warp_id;
    if (col >= N) {
        return;
    }

    // Each row of W_IQ3S has n_blocks * (110/4) words but actually packed as
    // row_stride_w words per row. Block n in row `col` starts at:
    //   row_word_base + n * (110/4) = col * row_stride_w + n * 28? (110 isn't u32 aligned)
    // The packing convention in backpack is row-aligned to u32 strides; the
    // host-side packer (TODO: add pack_iq3s) sets row_stride_w. For now we
    // assume row_stride_w covers n_blocks blocks with no padding between blocks.
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        // Cooperative X load for this super-block — all 256 threads each load 1 float
        let x_idx = b * QK_K + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx];
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        // The block starts at (row_w_base + b * 28) words.
        // 28 words = 112 bytes (we pad 110 → 112 host-side).
        let blk_word_base = row_w_base + b * 28u;

        // Read super-block scale d (fp16 in bytes 0..1)
        let d_u32 = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let d     = fp16_to_f32(d_u32);

        // Each lane within a warp does 8 elements (8 lanes per 32-element ib32).
        // ib32 = warp_id_inside_block; we use lane / 8 to pick the inner offset (0..3).
        // Simpler: each of the 32 lanes does one ib32, computes 8 dequant values, MACs.
        let ib32 = lane;
        if (ib32 < 8u) {
            // unrolled-ish; let WGSL inline the inner work
            var partial: f32 = 0.0;
            for (var inner = 0u; inner < 32u; inner = inner + 1u) {
                let w = dq_one_iq3s(blk_word_base, d, ib32, inner);
                let k_offset = b * QK_K + ib32 * 32u + inner;
                partial = partial + w * smem_x[ib32 * 32u + inner];
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    // Reduce 32 lane partials in this warp (only first 8 lanes had work)
    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] iq3s_matmul_moe
static const char* WGSL_IQ3S_MATMUL_MOE = R"WGSL(
// IQ3_S MoE matmul — indirect-via-buffer expert offset
//
// Differs from iq3s_matmul: reads expert_row_off from a separate GPU buffer
// indexed by slot_idx. This lets the dispatch fix slot_idx at build-time
// while the actual expert offset is computed per-decode by moe_compute_offsets
// from the routing decision.
//
// Bindings (7):
//   0: X        f32         — input row
//   1: W_IQ3S   u32         — fused-expert IQ3_S weights
//   2: Codebook u32         — iq3s_grid
//   3: Bias     f32
//   4: Y        f32         — output (per-expert result, accumulated by host)
//   5: offsets  u32         — [k] row offsets, one per slot
//   6: _params_ u32         — [K, N, n_blocks, row_stride_words, y_offset, slot_idx]

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ3S:   array<u32>;
@group(0) @binding(2) var<storage, read>       Codebook: array<u32>;
@group(0) @binding(3) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(4) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(5) var<storage, read>       offsets:  array<u32>;
@group(0) @binding(6) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 28u;

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ3S[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) { f = (sign << 31u) | (mant << 13u); }
    else if (exp == 31u) { f = (sign << 31u) | 0x7F800000u | (mant << 13u); }
    else { f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u); }
    return bitcast<f32>(f);
}

fn dq_one_iq3s(base_word: u32, d_scale: f32, ib32: u32, inner: u32) -> f32 {
    let lp = inner / 4u;
    let l  = lp / 2u;
    let p  = lp & 1u;
    let j  = inner & 3u;
    let qs_byte = get_u8(base_word, 2u + ib32 * 8u + 2u * l + p);
    let qh_byte = get_u8(base_word, 66u + ib32);
    let qh_bit  = (qh_byte >> (2u * l + p)) & 1u;
    let grid_idx = qs_byte | (qh_bit << 8u);
    let grid32   = Codebook[grid_idx];
    let mag      = (grid32 >> (8u * j)) & 0xFFu;
    let signs_byte = get_u8(base_word, 74u + ib32 * 4u + l);
    let sign_bit = (signs_byte >> (4u * p + j)) & 1u;
    let scales_byte = get_u8(base_word, 106u + ib32 / 2u);
    let sub_4bit    = (scales_byte >> (4u * (ib32 & 1u))) & 0xFu;
    let db          = d_scale * (1.0 + 2.0 * f32(sub_4bit));
    let val_unsigned = db * f32(mag);
    return select(val_unsigned, -val_unsigned, sign_bit == 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];
    let slot_idx       = _params_[5];
    let expert_row_off = offsets[slot_idx];  // buffer-driven, set per-decode

    let col = col_base + warp_id;

    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_word_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let d     = fp16_to_f32(d_u32);

        let ib32 = lane;
        if (ib32 < 8u) {
            var partial: f32 = 0.0;
            for (var inner = 0u; inner < 32u; inner = inner + 1u) {
                let w = dq_one_iq3s(blk_word_base, d, ib32, inner);
                partial = partial + w * smem_x[ib32 * 32u + inner];
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] iq4xs_matmul
static const char* WGSL_IQ4XS_MATMUL = R"WGSL(
// IQ4_XS matmul — Phase 4
//
// Block: 136 bytes / 256 elements
//   [0..1]   fp16 d
//   [2..3]   scales_h (u16)
//   [4..7]   scales_l[4]    — packed 4-bit + 4-bit per sub-block
//   [8..135] qs[128]        — 4-bit quants, packed 2 per byte
//
// Per sub-block ib in 0..7:
//   ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)
//   dl = d * (ls - 32)
//   For 32 quants in this sub-block: y[j] = dl * kvalues_iq4nl[nibble]
//
// kvalues_iq4nl is a 16-entry table; embedded as const array.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ4XS:  array<u32>;
@group(0) @binding(2) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 34u;  // 136 bytes / 4

var<workgroup> smem_x: array<f32, 256>;

// kvalues_iq4nl
const KV0:f32=-127.0; const KV1:f32=-104.0; const KV2:f32=-83.0; const KV3:f32=-65.0;
const KV4:f32=-49.0;  const KV5:f32=-35.0;  const KV6:f32=-22.0; const KV7:f32=-10.0;
const KV8:f32=1.0;    const KV9:f32=13.0;   const KVA:f32=25.0;  const KVB:f32=38.0;
const KVC:f32=53.0;   const KVD:f32=69.0;   const KVE:f32=89.0;  const KVF:f32=113.0;

fn iq4nl(n: u32) -> f32 {
    let arr = array<f32, 16>(KV0,KV1,KV2,KV3,KV4,KV5,KV6,KV7,KV8,KV9,KVA,KVB,KVC,KVD,KVE,KVF);
    return arr[n & 0xfu];
}

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ4XS[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) { f = (sign << 31u) | (mant << 13u); }
    else if (exp == 31u) { f = (sign << 31u) | 0x7F800000u | (mant << 13u); }
    else { f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u); }
    return bitcast<f32>(f);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;
    let col = col_base + warp_id;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];
    let expert_row_off = _params_[5];


    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_base, 0u) | (get_u8(blk_base, 1u) << 8u);
        let d = fp16_to_f32(d_u32);
        let scales_h = get_u8(blk_base, 2u) | (get_u8(blk_base, 3u) << 8u);

        // One lane per sub-block (8 sub-blocks × 32 elements)
        let ib = lane;
        if (ib < 8u) {
            let scales_l_byte = get_u8(blk_base, 4u + ib / 2u);
            let ls_lo = (scales_l_byte >> (4u * (ib & 1u))) & 0xfu;
            let ls_hi = (scales_h >> (2u * ib)) & 3u;
            let ls = ls_lo | (ls_hi << 4u);
            let dl = d * (f32(ls) - 32.0);
            var partial: f32 = 0.0;
            for (var j = 0u; j < 16u; j = j + 1u) {
                let q_byte = get_u8(blk_base, 8u + ib * 16u + j);
                let n_lo = q_byte & 0xfu;
                let n_hi = q_byte >> 4u;
                let v_lo = dl * iq4nl(n_lo);
                let v_hi = dl * iq4nl(n_hi);
                partial = partial + v_lo * smem_x[ib * 32u + j];
                partial = partial + v_hi * smem_x[ib * 32u + 16u + j];
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] iq4xs_matmul_moe
static const char* WGSL_IQ4XS_MATMUL_MOE = R"WGSL(
// IQ4_XS MoE matmul — indirect-via-buffer expert offset

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_IQ4XS:  array<u32>;
@group(0) @binding(2) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(4) var<storage, read>       offsets:  array<u32>;
@group(0) @binding(5) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 34u;

var<workgroup> smem_x: array<f32, 256>;

const KV0:f32=-127.0; const KV1:f32=-104.0; const KV2:f32=-83.0; const KV3:f32=-65.0;
const KV4:f32=-49.0;  const KV5:f32=-35.0;  const KV6:f32=-22.0; const KV7:f32=-10.0;
const KV8:f32=1.0;    const KV9:f32=13.0;   const KVA:f32=25.0;  const KVB:f32=38.0;
const KVC:f32=53.0;   const KVD:f32=69.0;   const KVE:f32=89.0;  const KVF:f32=113.0;

fn iq4nl(n: u32) -> f32 {
    let arr = array<f32, 16>(KV0,KV1,KV2,KV3,KV4,KV5,KV6,KV7,KV8,KV9,KVA,KVB,KVC,KVD,KVE,KVF);
    return arr[n & 0xfu];
}

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_IQ4XS[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) { f = (sign << 31u) | (mant << 13u); }
    else if (exp == 31u) { f = (sign << 31u) | 0x7F800000u | (mant << 13u); }
    else { f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u); }
    return bitcast<f32>(f);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;
    let col = col_base + warp_id;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];
    let slot_idx       = _params_[5];
    let expert_row_off = offsets[slot_idx];


    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = (col + expert_row_off) * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_base, 0u) | (get_u8(blk_base, 1u) << 8u);
        let d = fp16_to_f32(d_u32);
        let scales_h = get_u8(blk_base, 2u) | (get_u8(blk_base, 3u) << 8u);

        let ib = lane;
        if (ib < 8u) {
            let scales_l_byte = get_u8(blk_base, 4u + ib / 2u);
            let ls_lo = (scales_l_byte >> (4u * (ib & 1u))) & 0xfu;
            let ls_hi = (scales_h >> (2u * ib)) & 3u;
            let ls = ls_lo | (ls_hi << 4u);
            let dl = d * (f32(ls) - 32.0);
            var partial: f32 = 0.0;
            for (var j = 0u; j < 16u; j = j + 1u) {
                let q_byte = get_u8(blk_base, 8u + ib * 16u + j);
                partial = partial + dl * iq4nl(q_byte & 0xfu) * smem_x[ib * 32u + j];
                partial = partial + dl * iq4nl(q_byte >> 4u) * smem_x[ib * 32u + 16u + j];
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] mxfp4_matmul
static const char* WGSL_MXFP4_MATMUL = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_blocks: array<i32>;
@group(0) @binding(2) var<storage, read> W_scales: array<i32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

// FP4 E2M1 lookup table (indexed by 3-bit abs value)
// 0→0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
const FP4_LUT = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];
    let N = _params_[1];
    let stride_blocks = _params_[2];
    let stride_scales = _params_[3];

    let col = gid.x;  // output column (N dimension)
    let row = wid.y;   // token index (T dimension)
    if (col >= N) { return; }

    let n_chunks = K / 32u;  // MXFP4 block size = 32
    let blocks_base = col * stride_blocks;
    let scales_base = col * stride_scales;
    let x_base = row * K;

    var acc: f32 = 0.0;

    for (var chunk_i = 0u; chunk_i < n_chunks; chunk_i++) {
        // Load E8M0 scale: 4 bytes packed per i32
        let scale_word = W_scales[scales_base + chunk_i / 4u];
        let scale_byte = (u32(scale_word) >> ((chunk_i % 4u) * 8u)) & 0xFFu;
        let scale = exp2(f32(scale_byte) - 127.0);

        let off = chunk_i * 32u;

        // Process 32 FP4 elements (4 i32 words × 8 nibbles each)
        for (var w_i = 0u; w_i < 4u; w_i++) {
            let word = u32(W_blocks[blocks_base + (off / 8u) + w_i]);

            // Unroll 8 nibbles per word
            for (var nib = 0u; nib < 8u; nib++) {
                let k = off + w_i * 8u + nib;
                let nibble = (word >> (nib * 4u)) & 0xFu;
                let sign = f32(nibble >> 3u);
                let abs_val = FP4_LUT[nibble & 7u];
                let w = abs_val * (1.0 - 2.0 * sign);
                acc += X[x_base + k] * w * scale;
            }
        }
    }

    Y[row * N + col] = acc + Bias[col];
}
)WGSL";

// [quant_kq] q2k_matmul
static const char* WGSL_Q2K_MATMUL = R"WGSL(
// Q2_K matmul — Phase 4
//
// Block: 84 bytes / 256 elements
//   [0..1]   fp16 d
//   [2..3]   fp16 dmin
//   [4..19]  scales[16]    — 4-bit scale + 4-bit min per 16-element sub-block
//   [20..83] qs[64]        — 2-bit quants, packed 4 per byte
//
// Per llama.cpp dequant: 8 sub-blocks per super-block. Each sub-block uses
//   sc = scales[is]
//   dl = d * (sc & 0xf), ml = dmin * (sc >> 4)
//   val = dl * (int8)((q[l] >> shift) & 3) - ml
// with sub-blocks of 16 elements organized in groups of 128 with 4 shifts.
//
// NOTE: not yet registered or dispatched.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_Q2K:    array<u32>;
@group(0) @binding(2) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 21u;  // 84 bytes / 4

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q2K[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) { f = (sign << 31u) | (mant << 13u); }
    else if (exp == 31u) { f = (sign << 31u) | 0x7F800000u | (mant << 13u); }
    else { f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u); }
    return bitcast<f32>(f);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;
    let col = col_base + warp_id;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];


    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = col * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_word_base = row_w_base + b * BLOCK_WORDS;
        let d_u32    = get_u8(blk_word_base, 0u) | (get_u8(blk_word_base, 1u) << 8u);
        let dmin_u32 = get_u8(blk_word_base, 2u) | (get_u8(blk_word_base, 3u) << 8u);
        let d    = fp16_to_f32(d_u32);
        let dmin = fp16_to_f32(dmin_u32);

        // One lane per sub-block (16 sub-blocks of 16 elements each)
        let isb = lane;
        if (isb < 16u) {
            let sc = get_u8(blk_word_base, 4u + isb);
            let dl = d    * f32(sc & 0xfu);
            let ml = dmin * f32(sc >> 4u);
            // shift selects which 2-bit field (0,2,4,6) based on isb's chunk
            let chunk = isb / 4u;          // 0..3
            let shift = (chunk * 2u);      // 0,2,4,6
            let within = isb % 4u;         // 0..3
            // qs layout: 4 chunks × 16-byte groups; within-chunk picks high vs low halves
            // Following llama.cpp: each chunk has 32 bytes of qs covering 128 elements
            // (4 sub-blocks × 16 elements × 2 bits = 128 bits = 16 bytes... wait).
            // Reference walks q ptr by 32 each n+=128 iter.
            //
            // Use simpler indexing: for sub-block isb, the 16 quants are at
            //   q[(isb/8)*32 + offset]  with shift = (isb%8)/2 * 2
            //   offset = (isb & 1) * 16
            let group = isb / 8u;          // 0 or 1 (which 128-elt group)
            let sub_in_group = isb % 8u;
            let q_shift = (sub_in_group / 2u) * 2u;
            let q_offset = (sub_in_group & 1u) * 16u;
            var partial: f32 = 0.0;
            for (var l = 0u; l < 16u; l = l + 1u) {
                let q_byte = get_u8(blk_word_base, 20u + group * 32u + q_offset + l);
                let q2 = (q_byte >> q_shift) & 3u;
                // q is unsigned 0..3 but cast to int8 in reference. Here non-negative so just f32.
                let val = dl * f32(q2) - ml;
                let k_off = group * 128u + sub_in_group * 16u + l;
                partial = partial + val * smem_x[k_off];
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] q3k_matmul
static const char* WGSL_Q3K_MATMUL = R"WGSL(
// Q3_K matmul — Phase 4
//
// Block: 110 bytes / 256 elements
//   [0..1]    fp16 d
//   [2..33]   hmask[32]     — 1 high bit per quant
//   [34..97]  qs[64]        — 2-bit low quants
//   [98..109] scales[12]    — packed 16 × 6-bit signed scales (offset by 32)
//
// Sub-blocks: 16 of 16 elements each. Per element:
//   sc = unpacked_scale[is] - 32  (signed)
//   dl = d * sc
//   val = dl * ( (q[l] >> shift) & 3 - (hm[l] & mask ? 0 : 4) )
//
// NOTE: not yet registered or dispatched.

enable subgroups;

@group(0) @binding(0) var<storage, read>       X:        array<f32>;
@group(0) @binding(1) var<storage, read>       W_Q3K:    array<u32>;
@group(0) @binding(2) var<storage, read>       Bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> Y:        array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K:   u32 = 256u;
const BLOCK_WORDS: u32 = 28u;  // 110 bytes padded to 112

var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_word: u32, byte_off: u32) -> u32 {
    let wi = base_word + byte_off / 4u;
    let sh = (byte_off % 4u) * 8u;
    return (W_Q3K[wi] >> sh) & 0xFFu;
}

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp  = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: u32 = 0u;
    if (exp == 0u) { f = (sign << 31u) | (mant << 13u); }
    else if (exp == 31u) { f = (sign << 31u) | 0x7F800000u | (mant << 13u); }
    else { f = (sign << 31u) | ((exp + 112u) << 23u) | (mant << 13u); }
    return bitcast<f32>(f);
}

// Unpack 12 bytes of scales into 16 signed 6-bit values stored as u32.
// Mirrors llama.cpp aux[] computation. Returns scale[is] in 0..63 (need to subtract 32).
fn unpack_scale(blk_base: u32, is: u32) -> u32 {
    // Read aux[0..3] = scales[0..11] reorganized into 4 u32s.
    let s0_a = get_u8(blk_base, 98u + 0u);
    let s0_b = get_u8(blk_base, 98u + 1u);
    let s0_c = get_u8(blk_base, 98u + 2u);
    let s0_d = get_u8(blk_base, 98u + 3u);
    let s1_a = get_u8(blk_base, 98u + 4u);
    let s1_b = get_u8(blk_base, 98u + 5u);
    let s1_c = get_u8(blk_base, 98u + 6u);
    let s1_d = get_u8(blk_base, 98u + 7u);
    let s2_a = get_u8(blk_base, 98u + 8u);
    let s2_b = get_u8(blk_base, 98u + 9u);
    let s2_c = get_u8(blk_base, 98u + 10u);
    let s2_d = get_u8(blk_base, 98u + 11u);
    let aux0 = s0_a | (s0_b << 8u) | (s0_c << 16u) | (s0_d << 24u);
    let aux1 = s1_a | (s1_b << 8u) | (s1_c << 16u) | (s1_d << 24u);
    let aux2 = s2_a | (s2_b << 8u) | (s2_c << 16u) | (s2_d << 24u);
    let kmask1: u32 = 0x03030303u;
    let kmask2: u32 = 0x0f0f0f0fu;
    let tmp = aux2;
    let n0 = (aux0 & kmask2) | (((tmp >>  0u) & kmask1) << 4u);
    let n1 = (aux1 & kmask2) | (((tmp >>  2u) & kmask1) << 4u);
    let n2 = ((aux0 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    let n3 = ((aux1 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
    // Pick byte `is` from {n0, n1, n2, n3}
    let nibbles = array<u32, 4>(n0, n1, n2, n3);
    let group = is / 4u;          // 0..3
    let within = is & 3u;
    let n = nibbles[group];
    return (n >> (within * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane    = tid % 32u;
    let col_base = wid.y * TILE_N;
    let col = col_base + warp_id;

    let K              = _params_[0];
    let N              = _params_[1];
    let n_blocks       = _params_[2];
    let row_stride_w   = _params_[3];
    let y_offset       = _params_[4];


    let x_base = row * K;
    var acc: f32 = 0.0;
    let row_w_base = col * row_stride_w;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let x_idx = b * QK_K + tid;
        smem_x[tid] = select(0.0, X[x_base + x_idx], x_idx < K);
        workgroupBarrier();

        let blk_base = row_w_base + b * BLOCK_WORDS;
        let d_u32 = get_u8(blk_base, 0u) | (get_u8(blk_base, 1u) << 8u);
        let d = fp16_to_f32(d_u32);

        // One lane per sub-block (16 sub-blocks × 16 elements)
        let isb = lane;
        if (isb < 16u) {
            let raw_scale = unpack_scale(blk_base, isb);
            let dl = d * (f32(raw_scale) - 32.0);
            // Following the reference's nested loop: per 128-elt outer (n+=128)
            // with 4 sub-blocks (j=0..3) × 2 inner (l+0, l+16) × shift+=2
            // and m doubles each j. Map isb in 0..15 → (n_group, j, half):
            //   n_group = isb / 8  (0 or 1)
            //   sub_in_group = (isb % 8)        // 0..7
            //   j = sub_in_group / 2            // 0..3 — picks shift
            //   half = sub_in_group & 1         // 0=low (l 0..15), 1=high (l 16..31)
            //   m = 1u << ((isb % 8) / 2 + n_group * 4)   — hmask bit for THIS group
            let n_group = isb / 8u;
            let sub_in_group = isb % 8u;
            let j = sub_in_group / 2u;
            let half = sub_in_group & 1u;
            let shift = j * 2u;
            let m: u32 = 1u << (j + n_group * 4u);
            let q_base_off = 34u + n_group * 32u;
            let hm_base_off = 2u;
            var partial: f32 = 0.0;
            for (var l = 0u; l < 16u; l = l + 1u) {
                let q_off = half * 16u + l;
                let q_byte = get_u8(blk_base, q_base_off + q_off);
                let hm_byte = get_u8(blk_base, hm_base_off + q_off);
                let q_low = (q_byte >> shift) & 3u;
                let h_sub = select(4u, 0u, (hm_byte & m) != 0u);
                let q_signed = i32(q_low) - i32(h_sub);
                let val = dl * f32(q_signed);
                let k_off = n_group * 128u + j * 32u + half * 16u + l;
                partial = partial + val * smem_x[k_off];
            }
            acc = acc + partial;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[y_offset + row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] q4k_down_gelu
static const char* WGSL_Q4K_DOWN_GELU = R"WGSL(
enable subgroups;

// Fused GELU-mul + Q4_K matmul (down projection)
// Reads gate/up from GateUp buffer, computes GELU(gate)*up on-the-fly
// into shared memory, then performs Q4_K matrix-vector multiply.
// Replaces 2 dispatches: gelu_mul_fused + q4k_matmul
//
// Grid: (1, ceil(N/8), 1) where N = nEmbd (output dim)
// WG: 256 threads = 8 warps, each warp handles one output column

@group(0) @binding(0) var<storage, read> GateUp: array<f32>; // 2*K floats (gate || up)
@group(0) @binding(1) var<storage, read> W_Q4K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 36u;

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

    let K = _params_[0];  // intermediate_size
    let N = _params_[1];  // nEmbd
    let n_blocks = _params_[2];
    let row_stride_words = _params_[3];
    let y_offset = _params_[4];

    let x_base = row * K;
    var acc: f32 = 0.0;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let k_start = b * QK_K;

        // Fused GELU-mul + cooperative X load
        let x_idx = k_start + tid;
        if (x_idx < K) {
            let g = GateUp[x_idx];
            let u = GateUp[K + x_idx];
            let inner = 0.7978845608 * (g + 0.044715 * g * g * g);
            let gelu = 0.5 * g * (1.0 + tanh(inner));
            smem_x[tid] = gelu * u;
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        if (col < N) {
            let block_base = col * row_stride_words + b * BLOCK_WORDS;

            let dd = unpack2x16float(W_Q4K[block_base]);
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
                let g_idx = sb / 2u;
                let hi = (sb & 1u) == 1u;

                let i = lane;
                let local_idx = sb * 32u + i;
                let qb = get_u8(block_base, 16u + g_idx * 32u + i);
                let q = select(qb & 0x0Fu, (qb >> 4u) & 0x0Fu, hi);
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
)WGSL";

// [quant_kq] q4k_matmul
// [quant_kq] q8_quantize_dp4a
static const char* WGSL_Q8_QUANTIZE_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Quantize one f32 activation vector into the Q8 layout consumed by the
// prequantized Q4_K DP4A matvec. One workgroup handles 256 values.
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(2) var<storage, read_write> XS: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let lane = tid & 31u;
    let block32 = tid / 32u;
    let pack_lane = lane & 3u;
    let pack_group = lane / 4u;
    let K = P[0];
    let k = wid.x * 256u + tid;
    let xv = select(0.0, X[k], k < K);
    var amax = abs(xv);
    amax = max(amax, subgroupShuffleXor(amax, 16u));
    amax = max(amax, subgroupShuffleXor(amax, 8u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    let scale = amax / 127.0;
    let global_block = wid.x * 8u + block32;
    if (lane == 0u) { XS[global_block] = scale; }
    let safe_scale = select(1.0, scale, scale != 0.0);
    let qi = clamp(i32(round(xv / safe_scale)), -127, 127);
    var packed = u32(qi & 255) << (pack_lane * 8u);
    packed |= subgroupShuffleXor(packed, 1u);
    packed |= subgroupShuffleXor(packed, 2u);
    if (pack_lane == 0u) {
        XQ[wid.x * 64u + block32 * 8u + pack_group] = packed;
    }
}
)WGSL";

// Quantize every row of a prefill activation matrix once. XQ is row-major
// packed i8 and XS stores one scale per 32 activation values.
static const char* WGSL_Q8_QUANTIZE_BATCHED_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0) var<storage,read>X:array<f32>;
@group(0) @binding(1) var<storage,read_write>XQ:array<u32>;
@group(0) @binding(2) var<storage,read_write>XS:array<f32>;
@group(0) @binding(3) var<storage,read>P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let lane=tid&31u;let block32=tid/32u;let packLane=lane&3u;let packGroup=lane/4u;
 let K=P[0];let M=P[2];let row=wid.y;if(row>=M){return;}
 let k=wid.x*256u+tid;let xv=select(0.0,X[row*K+k],k<K);var amax=abs(xv);
 amax=max(amax,subgroupShuffleXor(amax,16u));amax=max(amax,subgroupShuffleXor(amax,8u));
 amax=max(amax,subgroupShuffleXor(amax,4u));amax=max(amax,subgroupShuffleXor(amax,2u));
 amax=max(amax,subgroupShuffleXor(amax,1u));let scale=amax/127.0;
 let blocks32=(K+31u)/32u;if(lane==0u){XS[row*blocks32+wid.x*8u+block32]=scale;}
 let safe=select(1.0,scale,scale!=0.0);let qi=clamp(i32(round(xv/safe)),-127,127);
 var packed=u32(qi&255)<<(packLane*8u);packed|=subgroupShuffleXor(packed,1u);packed|=subgroupShuffleXor(packed,2u);
 if(packLane==0u){XQ[row*((K+3u)/4u)+wid.x*64u+block32*8u+packGroup]=packed;}
}
)WGSL";

// Prefill Q4_K matmul consuming the row-major Q8 matrix above. Each logical
// 32-lane warp produces one output column for eight prompt rows, so each
// packed weight fragment is loaded once and reused across the eight-row tile.
static const char* WGSL_Q4K_MATMUL_PREQUANT_BATCHED_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0)var<storage,read>XQ:array<u32>;
@group(0) @binding(1)var<storage,read>XS:array<f32>;
@group(0) @binding(2)var<storage,read>W:array<u32>;
@group(0) @binding(3)var<storage,read>Bias:array<f32>;
@group(0) @binding(4)var<storage,read_write>Y:array<f32>;
@group(0) @binding(5)var<storage,read>P:array<u32>;
const BLOCK_WORDS:u32=36u;
var<workgroup>sxq:array<u32,512>;var<workgroup>sxs:array<f32,64>;var<workgroup>sxsum:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let row0=wid.x*8u;let col=wid.y*8u+warp;
 let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];var acc:array<f32,8>;
 let qStride=(K+3u)/4u;let sStride=(K+31u)/32u;
 for(var b=0u;b<nb;b++){
  let lm=tid/32u;let ll=tid&31u;let lr=row0+lm;let lq=lm*64u+ll*2u;
  if(lr<M){let qb=lr*qStride+b*64u+ll*2u;sxq[lq]=XQ[qb];sxq[lq+1u]=XQ[qb+1u];}else{sxq[lq]=0u;sxq[lq+1u]=0u;}
  if(tid<64u){let sm=tid/8u;let ss=tid&7u;let sr=row0+sm;if(sr<M){sxs[tid]=XS[sr*sStride+b*8u+ss];}else{sxs[tid]=0.0;}}
  workgroupBarrier();let sa0=sxq[lq];let sa1=sxq[lq+1u];sxsum[tid]=f32(dot4I8Packed(sa0,0x01010101u)+dot4I8Packed(sa1,0x01010101u));workgroupBarrier();
  if(col<N){
  let sb=lane/4u;let j=sb&3u;let qgroup=sb/2u;let high=(sb&1u)!=0u;let elem0=(lane&3u)*8u;
  let base=col*rs+b*BLOCK_WORDS;let dm=unpack2x16float(W[base]);let shift=j*8u;
  let dv=(W[base+1u]>>shift)&255u;let mv=(W[base+2u]>>shift)&255u;var sc:u32;var mn:u32;
  if(sb<4u){sc=dv&63u;mn=mv&63u;}else{let md=(W[base+3u]>>shift)&255u;sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}
  let payload=base+4u+qgroup*8u+elem0/4u;let p0=W[payload];let p1=W[payload+1u];let mask=0x0F0F0F0Fu;
  let w0=select(p0&mask,(p0>>4u)&mask,high);let w1=select(p1&mask,(p1>>4u)&mask,high);
  for(var m=0u;m<8u;m++){let row=row0+m;if(row<M){
   let aq0=sxq[m*64u+lane*2u];let aq1=sxq[m*64u+lane*2u+1u];let asum=sxsum[m*32u+lane];
   let dot=dot4I8Packed(aq0,w0)+dot4I8Packed(aq1,w1);
   acc[m]+=sxs[m*8u+sb]*(dm.x*f32(sc)*f32(dot)-dm.y*f32(mn)*asum);
  }}
 }workgroupBarrier();}
 for(var m=0u;m<8u;m++){let total=subgroupAdd(acc[m]);let row=row0+m;if(lane==0u&&col<N&&row<M){Y[row*N+col]=total+Bias[col];}}
}
)WGSL";

// [quant_kq] q4k_matmul_prequant_dp4a
static const char* WGSL_Q4K_MATMUL_PREQUANT_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Q4_K matvec over an activation quantized once by q8_quantize_dp4a.
// Each workgroup produces eight output rows (one per subgroup). Keeping one
// accumulator per subgroup isolates activation reuse from register-heavy
// multi-column tiling.
@group(0) @binding(0) var<storage, read> XQ: array<u32>;
@group(0) @binding(1) var<storage, read> XS: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;

const BLOCK_WORDS: u32 = 36u;
const COLS_PER_WARP: u32 = 1u;
var<workgroup> xq: array<u32, 64>;
var<workgroup> xs: array<f32, 8>;

fn u8_at(base: u32, off: u32) -> u32 {
    return (W[base + off / 4u] >> ((off & 3u) * 8u)) & 255u;
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
    let rs = P[3];
    var cols: array<u32, 1>;
    var valid: array<bool, 1>;
    var acc: array<f32, 1>;
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        cols[c] = wid.y * 8u + warp;
        valid[c] = cols[c] < N;
        acc[c] = 0.0;
    }

    for (var b = 0u; b < nb; b++) {
        if (tid < 64u) { xq[tid] = XQ[b * 64u + tid]; }
        if (tid < 8u) { xs[tid] = XS[b * 8u + tid]; }
        workgroupBarrier();

        let sb = lane / 4u;
        let j = sb & 3u;
        let qgroup = sb / 2u;
        let high = (sb & 1u) != 0u;
        let elem0 = (lane & 3u) * 8u;
        let aq0 = xq[lane * 2u];
        let aq1 = xq[lane * 2u + 1u];
        let asum = dot4I8Packed(aq0, 0x01010101u) +
                   dot4I8Packed(aq1, 0x01010101u);

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (valid[c]) {
                let base = cols[c] * rs + b * BLOCK_WORDS;
                let dm = unpack2x16float(W[base]);
                let shift = j * 8u;
                let dv = (W[base + 1u] >> shift) & 255u;
                let mv = (W[base + 2u] >> shift) & 255u;
                var sc: u32;
                var mn: u32;
                if (sb < 4u) {
                    sc = dv & 63u;
                    mn = mv & 63u;
                } else {
                    let md = (W[base + 3u] >> shift) & 255u;
                    sc = (md & 15u) | ((dv >> 2u) & 48u);
                    mn = (md >> 4u) | ((mv >> 2u) & 48u);
                }
                // Eight consecutive payload bytes are two aligned words.
                // Masking their low/high nibbles directly produces the packed
                // signed-dot operands without eight byte-addressed loads.
                let payload = base + 4u + qgroup * 8u + elem0 / 4u;
                let packed0 = W[payload];
                let packed1 = W[payload + 1u];
                let nibble_mask = 0x0F0F0F0Fu;
                let w0 = select(packed0 & nibble_mask,
                    (packed0 >> 4u) & nibble_mask, high);
                let w1 = select(packed1 & nibble_mask,
                    (packed1 >> 4u) & nibble_mask, high);
                let dot = dot4I8Packed(aq0, w0) + dot4I8Packed(aq1, w1);
                acc[c] += xs[sb] * (dm.x * f32(sc) * f32(dot) -
                                     dm.y * f32(mn) * f32(asum));
            }
        }
        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        let total = subgroupAdd(acc[c]);
        if (lane == 0u && valid[c]) {
            Y[cols[c]] = total + Bias[cols[c]];
        }
    }
}
)WGSL";

// AMD variant inspired by llama.cpp Vulkan's reduc16 K-quant matvec path.
// Two independent 16-lane rows share each logical wave, doubling the output
// rows per workgroup without keeping a second row accumulator per lane.
static const char* WGSL_Q4K_MATMUL_PREQUANT_DP4A_REDUC16 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0) var<storage,read> XQ:array<u32>;
@group(0) @binding(1) var<storage,read> XS:array<f32>;
@group(0) @binding(2) var<storage,read> W:array<u32>;
@group(0) @binding(3) var<storage,read> Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write> Y:array<f32>;
@group(0) @binding(5) var<storage,read> P:array<u32>;
const BLOCK_WORDS:u32=36u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(workgroup_id) wid:vec3<u32>){
 let tid=lid.x;let half_lane=tid&15u;let row=wid.y*16u+tid/16u;
 let K=P[0];let N=P[1];let nb=P[2];let rs=P[3];var acc=0.0;
 for(var b=0u;b<nb;b++){
  if(row<N){
   let sb=half_lane/2u;let j=sb&3u;let qgroup=sb/2u;
   let high=(sb&1u)!=0u;let part=(half_lane&1u)*4u;
   let aqb=b*64u+sb*8u+part;let aq0=XQ[aqb];let aq1=XQ[aqb+1u];
   let aq2=XQ[aqb+2u];let aq3=XQ[aqb+3u];
   let asum=dot4I8Packed(aq0,0x01010101u)+dot4I8Packed(aq1,0x01010101u)
           +dot4I8Packed(aq2,0x01010101u)+dot4I8Packed(aq3,0x01010101u);
   let base=row*rs+b*BLOCK_WORDS;let dm=unpack2x16float(W[base]);
   let shift=j*8u;let dv=(W[base+1u]>>shift)&255u;
   let mv=(W[base+2u]>>shift)&255u;var sc:u32;var mn:u32;
   if(sb<4u){sc=dv&63u;mn=mv&63u;}else{
    let md=(W[base+3u]>>shift)&255u;
    sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);
   }
   let payload=base+4u+qgroup*8u+part;
   let p0=W[payload];let p1=W[payload+1u];
   let p2=W[payload+2u];let p3=W[payload+3u];
   let mask=0x0F0F0F0Fu;
   let w0=select(p0&mask,(p0>>4u)&mask,high);
   let w1=select(p1&mask,(p1>>4u)&mask,high);
   let w2=select(p2&mask,(p2>>4u)&mask,high);
   let w3=select(p3&mask,(p3>>4u)&mask,high);
   let dot=dot4I8Packed(aq0,w0)+dot4I8Packed(aq1,w1)
          +dot4I8Packed(aq2,w2)+dot4I8Packed(aq3,w3);
   acc+=XS[b*8u+sb]*(dm.x*f32(sc)*f32(dot)-dm.y*f32(mn)*f32(asum));
  }
 }
 acc+=subgroupShuffleXor(acc,8u);acc+=subgroupShuffleXor(acc,4u);
 acc+=subgroupShuffleXor(acc,2u);acc+=subgroupShuffleXor(acc,1u);
 if(half_lane==0u&&row<N){Y[row]=acc+Bias[row];}
}
)WGSL";


static const char* WGSL_Q4K_MATMUL_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 36u;
var<workgroup> xq: array<u32, 64>;
var<workgroup> xs: array<f32, 8>;

fn u8_at(base: u32, off: u32) -> u32 {
    return (W[base + off / 4u] >> ((off & 3u) * 8u)) & 255u;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let warp = tid / 32u;
    let lane = tid & 31u;
    let col = wid.y * 8u + warp;
    let K = P[0];
    let N = P[1];
    let nb = P[2];
    let rs = P[3];
    let row = wid.x;
    let block32 = tid / 32u;
    let elem32 = tid & 31u;
    let pack_lane = elem32 & 3u;
    let pack_group = elem32 / 4u;
    var acc = 0.0;

    for (var b = 0u; b < nb; b++) {
        let k = b * QK_K + tid;
        let xv = select(0.0, X[row * K + k], k < K);
        var amax = abs(xv);
        amax = max(amax, subgroupShuffleXor(amax, 16u));
        amax = max(amax, subgroupShuffleXor(amax, 8u));
        amax = max(amax, subgroupShuffleXor(amax, 4u));
        amax = max(amax, subgroupShuffleXor(amax, 2u));
        amax = max(amax, subgroupShuffleXor(amax, 1u));
        let scale = amax / 127.0;
        if (elem32 == 0u) { xs[block32] = scale; }
        let safe_scale = select(1.0, scale, scale != 0.0);
        let qi = clamp(i32(round(xv / safe_scale)), -127, 127);
        var packed = u32(qi & 255) << (pack_lane * 8u);
        packed |= subgroupShuffleXor(packed, 1u);
        packed |= subgroupShuffleXor(packed, 2u);
        if (pack_lane == 0u) {
            xq[block32 * 8u + pack_group] = packed;
        }
        workgroupBarrier();

        if (col < N) {
            let base = col * rs + b * BLOCK_WORDS;
            let dm = unpack2x16float(W[base]);
            let sb = lane / 4u;
            let j = sb & 3u;
            let shift = j * 8u;
            let dv = (W[base + 1u] >> shift) & 255u;
            let mv = (W[base + 2u] >> shift) & 255u;
            var sc: u32;
            var mn: u32;
            if (sb < 4u) {
                sc = dv & 63u;
                mn = mv & 63u;
            } else {
                let md = (W[base + 3u] >> shift) & 255u;
                sc = (md & 15u) | ((dv >> 2u) & 48u);
                mn = (md >> 4u) | ((mv >> 2u) & 48u);
            }
            let qgroup = sb / 2u;
            let high = (sb & 1u) != 0u;
            let elem0 = (lane & 3u) * 8u;
            let payload = base + 4u + qgroup * 8u + elem0 / 4u;
            let packed0 = W[payload];
            let packed1 = W[payload + 1u];
            let nibble_mask = 0x0F0F0F0Fu;
            let w0 = select(packed0 & nibble_mask,
                (packed0 >> 4u) & nibble_mask, high);
            let w1 = select(packed1 & nibble_mask,
                (packed1 >> 4u) & nibble_mask, high);
            let aq0 = xq[lane * 2u];
            let aq1 = xq[lane * 2u + 1u];
            let dot = dot4I8Packed(aq0, w0) + dot4I8Packed(aq1, w1);
            let sum = dot4I8Packed(aq0, 0x01010101u) +
                      dot4I8Packed(aq1, 0x01010101u);
            acc += xs[sb] * (dm.x * f32(sc) * f32(dot) -
                             dm.y * f32(mn) * f32(sum));
        }
        workgroupBarrier();
    }

    let total = subgroupAdd(acc);
    if (lane == 0u && col < N) {
        Y[row * N + col] = total + Bias[col];
    }
}

)WGSL";

static const char* WGSL_Q4K_MATMUL = R"WGSL(
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
)WGSL";

// [quant_kq] q4k_matmul_128
static const char* WGSL_Q4K_MATMUL_128 = R"WGSL(
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,256>;
fn u8at(b:u32,o:u32)->u32{let a=b*4u+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let nb=P[2];let rs=P[3];let yo=P[4];let col=wid.y*4u+warp;var acc=0.0;
 for(var b=0u;b<nb;b++){let kb=b*256u;let k0=kb+tid;let k1=k0+128u;sx[tid]=select(0.0,X[wid.x*K+k0],k0<K);sx[tid+128u]=select(0.0,X[wid.x*K+k1],k1<K);workgroupBarrier();
  if(col<N){let bb=col*rs+b*36u;let dd=unpack2x16float(W[bb]);for(var sb=0u;sb<8u;sb++){var sc:u32;var mn:u32;if(sb<4u){sc=u8at(bb,4u+sb)&63u;mn=u8at(bb,8u+sb)&63u;}else{let j=sb-4u;let dv=u8at(bb,4u+j);let mv=u8at(bb,8u+j);let md=u8at(bb,12u+j);sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}let qv=u8at(bb,16u+(sb/2u)*32u+lane);let q=select(qv&15u,qv>>4u,(sb&1u)==1u);acc+=sx[sb*32u+lane]*(dd.x*f32(sc)*f32(q)-dd.y*f32(mn));}}
  workgroupBarrier();}
 sx[tid]=acc;workgroupBarrier();
 for(var off=16u;off>0u;off=off/2u){if(lane<off){sx[tid]+=sx[tid+off];}workgroupBarrier();}
 let s=sx[warp*32u];if(lane==0u&&col<N){Y[wid.x*N+col+yo]=s+B[col];}
}
)WGSL";

// [quant_kq] q4k_matmul_batched4
static const char* WGSL_Q4K_MATMUL_BATCHED4 = R"WGSL(
enable subgroups;
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,1024>;
fn u8at(b:u32,o:u32)->u32{let a=b+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];
 let m0=wid.x*4u;let col=wid.y*8u+warp;var acc:array<f32,4>;
 for(var b=0u;b<nb;b++){
  let ki=b*256u+tid;for(var m=0u;m<4u;m++){sx[m*256u+tid]=select(0.0,X[(m0+m)*K+ki],m0+m<M&&ki<K);}workgroupBarrier();
  if(col<N){let bb=(col*rs+b*36u)*4u;let dd=unpack2x16float(W[bb/4u]);
   for(var sb=0u;sb<8u;sb++){var sc:u32;var mn:u32;if(sb<4u){sc=u8at(bb,4u+sb)&63u;mn=u8at(bb,8u+sb)&63u;}else{let j=sb-4u;let dv=u8at(bb,4u+j);let mv=u8at(bb,8u+j);let md=u8at(bb,12u+j);sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}
    let g=sb/2u;let qb=u8at(bb,16u+g*32u+lane);let q=select(qb&15u,qb>>4u,(sb&1u)==1u);let w=dd.x*f32(sc)*f32(q)-dd.y*f32(mn);let k=sb*32u+lane;
    for(var m=0u;m<4u;m++){acc[m]+=sx[m*256u+k]*w;}
   }
  }workgroupBarrier();
 }
 for(var m=0u;m<4u;m++){let s=subgroupAdd(acc[m]);if(lane==0u&&col<N&&m0+m<M){Y[(m0+m)*N+col]=s+B[col];}}
}
)WGSL";

// [quant_kq] q4k_matmul_batched8
static const char* WGSL_Q4K_MATMUL_BATCHED8 = R"WGSL(
enable subgroups;
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,2048>;
fn u8at(b:u32,o:u32)->u32{let a=b+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];
 let m0=wid.x*8u;let col=wid.y*8u+warp;var acc:array<f32,8>;
 for(var b=0u;b<nb;b++){
  let ki=b*256u+tid;for(var m=0u;m<8u;m++){sx[m*256u+tid]=select(0.0,X[(m0+m)*K+ki],m0+m<M&&ki<K);}workgroupBarrier();
  if(col<N){let bb=(col*rs+b*36u)*4u;let dd=unpack2x16float(W[bb/4u]);
   for(var sb=0u;sb<8u;sb++){var sc:u32;var mn:u32;if(sb<4u){sc=u8at(bb,4u+sb)&63u;mn=u8at(bb,8u+sb)&63u;}else{let j=sb-4u;let dv=u8at(bb,4u+j);let mv=u8at(bb,8u+j);let md=u8at(bb,12u+j);sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}
    let g=sb/2u;let qb=u8at(bb,16u+g*32u+lane);let q=select(qb&15u,qb>>4u,(sb&1u)==1u);let w=dd.x*f32(sc)*f32(q)-dd.y*f32(mn);let k=sb*32u+lane;
    for(var m=0u;m<8u;m++){acc[m]+=sx[m*256u+k]*w;}
   }
  }workgroupBarrier();
 }
 for(var m=0u;m<8u;m++){let s=subgroupAdd(acc[m]);if(lane==0u&&col<N&&m0+m<M){Y[(m0+m)*N+col]=s+B[col];}}
}
)WGSL";

// [quant_kq] q4k_matmul_norm
static const char* WGSL_Q4K_MATMUL_NORM = R"WGSL(
enable subgroups;

// Fused RMSNorm + Q4_K matmul for decode (M=1)
// Eliminates the separate rms_norm / rms_next dispatch before each layer's QKV matmul.
// Pass 1: Cooperative RMSNorm of X → shared memory (normed values)
// Pass 2: Q4_K matmul using normed X from shared memory
//
// Dispatch: (1, ceil(N/8), 1)
// WG: 256 threads = 8 warps, TILE_N=8

var<workgroup> smem_x: array<f32, 256>;
var<workgroup> _smem_reduce: array<i32, 8>;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q4K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;  // [K, N, n_blocks, row_stride_words, y_offset]
@group(0) @binding(5) var<storage, read> NormW: array<f32>;     // RMSNorm weight [K]
@group(0) @binding(6) var<storage, read> _norm_params_: array<u32>;  // [K, 0, 0, eps_as_u32]

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 36u;

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
    let y_offset = _params_[4];
    let eps = bitcast<f32>(_norm_params_[3]);

    let x_base = row * K;

    // ── Pass 1: Compute RMSNorm rstd across all K elements ──
    // Each thread accumulates sum_sq over strided K elements
    var sum_sq: f32 = 0.0;
    var k = tid;
    for (; k < K; k += 256u) {
        let v = X[x_base + k];
        sum_sq += v * v;
    }

    // Intra-warp reduction
    let warp_sum = subgroupAdd(sum_sq);
    if (lane == 0u) {
        _smem_reduce[warp_id] = bitcast<i32>(warp_sum);
    }
    workgroupBarrier();

    // Cross-warp reduction (thread 0..7 read, subgroupAdd across first warp)
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

    // ── Pass 2: Q4_K matmul with inline RMSNorm ──
    var acc: f32 = 0.0;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let k_start = b * QK_K;

        // Cooperative load: apply RMSNorm on-the-fly into shared memory
        let x_idx = k_start + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx] * rstd * NormW[x_idx];
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        if (col < N) {
            let block_base = col * row_stride_words + b * BLOCK_WORDS;

            let dd = unpack2x16float(W_Q4K[block_base]);
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
                let qb = get_u8(block_base, 16u + g * 32u + i);
                let q = select(qb & 0x0Fu, (qb >> 4u) & 0x0Fu, hi);
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
)WGSL";

// [quant_kq] q5k_matmul
static const char* WGSL_Q5K_MATMUL = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q5K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;
const BLOCK_WORDS: u32 = 44u; // 176 bytes / 4

// Shared memory: cache X for the current super-block (256 elements)
var<workgroup> smem_x: array<f32, 256>;

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

    let x_base = row * K;
    var acc: f32 = 0.0;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let k_start = b * QK_K;

        // Cooperative X load: all 256 threads load 1 float each into shared memory
        let x_idx = k_start + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx];
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

    // Keep the eight logical 32-lane columns independent on GPUs whose
    // hardware subgroup width is not 32.
    smem_x[tid] = acc;
    workgroupBarrier();
    for (var offset = 16u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) { smem_x[tid] += smem_x[tid + offset]; }
        workgroupBarrier();
    }
    let sum = smem_x[warp_id * 32u];
    if (lane == 0u && col < N) {
        Y[row * N + col + y_offset] = sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] q5k_matmul_batched4
static const char* WGSL_Q5K_MATMUL_BATCHED4 = R"WGSL(
enable subgroups;
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,1024>;
fn u8at(b:u32,o:u32)->u32{let a=b+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];
 let m0=wid.x*4u;let col=wid.y*8u+warp;var acc:array<f32,4>;
 for(var b=0u;b<nb;b++){
  let ki=b*256u+tid;for(var m=0u;m<4u;m++){sx[m*256u+tid]=select(0.0,X[(m0+m)*K+ki],m0+m<M&&ki<K);}workgroupBarrier();
  if(col<N){let bb=(col*rs+b*44u)*4u;let dd=unpack2x16float(W[bb/4u]);
   for(var sb=0u;sb<8u;sb++){var sc:u32;var mn:u32;if(sb<4u){sc=u8at(bb,4u+sb)&63u;mn=u8at(bb,8u+sb)&63u;}else{let j=sb-4u;let dv=u8at(bb,4u+j);let mv=u8at(bb,8u+j);let md=u8at(bb,12u+j);sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}
    let g=sb/2u;let qb=u8at(bb,48u+g*32u+lane);let ql=select(qb&15u,qb>>4u,(sb&1u)==1u);let qh=(u8at(bb,16u+lane)>>sb)&1u;let q=ql|(qh<<4u);let w=dd.x*f32(sc)*f32(q)-dd.y*f32(mn);let k=sb*32u+lane;
    for(var m=0u;m<4u;m++){acc[m]+=sx[m*256u+k]*w;}
   }
  }workgroupBarrier();
 }
 for(var m=0u;m<4u;m++){let s=subgroupAdd(acc[m]);if(lane==0u&&col<N&&m0+m<M){Y[(m0+m)*N+col]=s+B[col];}}
}
)WGSL";

// Q5_K prefill over the row-major Q8 activation matrix. Eight prompt rows
// share every packed weight fragment, matching the accepted Q4_K row tile.
static const char* WGSL_Q5K_MATMUL_PREQUANT_BATCHED_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0)var<storage,read>XQ:array<u32>;
@group(0) @binding(1)var<storage,read>XS:array<f32>;
@group(0) @binding(2)var<storage,read>W:array<u32>;
@group(0) @binding(3)var<storage,read>Bias:array<f32>;
@group(0) @binding(4)var<storage,read_write>Y:array<f32>;
@group(0) @binding(5)var<storage,read>P:array<u32>;
const BLOCK_WORDS:u32=44u;
var<workgroup>sxq:array<u32,512>;var<workgroup>sxs:array<f32,64>;var<workgroup>sxsum:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let row0=wid.x*8u;let col=wid.y*8u+warp;
 let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];var acc:array<f32,8>;
 let qStride=(K+3u)/4u;let sStride=(K+31u)/32u;
 for(var b=0u;b<nb;b++){
  let lm=tid/32u;let ll=tid&31u;let lr=row0+lm;let lq=lm*64u+ll*2u;
  if(lr<M){let qb=lr*qStride+b*64u+ll*2u;sxq[lq]=XQ[qb];sxq[lq+1u]=XQ[qb+1u];}else{sxq[lq]=0u;sxq[lq+1u]=0u;}
  if(tid<64u){let sm=tid/8u;let ss=tid&7u;let sr=row0+sm;if(sr<M){sxs[tid]=XS[sr*sStride+b*8u+ss];}else{sxs[tid]=0.0;}}
  workgroupBarrier();let sa0=sxq[lq];let sa1=sxq[lq+1u];sxsum[tid]=f32(dot4I8Packed(sa0,0x01010101u)+dot4I8Packed(sa1,0x01010101u));workgroupBarrier();
  if(col<N){
  let sb=lane/4u;let j=sb&3u;let qgroup=sb/2u;let high=(sb&1u)!=0u;let elem0=(lane&3u)*8u;
  let base=col*rs+b*BLOCK_WORDS;let dm=unpack2x16float(W[base]);let shift=j*8u;
  let dv=(W[base+1u]>>shift)&255u;let mv=(W[base+2u]>>shift)&255u;var sc:u32;var mn:u32;
  if(sb<4u){sc=dv&63u;mn=mv&63u;}else{let md=(W[base+3u]>>shift)&255u;sc=(md&15u)|((dv>>2u)&48u);mn=(md>>4u)|((mv>>2u)&48u);}
  let payload=base+12u+qgroup*8u+elem0/4u;let p0=W[payload];let p1=W[payload+1u];let mask=0x0F0F0F0Fu;
  let lo0=select(p0&mask,(p0>>4u)&mask,high);let lo1=select(p1&mask,(p1>>4u)&mask,high);
  let qh0=((W[base+4u+elem0/4u]>>sb)&0x01010101u)<<4u;
  let qh1=((W[base+5u+elem0/4u]>>sb)&0x01010101u)<<4u;
  let w0=lo0|qh0;let w1=lo1|qh1;
  for(var m=0u;m<8u;m++){let row=row0+m;if(row<M){
   let aq0=sxq[m*64u+lane*2u];let aq1=sxq[m*64u+lane*2u+1u];let asum=sxsum[m*32u+lane];
   let dot=dot4I8Packed(aq0,w0)+dot4I8Packed(aq1,w1);
   acc[m]+=sxs[m*8u+sb]*(dm.x*f32(sc)*f32(dot)-dm.y*f32(mn)*asum);
  }}
 }workgroupBarrier();}
 for(var m=0u;m<8u;m++){let total=subgroupAdd(acc[m]);let row=row0+m;if(lane==0u&&col<N&&row<M){Y[row*N+col]=total+Bias[col];}}
}
)WGSL";

// [quant_kq] q5k_matmul_norm
static const char* WGSL_Q5K_MATMUL_NORM = R"WGSL(
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
)WGSL";

// [quant_kq] q6k_down_gelu
static const char* WGSL_Q6K_DOWN_GELU = R"WGSL(
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
)WGSL";

// [quant_kq] q6k_gather
static const char* WGSL_Q6K_GATHER = R"WGSL(
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Token: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

fn u8_at(addr: u32) -> u32 {
    return (W[addr / 4u] >> ((addr & 3u) * 8u)) & 0xffu;
}

fn i8_at(addr: u32) -> i32 {
    let v = u8_at(addr);
    return select(i32(v), i32(v) - 256, v >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let K = P[0];
    if (k >= K) { return; }

    let row_stride_bytes = P[1] * 4u;
    let token = u32(max(Token[0], 0));
    let local = k & 255u;
    let group = local / 128u;
    let r = local & 127u;
    let quarter = r / 32u;
    let lane = r & 31u;
    let block = k / 256u;
    let base = token * row_stride_bytes + block * 210u;

    let ql_base = base + group * 64u;
    let qh = u8_at(base + 128u + group * 32u + lane);
    let ql = u8_at(ql_base + select(lane, 32u + lane,
                                    quarter == 1u || quarter == 3u));
    let low = select(ql & 15u, ql >> 4u, quarter >= 2u);
    let high_shift = quarter * 2u;
    let q = i32(low | (((qh >> high_shift) & 3u) << 4u)) - 32;

    let scale_index = group * 8u + lane / 16u + quarter * 2u;
    let scale = i8_at(base + 192u + scale_index);
    let dh = u8_at(base + 208u) | (u8_at(base + 209u) << 8u);
    let d = unpack2x16float(dh).x;
    X[k] = d * f32(scale) * f32(q);
}
)WGSL";

// [quant_kq] q6k_gather_batched
static const char* WGSL_Q6K_GATHER_BATCHED = R"WGSL(
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Tokens: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

fn u8_at(a:u32)->u32{return(W[a/4u]>>((a&3u)*8u))&255u;}
fn i8_at(a:u32)->i32{let v=u8_at(a);return select(i32(v),i32(v)-256,v>=128u);}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let idx=gid.x;let M=P[0];let K=P[1];if(idx>=M*K){return;}
 let m=idx/K;let k=idx-m*K;let rs=P[2]*4u;let token=u32(max(Tokens[m],0));
 let local=k&255u;let g=local/128u;let r=local&127u;let qtr=r/32u;let lane=r&31u;
 let base=token*rs+(k/256u)*210u;let qlb=base+g*64u;
 let qh=u8_at(base+128u+g*32u+lane);
 let ql=u8_at(qlb+select(lane,32u+lane,qtr==1u||qtr==3u));
 let low=select(ql&15u,ql>>4u,qtr>=2u);let q=i32(low|(((qh>>(qtr*2u))&3u)<<4u))-32;
 let sc=i8_at(base+192u+g*8u+lane/16u+qtr*2u);
 let dh=u8_at(base+208u)|(u8_at(base+209u)<<8u);
 X[idx]=unpack2x16float(dh).x*f32(sc)*f32(q);
}
)WGSL";

// [quant_kq] q6k_matmul
static const char* WGSL_Q6K_MATMUL = R"WGSL(
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q6K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

const TILE_N: u32 = 8u;
const QK_K: u32 = 256u;

// Shared memory: cache X for the current super-block (256 elements)
var<workgroup> smem_x: array<f32, 256>;

fn get_u8(base_byte: u32, byte_off: u32) -> u32 {
    let addr = base_byte + byte_off;
    let wi = addr / 4u;
    let sh = (addr % 4u) * 8u;
    return (W_Q6K[wi] >> sh) & 0xFFu;
}

fn get_i8(base_byte: u32, byte_off: u32) -> i32 {
    let u = get_u8(base_byte, byte_off);
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
    let y_offset = _params_[4];

    let x_base = row * K;
    var acc: f32 = 0.0;

    for (var b = 0u; b < n_blocks; b = b + 1u) {
        let k_start = b * QK_K;

        // Cooperative X load: all 256 threads load 1 float each into shared memory
        let x_idx = k_start + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx];
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        if (col < N) {
            // Q6_K blocks are 210 bytes and therefore alternate between word-
            // aligned and half-word-aligned starts. Address them in bytes.
            let block_base = col * row_stride_words * 4u + b * 210u;

            let d_u16 = get_u8(block_base, 208u) | (get_u8(block_base, 209u) << 8u);
            let d = unpack2x16float(d_u16).x;
            let l = lane;

            // Two groups of 128 values
            for (var group = 0u; group < 2u; group = group + 1u) {
                let ql_off = group * 64u;
                let qh_off = 128u + group * 32u;
                let sc_off = 192u + group * 8u;
                let local_base = group * 128u;  // offset within the 256-element block

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

                // Local indices within the 256-element super-block
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

    // Keep the eight logical 32-lane columns independent on GPUs whose
    // hardware subgroup width is not 32.
    smem_x[tid] = acc;
    workgroupBarrier();
    for (var offset = 16u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) { smem_x[tid] += smem_x[tid + offset]; }
        workgroupBarrier();
    }
    let sum = smem_x[warp_id * 32u];
    if (lane == 0u && col < N) {
        Y[row * N + col + y_offset] = sum + Bias[col];
    }
}
)WGSL";

// [quant_kq] q6k_matmul_prequant_dp4a
// llama.cpp-style Q8 activation x Q6_K weight matvec.  This kernel is used
// for the tied LM head after q8_quantize_dp4a has quantized the final hidden
// state once.  A logical 32-lane warp produces one vocabulary logit.
static const char* WGSL_Q6K_MATMUL_PREQUANT_DP4A = R"WGSL(
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
)WGSL";

// Intel logical-16 counterpart to the packed Q6_K LM-head kernel.  This
// follows llama.cpp Vulkan's reduc16 shape: one subgroup produces one logit,
// with four packed activation/weight fragments consumed by each lane.
static const char* WGSL_Q6K_MATMUL_PREQUANT_DP4A_REDUC16 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;
@group(0) @binding(0) var<storage,read> XQ:array<u32>;
@group(0) @binding(1) var<storage,read> XS:array<f32>;
@group(0) @binding(2) var<storage,read> W:array<u32>;
@group(0) @binding(3) var<storage,read> Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write> Y:array<f32>;
@group(0) @binding(5) var<storage,read> P:array<u32>;
fn load_u8(base:u32,off:u32)->u32{let a=base+off;return(W[a/4u]>>((a&3u)*8u))&255u;}
fn load_i8(base:u32,off:u32)->i32{let v=load_u8(base,off);return select(i32(v),i32(v)-256,v>=128u);}
fn load_u32(base:u32,off:u32)->u32{let a=base+off;let wi=a/4u;let s=(a&3u)*8u;
 if(s==0u){return W[wi];}return(W[wi]>>s)|(W[wi+1u]<<(32u-s));}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,
        @builtin(workgroup_id) wid:vec3<u32>){
 let tid=lid.x;let lane=tid&15u;let K=P[0];let N=P[1];let nb=P[2];
 let rs=P[3];let yoff=P[4];let col=wid.y*16u+tid/16u;var acc=0.0;
 if(col<N){for(var b=0u;b<nb;b++){
  let bb=col*rs*4u+b*210u;let dh=load_u8(bb,208u)|(load_u8(bb,209u)<<8u);
  let d=unpack2x16float(dh).x;
  for(var part=0u;part<4u;part++){
   let pack=lane+part*16u;let index=pack*4u;let group=index/128u;
   let within=index-group*128u;let quarter=within/32u;let local=within&31u;
   let qlo=group*64u+select(local,32u+local,quarter==1u||quarter==3u);
   let qho=128u+group*32u+local;let ql=load_u32(bb,qlo);let qh=load_u32(bb,qho);
   let low=select(ql&0x0F0F0F0Fu,(ql>>4u)&0x0F0F0F0Fu,quarter>=2u);
   let values=low|(((qh>>(quarter*2u))&0x03030303u)<<4u);
   let signed_values=((values^0x80808080u)-0x20202020u)^0x80808080u;
   let si=group*8u+quarter*2u+local/16u;
   let ws=f32(load_i8(bb,192u+si));let xs=XS[b*8u+index/32u];
   acc+=f32(dot4I8Packed(XQ[b*64u+pack],signed_values))*xs*d*ws;
  }
 }}
 acc+=subgroupShuffleXor(acc,8u);acc+=subgroupShuffleXor(acc,4u);
 acc+=subgroupShuffleXor(acc,2u);acc+=subgroupShuffleXor(acc,1u);
 if(lane==0u&&col<N){Y[wid.x*N+col+yoff]=acc+Bias[col];}
}
)WGSL";

// [quant_kq] q6k_matmul_batched4
static const char* WGSL_Q6K_MATMUL_BATCHED4 = R"WGSL(
enable subgroups;
@group(0) @binding(0)var<storage,read>X:array<f32>;
@group(0) @binding(1)var<storage,read>W:array<u32>;
@group(0) @binding(2)var<storage,read>B:array<f32>;
@group(0) @binding(3)var<storage,read_write>Y:array<f32>;
@group(0) @binding(4)var<storage,read>P:array<u32>;
var<workgroup>sx:array<f32,1024>;
fn u8at(b:u32,o:u32)->u32{let a=b+o;return(W[a/4u]>>((a&3u)*8u))&255u;}
fn i8at(b:u32,o:u32)->i32{let u=u8at(b,o);return select(i32(u),i32(u)-256,u>=128u);}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let K=P[0];let N=P[1];let M=P[2];let nb=P[3];let rs=P[4];
 let m0=wid.x*4u;let col=wid.y*8u+warp;var acc:array<f32,4>;
 for(var b=0u;b<nb;b++){
  let ki=b*256u+tid;for(var m=0u;m<4u;m++){sx[m*256u+tid]=select(0.0,X[(m0+m)*K+ki],m0+m<M&&ki<K);}workgroupBarrier();
  if(col<N){let bb=col*rs*4u+b*210u;let dh=u8at(bb,208u)|(u8at(bb,209u)<<8u);let d=unpack2x16float(dh).x;let si=lane/16u;
   for(var g=0u;g<2u;g++){let qlo=g*64u;let qho=128u+g*32u;let sco=192u+g*8u;let kb=g*128u;let ql0=u8at(bb,qlo+lane);let ql1=u8at(bb,qlo+32u+lane);let qh=u8at(bb,qho+lane);
    let q1=i32((ql0&15u)|(((qh>>0u)&3u)<<4u))-32;let q2=i32((ql1&15u)|(((qh>>2u)&3u)<<4u))-32;let q3=i32((ql0>>4u)|(((qh>>4u)&3u)<<4u))-32;let q4=i32((ql1>>4u)|(((qh>>6u)&3u)<<4u))-32;
    let w1=d*f32(i8at(bb,sco+si))*f32(q1);let w2=d*f32(i8at(bb,sco+si+2u))*f32(q2);let w3=d*f32(i8at(bb,sco+si+4u))*f32(q3);let w4=d*f32(i8at(bb,sco+si+6u))*f32(q4);
    for(var m=0u;m<4u;m++){acc[m]+=sx[m*256u+kb+lane]*w1+sx[m*256u+kb+32u+lane]*w2+sx[m*256u+kb+64u+lane]*w3+sx[m*256u+kb+96u+lane]*w4;}
   }
  }workgroupBarrier();
 }
 for(var m=0u;m<4u;m++){let s=subgroupAdd(acc[m]);if(lane==0u&&col<N&&m0+m<M){Y[(m0+m)*N+col]=s+B[col];}}
}
)WGSL";

// [quant_kq] q6k_matmul_norm
static const char* WGSL_Q6K_MATMUL_NORM = R"WGSL(
enable subgroups;

// Fused RMSNorm + Q6_K matmul for decode (M=1)
// Eliminates the separate rms_norm / rms_next dispatch before each layer's QKV matmul.
// Pass 1: Cooperative RMSNorm of X → rstd
// Pass 2: Q6_K matmul with inline RMSNorm from shared memory
//
// Dispatch: (1, ceil(N/8), 1)
// WG: 256 threads = 8 warps, TILE_N=8

var<workgroup> smem_x: array<f32, 256>;
var<workgroup> _smem_reduce: array<i32, 8>;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_Q6K: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;  // [K, N, n_blocks, row_stride_words, y_offset]
@group(0) @binding(5) var<storage, read> NormW: array<f32>;     // RMSNorm weight [K]
@group(0) @binding(6) var<storage, read> _norm_params_: array<u32>;  // [K, 0, 0, eps_as_u32]

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
    let y_offset = _params_[4];
    let eps = bitcast<f32>(_norm_params_[3]);

    let x_base = row * K;

    // ── Pass 1: Compute RMSNorm rstd across all K elements ──
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

        // Cooperative load: apply RMSNorm on-the-fly into shared memory
        let x_idx = k_start + tid;
        if (x_idx < K) {
            smem_x[tid] = X[x_base + x_idx] * rstd * NormW[x_idx];
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
)WGSL";

// [quant_kq] q6k_matmul_wide
static const char* WGSL_Q6K_MATMUL_WIDE = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

var<workgroup> sx: array<f32, 256>;

fn u8_at(base_byte: u32, off: u32) -> u32 {
    let a = base_byte + off;
    return (W[a / 4u] >> ((a & 3u) * 8u)) & 255u;
}
fn i8_at(base_byte: u32, off: u32) -> i32 {
    let v=u8_at(base_byte,off);return select(i32(v),i32(v)-256,v>=128u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid=lid.x;let warp=tid/32u;let lane=tid&31u;
    let K=P[0];let N=P[1];let nb=P[2];let rs=P[3];let yo=P[4];
    let col0=wid.y*16u+warp*2u;
    var acc: array<f32,2>;
    acc[0]=0.0;acc[1]=0.0;
    for(var b=0u;b<nb;b++){
        let ki=b*256u+tid;sx[tid]=select(0.0,X[wid.x*K+ki],ki<K);
        workgroupBarrier();
        for(var c=0u;c<2u;c++){
            let col=col0+c;
            if(col<N){
                let bb=col*rs*4u+b*210u;
                let dh=u8_at(bb,208u)|(u8_at(bb,209u)<<8u);
                let d=unpack2x16float(dh).x;
                for(var g=0u;g<2u;g++){
                    let qlo=g*64u;let qho=128u+g*32u;let sco=192u+g*8u;
                    let ql0=u8_at(bb,qlo+lane);let ql1=u8_at(bb,qlo+32u+lane);
                    let qh=u8_at(bb,qho+lane);let si=lane/16u;
                    let q0=i32((ql0&15u)|(((qh>>0u)&3u)<<4u))-32;
                    let q1=i32((ql1&15u)|(((qh>>2u)&3u)<<4u))-32;
                    let q2=i32((ql0>>4u)|(((qh>>4u)&3u)<<4u))-32;
                    let q3=i32((ql1>>4u)|(((qh>>6u)&3u)<<4u))-32;
                    let lb=g*128u;
                    acc[c]+=sx[lb+lane]*d*f32(i8_at(bb,sco+si))*f32(q0);
                    acc[c]+=sx[lb+32u+lane]*d*f32(i8_at(bb,sco+si+2u))*f32(q1);
                    acc[c]+=sx[lb+64u+lane]*d*f32(i8_at(bb,sco+si+4u))*f32(q2);
                    acc[c]+=sx[lb+96u+lane]*d*f32(i8_at(bb,sco+si+6u))*f32(q3);
                }
            }
        }
        workgroupBarrier();
    }
    for(var c=0u;c<2u;c++){
        let col=col0+c;let sum=subgroupAdd(acc[c]);
        if(lane==0u&&col<N){Y[wid.x*N+col+yo]=sum+Bias[col];}
    }
}
)WGSL";

// ─── [quant_q8] ────────────────────────────────────────────────────

// [quant_q8] q8_down_silu_add
static const char* WGSL_Q8_DOWN_SILU_ADD = R"WGSL(
// Fused: SiLU·mul + Q8_0 matmul + residual add.
// Reads gateUpBuf (2*IM elements), applies silu(gate)*up on-the-fly,
// then multiplies by W_down and adds to residual Y.
// Eliminates the separate silu_mul dispatch entirely.
//
// Input X layout: [gate_0..gate_{IM-1}, up_0..up_{IM-1}]
// Effective input after silu: silu(gate_i) * up_i for i in [0, IM)
//
// Bindings:
//   0: GateUp (read) — 2*IM floats (gate || up concatenated)
//   1: W_Q8 (read) — quantized weight matrix (N × K/4 u32)
//   2: Scales (read) — fp16 scales packed as u32
//   3: Bias (read) — per-output bias (zeros for no bias)
//   4: Y (read_write) — output += matmul result (residual add)
//   5: params — [K=IM, N=E, IM, 0]

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params {
    K: u32,
    N: u32,
    IM: u32,
    pad: u32,
};
@group(0) @binding(5) var<uniform> params: Params;

const TILE_N: u32 = 8u;
const STRIDE: u32 = 256u;

var<workgroup> smem_x: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = params.K;    // IM (intermediate_size)
    let N = params.N;    // E (n_embd)
    let IM = params.IM;  // same as K, used for up offset

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let valid = col < N;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    let n_strides = (K + STRIDE - 1u) / STRIDE;
    var acc: f32 = 0.0;

    for (var g = 0u; g < n_strides; g = g + 1u) {
        let k_off = g * STRIDE;
        let elem_in_range = k_off + tid < K;

        // Load gate and up values, apply silu·mul, store to shared memory
        if (elem_in_range) {
            let idx = k_off + tid;
            let gate = GateUp[row * 2u * IM + idx];         // gate[idx]
            let up   = GateUp[row * 2u * IM + IM + idx];    // up[idx]
            // silu(gate) * up = gate / (1 + exp(-gate)) * up
            let silu_gate = gate / (1.0 + exp(-gate));
            smem_x[tid] = silu_gate * up;
        } else {
            smem_x[tid] = 0.0;
        }
        workgroupBarrier();

        let lane_in_range = k_off + lane * 8u < K;
        if (valid && lane_in_range) {
            let k_base = lane * 8u;
            let xv0 = vec4<f32>(smem_x[k_base], smem_x[k_base+1u],
                                smem_x[k_base+2u], smem_x[k_base+3u]);
            let xv1 = vec4<f32>(smem_x[k_base+4u], smem_x[k_base+5u],
                                smem_x[k_base+6u], smem_x[k_base+7u]);

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
        workgroupBarrier();
    }

    smem_x[tid] = acc;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            smem_x[tid] += smem_x[tid + stride];
        }
        workgroupBarrier();
    }
    let warp_sum = smem_x[warp_id * 32u];
    if (lane == 0u && valid) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_down_silu_add_batched
static const char* WGSL_Q8_DOWN_SILU_ADD_BATCHED = R"WGSL(
enable subgroups;

// Batched fused: SiLU·mul + Q8_0 down-proj + residual add for prefill.
// Y[T×N] += (silu(gate) * up) × W_down^T
// GateUp layout: [T rows × 2*IM cols] where gate=[0..IM), up=[IM..2*IM)
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 64u;  // supports up to K=16384

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];   // IM
    let N = _params_[1];   // E
    let IM = _params_[2];  // same as K
    let T = _params_[3];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    var acc: array<f32, 8>;
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        var wv0: vec4<f32>;
        var wv1: vec4<f32>;
        var scale0: f32 = 0.0;
        var scale1: f32 = 0.0;

        if (valid && in_k) {
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
            wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);
        }

        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                // Apply silu·mul on-the-fly: silu(gate[k]) * up[k]
                let gu_base = row * 2u * IM;
                var sv0: vec4<f32>;
                var sv1: vec4<f32>;
                for (var e = 0u; e < 4u; e++) {
                    let idx = k_base + e;
                    let gate = GateUp[gu_base + idx];
                    let up   = GateUp[gu_base + IM + idx];
                    sv0[e] = gate / (1.0 + exp(-gate)) * up;
                }
                for (var e = 0u; e < 4u; e++) {
                    let idx = k_base + 4u + e;
                    let gate = GateUp[gu_base + idx];
                    let up   = GateUp[gu_base + IM + idx];
                    sv1[e] = gate / (1.0 + exp(-gate)) * up;
                }
                acc[m] += dot(sv0, wv0) * scale0 + dot(sv1, wv1) * scale1;
            }
        }
    }

    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] += warp_sum + Bias[col];
        }
    }
}
)WGSL";

// [quant_q8] q8_down_silu_add_d3d12
static const char* WGSL_Q8_DOWN_SILU_ADD_D3D12 = R"WGSL(
enable subgroups;

// D3D12-optimized fused SiLU+down+residual: double-buffered smem.
//   Y[T×N] += silu_mul(GateUp[T×2*IM]) × W_down[N×K]^T
//
// Same as q8_matmul_d3d12 but computes SiLU into smem instead of X.
// params: [K=IM, N=E, IM, T]
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK: u32 = 256u;
const WG: u32 = 256u;
const MAX_ITERS: u32 = 64u;  // supports up to K=16384

var<workgroup> smem_s: array<array<f32, 2048>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let IM = _params_[2];
    let T = _params_[3];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;

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

    // Prefetch first SiLU tile
    for (var r = 0u; r < TILE_M; r++) {
        let row = tile_row * TILE_M + r;
        let idx = r * BK + tid;
        if (row < T && tid < K) {
            let gu_base = row * 2u * IM;
            let gate = GateUp[gu_base + tid];
            let up = GateUp[gu_base + IM + tid];
            smem_s[0][idx] = gate / (1.0 + exp(-gate)) * up;
        } else {
            smem_s[0][idx] = 0.0;
        }
    }
    workgroupBarrier();

    for (var g = 0u; g < MAX_ITERS; g++) {
        let cur = g & 1u;
        let nxt = 1u - cur;
        let k_off = g * BK;
        let in_k = k_off < K;
        let k_next = (g + 1u) * BK;

        if (g + 1u < MAX_ITERS) {
            for (var r = 0u; r < TILE_M; r++) {
                let row = tile_row * TILE_M + r;
                let idx = r * BK + tid;
                if (row < T && k_next + tid < K) {
                    let gu_base = row * 2u * IM;
                    let gate = GateUp[gu_base + k_next + tid];
                    let up = GateUp[gu_base + IM + k_next + tid];
                    smem_s[nxt][idx] = gate / (1.0 + exp(-gate)) * up;
                } else {
                    smem_s[nxt][idx] = 0.0;
                }
            }
        }

        if (in_k) {
        let k_base = lane * 8u;
        for (var c = 0u; c < COLS_PER_WARP; c++) {
            var wv0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var wv1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var scale0: f32 = 0.0;
            var scale1: f32 = 0.0;

            if (col_valid[c]) {
                let w_off = w_bases[c] + g * 64u + lane * 2u;
                let pw0 = W_Q8[w_off];
                let pw1 = W_Q8[w_off + 1u];
                wv0 = vec4<f32>(
                    f32(extractBits(i32(pw0), 0u, 8u)), f32(extractBits(i32(pw0), 8u, 8u)),
                    f32(extractBits(i32(pw0), 16u, 8u)), f32(extractBits(i32(pw0), 24u, 8u)));
                wv1 = vec4<f32>(
                    f32(extractBits(i32(pw1), 0u, 8u)), f32(extractBits(i32(pw1), 8u, 8u)),
                    f32(extractBits(i32(pw1), 16u, 8u)), f32(extractBits(i32(pw1), 24u, 8u)));
                let block0 = g * 8u + (lane * 8u) / 32u;
                let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
            }

            for (var m = 0u; m < TILE_M; m++) {
                let smem_base = m * BK + k_base;
                let sv0 = vec4<f32>(smem_s[cur][smem_base], smem_s[cur][smem_base + 1u],
                                    smem_s[cur][smem_base + 2u], smem_s[cur][smem_base + 3u]);
                let sv1 = vec4<f32>(smem_s[cur][smem_base + 4u], smem_s[cur][smem_base + 5u],
                                    smem_s[cur][smem_base + 6u], smem_s[cur][smem_base + 7u]);
                acc[m][c] += dot(sv0, wv0) * scale0 + dot(sv1, wv1) * scale1;
            }
        }
        }
        workgroupBarrier();
    }

    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < T) {
                Y[row * N + cols[c]] += warp_sum + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [quant_q8] q8_down_silu_add_dp4a_d3d12
static const char* WGSL_Q8_DOWN_SILU_ADD_DP4A_D3D12 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// D3D12 DP4A-accelerated fused SiLU+down+residual for batched prefill.
//   Y[T×N] += silu_mul(GateUp[T×2*IM]) × W_down[N×K]^T
//
// Same as q8_matmul_dp4a_d3d12 but computes SiLU activation and
// quantizes the result to int8 in smem instead of loading X directly.
// params struct: {K=IM, N=E, IM, M=T}
//
// Grid: (ceil(N/TILE_N), ceil(T/TILE_M), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { K: u32, N: u32, IM: u32, M: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK: u32 = 256u;
const WG: u32 = 256u;

var<workgroup> smem_xq: array<array<u32, 512>, 2>;
var<workgroup> smem_xs: array<array<f32, 64>, 2>;

// Compute SiLU(gate) * up for one element
fn silu_mul(gu: ptr<storage, array<f32>, read_write>,
            row: u32, k: u32, IM: u32) -> f32 {
    let base = row * 2u * IM;
    let gate = (*gu)[base + k];
    let up = (*gu)[base + IM + k];
    return gate / (1.0 + exp(-gate)) * up;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_col = wid.x;
    let tile_row = wid.y;
    let tid = lid.x;

    let K = params.K;
    let N = params.N;
    let IM = params.IM;
    let T = params.M;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    var acc: array<array<f32, 4>, 8>;
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) { acc[m][c] = 0.0; }
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

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    // ── Quantize first SiLU tile into buffer 0 ──────────────────────
    let nk = K / BK;
    for (var r = 0u; r < TILE_M; r++) {
        let row = tile_row * TILE_M + r;
        let x_val = select(0.0, silu_mul(&GateUp, row, tid, IM), row < T && tid < K);

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

        if (g + 1u < nk) {
            for (var r = 0u; r < TILE_M; r++) {
                let row = tile_row * TILE_M + r;
                let x_val = select(0.0, silu_mul(&GateUp, row, k_next + tid, IM),
                                   row < T && k_next + tid < K);

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

        // DP4A compute
        let x_block = lane / 4u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            if (col_valid[c]) {
                let w_off = w_bases[c] + g * 64u + lane * 2u;
                let wq0 = W_Q8[w_off];
                let wq1 = W_Q8[w_off + 1u];

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

    // ── Write output (residual add) ──────────────────────────────────
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < T) {
                Y[row * N + cols[c]] += warp_sum + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [quant_q8] q8_down_silu_add_tiled
static const char* WGSL_Q8_DOWN_SILU_ADD_TILED = R"WGSL(
enable subgroups;

// Tiled fused: SiLU·mul + Q8_0 down-proj + residual add for prefill.
//   Y[T×N] += silu_mul(GateUp[T×2*IM]) × W_down[N×K]^T
//
// Tile: TILE_M(8) rows × TILE_N(32) columns per workgroup.
// 256 threads = 8 warps, each warp processes 4 output columns.
//
// Shared memory caches the SiLU-activated input: silu(gate) * up.
// This is computed once cooperatively per K-block, then reused
// across all 32 output columns — eliminates redundant SiLU eval.
//
// GateUp layout: [T × 2*IM], gate=[0..IM), up=[IM..2*IM)
// params: [K=IM, N=E, IM, T]
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK_LOAD: u32 = 256u;
const MAX_ITERS: u32 = 64u;  // supports up to K=16384

// Shared memory: SiLU-activated values [TILE_M × BK_LOAD]
var<workgroup> smem_s: array<f32, 2048>;  // 8 × 256

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];   // IM
    let N = _params_[1];   // E
    let IM = _params_[2];  // same as K
    let T = _params_[3];

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

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

    for (var g = 0u; g < MAX_ITERS; g = g + 1u) {
        let k_off = g * BK_LOAD;
        let in_k = k_off < K;

        // ── Cooperative load: compute SiLU·mul into smem ────────────
        // 256 threads, 8 rows × 256 K elements per iteration
        for (var r = 0u; r < TILE_M; r++) {
            let row = tile_row * TILE_M + r;
            let smem_idx = r * BK_LOAD + tid;
            if (in_k && row < T) {
                let idx = k_off + tid;
                let gu_base = row * 2u * IM;
                let gate = GateUp[gu_base + idx];
                let up   = GateUp[gu_base + IM + idx];
                smem_s[smem_idx] = gate / (1.0 + exp(-gate)) * up;
            } else {
                smem_s[smem_idx] = 0.0;
            }
        }
        workgroupBarrier();

        // ── Compute: dot SiLU-activated rows (smem) with W (global) ─
        if (in_k) {
            let k_base = lane * 8u;

            for (var c = 0u; c < COLS_PER_WARP; c++) {
                var wv0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var wv1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var scale0: f32 = 0.0;
                var scale1: f32 = 0.0;

                if (col_valid[c]) {
                    let w_off = w_bases[c] + g * 64u + lane * 2u;
                    let pw0 = W_Q8[w_off];
                    let pw1 = W_Q8[w_off + 1u];

                    wv0 = vec4<f32>(
                        f32(extractBits(i32(pw0), 0u, 8u)),
                        f32(extractBits(i32(pw0), 8u, 8u)),
                        f32(extractBits(i32(pw0), 16u, 8u)),
                        f32(extractBits(i32(pw0), 24u, 8u)));
                    wv1 = vec4<f32>(
                        f32(extractBits(i32(pw1), 0u, 8u)),
                        f32(extractBits(i32(pw1), 8u, 8u)),
                        f32(extractBits(i32(pw1), 16u, 8u)),
                        f32(extractBits(i32(pw1), 24u, 8u)));

                    let block0 = g * 8u + (lane * 8u) / 32u;
                    let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                    let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                    scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                    let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                    scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
                }

                for (var m = 0u; m < TILE_M; m++) {
                    let smem_base = m * BK_LOAD + k_base;
                    let sv0 = vec4<f32>(smem_s[smem_base],
                                        smem_s[smem_base + 1u],
                                        smem_s[smem_base + 2u],
                                        smem_s[smem_base + 3u]);
                    let sv1 = vec4<f32>(smem_s[smem_base + 4u],
                                        smem_s[smem_base + 5u],
                                        smem_s[smem_base + 6u],
                                        smem_s[smem_base + 7u]);
                    acc[m][c] += dot(sv0, wv0) * scale0 + dot(sv1, wv1) * scale1;
                }
            }
        }

        workgroupBarrier();
    }

    // Reduce + residual add
    for (var c = 0u; c < COLS_PER_WARP; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            let warp_sum = subgroupAdd(acc[m][c]);
            let row = tile_row * TILE_M + m;
            if (lane == 0u && col_valid[c] && row < T) {
                Y[row * N + cols[c]] += warp_sum + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [quant_q8] q8_down_silu_add_vulkan
static const char* WGSL_Q8_DOWN_SILU_ADD_VULKAN = R"WGSL(
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// Double-buffered MMA fused SiLU+down+residual:
//   Y[M×N] += silu_mul(GateUp[M×2*IM]) × W[N×K]^T + Bias[N]
//
// Same double-buffer + TILE_K=32 as q8_matmul_mma but with SiLU
// activation computed during tile_A staging.
//
// params: [K=IM, N=E, IM, M=T]
// Grid: (ceil(N/32), ceil(M/32))

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { K: u32, N: u32, IM: u32, M: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const MK: u32 = 16u;
const SB: u32 = 32u;
const WG: u32 = 512u;
const AB_A: u32 = 2048u;
const AB_B: u32 = 2048u;

var<workgroup> tA: array<array<f16, 2048>, 2>;
var<workgroup> tB: array<array<f16, 2048>, 2>;
var<workgroup> tC: array<f32, 4096>;

fn load_silu(gu: ptr<storage, array<f32>, read_write>, row: u32, k: u32, IM: u32) -> f16 {
    let base = row * 2u * IM;
    let gate = (*gu)[base + k];
    let up = (*gu)[base + IM + k];
    return f16(gate / (1.0 + exp(-gate)) * up);
}

@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let K = params.K;  let N = params.N;
    let IM = params.IM;  let M = params.M;
    let lx = lid.x;
    let rb = wid.y * TM;
    let cb = wid.x * TN;
    let sr = sg_id & 3u;
    let sc = sg_id >> 2u;
    let ws = K / 4u;
    let nkt = K / TK;

    var mC: subgroup_matrix_result<f32, 16, 16>;

    // Prefetch tile 0
    for (var i = lx; i < AB_A; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gr = rb + r;
        tA[0][i] = select(0.0h, load_silu(&GateUp, gr, c, IM), gr < M);
    }
    for (var i = lx; i < AB_B; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gc = cb + r;
        if (gc < N) {
            let pk = W_Q8[gc * ws + c / 4u];
            let q = f32(extractBits(i32(pk), (c & 3u) * 8u, 8u));
            let si = gc * (K / SB) + c / SB;
            let sp = unpack2x16float(Scales[si / 2u]);
            tB[0][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
        } else { tB[0][i] = 0.0h; }
    }
    workgroupBarrier();

    for (var ti = 0u; ti < nkt; ti++) {
        let cur = ti & 1u;
        let nxt = 1u - cur;
        let kn = (ti + 1u) * TK;

        if (ti + 1u < nkt) {
            for (var i = lx; i < AB_A; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gr = rb + r;
                tA[nxt][i] = select(0.0h, load_silu(&GateUp, gr, kn + c, IM), gr < M);
            }
            for (var i = lx; i < AB_B; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gc = cb + r;  let gk = kn + c;
                if (gc < N) {
                    let pk = W_Q8[gc * ws + gk / 4u];
                    let q = f32(extractBits(i32(pk), (gk & 3u) * 8u, 8u));
                    let si = gc * (K / SB) + gk / SB;
                    let sp = unpack2x16float(Scales[si / 2u]);
                    tB[nxt][i] = f16(q * select(sp.x, sp.y, (si & 1u) != 0u));
                } else { tB[nxt][i] = 0.0h; }
            }
        }

        let ao = sr * 16u * TK;
        let bo = sc * 16u * TK;
        let mA0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&tA[cur], ao, false, TK);
        let mB0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&tB[cur], bo, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA0, mB0, mC);

        let mA1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&tA[cur], ao + MK, false, TK);
        let mB1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&tB[cur], bo + MK, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA1, mB1, mC);

        workgroupBarrier();
    }

    subgroupMatrixStore(&tC, sr * 16u * TN + sc * 16u, mC, false, TN);
    workgroupBarrier();

    for (var i = lx; i < TM * TN; i += WG) {
        let r = i / TN;  let c = i % TN;
        let gr = rb + r;  let gc = cb + c;
        if (gr < M && gc < N) {
            Y[gr * N + gc] += tC[i] + Bias[gc];
        }
    }
}
)WGSL";

// [quant_q8] q8_gather_batched
static const char* WGSL_Q8_GATHER_BATCHED = R"WGSL(
// Gather rows from a Q8_0 table into fp32. Grid: ceil(T*D/256).
@group(0) @binding(0) var<storage,read> W:array<u32>;
@group(0) @binding(1) var<storage,read> S:array<u32>;
@group(0) @binding(2) var<storage,read> Tokens:array<i32>;
@group(0) @binding(3) var<storage,read_write> O:array<f32>;
@group(0) @binding(4) var<storage,read> P:array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let T=P[0];let D=P[1];let V=P[2];let i=gid.x;if(i>=T*D){return;}
 let t=i/D;let d=i%D;let raw=Tokens[t];let tok=select(0u,u32(raw),raw>=0&&u32(raw)<V);
 let p=W[tok*(D/4u)+d/4u];let q=f32(extractBits(i32(p),8u*(d&3u),8u));
 let si=tok*(D/32u)+d/32u;let sp=unpack2x16float(S[si/2u]);
 O[i]=q*select(sp.x,sp.y,(si&1u)!=0u);
}
)WGSL";

// [quant_q8] q8_matmul
static const char* WGSL_Q8_MATMUL = R"WGSL(
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
var<workgroup> reduce_scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // K_PER_ITER=8: each lane processes 8 elements per 256-element stride
    // (32 lanes × 8 elem/lane = 256 elements = 8 Q8_0 blocks per stride)
    let n_strides = K / 256u;
    let stride_w = K / 4u;  // u32 per weight row

    var acc: f32 = 0.0;

    let col_valid = col < N;

    if (col_valid) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations (two vec4 loads)
            let k_base = g * 256u + lane * 8u;
            let xv0 = vec4<f32>(X[x_base + k_base],
                                X[x_base + k_base + 1u],
                                X[x_base + k_base + 2u],
                                X[x_base + k_base + 3u]);
            let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                X[x_base + k_base + 5u],
                                X[x_base + k_base + 6u],
                                X[x_base + k_base + 7u]);

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 and form vec4 for dot product
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales: 2 blocks per 8 elements
            // Correction: lane * 8 spans TWO 32-element blocks when lane >= 4
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            // vec4 dot products with per-block scaling
            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    acc = select(0.0, acc, col_valid);

    reduce_scratch[tid] = acc;
    workgroupBarrier();
    for (var offset = 16u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) { reduce_scratch[tid] += reduce_scratch[tid + offset]; }
        workgroupBarrier();
    }
    let warp_sum = reduce_scratch[warp_id * 32u];

    if (lane == 0u && col_valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_add
static const char* WGSL_Q8_MATMUL_ADD = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    // 8 warps × 32 threads: each warp computes one output
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    // K_PER_ITER=8: each lane processes 8 elements per 256-element stride
    // (32 lanes × 8 elem/lane = 256 elements = 8 Q8_0 blocks per stride)
    let n_strides = K / 256u;
    let stride_w = K / 4u;  // u32 per weight row

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations (two vec4 loads)
            let k_base = g * 256u + lane * 8u;
            let xv0 = vec4<f32>(X[x_base + k_base],
                                X[x_base + k_base + 1u],
                                X[x_base + k_base + 2u],
                                X[x_base + k_base + 3u]);
            let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                X[x_base + k_base + 5u],
                                X[x_base + k_base + 6u],
                                X[x_base + k_base + 7u]);

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 and form vec4 for dot product
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales: 2 blocks per 8 elements
            // block_idx = (g * 256 + lane * 8) / 32
            let bi = g * 8u + lane / 4u;
            let si0 = s_base + bi;
            let si1 = si0 + 1u;
            // Correction: lane * 8 spans TWO 32-element blocks when lane >= 4
            // Block for first 4 elements: (g*256 + lane*8) / 32 = g*8 + lane/4
            // Block for second 4 elements: (g*256 + lane*8 + 4) / 32
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            // vec4 dot products with per-block scaling
            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_add_batched
static const char* WGSL_Q8_MATMUL_ADD_BATCHED = R"WGSL(
enable subgroups;

// Batched Q8_0 matmul + residual add for prefill.
// Y[T×N] += X[T×K] × W[K×N]^T + Bias
// Same as q8_matmul_batched but adds to output (residual connection).

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 24u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    var acc: array<f32, 8>;
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        var wv0: vec4<f32>;
        var wv1: vec4<f32>;
        var scale0: f32 = 0.0;
        var scale1: f32 = 0.0;

        if (valid && in_k) {
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
            wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);
        }

        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                let x_base = row * K;
                let xv0 = vec4<f32>(X[x_base + k_base],
                                    X[x_base + k_base + 1u],
                                    X[x_base + k_base + 2u],
                                    X[x_base + k_base + 3u]);
                let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                    X[x_base + k_base + 5u],
                                    X[x_base + k_base + 6u],
                                    X[x_base + k_base + 7u]);
                acc[m] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
            }
        }
    }

    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] += warp_sum + Bias[col];
        }
    }
}
)WGSL";

// [quant_q8] q8_matmul_add_fast
static const char* WGSL_Q8_MATMUL_ADD_FAST = R"WGSL(
enable subgroups;
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;
const TILE_N: u32 = 8u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x; let tile_col = wid.y; let tid = lid.x;
    let K = _params_[0]; let N = _params_[1]; let x_base = row * K;
    let warp_id = tid/32u; let lane = tid%32u;
    let col = tile_col * TILE_N + warp_id;
    let n_strides = K / 512u; let stride_w = K / 4u;
    var acc: f32 = 0.0;
    if (col < N) {
        let w_base = col * stride_w; let n_blocks = K / 32u; let s_base = col * n_blocks;
        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g*512u + lane*16u;
            let xv0 = vec4<f32>(X[x_base+k_base],X[x_base+k_base+1u],X[x_base+k_base+2u],X[x_base+k_base+3u]);
            let xv1 = vec4<f32>(X[x_base+k_base+4u],X[x_base+k_base+5u],X[x_base+k_base+6u],X[x_base+k_base+7u]);
            let xv2 = vec4<f32>(X[x_base+k_base+8u],X[x_base+k_base+9u],X[x_base+k_base+10u],X[x_base+k_base+11u]);
            let xv3 = vec4<f32>(X[x_base+k_base+12u],X[x_base+k_base+13u],X[x_base+k_base+14u],X[x_base+k_base+15u]);
            let w_off = w_base + g*128u + lane*4u;
            let pw0=W_Q8[w_off]; let pw1=W_Q8[w_off+1u]; let pw2=W_Q8[w_off+2u]; let pw3=W_Q8[w_off+3u];
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));
            let wv2 = vec4<f32>(f32(extractBits(i32(pw2),0u,8u)),f32(extractBits(i32(pw2),8u,8u)),f32(extractBits(i32(pw2),16u,8u)),f32(extractBits(i32(pw2),24u,8u)));
            let wv3 = vec4<f32>(f32(extractBits(i32(pw3),0u,8u)),f32(extractBits(i32(pw3),8u,8u)),f32(extractBits(i32(pw3),16u,8u)),f32(extractBits(i32(pw3),24u,8u)));
            let b0=g*16u+(lane*16u)/32u; let b1=g*16u+(lane*16u+4u)/32u;
            let b2=g*16u+(lane*16u+8u)/32u; let b3=g*16u+(lane*16u+12u)/32u;
            let sp0=unpack2x16float(Scales[(s_base+b0)/2u]); let sc0=select(sp0.x,sp0.y,((s_base+b0)&1u)!=0u);
            let sp1=unpack2x16float(Scales[(s_base+b1)/2u]); let sc1=select(sp1.x,sp1.y,((s_base+b1)&1u)!=0u);
            let sp2=unpack2x16float(Scales[(s_base+b2)/2u]); let sc2=select(sp2.x,sp2.y,((s_base+b2)&1u)!=0u);
            let sp3=unpack2x16float(Scales[(s_base+b3)/2u]); let sc3=select(sp3.x,sp3.y,((s_base+b3)&1u)!=0u);
            acc += dot(xv0,wv0)*sc0+dot(xv1,wv1)*sc1+dot(xv2,wv2)*sc2+dot(xv3,wv3)*sc3;
        }
    }
    let warp_sum = subgroupAdd(acc);
    if (lane==0u && col<N) { Y[row*N+col] += warp_sum + Bias[col]; }
}
)WGSL";

// [quant_q8] q8_matmul_add_lite
static const char* WGSL_Q8_MATMUL_ADD_LITE = R"WGSL(
enable subgroups;

// Q8_0 matmul + residual add with single output per workgroup (TILE_N=1, WG=32).
// Same as q8_matmul_lite but adds the result to Y (residual connection).

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let col = wid.y;
    let lane = lid.x;

    let K = _params_[0];
    let N = _params_[1];

    let valid = col < N;

    let x_base = row * K;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    let n_strides = K / 256u;

    var acc: f32 = 0.0;

    for (var g = 0u; g < n_strides; g = g + 1u) {
        if (valid) {
        let k_base = g * 256u + lane * 8u;
        let xv0 = vec4<f32>(X[x_base + k_base],
                            X[x_base + k_base + 1u],
                            X[x_base + k_base + 2u],
                            X[x_base + k_base + 3u]);
        let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                            X[x_base + k_base + 5u],
                            X[x_base + k_base + 6u],
                            X[x_base + k_base + 7u]);

        let w_off = w_base + g * 64u + lane * 2u;
        let pw0 = W_Q8[w_off];
        let pw1 = W_Q8[w_off + 1u];

        let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
        let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

        let block0 = g * 8u + (lane * 8u) / 32u;
        let block1 = g * 8u + (lane * 8u + 4u) / 32u;
        let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
        let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
        let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
        let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

        acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && valid) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_add_norm
static const char* WGSL_Q8_MATMUL_ADD_NORM = R"WGSL(
enable subgroups;

// Fused: Residual Add + RMSNorm + Q8_0 matmul.
// Combines add_rms_norm + q8 matmul into a single dispatch.
// Pass 1: X += Residual, compute sum_sq (for RMSNorm)
// Pass 2: Re-read X (L1 cached), apply rstd * NormW, dot with W_Q8
//
// Bindings:
//   0: X (read_write) — hidden state, updated with residual add
//   1: Residual (read) — residual to add to X
//   2: W_Q8 (read) — quantized weight matrix (N × K/4 u32)
//   3: Scales (read) — fp16 scales packed as u32
//   4: Bias (read) — per-output bias
//   5: Y (write) — output
//   6: _params_ — [K, N, 0, eps_as_u32]
//   7: NormW (read) — norm weight vector, E floats

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(3) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@group(0) @binding(6) var<storage, read_write> _params_: array<u32>;
@group(0) @binding(7) var<storage, read_write> NormW: array<f32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let eps = bitcast<f32>(_params_[3]);
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;

    // ── Pass 1: Residual add + compute sum of squares ──────────────────
    // All 8 warps read X + Residual, add them, store back, compute sum_sq.
    // X is 12KB for K=3072 — fits in L1 cache for pass 2.
    var sum_sq: f32 = 0.0;
    for (var g = 0u; g < n_strides; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let idx0 = x_base + k_base;

        // Read X + Residual, add
        var xv0 = vec4<f32>(X[idx0],     X[idx0 + 1u],
                            X[idx0 + 2u], X[idx0 + 3u]);
        var xv1 = vec4<f32>(X[idx0 + 4u], X[idx0 + 5u],
                            X[idx0 + 6u], X[idx0 + 7u]);
        let rv0 = vec4<f32>(Residual[idx0],     Residual[idx0 + 1u],
                            Residual[idx0 + 2u], Residual[idx0 + 3u]);
        let rv1 = vec4<f32>(Residual[idx0 + 4u], Residual[idx0 + 5u],
                            Residual[idx0 + 6u], Residual[idx0 + 7u]);
        xv0 = xv0 + rv0;
        xv1 = xv1 + rv1;

        // Store updated X
        X[idx0]     = xv0.x; X[idx0 + 1u] = xv0.y;
        X[idx0 + 2u] = xv0.z; X[idx0 + 3u] = xv0.w;
        X[idx0 + 4u] = xv1.x; X[idx0 + 5u] = xv1.y;
        X[idx0 + 6u] = xv1.z; X[idx0 + 7u] = xv1.w;

        sum_sq += dot(xv0, xv0) + dot(xv1, xv1);
    }
    let total_sq = subgroupAdd(sum_sq);
    let rstd = 1.0 / sqrt(total_sq / f32(K) + eps);

    // ── Pass 2: Normalized matmul ───────────────────────────────────────
    // Re-read X (L1 cached), apply RMSNorm weight, dot with Q8 weights.
    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g * 256u + lane * 8u;

            // Re-read X from L1 cache + apply norm
            let raw0 = vec4<f32>(X[x_base + k_base],
                                 X[x_base + k_base + 1u],
                                 X[x_base + k_base + 2u],
                                 X[x_base + k_base + 3u]);
            let raw1 = vec4<f32>(X[x_base + k_base + 4u],
                                 X[x_base + k_base + 5u],
                                 X[x_base + k_base + 6u],
                                 X[x_base + k_base + 7u]);
            let nw0 = vec4<f32>(NormW[k_base],     NormW[k_base + 1u],
                                NormW[k_base + 2u], NormW[k_base + 3u]);
            let nw1 = vec4<f32>(NormW[k_base + 4u], NormW[k_base + 5u],
                                NormW[k_base + 6u], NormW[k_base + 7u]);
            let xv0 = raw0 * rstd * nw0;
            let xv1 = raw1 * rstd * nw1;

            // Read Q8 weights
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_add_smem
static const char* WGSL_Q8_MATMUL_ADD_SMEM = R"WGSL(
enable subgroups;

// Q8_0 matmul + residual add with shared-memory X caching.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const STRIDE: u32 = 256u;
const MAX_STRIDES: u32 = 24u;

var<workgroup> smem_x: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;
    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let valid = col < N;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    var acc: f32 = 0.0;

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * STRIDE;
        let in_range = k_off < K;

        if (in_range) {
            smem_x[tid] = X[x_base + k_off + tid];
        }
        workgroupBarrier();

        if (valid && in_range) {
            let k_base = lane * 8u;
            let xv0 = vec4<f32>(smem_x[k_base], smem_x[k_base+1u],
                                smem_x[k_base+2u], smem_x[k_base+3u]);
            let xv1 = vec4<f32>(smem_x[k_base+4u], smem_x[k_base+5u],
                                smem_x[k_base+6u], smem_x[k_base+7u]);

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        Y[row * N + col] += warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_batched
static const char* WGSL_Q8_MATMUL_BATCHED = R"WGSL(
enable subgroups;

// Batched Q8_0 matmul for prefill: Y[T×N] = X[T×K] × W[K×N]^T
// Each workgroup computes a TILE_M × TILE_N output tile.
// Weight row is Q8_0: read once, reused across all T (M) rows.
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)
// WG: 256 threads = 8 warps
//
// Tile: TILE_M=8 rows × TILE_N=8 cols, K iterated in blocks of 32
// Each warp handles one output column, all TILE_M rows.
//
// Bindings:
//   0: X — fp32 activations [T × K]
//   1: W_Q8 — int8 weights packed as u32 [N × K/4]
//   2: Scales — fp16 weight scales packed as u32
//   3: Bias — per-output bias [N]
//   4: Y — output [T × N]
//   5: _params_ — [K, N, T, 0]

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const TILE_M: u32 = 8u;
const MAX_STRIDES: u32 = 24u;  // ceil(6144/256)

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;   // which TILE_M block of rows
    let tile_col = wid.y;   // which TILE_N block of columns
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    // Each warp accumulates TILE_M rows for its output column
    var acc: array<f32, 8>;  // TILE_M accumulators
    for (var m = 0u; m < TILE_M; m++) { acc[m] = 0.0; }

    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let in_k = g * 256u < K;

        // Read 2 packed u32 weights (8 int8 values) — same for all rows
        var wv0: vec4<f32>;
        var wv1: vec4<f32>;
        var scale0: f32 = 0.0;
        var scale1: f32 = 0.0;

        if (valid && in_k) {
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
            wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);
        }

        // For each of TILE_M rows, read X and dot with same weights
        for (var m = 0u; m < TILE_M; m++) {
            let row = tile_row * TILE_M + m;
            if (row < T && valid && in_k) {
                let x_base = row * K;
                let xv0 = vec4<f32>(X[x_base + k_base],
                                    X[x_base + k_base + 1u],
                                    X[x_base + k_base + 2u],
                                    X[x_base + k_base + 3u]);
                let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                                    X[x_base + k_base + 5u],
                                    X[x_base + k_base + 6u],
                                    X[x_base + k_base + 7u]);
                acc[m] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
            }
        }
    }

    // Reduce across lanes within each warp
    for (var m = 0u; m < TILE_M; m++) {
        let warp_sum = subgroupAdd(acc[m]);
        let row = tile_row * TILE_M + m;
        if (lane == 0u && valid && row < T) {
            Y[row * N + col] = warp_sum + Bias[col];
        }
    }
}
)WGSL";

// [quant_q8] q8_matmul_batched_dp4a
static const char* WGSL_Q8_MATMUL_BATCHED_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// DP4A Q8 prefill matmul. A workgroup computes 32 columns x 4 rows and
// quantizes each activation block once for all columns.
@group(0) @binding(0) var<storage,read>X:array<f32>;
@group(0) @binding(1) var<storage,read>W:array<u32>;
@group(0) @binding(2) var<storage,read>S:array<u32>;
@group(0) @binding(3) var<storage,read>Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write>Y:array<f32>;
@group(0) @binding(5) var<storage,read>P:array<u32>;
const ROWS:u32=4u;const COLS:u32=4u;const BK:u32=256u;
var<workgroup>xq:array<u32,256>;var<workgroup>xs:array<f32,32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id)lid:vec3<u32>,@builtin(workgroup_id)wid:vec3<u32>){
 let K=P[0];let N=P[1];let M=P[2];let tid=lid.x;let warp=tid/32u;let lane=tid&31u;let row0=wid.x*ROWS;
 let blocks=K/32u;let words=K/4u;var cols:array<u32,COLS>;var valid:array<bool,COLS>;
 for(var c=0u;c<COLS;c++){cols[c]=wid.y*32u+warp*COLS+c;valid[c]=cols[c]<N;}
 var acc:array<f32,ROWS*COLS>;for(var i=0u;i<ROWS*COLS;i++){acc[i]=0.0;}
 for(var kb=0u;kb<K;kb+=BK){let block=tid/32u;let e=tid&31u;let pl=e&3u;let pg=e/4u;
  for(var r=0u;r<ROWS;r++){let row=row0+r;let xv=select(0.0,X[row*K+kb+tid],row<M);var am=abs(xv);
   am=max(am,subgroupShuffleXor(am,16u));am=max(am,subgroupShuffleXor(am,8u));am=max(am,subgroupShuffleXor(am,4u));am=max(am,subgroupShuffleXor(am,2u));am=max(am,subgroupShuffleXor(am,1u));
   let sc=am/127.0;if(e==0u){xs[r*8u+block]=sc;}let safe=select(1.0,sc,sc!=0.0);let q=u32(clamp(i32(round(xv/safe)),-127,127))&255u;var pack=q<<(pl*8u);pack|=subgroupShuffleXor(pack,1u);pack|=subgroupShuffleXor(pack,2u);if(pl==0u){xq[r*64u+block*8u+pg]=pack;}}
  workgroupBarrier();let xb=lane/4u;let wb=kb/32u+xb;
  for(var c=0u;c<COLS;c++){if(valid[c]){let col=cols[c];let off=col*words+kb/4u+lane*2u;let w0=W[off];let w1=W[off+1u];let si=col*blocks+wb;let sp=unpack2x16float(S[si/2u]);let ws=select(sp.x,sp.y,(si&1u)!=0u);
    for(var r=0u;r<ROWS;r++){let base=r*64u+lane*2u;let d=dot4I8Packed(xq[base],w0)+dot4I8Packed(xq[base+1u],w1);acc[c*ROWS+r]+=f32(d)*ws*xs[r*8u+xb];}}}workgroupBarrier();}
 for(var c=0u;c<COLS;c++){
  for(var r=0u;r<ROWS;r++){let sum=subgroupAdd(acc[c*ROWS+r]);if(lane==0u&&valid[c]){let row=row0+r;if(row<M){Y[row*N+cols[c]]=sum+Bias[cols[c]];}}}
 }
}
)WGSL";

// [quant_q8] q8_matmul_d3d12
static const char* WGSL_Q8_MATMUL_D3D12 = R"WGSL(
enable subgroups;

// D3D12-optimized Q8_0 GEMM: double-buffered smem, no MMA.
//   Y[T×N] = X[T×K] × W[N×K]^T + Bias[N]
//
// Same binding layout as q8_matmul_tiled but with double-buffered
// X caching to overlap next tile load with current compute.
//
// TILE_M=8, TILE_N=32 (4 cols per warp), 256 threads.
// params: [K, N, T, 0]
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const TILE_M: u32 = 8u;
const COLS_PER_WARP: u32 = 4u;
const BK: u32 = 256u;
const WG: u32 = 256u;
const MAX_ITERS: u32 = 24u;  // ceil(6144/256)

// Double-buffered shared memory for X tile
var<workgroup> smem_x: array<array<f32, 2048>, 2>;  // [2][8×256]

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

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

    // Prefetch first tile into buffer 0
    for (var r = 0u; r < TILE_M; r++) {
        let row = tile_row * TILE_M + r;
        let idx = r * BK + tid;
        smem_x[0][idx] = select(0.0, X[row * K + tid], row < T && tid < K);
    }
    workgroupBarrier();

    for (var g = 0u; g < MAX_ITERS; g++) {
        let cur = g & 1u;
        let nxt = 1u - cur;
        let k_off = g * BK;
        let in_k = k_off < K;
        let k_next = (g + 1u) * BK;

        // Load next tile into nxt buffer (if not last)
        if (g + 1u < MAX_ITERS) {
            for (var r = 0u; r < TILE_M; r++) {
                let row = tile_row * TILE_M + r;
                let idx = r * BK + tid;
                smem_x[nxt][idx] = select(0.0, X[row * K + k_next + tid], row < T && k_next + tid < K);
            }
        }

        // Compute: each warp processes 4 columns from global W,
        // dotting with X rows from shared memory
        if (in_k) {
        let k_base = lane * 8u;

        for (var c = 0u; c < COLS_PER_WARP; c++) {
            var wv0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var wv1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var scale0: f32 = 0.0;
            var scale1: f32 = 0.0;

            if (col_valid[c]) {
                let w_off = w_bases[c] + g * 64u + lane * 2u;
                let pw0 = W_Q8[w_off];
                let pw1 = W_Q8[w_off + 1u];

                wv0 = vec4<f32>(
                    f32(extractBits(i32(pw0), 0u, 8u)),
                    f32(extractBits(i32(pw0), 8u, 8u)),
                    f32(extractBits(i32(pw0), 16u, 8u)),
                    f32(extractBits(i32(pw0), 24u, 8u)));
                wv1 = vec4<f32>(
                    f32(extractBits(i32(pw1), 0u, 8u)),
                    f32(extractBits(i32(pw1), 8u, 8u)),
                    f32(extractBits(i32(pw1), 16u, 8u)),
                    f32(extractBits(i32(pw1), 24u, 8u)));

                let block0 = g * 8u + (lane * 8u) / 32u;
                let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
            }

            for (var m = 0u; m < TILE_M; m++) {
                let smem_base = m * BK + k_base;
                let xv0 = vec4<f32>(smem_x[cur][smem_base],
                                    smem_x[cur][smem_base + 1u],
                                    smem_x[cur][smem_base + 2u],
                                    smem_x[cur][smem_base + 3u]);
                let xv1 = vec4<f32>(smem_x[cur][smem_base + 4u],
                                    smem_x[cur][smem_base + 5u],
                                    smem_x[cur][smem_base + 6u],
                                    smem_x[cur][smem_base + 7u]);
                acc[m][c] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
            }
        }
        }

        workgroupBarrier();
    }

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
)WGSL";

// [quant_q8] q8_matmul_decode_dp4a_d3d12
static const char* WGSL_Q8_MATMUL_DECODE_DP4A_D3D12 = R"WGSL(
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
const TILE_M: u32 = 1u;
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
)WGSL";

// [quant_q8] q8_matmul_dp4a
static const char* WGSL_Q8_MATMUL_DP4A = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Q8_0 matmul using DP4A (dot4I8Packed) for integer dot products.
// W8A32: INT8 weights × FP32 activations.
//
// Strategy: quantize the fp32 activation vector to int8 per-block
// (same blocking as Q8_0 weights: blocks of 32), then use dp4a
// for the integer dot product, and rescale with both scales.
//
// Each lane processes 8 elements (2 dp4a calls per iteration).

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const MAX_STRIDES: u32 = 8u;   // ceil(2048 / 256) — supports up to K=2048

var<workgroup> smem_x_q: array<u32, 64>;   // quantized X: 256 int8 = 64 u32
var<workgroup> smem_x_s: array<f32, 8>;    // per-block scales for X: 256/32 = 8

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let stride_w = K / 4u;

    var acc: f32 = 0.0;
    let valid = col < N;
    let w_base = select(0u, col * stride_w, valid);
    let n_blocks = K / 32u;
    let s_base = select(0u, col * n_blocks, valid);

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * 256u;
        let in_range = k_off < K;

        // ── Phase 1: Cooperatively quantize X to int8 ──────────────────
        {
            let block_id = tid / 32u;  // 0..7
            let elem_in_block = tid % 32u;
            let x_val = select(0.0, X[x_base + k_off + tid], in_range);
            let abs_val = abs(x_val);

            // Reduce absmax within warp
            var max_val = abs_val;
            max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
            max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

            let x_scale = max_val / 127.0;
            if (elem_in_block == 0u) {
                smem_x_s[block_id] = x_scale;
            }

            // Quantize this element to int8
            let safe_scale = select(1.0, x_scale, x_scale != 0.0);
            let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

            // Pack 4 int8 values into u32 using subgroup shuffle
            let pack_lane = elem_in_block % 4u;
            let pack_group = elem_in_block / 4u;
            let byte_val = u32(q_val & 0xFF);
            let shifted = byte_val << (pack_lane * 8u);

            var packed = shifted;
            packed = packed | subgroupShuffleXor(packed, 1u);
            packed = packed | subgroupShuffleXor(packed, 2u);

            if (pack_lane == 0u) {
                smem_x_q[block_id * 8u + pack_group] = packed;
            }
        }
        workgroupBarrier();

        // ── Phase 2: DP4A matmul using quantized X ─────────────────────
        if (valid && in_range) {
            let k_base_in_stride = lane * 8u;
            let xq_off = k_base_in_stride / 4u;

            let xq0 = smem_x_q[xq_off];
            let xq1 = smem_x_q[xq_off + 1u];

            let w_off = w_base + g * 64u + lane * 2u;
            let wq0 = W_Q8[w_off];
            let wq1 = W_Q8[w_off + 1u];

            // DP4A: dot product of 4 packed int8 values
            let idot0 = dot4I8Packed(xq0, wq0);
            let idot1 = dot4I8Packed(xq1, wq1);

            // Rescale: result = int_sum * x_scale * w_scale
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;

            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let w_scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let w_scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            let x_block0 = k_base_in_stride / 32u;
            let x_block1 = (k_base_in_stride + 4u) / 32u;
            let x_scale0 = smem_x_s[x_block0];
            let x_scale1 = smem_x_s[x_block1];

            acc += f32(idot0) * w_scale0 * x_scale0
                 + f32(idot1) * w_scale1 * x_scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_dp4a_d3d12
static const char* WGSL_Q8_MATMUL_DP4A_D3D12 = R"WGSL(
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
)WGSL";

// [quant_q8] q8_matmul_fast
static const char* WGSL_Q8_MATMUL_FAST = R"WGSL(
enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 512u;
    let stride_w = K / 4u;
    var acc: f32 = 0.0;

    let col_valid = col < N;

    if (col_valid) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;
        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g * 512u + lane * 16u;
            let xv0 = vec4<f32>(X[x_base+k_base], X[x_base+k_base+1u],
                                X[x_base+k_base+2u], X[x_base+k_base+3u]);
            let xv1 = vec4<f32>(X[x_base+k_base+4u], X[x_base+k_base+5u],
                                X[x_base+k_base+6u], X[x_base+k_base+7u]);
            let xv2 = vec4<f32>(X[x_base+k_base+8u], X[x_base+k_base+9u],
                                X[x_base+k_base+10u], X[x_base+k_base+11u]);
            let xv3 = vec4<f32>(X[x_base+k_base+12u], X[x_base+k_base+13u],
                                X[x_base+k_base+14u], X[x_base+k_base+15u]);

            let w_off = w_base + g * 128u + lane * 4u;
            let pw0 = W_Q8[w_off]; let pw1 = W_Q8[w_off+1u];
            let pw2 = W_Q8[w_off+2u]; let pw3 = W_Q8[w_off+3u];
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));
            let wv2 = vec4<f32>(f32(extractBits(i32(pw2),0u,8u)),f32(extractBits(i32(pw2),8u,8u)),
                                f32(extractBits(i32(pw2),16u,8u)),f32(extractBits(i32(pw2),24u,8u)));
            let wv3 = vec4<f32>(f32(extractBits(i32(pw3),0u,8u)),f32(extractBits(i32(pw3),8u,8u)),
                                f32(extractBits(i32(pw3),16u,8u)),f32(extractBits(i32(pw3),24u,8u)));

            let block0 = g*16u + (lane*16u)/32u;
            let block1 = g*16u + (lane*16u+4u)/32u;
            let block2 = g*16u + (lane*16u+8u)/32u;
            let block3 = g*16u + (lane*16u+12u)/32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let sc0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let sc1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);
            let sp2 = unpack2x16float(Scales[(s_base+block2)/2u]);
            let sc2 = select(sp2.x, sp2.y, ((s_base+block2)&1u)!=0u);
            let sp3 = unpack2x16float(Scales[(s_base+block3)/2u]);
            let sc3 = select(sp3.x, sp3.y, ((s_base+block3)&1u)!=0u);
            acc += dot(xv0,wv0)*sc0 + dot(xv1,wv1)*sc1
                 + dot(xv2,wv2)*sc2 + dot(xv3,wv3)*sc3;
        }
    }

    acc = select(0.0, acc, col_valid);
    var warp_sum = acc;
    warp_sum += subgroupShuffleXor(warp_sum, 16u);
    warp_sum += subgroupShuffleXor(warp_sum, 8u);
    warp_sum += subgroupShuffleXor(warp_sum, 4u);
    warp_sum += subgroupShuffleXor(warp_sum, 2u);
    warp_sum += subgroupShuffleXor(warp_sum, 1u);

    if (lane==0u && col_valid) {
        Y[row*N+col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_lite
static const char* WGSL_Q8_MATMUL_LITE = R"WGSL(
enable subgroups;

// Q8_0 matmul with single output per workgroup (TILE_N=1, WG=32).
// Optimized for small K where high workgroup count = more GPU occupancy.
// Each workgroup = 1 warp = 32 threads computing 1 output element.
// K_PER_ITER=8: each thread processes 8 elements per 256-element stride.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let col = wid.y;
    let lane = lid.x;

    let K = _params_[0];
    let N = _params_[1];

    let valid = col < N;

    let x_base = row * K;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    let n_strides = K / 256u;

    var acc: f32 = 0.0;

    for (var g = 0u; g < n_strides; g = g + 1u) {
        if (valid) {
        // Read 8 fp32 activations
        let k_base = g * 256u + lane * 8u;
        let xv0 = vec4<f32>(X[x_base + k_base],
                            X[x_base + k_base + 1u],
                            X[x_base + k_base + 2u],
                            X[x_base + k_base + 3u]);
        let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                            X[x_base + k_base + 5u],
                            X[x_base + k_base + 6u],
                            X[x_base + k_base + 7u]);

        // Read 2 packed u32 weights (8 int8 values)
        let w_off = w_base + g * 64u + lane * 2u;
        let pw0 = W_Q8[w_off];
        let pw1 = W_Q8[w_off + 1u];

        let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                            f32(extractBits(i32(pw0), 8u, 8u)),
                            f32(extractBits(i32(pw0), 16u, 8u)),
                            f32(extractBits(i32(pw0), 24u, 8u)));
        let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                            f32(extractBits(i32(pw1), 8u, 8u)),
                            f32(extractBits(i32(pw1), 16u, 8u)),
                            f32(extractBits(i32(pw1), 24u, 8u)));

        // Per-block scales
        let block0 = g * 8u + (lane * 8u) / 32u;
        let block1 = g * 8u + (lane * 8u + 4u) / 32u;
        let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
        let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
        let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
        let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

        acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_norm
static const char* WGSL_Q8_MATMUL_NORM = R"WGSL(
// Fused: RMSNorm + Q8_0 matmul.
// Reads raw X, computes RMSNorm inline (two-pass over X — second from L1 cache),
// then multiplies normalized X by Q8 weight matrix.
// Eliminates the separate rms_norm/rms_next dispatch.
//
// Pass 1: Load X, accumulate sum_sq per warp via subgroupAdd
// Pass 2: Re-load X (L1 cached, 8KB), multiply by rstd * NormW, dot with W_Q8
//
// Bindings:
//   0: X (read) — raw input vector (pre-norm), E floats
//   1: W_Q8 (read) — quantized weight matrix (N × K/4 u32)
//   2: Scales (read) — fp16 scales packed as u32
//   3: Bias (read) — per-output bias
//   4: Y (write) — output
//   5: _params_ — [K, N, 0, eps_as_u32]
//   6: NormW (read) — norm weight vector, E floats

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;
@group(0) @binding(6) var<storage, read_write> NormW: array<f32>;

const TILE_N: u32 = 8u;
var<workgroup> reduce_scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let eps = bitcast<f32>(_params_[3]);
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;

    // ── Pass 1: Compute sum of squares for RMSNorm ──────────────────────
    // Each warp independently reads the full X and computes sum_sq.
    // X is 8KB (K=2048 × 4 bytes) — fits in L1 cache for pass 2.
    var sum_sq: f32 = 0.0;
    for (var g = 0u; g < n_strides; g = g + 1u) {
        let k_base = g * 256u + lane * 8u;
        let xv0 = vec4<f32>(X[x_base + k_base],
                            X[x_base + k_base + 1u],
                            X[x_base + k_base + 2u],
                            X[x_base + k_base + 3u]);
        let xv1 = vec4<f32>(X[x_base + k_base + 4u],
                            X[x_base + k_base + 5u],
                            X[x_base + k_base + 6u],
                            X[x_base + k_base + 7u]);
        sum_sq += dot(xv0, xv0) + dot(xv1, xv1);
    }
    reduce_scratch[tid] = sum_sq;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            reduce_scratch[tid] += reduce_scratch[tid + stride];
        }
        workgroupBarrier();
    }
    let total_sq = reduce_scratch[warp_id * 32u];
    let rstd = 1.0 / sqrt(total_sq / f32(K) + eps);
    // Every logical warp must consume its RMS result before the same scratch
    // array is reused for the matmul reduction below. Wider/non-lockstep
    // hardware subgroups otherwise expose a write-after-read race.
    workgroupBarrier();

    // ── Pass 2: Normalized matmul ───────────────────────────────────────
    // Re-read X (L1 cached), apply RMSNorm weight, dot with Q8 weights.
    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_base = g * 256u + lane * 8u;

            // Re-read X from L1 cache + apply norm
            let raw0 = vec4<f32>(X[x_base + k_base],
                                 X[x_base + k_base + 1u],
                                 X[x_base + k_base + 2u],
                                 X[x_base + k_base + 3u]);
            let raw1 = vec4<f32>(X[x_base + k_base + 4u],
                                 X[x_base + k_base + 5u],
                                 X[x_base + k_base + 6u],
                                 X[x_base + k_base + 7u]);
            let nw0 = vec4<f32>(NormW[k_base],     NormW[k_base + 1u],
                                NormW[k_base + 2u], NormW[k_base + 3u]);
            let nw1 = vec4<f32>(NormW[k_base + 4u], NormW[k_base + 5u],
                                NormW[k_base + 6u], NormW[k_base + 7u]);
            let xv0 = raw0 * rstd * nw0;
            let xv1 = raw1 * rstd * nw1;

            // Read Q8 weights
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    reduce_scratch[tid] = acc;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            reduce_scratch[tid] += reduce_scratch[tid + stride];
        }
        workgroupBarrier();
    }
    let warp_sum = reduce_scratch[warp_id * 32u];

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_prequant_add_d3d12
static const char* WGSL_Q8_MATMUL_PREQUANT_ADD_D3D12 = R"WGSL(
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
)WGSL";

// [quant_q8] q8_matmul_prequant_add_wide_d3d12
static const char* WGSL_Q8_MATMUL_PREQUANT_ADD_WIDE_D3D12 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Wider D3D12 DP4A batched matmul over pre-quantized activations with residual add.
// Intended for the hotspot FFN down projection during prefill.
//   Y[M×N] += Xq[M×K] × Wq[N×K]^T + Bias[N]
// Grid: (ceil(N/64), ceil(M/4), 1)

@group(0) @binding(0) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(1) var<storage, read_write> XS: array<f32>;
@group(0) @binding(2) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(3) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

struct Params { M: u32, N: u32, K: u32, pad: u32, };
@group(0) @binding(6) var<uniform> params: Params;

const TILE_N: u32 = 64u;
const TILE_M: u32 = 4u;
const COLS_PER_WARP: u32 = 8u;
const BK: u32 = 256u;

var<workgroup> smem_xq: array<u32, 256>;
var<workgroup> smem_xs: array<f32, 32>;

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
    let lane_k = lane % 16u;
    let lane_c_grp = lane / 16u;
    let stride_xq = K / 4u;
    let stride_xs = K / 32u;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let ng = K / BK;
    let x_block = lane_k / 2u;

    var acc: array<array<f32, 4>, 4>;
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < 4u; c++) {
            acc[m][c] = 0.0;
        }
    }

    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    var w_bases: array<u32, 4>;
    var s_bases: array<u32, 4>;
    let c_start = lane_c_grp * 4u;
    for (var c = 0u; c < 4u; c++) {
        let global_c = tile_col * TILE_N + warp_id * COLS_PER_WARP + c_start + c;
        cols[c] = global_c;
        col_valid[c] = global_c < N;
        w_bases[c] = select(0u, cols[c] * stride_w, col_valid[c]);
        s_bases[c] = select(0u, cols[c] * n_blocks, col_valid[c]);
    }

    for (var g = 0u; g < ng; g++) {
        let x_pack_off = g * 64u;
        let x_scale_off = g * 8u;

        let row = tile_row * TILE_M + tid / 64u;
        let xj = tid % 64u;
        if (row < M) {
            smem_xq[tid] = XQ[row * stride_xq + x_pack_off + xj];
        } else {
            smem_xq[tid] = 0u;
        }

        if (tid < 32u) {
            let sm = tid / 8u;
            let sj = tid % 8u;
            let srow = tile_row * TILE_M + sm;
            if (srow < M) {
                smem_xs[tid] = XS[srow * stride_xs + x_scale_off + sj];
            } else {
                smem_xs[tid] = 0.0;
            }
        }
        workgroupBarrier();

        for (var c = 0u; c < 4u; c++) {
            if (col_valid[c]) {
                let w_off = w_bases[c] + x_pack_off + lane_k * 4u;
                let wq0 = W_Q8[w_off];
                let wq1 = W_Q8[w_off + 1u];
                let wq2 = W_Q8[w_off + 2u];
                let wq3 = W_Q8[w_off + 3u];

                let block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(s_bases[c] + block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((s_bases[c] + block) & 1u) != 0u);

                for (var m = 0u; m < TILE_M; m++) {
                    let xm_base = m * 64u + lane_k * 4u;
                    let xq0 = smem_xq[xm_base];
                    let xq1 = smem_xq[xm_base + 1u];
                    let xq2 = smem_xq[xm_base + 2u];
                    let xq3 = smem_xq[xm_base + 3u];

                    let xsc = smem_xs[m * 8u + x_block];

                    let idot = dot4I8Packed(xq0, wq0) +
                               dot4I8Packed(xq1, wq1) +
                               dot4I8Packed(xq2, wq2) +
                               dot4I8Packed(xq3, wq3);

                    acc[m][c] += f32(idot) * w_scale * xsc;
                }
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < 4u; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            var val = acc[m][c];
            val += subgroupShuffleXor(val, 8u);
            val += subgroupShuffleXor(val, 4u);
            val += subgroupShuffleXor(val, 2u);
            val += subgroupShuffleXor(val, 1u);

            let row = tile_row * TILE_M + m;
            if (lane_k == 0u && col_valid[c] && row < M) {
                Y[row * N + cols[c]] += val + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [quant_q8] q8_matmul_prequant_d3d12
static const char* WGSL_Q8_MATMUL_PREQUANT_D3D12 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// D3D12 DP4A batched matmul over pre-quantized activations.
//   Y[M×N] = Xq[M×K] × Wq[N×K]^T + Bias[N]
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
                Y[row * N + cols[c]] = warp_sum + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [quant_q8] q8_matmul_prequant_wide_d3d12
static const char* WGSL_Q8_MATMUL_PREQUANT_WIDE_D3D12 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Wider D3D12 DP4A batched matmul over pre-quantized activations.
// Intended only for hotspot projections like qkv and gateup.
//   Y[M×N] = Xq[M×K] × Wq[N×K]^T + Bias[N]
// Grid: (ceil(N/64), ceil(M/4), 1)

@group(0) @binding(0) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(1) var<storage, read_write> XS: array<f32>;
@group(0) @binding(2) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(3) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

struct Params { M: u32, N: u32, K: u32, pad: u32, };
@group(0) @binding(6) var<uniform> params: Params;

const TILE_N: u32 = 64u;
const TILE_M: u32 = 4u;
const COLS_PER_WARP: u32 = 8u;
const BK: u32 = 256u;

var<workgroup> smem_xq: array<u32, 256>;
var<workgroup> smem_xs: array<f32, 32>;

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
    let lane_k = lane % 16u;
    let lane_c_grp = lane / 16u;
    let stride_xq = K / 4u;
    let stride_xs = K / 32u;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let ng = K / BK;
    let x_block = lane_k / 2u;

    var acc: array<array<f32, 4>, 4>;
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < 4u; c++) {
            acc[m][c] = 0.0;
        }
    }

    var cols: array<u32, 4>;
    var col_valid: array<bool, 4>;
    var w_bases: array<u32, 4>;
    var s_bases: array<u32, 4>;
    let c_start = lane_c_grp * 4u;
    for (var c = 0u; c < 4u; c++) {
        let global_c = tile_col * TILE_N + warp_id * COLS_PER_WARP + c_start + c;
        cols[c] = global_c;
        col_valid[c] = global_c < N;
        w_bases[c] = select(0u, cols[c] * stride_w, col_valid[c]);
        s_bases[c] = select(0u, cols[c] * n_blocks, col_valid[c]);
    }

    for (var g = 0u; g < ng; g++) {
        let x_pack_off = g * 64u;
        let x_scale_off = g * 8u;

        let row = tile_row * TILE_M + tid / 64u;
        let xj = tid % 64u;
        if (row < M) {
            smem_xq[tid] = XQ[row * stride_xq + x_pack_off + xj];
        } else {
            smem_xq[tid] = 0u;
        }

        if (tid < 32u) {
            let sm = tid / 8u;
            let sj = tid % 8u;
            let srow = tile_row * TILE_M + sm;
            if (srow < M) {
                smem_xs[tid] = XS[srow * stride_xs + x_scale_off + sj];
            } else {
                smem_xs[tid] = 0.0;
            }
        }
        workgroupBarrier();

        for (var c = 0u; c < 4u; c++) {
            if (col_valid[c]) {
                let w_off = w_bases[c] + x_pack_off + lane_k * 4u;
                let wq0 = W_Q8[w_off];
                let wq1 = W_Q8[w_off + 1u];
                let wq2 = W_Q8[w_off + 2u];
                let wq3 = W_Q8[w_off + 3u];

                let block = g * 8u + x_block;
                let sp = unpack2x16float(Scales[(s_bases[c] + block) / 2u]);
                let w_scale = select(sp.x, sp.y, ((s_bases[c] + block) & 1u) != 0u);

                for (var m = 0u; m < TILE_M; m++) {
                    let xm_base = m * 64u + lane_k * 4u;
                    let xq0 = smem_xq[xm_base];
                    let xq1 = smem_xq[xm_base + 1u];
                    let xq2 = smem_xq[xm_base + 2u];
                    let xq3 = smem_xq[xm_base + 3u];

                    let xsc = smem_xs[m * 8u + x_block];

                    let idot = dot4I8Packed(xq0, wq0) +
                               dot4I8Packed(xq1, wq1) +
                               dot4I8Packed(xq2, wq2) +
                               dot4I8Packed(xq3, wq3);

                    acc[m][c] += f32(idot) * w_scale * xsc;
                }
            }
        }

        workgroupBarrier();
    }

    for (var c = 0u; c < 4u; c++) {
        for (var m = 0u; m < TILE_M; m++) {
            var val = acc[m][c];
            val += subgroupShuffleXor(val, 8u);
            val += subgroupShuffleXor(val, 4u);
            val += subgroupShuffleXor(val, 2u);
            val += subgroupShuffleXor(val, 1u);

            let out_row = tile_row * TILE_M + m;
            if (lane_k == 0u && col_valid[c] && out_row < M) {
                Y[out_row * N + cols[c]] = val + Bias[cols[c]];
            }
        }
    }
}
)WGSL";

// [quant_q8] q8_matmul_smem
static const char* WGSL_Q8_MATMUL_SMEM = R"WGSL(
enable subgroups;

// Q8_0 matmul with shared-memory X caching.
// All 256 threads cooperatively load X, all participate in barriers.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;
const STRIDE: u32 = 256u;
const MAX_STRIDES: u32 = 24u;

var<workgroup> smem_x: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;
    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;
    let valid = col < N;
    let stride_w = K / 4u;
    let n_blocks = K / 32u;
    let w_base = select(0u, col * stride_w, valid);
    let s_base = select(0u, col * n_blocks, valid);
    var acc: f32 = 0.0;

    for (var g = 0u; g < MAX_STRIDES; g = g + 1u) {
        let k_off = g * STRIDE;
        // Guard: K is uniform across workgroup so this is safe
        let in_range = k_off < K;

        // All threads participate in shared memory load + barrier
        if (in_range) {
            smem_x[tid] = X[x_base + k_off + tid];
        }
        workgroupBarrier();

        if (valid && in_range) {
            let k_base = lane * 8u;
            let xv0 = vec4<f32>(smem_x[k_base], smem_x[k_base+1u],
                                smem_x[k_base+2u], smem_x[k_base+3u]);
            let xv1 = vec4<f32>(smem_x[k_base+4u], smem_x[k_base+5u],
                                smem_x[k_base+6u], smem_x[k_base+7u]);

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0),0u,8u)),f32(extractBits(i32(pw0),8u,8u)),
                                f32(extractBits(i32(pw0),16u,8u)),f32(extractBits(i32(pw0),24u,8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1),0u,8u)),f32(extractBits(i32(pw1),8u,8u)),
                                f32(extractBits(i32(pw1),16u,8u)),f32(extractBits(i32(pw1),24u,8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base+block0)/2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base+block0)&1u)!=0u);
            let sp1 = unpack2x16float(Scales[(s_base+block1)/2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base+block1)&1u)!=0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
        workgroupBarrier();
    }

    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_tiled
static const char* WGSL_Q8_MATMUL_TILED = R"WGSL(
enable subgroups;

// Shared-memory tiled Q8_0 GEMM for batched prefill:
//   Y[T×N] = X[T×K] × W[N×K]^T + Bias[N]
//
// Tile: TILE_M(8) rows × TILE_N(32) columns per workgroup.
// 256 threads = 8 warps, each warp processes 4 output columns.
// X rows are loaded into shared memory and reused across all 32 columns.
//
// vs. q8_matmul_batched (TILE_N=8): 4× fewer workgroups, 4× more X reuse.
//
// Grid: (ceil(T/TILE_M), ceil(N/TILE_N), 1)
// Workgroup: 256 threads
//
// Shared memory: TILE_M × 256 = 2048 floats = 8 KB per K-block.

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;  // output columns per tile (4 per warp)
const TILE_M: u32 = 8u;   // output rows per tile
const COLS_PER_WARP: u32 = 4u;
const BK_LOAD: u32 = 256u;
const MAX_ITERS: u32 = 24u;

// Shared memory: TILE_M rows × BK_LOAD columns of X
var<workgroup> smem_x: array<f32, 2048>;  // 8 × 256

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let T = _params_[2];

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    let stride_w = K / 4u;
    let n_blocks = K / 32u;

    // Each warp handles COLS_PER_WARP=4 output columns
    var acc: array<array<f32, 4>, 8>;  // [TILE_M][COLS_PER_WARP]
    for (var m = 0u; m < TILE_M; m++) {
        for (var c = 0u; c < COLS_PER_WARP; c++) {
            acc[m][c] = 0.0;
        }
    }

    // Pre-compute column indices and validity
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

    for (var g = 0u; g < MAX_ITERS; g = g + 1u) {
        let k_off = g * BK_LOAD;
        let in_k = k_off < K;

        // ── Cooperative load: X tile [TILE_M × BK_LOAD] into smem ───
        for (var r = 0u; r < TILE_M; r++) {
            let row = tile_row * TILE_M + r;
            let smem_idx = r * BK_LOAD + tid;
            if (in_k && row < T) {
                smem_x[smem_idx] = X[row * K + k_off + tid];
            } else {
                smem_x[smem_idx] = 0.0;
            }
        }
        workgroupBarrier();

        // ── Compute: each warp processes 4 columns from global W,
        //    dotting with X rows from shared memory ───────────────────
        if (in_k) {
            let k_base = lane * 8u;

            for (var c = 0u; c < COLS_PER_WARP; c++) {
                // Guard invalid columns but keep all threads active
                // (no divergent control flow before subgroupAdd)
                var wv0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var wv1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var scale0: f32 = 0.0;
                var scale1: f32 = 0.0;

                if (col_valid[c]) {
                    let w_off = w_bases[c] + g * 64u + lane * 2u;
                    let pw0 = W_Q8[w_off];
                    let pw1 = W_Q8[w_off + 1u];

                    wv0 = vec4<f32>(
                        f32(extractBits(i32(pw0), 0u, 8u)),
                        f32(extractBits(i32(pw0), 8u, 8u)),
                        f32(extractBits(i32(pw0), 16u, 8u)),
                        f32(extractBits(i32(pw0), 24u, 8u)));
                    wv1 = vec4<f32>(
                        f32(extractBits(i32(pw1), 0u, 8u)),
                        f32(extractBits(i32(pw1), 8u, 8u)),
                        f32(extractBits(i32(pw1), 16u, 8u)),
                        f32(extractBits(i32(pw1), 24u, 8u)));

                    let block0 = g * 8u + (lane * 8u) / 32u;
                    let block1 = g * 8u + (lane * 8u + 4u) / 32u;
                    let sp0 = unpack2x16float(Scales[(s_bases[c] + block0) / 2u]);
                    scale0 = select(sp0.x, sp0.y, ((s_bases[c] + block0) & 1u) != 0u);
                    let sp1 = unpack2x16float(Scales[(s_bases[c] + block1) / 2u]);
                    scale1 = select(sp1.x, sp1.y, ((s_bases[c] + block1) & 1u) != 0u);
                }

                for (var m = 0u; m < TILE_M; m++) {
                    let smem_base = m * BK_LOAD + k_base;
                    let xv0 = vec4<f32>(smem_x[smem_base],
                                        smem_x[smem_base + 1u],
                                        smem_x[smem_base + 2u],
                                        smem_x[smem_base + 3u]);
                    let xv1 = vec4<f32>(smem_x[smem_base + 4u],
                                        smem_x[smem_base + 5u],
                                        smem_x[smem_base + 6u],
                                        smem_x[smem_base + 7u]);
                    acc[m][c] += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
                }
            }
        }

        workgroupBarrier();
    }

    // Reduce across lanes within each warp, write results
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
)WGSL";

// [quant_q8] q8_matmul_vec4
static const char* WGSL_Q8_MATMUL_VEC4 = R"WGSL(
enable subgroups;

// Optimized Q8_0 matmul with vec4 loads for X (activation vector).
// Uses array<vec4<f32>> binding to enable 128-bit coalesced memory reads.
// K_PER_ITER=8: each lane processes 8 elements per 256-element stride.
// TILE_N=8: each workgroup computes 8 output elements (one per warp).

@group(0) @binding(0) var<storage, read_write> X_v4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;

    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        // X base in vec4 units: row * K / 4
        let x_base_v4 = row * (K / 4u);

        for (var g = 0u; g < n_strides; g = g + 1u) {
            // Read 8 fp32 activations via 2 × vec4 loads (128-bit each)
            let k_v4_base = g * 64u + lane * 2u;  // 256 elements / 4 = 64 vec4s per stride
            let xv0 = X_v4[x_base_v4 + k_v4_base];
            let xv1 = X_v4[x_base_v4 + k_v4_base + 1u];

            // Read 2 packed u32 weights (8 int8 values)
            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            // Extract int8 → f32 via extractBits (sign-extended)
            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            // Per-block scales (2 blocks per 8 elements)
            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    let warp_sum = subgroupAdd(acc);

    if (lane == 0u && col < N) {
        Y[row * N + col] = warp_sum + Bias[col];
    }
}
)WGSL";

// [quant_q8] q8_matmul_vulkan
static const char* WGSL_Q8_MATMUL_VULKAN = R"WGSL(
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

// Double-buffered subgroup-matrix Q8_0 GEMM:
//   Y[M×N] = X[M×K] × W[N×K]^T + Bias[N]
//
// Optimizations:
//   - Double-buffered smem: load next tile while computing current
//   - TILE_K=32: 2 MMA calls per K-block, halves barrier count
//   - Uniform params: enable early loop exit at actual K
//   - Vectorized 4-at-a-time Q8 dequant from packed u32
//
// 4 subgroups, 32×32 output tile. Grid: (ceil(N/32), ceil(M/32))

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W_Q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

struct Params { M: u32, N: u32, K: u32, pad: u32, };
@group(0) @binding(5) var<uniform> params: Params;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const MK: u32 = 16u;     // MMA tile K
const SB: u32 = 32u;     // Q8 scale block size
const WG: u32 = 512u;
const AB_A: u32 = 2048u; // TM * TK = 64*32
const AB_B: u32 = 2048u; // TN * TK = 64*32

var<workgroup> tA: array<array<f16, 2048>, 2>;  // double-buf A [64×32]
var<workgroup> tB: array<array<f16, 2048>, 2>;  // double-buf B [64×32]
var<workgroup> tC: array<f32, 4096>;             // output [64×64]

@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
    let M = params.M;  let N = params.N;  let K = params.K;
    let lx = lid.x;
    let rb = wid.y * TM;
    let cb = wid.x * TN;
    let sr = sg_id & 3u;   // 4 rows of 16 = 64
    let sc = sg_id >> 2u;  // 4 cols of 16 = 64
    let ws = K / 4u;
    let nkt = K / TK;

    var mC: subgroup_matrix_result<f32, 16, 16>;

    // ── Prefetch tile 0 into buffer 0 ────────────────────────────────
    for (var i = lx; i < AB_A; i += WG) {
        let r = i / TK;  let c = i % TK;
        let gr = rb + r;
        tA[0][i] = select(0.0h, f16(X[gr * K + c]), gr < M);
    }
    // Vectorized weight dequant: load 1 u32 (4 packed i8), write 4 f16
    // AB_B=2048 f16 = 512 u32 weights. 512 threads → 1 u32 per thread.
    {
        let wi = lx;  // 0..511 → one u32 per thread
        let row = wi / (TK / 4u);  // which of 64 weight rows
        let col4 = wi % (TK / 4u);  // which group of 4 i8 [0..7]
        let gc = cb + row;
        let base_k = col4 * 4u;
        if (gc < N) {
            let pk = W_Q8[gc * ws + col4];
            let si = gc * (K / SB) + base_k / SB;
            let sp = unpack2x16float(Scales[si / 2u]);
            let scale = select(sp.x, sp.y, (si & 1u) != 0u);
            let smem_base = row * TK + base_k;
            tB[0][smem_base]      = f16(f32(extractBits(i32(pk), 0u, 8u)) * scale);
            tB[0][smem_base + 1u] = f16(f32(extractBits(i32(pk), 8u, 8u)) * scale);
            tB[0][smem_base + 2u] = f16(f32(extractBits(i32(pk), 16u, 8u)) * scale);
            tB[0][smem_base + 3u] = f16(f32(extractBits(i32(pk), 24u, 8u)) * scale);
        } else {
            let smem_base = row * TK + base_k;
            tB[0][smem_base] = 0.0h; tB[0][smem_base + 1u] = 0.0h;
            tB[0][smem_base + 2u] = 0.0h; tB[0][smem_base + 3u] = 0.0h;
        }
    }
    workgroupBarrier();

    // ── Main loop: compute cur tile, load next ───────────────────────
    for (var ti = 0u; ti < nkt; ti++) {
        let cur = ti & 1u;
        let nxt = 1u - cur;
        let kn = (ti + 1u) * TK;

        // Load next tile (if not last iteration)
        if (ti + 1u < nkt) {
            for (var i = lx; i < AB_A; i += WG) {
                let r = i / TK;  let c = i % TK;
                let gr = rb + r;
                tA[nxt][i] = select(0.0h, f16(X[gr * K + kn + c]), gr < M);
            }
            // Vectorized weight dequant for next tile
            {
                let wi = lx;
                let row = wi / (TK / 4u);
                let col4 = wi % (TK / 4u);
                let gc = cb + row;
                let base_k = kn + col4 * 4u;
                if (gc < N && base_k < K) {
                    let pk = W_Q8[gc * ws + base_k / 4u];
                    let si = gc * (K / SB) + base_k / SB;
                    let sp = unpack2x16float(Scales[si / 2u]);
                    let scale = select(sp.x, sp.y, (si & 1u) != 0u);
                    let smem_base = row * TK + col4 * 4u;
                    tB[nxt][smem_base]      = f16(f32(extractBits(i32(pk), 0u, 8u)) * scale);
                    tB[nxt][smem_base + 1u] = f16(f32(extractBits(i32(pk), 8u, 8u)) * scale);
                    tB[nxt][smem_base + 2u] = f16(f32(extractBits(i32(pk), 16u, 8u)) * scale);
                    tB[nxt][smem_base + 3u] = f16(f32(extractBits(i32(pk), 24u, 8u)) * scale);
                } else {
                    let smem_base = row * TK + col4 * 4u;
                    tB[nxt][smem_base] = 0.0h; tB[nxt][smem_base + 1u] = 0.0h;
                    tB[nxt][smem_base + 2u] = 0.0h; tB[nxt][smem_base + 3u] = 0.0h;
                }
            }
        }

        // Compute: 2 MMA calls (K=32 = 2×16)
        let ao = sr * 16u * TK;
        let bo = sc * 16u * TK;

        let mA0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tA[cur], ao, false, TK);
        let mB0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tB[cur], bo, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA0, mB0, mC);

        let mA1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(
            &tA[cur], ao + MK, false, TK);
        let mB1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(
            &tB[cur], bo + MK, true, TK);
        mC = subgroupMatrixMultiplyAccumulate(mA1, mB1, mC);

        workgroupBarrier();
    }

    // ── Store result ─────────────────────────────────────────────────
    subgroupMatrixStore(&tC, sr * 16u * TN + sc * 16u, mC, false, TN);
    workgroupBarrier();

    for (var i = lx; i < TM * TN; i += WG) {
        let r = i / TN;  let c = i % TN;
        let gr = rb + r;  let gc = cb + c;
        if (gr < M && gc < N) {
            Y[gr * N + gc] = tC[i] + Bias[gc];
        }
    }
}
)WGSL";

// [quant_q8] quantize_fp32_rows_d3d12
static const char* WGSL_QUANTIZE_FP32_ROWS_D3D12 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Quantize fp32 activations [M×K] into packed int8 + per-block scales.
// Grid: (ceil(K/256), M, 1)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(2) var<storage, read_write> XS: array<f32>;

struct Params { M: u32, K: u32, pad0: u32, pad1: u32, };
@group(0) @binding(3) var<uniform> params: Params;

const BK: u32 = 256u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let block_group = wid.x;
    let row = wid.y;
    let tid = lid.x;

    let M = params.M;
    let K = params.K;
    let row_valid = row < M;
    let gk = block_group * BK + tid;
    let in_range = row_valid && gk < K;

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    let x_val = select(0.0, X[row * K + gk], in_range);
    var max_val = abs(x_val);
    max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

    let x_scale = max_val / 127.0;
    let safe_scale = select(1.0, x_scale, x_scale != 0.0);
    let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

    let byte_val = u32(q_val & 0xFF);
    let shifted = byte_val << (pack_lane * 8u);
    var packed = shifted;
    packed = packed | subgroupShuffleXor(packed, 1u);
    packed = packed | subgroupShuffleXor(packed, 2u);

    let stride_q = K / 4u;
    let stride_s = K / 32u;
    let block_base_q = block_group * 64u + block_id * 8u + pack_group;
    let block_base_s = block_group * 8u + block_id;

    if (row_valid && elem_in_block == 0u && block_base_s < stride_s) {
        XS[row * stride_s + block_base_s] = x_scale;
    }
    if (row_valid && pack_lane == 0u && block_base_q < stride_q) {
        XQ[row * stride_q + block_base_q] = packed;
    }
}
)WGSL";

// [quant_q8] silu_quantize_rows_d3d12
static const char* WGSL_SILU_QUANTIZE_ROWS_D3D12 = R"WGSL(
requires packed_4x8_integer_dot_product;
enable subgroups;

// Compute SiLU(gate) * up, then quantize to packed int8 + per-block scales.
// Grid: (ceil(IM/256), M, 1)

@group(0) @binding(0) var<storage, read_write> GateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> XQ: array<u32>;
@group(0) @binding(2) var<storage, read_write> XS: array<f32>;

struct Params { M: u32, K: u32, pad0: u32, pad1: u32, };
@group(0) @binding(3) var<uniform> params: Params;

const BK: u32 = 256u;

fn silu_mul(gu: ptr<storage, array<f32>, read_write>, row: u32, k: u32, IM: u32) -> f32 {
    let base = row * 2u * IM;
    let gate = (*gu)[base + k];
    let up = (*gu)[base + IM + k];
    return gate / (1.0 + exp(-gate)) * up;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let block_group = wid.x;
    let row = wid.y;
    let tid = lid.x;

    let M = params.M;
    let IM = params.K;
    let row_valid = row < M;
    let gk = block_group * BK + tid;
    let in_range = row_valid && gk < IM;

    let block_id = tid / 32u;
    let elem_in_block = tid % 32u;
    let pack_lane = elem_in_block % 4u;
    let pack_group = elem_in_block / 4u;

    let x_val = select(0.0, silu_mul(&GateUp, row, gk, IM), in_range);
    var max_val = abs(x_val);
    max_val = max(max_val, subgroupShuffleXor(max_val, 16u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 8u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 4u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 2u));
    max_val = max(max_val, subgroupShuffleXor(max_val, 1u));

    let x_scale = max_val / 127.0;
    let safe_scale = select(1.0, x_scale, x_scale != 0.0);
    let q_val = clamp(i32(round(x_val / safe_scale)), -127, 127);

    let byte_val = u32(q_val & 0xFF);
    let shifted = byte_val << (pack_lane * 8u);
    var packed = shifted;
    packed = packed | subgroupShuffleXor(packed, 1u);
    packed = packed | subgroupShuffleXor(packed, 2u);

    let stride_q = IM / 4u;
    let stride_s = IM / 32u;
    let block_base_q = block_group * 64u + block_id * 8u + pack_group;
    let block_base_s = block_group * 8u + block_id;

    if (row_valid && elem_in_block == 0u && block_base_s < stride_s) {
        XS[row * stride_s + block_base_s] = x_scale;
    }
    if (row_valid && pack_lane == 0u && block_base_q < stride_q) {
        XQ[row * stride_q + block_base_q] = packed;
    }
}
)WGSL";

// ─── [shape] ───────────────────────────────────────────────────────

// [shape] argmax
static const char* WGSL_ARGMAX = R"WGSL(
enable subgroups;

// Multi-workgroup argmax — Phase 1: each workgroup scans a chunk and writes
// partial (max_val, max_idx) to the Partials buffer.
//
// Grid: (NUM_WG, 1, 1) where NUM_WG = params[1]
// Bindings: 0=Logits(f32), 1=Result(i32), 2=Params(u32), 3=Partials(u32)

@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> Result: array<i32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;
@group(0) @binding(3) var<storage, read_write> Partials: array<u32>;

const WG_SIZE: u32 = 256u;

var<workgroup> wg_max_val: array<f32, 8>;
var<workgroup> wg_max_idx: array<i32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let NUM_WG = _params_[1];
    let tid = lid.x;
    let gid = wid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each workgroup scans its chunk
    let chunk = (N + NUM_WG - 1u) / NUM_WG;
    let start = gid * chunk;
    let end = min(start + chunk, N);

    var local_max: f32 = -1e30;
    var local_idx: i32 = 0;

    // Strided scan within chunk for coalesced memory access
    var i = start + tid;
    for (; i < end; i = i + WG_SIZE) {
        let v = Logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i32(i);
        }
    }

    // Warp reduce
    for (var offset = 16u; offset > 0u; offset = offset >> 1u) {
        let other_val = subgroupShuffleXor(bitcast<i32>(local_max), offset);
        let other_idx = subgroupShuffleXor(local_idx, offset);
        let ov = bitcast<f32>(other_val);
        if (ov > local_max) {
            local_max = ov;
            local_idx = other_idx;
        }
    }

    // Workgroup reduce across 8 warps
    if (lane == 0u) {
        wg_max_val[warp_id] = local_max;
        wg_max_idx[warp_id] = local_idx;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var best_val = wg_max_val[0];
        var best_idx = wg_max_idx[0];
        for (var w = 1u; w < 8u; w = w + 1u) {
            if (wg_max_val[w] > best_val) {
                best_val = wg_max_val[w];
                best_idx = wg_max_idx[w];
            }
        }
        // Write partial result: [val_u32, idx_i32] per workgroup
        Partials[gid * 2u] = bitcast<u32>(best_val);
        Partials[gid * 2u + 1u] = bitcast<u32>(best_idx);

        // If single workgroup, write result directly
        if (NUM_WG == 1u) {
            Result[0] = best_idx;
        }
    }
}
)WGSL";

// [shape] argmax_reduce
static const char* WGSL_ARGMAX_REDUCE = R"WGSL(
enable subgroups;

// Argmax Phase 2: reduce NUM_WG partial results from Phase 1.
// Grid: (1, 1, 1) — single workgroup of 256 threads.
// Each partial is 2 u32s: [bitcast<u32>(max_val), bitcast<u32>(max_idx)].
//
// Bindings: 0=Partials(u32), 1=Result(i32), 2=Params(u32)

@group(0) @binding(0) var<storage, read> Partials: array<u32>;
@group(0) @binding(1) var<storage, read_write> Result: array<i32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

var<workgroup> wg_max_val: array<f32, 8>;
var<workgroup> wg_max_idx: array<i32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let NUM_WG = _params_[0];
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each thread scans a portion of the partial results
    var local_max: f32 = -1e30;
    var local_idx: i32 = 0;

    var i = tid;
    for (; i < NUM_WG; i = i + 256u) {
        let pv = bitcast<f32>(Partials[i * 2u]);
        let pi = bitcast<i32>(Partials[i * 2u + 1u]);
        if (pv > local_max) {
            local_max = pv;
            local_idx = pi;
        }
    }

    // Warp reduce
    for (var offset = 16u; offset > 0u; offset = offset >> 1u) {
        let other_val = subgroupShuffleXor(bitcast<i32>(local_max), offset);
        let other_idx = subgroupShuffleXor(local_idx, offset);
        let ov = bitcast<f32>(other_val);
        if (ov > local_max) {
            local_max = ov;
            local_idx = other_idx;
        }
    }

    // Workgroup reduce across 8 warps
    if (lane == 0u) {
        wg_max_val[warp_id] = local_max;
        wg_max_idx[warp_id] = local_idx;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var best_val = wg_max_val[0];
        var best_idx = wg_max_idx[0];
        for (var w = 1u; w < 8u; w = w + 1u) {
            if (wg_max_val[w] > best_val) {
                best_val = wg_max_val[w];
                best_idx = wg_max_idx[w];
            }
        }
        Result[0] = best_idx;
    }
}
)WGSL";

// [shape] concat_2input — f32 instantiated
static const char* WGSL_CONCAT_2INPUT = R"WGSL(
// Concat 2 inputs along any axis.
// Params: [0]=total_elements, [1]=axis_dim_A, [2]=axis_dim_out, [3]=inner_size
// Dispatch: ceil(total_elements / 512) — each thread handles 2 elements


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let a_split = _params_[1];
    let total_split = _params_[2];
    let inner = _params_[3];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    var v0: f32 = 0.0;
    var v1: f32 = 0.0;
    for (var k = 0u; k < 2u; k++) {
        let idx = base + k;
        if (idx >= N) { break; }
        let outer = idx / (total_split * inner);
        let rem = idx % (total_split * inner);
        let split_pos = rem / inner;
        let inner_pos = rem % inner;
        var val: f32;
        if (split_pos < a_split) {
            val = t_read(&A, outer * a_split * inner + split_pos * inner + inner_pos);
        } else {
            let b_split = total_split - a_split;
            val = t_read(&B, outer * b_split * inner + (split_pos - a_split) * inner + inner_pos);
        }
        if (k == 0u) { v0 = val; } else { v1 = val; }
    }
    t_write2(&Out, base, v0, v1);
}
)WGSL";

// [shape] concat_2input — dtype template
static const char* WGSL_CONCAT_2INPUT_T = R"WGSL(
// Concat 2 inputs along any axis.
// Params: [0]=total_elements, [1]=axis_dim_A, [2]=axis_dim_out, [3]=inner_size
// Dispatch: ceil(total_elements / 512) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> Out: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let a_split = _params_[1];
    let total_split = _params_[2];
    let inner = _params_[3];

    let base = gid.x * 2u;
    if (base >= N) { return; }

    var v0: f32 = 0.0;
    var v1: f32 = 0.0;
    for (var k = 0u; k < 2u; k++) {
        let idx = base + k;
        if (idx >= N) { break; }
        let outer = idx / (total_split * inner);
        let rem = idx % (total_split * inner);
        let split_pos = rem / inner;
        let inner_pos = rem % inner;
        var val: f32;
        if (split_pos < a_split) {
            val = t_read(&A, outer * a_split * inner + split_pos * inner + inner_pos);
        } else {
            let b_split = total_split - a_split;
            val = t_read(&B, outer * b_split * inner + (split_pos - a_split) * inner + inner_pos);
        }
        if (k == 0u) { v0 = val; } else { v1 = val; }
    }
    t_write2(&Out, base, v0, v1);
}
)WGSL";

// [shape] embed_gather — f32 instantiated
static const char* WGSL_EMBED_GATHER = R"WGSL(
// Embedding gather: out[i] = table[token_id * E + i] * normalizer
// Dispatch: (ceil(E/512), 1, 1) — each thread handles 2 elements
// _params_[0] = E (embedding dim), _params_[1] = normalizer (bitcast f32)


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> EmbeddingTable: array<f32>;
@group(0) @binding(1) var<storage, read> TokenId: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let normalizer_bits = _params_[1];
    let normalizer = select(bitcast<f32>(normalizer_bits), 1.0,
                            normalizer_bits == 0u);
    let token = u32(TokenId[0]);
    let base_offset = token * E;

    let base = gid.x * 2u;
    if (base >= E) { return; }

    let v0 = t_read(&EmbeddingTable, base_offset + base) * normalizer;
    var v1: f32 = 0.0;
    if (base + 1u < E) {
        v1 = t_read(&EmbeddingTable, base_offset + base + 1u) * normalizer;
    }
    t_write2(&X, base, v0, v1);
}
)WGSL";

// [shape] embed_gather — dtype template
static const char* WGSL_EMBED_GATHER_T = R"WGSL(
// Embedding gather: out[i] = table[token_id * E + i] * normalizer
// Dispatch: (ceil(E/512), 1, 1) — each thread handles 2 elements
// _params_[0] = E (embedding dim), _params_[1] = normalizer (bitcast f32)

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> EmbeddingTable: array<${T}>;
@group(0) @binding(1) var<storage, read> TokenId: array<i32>;
@group(0) @binding(2) var<storage, read_write> X: array<${T}>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let E = _params_[0];
    let normalizer_bits = _params_[1];
    let normalizer = select(bitcast<f32>(normalizer_bits), 1.0,
                            normalizer_bits == 0u);
    let token = u32(TokenId[0]);
    let base_offset = token * E;

    let base = gid.x * 2u;
    if (base >= E) { return; }

    let v0 = t_read(&EmbeddingTable, base_offset + base) * normalizer;
    var v1: f32 = 0.0;
    if (base + 1u < E) {
        v1 = t_read(&EmbeddingTable, base_offset + base + 1u) * normalizer;
    }
    t_write2(&X, base, v0, v1);
}
)WGSL";

// [shape] expand — f32 instantiated
static const char* WGSL_EXPAND = R"WGSL(
// Expand — broadcast tensor to larger shape
// Dispatch: (ceil(total/512), 1, 1) — each thread handles 2 elements
// Params: [total, ndim, 0, 0, out_strides[ndim], in_dims[ndim], in_strides[ndim]]


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_in_idx(out_idx: u32, ndim: u32) -> u32 {
    var remaining = out_idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_dim = _params_[4u + ndim + d];
        let in_stride = _params_[4u + 2u * ndim + d];
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;
        in_flat += (coord % in_dim) * in_stride;
    }
    return in_flat;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let ndim = _params_[1];

    let base = gid.x * 2u;
    if (base >= total) { return; }

    let v0 = t_read(&X, compute_in_idx(base, ndim));
    var v1: f32 = 0.0;
    if (base + 1u < total) {
        v1 = t_read(&X, compute_in_idx(base + 1u, ndim));
    }
    t_write2(&Y, base, v0, v1);
}
)WGSL";

// [shape] expand — dtype template
static const char* WGSL_EXPAND_T = R"WGSL(
// Expand — broadcast tensor to larger shape
// Dispatch: (ceil(total/512), 1, 1) — each thread handles 2 elements
// Params: [total, ndim, 0, 0, out_strides[ndim], in_dims[ndim], in_strides[ndim]]

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_in_idx(out_idx: u32, ndim: u32) -> u32 {
    var remaining = out_idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_dim = _params_[4u + ndim + d];
        let in_stride = _params_[4u + 2u * ndim + d];
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;
        in_flat += (coord % in_dim) * in_stride;
    }
    return in_flat;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let ndim = _params_[1];

    let base = gid.x * 2u;
    if (base >= total) { return; }

    let v0 = t_read(&X, compute_in_idx(base, ndim));
    var v1: f32 = 0.0;
    if (base + 1u < total) {
        v1 = t_read(&X, compute_in_idx(base + 1u, ndim));
    }
    t_write2(&Y, base, v0, v1);
}
)WGSL";

// [shape] gather
static const char* WGSL_GATHER = R"WGSL(
// Gather — index-based lookup along any axis
// Data is read as u32 (works for f32, packed fp16, etc.)
// Params: [outer, axisDim, innerU32, nIdx]
// Dispatch: (ceil(outer * nIdx * innerU32 / 256), 1, 1)

@group(0) @binding(0) var<storage, read> Data: array<u32>;
@group(0) @binding(1) var<storage, read> Indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outer = _params_[0];
    let axisDim = _params_[1];
    let innerU32 = _params_[2];
    let nIdx = _params_[3];

    let total = outer * nIdx * innerU32;
    let idx = gid.x;
    if (idx >= total) { return; }

    let j = idx % innerU32;
    let row = idx / innerU32;
    let i = row % nIdx;
    let outerIdx = row / nIdx;
    var gathered = Indices[i];
    if (gathered < 0) { gathered += i32(axisDim); }
    if (gathered < 0 || gathered >= i32(axisDim)) {
        Out[idx] = 0u;
        return;
    }
    let dataIdx = (outerIdx * axisDim + u32(gathered)) * innerU32 + j;
    Out[idx] = Data[dataIdx];
}
)WGSL";

// [shape] logit_softcap
static const char* WGSL_LOGIT_SOFTCAP = R"WGSL(
// Logit softcapping: Y[i] = tanh(Y[i] / cap) * cap
// Applied in-place after LM head matmul, before argmax.
//
// Dispatch: (ceil(N/256), 1, 1)
//
// Bindings:
//   0: Y (read_write) — logits [N] fp32, modified in-place
//   1: _params_ — [N, cap_as_u32, 0, 0]

@group(0) @binding(0) var<storage, read_write> Y: array<f32>;
@group(0) @binding(1) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let cap = bitcast<f32>(_params_[1]);
    let i = gid.x;
    if (i >= N) { return; }
    Y[i] = tanh(Y[i] / cap) * cap;
}
)WGSL";

// [shape] slice
static const char* WGSL_SLICE = R"WGSL(
// Slice — general N-dimensional slicing
// Params: [total, ndim, 0, 0, out_strides[ndim], in_strides[ndim], starts[ndim], steps[ndim]]
// Dispatch: (ceil(total/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= total) { return; }

    var remaining = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let start = _params_[4u + 2u * ndim + d];
        let step = _params_[4u + 3u * ndim + d];
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;
        let in_coord = start + coord * step;
        in_flat += in_coord * in_stride;
    }
    Y[idx] = X[in_flat];
}
)WGSL";

// [shape] slice_t — f32 instantiated
static const char* WGSL_SLICE_T = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= total) { return; }

    var remaining = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let start = _params_[4u + 2u * ndim + d];
        let step = _params_[4u + 3u * ndim + d];
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;
        let in_coord = start + coord * step;
        in_flat += in_coord * in_stride;
    }
    let val = t_read(&X, in_flat);
    t_write(&Y, idx, val);
}
)WGSL";

// [shape] slice_t — dtype template
static const char* WGSL_SLICE_T_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= total) { return; }

    var remaining = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let start = _params_[4u + 2u * ndim + d];
        let step = _params_[4u + 3u * ndim + d];
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;
        let in_coord = start + coord * step;
        in_flat += in_coord * in_stride;
    }
    let val = t_read(&X, in_flat);
    t_write(&Y, idx, val);
}
)WGSL";

// [shape] softmax — f32 instantiated
static const char* WGSL_SOFTMAX = R"WGSL(
// Softmax — numerically stable per-row softmax
// Dispatch: (ceil(nRows/256), 1, 1)
// Internal accumulation in f32 for numerical stability.
// Uses paired writes for fp16 correctness (u32 packing).


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nRows = _params_[0];
    let rowLen = _params_[1];
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * rowLen;

    // Find max (f32)
    var mx: f32 = -1e30;
    for (var i = 0u; i < rowLen; i++) {
        mx = max(mx, t_read(&X, base + i));
    }
    // Exp sum
    var expSum: f32 = 0.0;
    for (var i = 0u; i < rowLen; i++) {
        expSum += exp(t_read(&X, base + i) - mx);
    }
    let invSum = 1.0 / max(expSum, 1e-10);
    // Write output in pairs
    let pairs = rowLen / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = exp(t_read(&X, base + i0) - mx) * invSum;
        let v1 = exp(t_read(&X, base + i0 + 1u) - mx) * invSum;
        t_write2(&Y, base + i0, v0, v1);
    }
    if ((rowLen & 1u) != 0u) {
        let last = rowLen - 1u;
        t_write2(&Y, base + last, exp(t_read(&X, base + last) - mx) * invSum, 0.0);
    }
}
)WGSL";

// [shape] softmax — dtype template
static const char* WGSL_SOFTMAX_T = R"WGSL(
// Softmax — numerically stable per-row softmax
// Dispatch: (ceil(nRows/256), 1, 1)
// Internal accumulation in f32 for numerical stability.
// Uses paired writes for fp16 correctness (u32 packing).

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nRows = _params_[0];
    let rowLen = _params_[1];
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * rowLen;

    // Find max (f32)
    var mx: f32 = -1e30;
    for (var i = 0u; i < rowLen; i++) {
        mx = max(mx, t_read(&X, base + i));
    }
    // Exp sum
    var expSum: f32 = 0.0;
    for (var i = 0u; i < rowLen; i++) {
        expSum += exp(t_read(&X, base + i) - mx);
    }
    let invSum = 1.0 / max(expSum, 1e-10);
    // Write output in pairs
    let pairs = rowLen / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = exp(t_read(&X, base + i0) - mx) * invSum;
        let v1 = exp(t_read(&X, base + i0 + 1u) - mx) * invSum;
        t_write2(&Y, base + i0, v0, v1);
    }
    if ((rowLen & 1u) != 0u) {
        let last = rowLen - 1u;
        t_write2(&Y, base + last, exp(t_read(&X, base + last) - mx) * invSum, 0.0);
    }
}
)WGSL";

// [shape] topk — f32 instantiated
static const char* WGSL_TOPK = R"WGSL(
// TopK — selection sort for K largest/smallest values.
// Params: [0]=totalSlices, [1]=dimSize, [2]=k, [3]=largest
// Dispatch: (totalSlices, 1, 1) — one thread per slice


fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<i32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let totalSlices = _params_[0];
    let dimSize = _params_[1];
    let k = _params_[2];
    let largest = _params_[3];
    let slice = gid.x;
    if (slice >= totalSlices) { return; }

    let base_in = slice * dimSize;
    let base_out = slice * k;

    for (var ki = 0u; ki < k; ki++) {
        var best_val: f32 = select(1e30, -1e30, largest != 0u);
        var best_idx: u32 = 0u;

        for (var d = 0u; d < dimSize; d++) {
            let v = t_read(&data, base_in + d);
            var already = false;
            for (var p = 0u; p < ki; p++) {
                if (u32(out_indices[base_out + p]) == d) { already = true; break; }
            }
            if (already) { continue; }

            if (largest != 0u) {
                if (v > best_val) { best_val = v; best_idx = d; }
            } else {
                if (v < best_val) { best_val = v; best_idx = d; }
            }
        }

        t_write(&out_values, base_out + ki, best_val);
        out_indices[base_out + ki] = i32(best_idx);
    }
}
)WGSL";

// [shape] topk — dtype template
static const char* WGSL_TOPK_T = R"WGSL(
// TopK — selection sort for K largest/smallest values.
// Params: [0]=totalSlices, [1]=dimSize, [2]=k, [3]=largest
// Dispatch: (totalSlices, 1, 1) — one thread per slice

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> data: array<${T}>;
@group(0) @binding(1) var<storage, read_write> out_values: array<${T}>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<i32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let totalSlices = _params_[0];
    let dimSize = _params_[1];
    let k = _params_[2];
    let largest = _params_[3];
    let slice = gid.x;
    if (slice >= totalSlices) { return; }

    let base_in = slice * dimSize;
    let base_out = slice * k;

    for (var ki = 0u; ki < k; ki++) {
        var best_val: f32 = select(1e30, -1e30, largest != 0u);
        var best_idx: u32 = 0u;

        for (var d = 0u; d < dimSize; d++) {
            let v = t_read(&data, base_in + d);
            var already = false;
            for (var p = 0u; p < ki; p++) {
                if (u32(out_indices[base_out + p]) == d) { already = true; break; }
            }
            if (already) { continue; }

            if (largest != 0u) {
                if (v > best_val) { best_val = v; best_idx = d; }
            } else {
                if (v < best_val) { best_val = v; best_idx = d; }
            }
        }

        t_write(&out_values, base_out + ki, best_val);
        out_indices[base_out + ki] = i32(best_idx);
    }
}
)WGSL";

// [shape] transpose
static const char* WGSL_TRANSPOSE = R"WGSL(
// Transpose — general N-dimensional permutation
// Works on u32 elements (f32 or packed fp16 pairs).
// Params: [total, ndim, 0, 0, out_strides[ndim], in_strides[ndim]]
//
// Dispatch: (ceil(total/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    var out_idx = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let coord = out_idx / out_stride;
        out_idx = out_idx % out_stride;
        in_flat += coord * in_stride;
    }
    Y[idx] = X[in_flat];
}
)WGSL";

// [shape] transpose_t — f32 instantiated
static const char* WGSL_TRANSPOSE_T = R"WGSL(
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}


fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}


@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn flat_to_in(idx: u32, ndim: u32) -> u32 {
    var out_idx = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let coord = out_idx / out_stride;
        out_idx = out_idx % out_stride;
        in_flat += coord * in_stride;
    }
    return in_flat;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let in_idx = flat_to_in(idx, ndim);
    let val = t_read(&X, in_idx);
    t_write(&Y, idx, val);
}
)WGSL";

// [shape] transpose_t — dtype template
static const char* WGSL_TRANSPOSE_T_T = R"WGSL(
${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn flat_to_in(idx: u32, ndim: u32) -> u32 {
    var out_idx = idx;
    var in_flat: u32 = 0u;
    for (var d = 0u; d < ndim; d++) {
        let out_stride = _params_[4u + d];
        let in_stride = _params_[4u + ndim + d];
        let coord = out_idx / out_stride;
        out_idx = out_idx % out_stride;
        in_flat += coord * in_stride;
    }
    return in_flat;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let ndim = _params_[1];
    let idx = gid.x;
    if (idx >= N) { return; }
    let in_idx = flat_to_in(idx, ndim);
    let val = t_read(&X, in_idx);
    t_write(&Y, idx, val);
}
)WGSL";

// ─── [shared] ──────────────────────────────────────────────────────

// [shared] copy_buffer
static const char* WGSL_COPY_BUFFER = R"WGSL(
// Copy one f32 buffer to another (utility for layout shuffles + residuals)
//
// Bindings:
//   0: src      array<f32> (read)
//   1: dst      array<f32> (write)
//   2: _params_ — [N, src_offset_words, dst_offset_words]

@group(0) @binding(0) var<storage, read>       src:     array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:     array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N        = _params_[0];
    let src_off  = _params_[1];
    let dst_off  = _params_[2];
    let i = gid.x;
    if (i >= N) { return; }
    dst[dst_off + i] = src[src_off + i];
}
)WGSL";

// [shared] mul
static const char* WGSL_MUL = R"WGSL(
// F32 elementwise multiply: out = a * b
//
// Bindings:
//   0: a       array<f32> (read)
//   1: b       array<f32> (read)
//   2: out     array<f32> (write — may alias a or b)
//   3: _params_ — [N]

@group(0) @binding(0) var<storage, read>       a:       array<f32>;
@group(0) @binding(1) var<storage, read>       b:       array<f32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i >= N) { return; }
    out[i] = a[i] * b[i];
}
)WGSL";

// [shared] silu
static const char* WGSL_SILU = R"WGSL(
// SiLU activation (in-place or to separate output) — used by Mamba gate
//   out[i] = x[i] * sigmoid(x[i])
//
// Bindings:
//   0: x         array<f32>  (read)
//   1: out       array<f32>  (write — may alias x for in-place)
//   2: _params_              — [N]

@group(0) @binding(0) var<storage, read>       x:       array<f32>;
@group(0) @binding(1) var<storage, read_write> out:     array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i >= N) { return; }
    let v = x[i];
    let sig = 1.0 / (1.0 + exp(-v));
    out[i] = v * sig;
}
)WGSL";

// [shared] zero_init
static const char* WGSL_ZERO_INIT = R"WGSL(
// Zero-init a buffer (used by SSM h_state reset, MoE intermediate clears, etc.)
//
// Bindings:
//   0: buf   array<f32> (write)
//   1: _params_ — [N]

@group(0) @binding(0) var<storage, read_write> buf:     array<f32>;
@group(0) @binding(1) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let i = gid.x;
    if (i < N) { buf[i] = 0.0; }
}
)WGSL";

// ─── [ssm] ─────────────────────────────────────────────────────────

// [ssm] conv1d_decode
static const char* WGSL_CONV1D_DECODE = R"WGSL(
// SSM conv1d (depthwise) — Mamba decode primitive
//
// For decode mode (single new token), the SSM conv state is a rolling
// buffer of the last `conv_k` input vectors per channel, ordered oldest to
// newest like llama.cpp's ggml_ssm_conv input. This kernel
// performs the FIR filter:
//   out[c] = sum_{k=0..K-1} weights[c, k] * state[c, k]
// per channel c in 0..d_inner.
//
// Bindings:
//   0: state   [d_inner * K]  — rolling buffer (fp32)
//   1: weights [d_inner * K]  — depthwise conv1d kernel (fp32, from ssm_conv1d.weight)
//   2: bias    [d_inner]      — optional bias (fp32, may be zero buffer)
//   3: out     [d_inner]      — fp32 output
//   4: _params_                — [d_inner, K]
//
// Workgroup: 256 threads, each handles ceil(d_inner / 256) channels.

@group(0) @binding(0) var<storage, read>       state:    array<f32>;
@group(0) @binding(1) var<storage, read>       weights:  array<f32>;
@group(0) @binding(2) var<storage, read>       bias:     array<f32>;
@group(0) @binding(3) var<storage, read_write> out:      array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let K       = _params_[1];
    let c = gid.x;
    if (c >= d_inner) { return; }
    var acc: f32 = bias[c];
    // weights laid out as [d_inner, K]: weights[c*K + k]
    // state laid out as [d_inner, K], oldest to newest: state[c*K + k]
    let base = c * K;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        acc = acc + weights[base + k] * state[base + k];
    }
    out[c] = acc;
}
)WGSL";

// [ssm] conv_state_update
static const char* WGSL_CONV_STATE_UPDATE = R"WGSL(
// SSM conv state update — shift-in new sample
//
// For decode mode, the conv1d state is a rolling buffer of the last K input
// vectors per channel. Before each conv1d_decode dispatch, we shift the
// state by 1 (drop oldest, append new x) per channel.
//
// Layout matches llama.cpp's ggml_ssm_conv input window: state[c*K + k] where
// k=0 is the oldest sample and k=K-1 is the newest.
// After update: state[c*K + k] = old_state[c*K + (k+1)] for k<K-1,
// state[c*K + K-1] = x[c].
//
// Bindings:
//   0: state  [d_inner * K]   (read+write)
//   1: x      [d_inner]       new sample
//   2: _params_              — [d_inner, K]

@group(0) @binding(0) var<storage, read_write> state:   array<f32>;
@group(0) @binding(1) var<storage, read>       x:       array<f32>;
@group(0) @binding(2) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let K       = _params_[1];
    let c = gid.x;
    if (c >= d_inner) { return; }
    let base = c * K;
    // Shift left (drop oldest) and append newest at the end.
    for (var k: u32 = 0u; k + 1u < K; k = k + 1u) {
        state[base + k] = state[base + k + 1u];
    }
    state[base + K - 1u] = x[c];
}
)WGSL";

// [ssm] delta_net_decode
static const char* WGSL_DELTA_NET_DECODE = R"WGSL(
// DeltaNet decode step — matrix-state recurrence for qwen35 linear attention
//
// Per-head state is a matrix [head_k_dim, head_v_dim]. Per decode step:
//   kv      = dot(k[h], exp(g[h]) * state[h, :, v_col])
//   delta   = (v[h, v_col] - kv) * beta[h]
//   state'  = exp(g[h]) * state + outer(k[h], delta)
//   y       = (q[h] / sqrt(head_k_dim)) @ state'
//
// Heads layout: num_v_heads outputs; K is shared in groups (num_k_heads <= num_v_heads).
// Q and K are num_k_heads-wide; Q/K head index is derived from the V head.
//
// Bindings:
//   0: q       [num_k_heads * head_k_dim]      f32
//   1: k       [num_k_heads * head_k_dim]      f32  (shared groups of v_heads / k_heads)
//   2: v       [num_v_heads * head_v_dim]      f32
//   3: beta    [num_v_heads]                   f32
//   4: gate    [num_v_heads]                   f32  (exp applied in-kernel)
//   5: state   [num_v_heads * head_k_dim * head_v_dim]  f32  (read+write)
//   6: y       [num_v_heads * head_v_dim]      f32  (write)
//   7: _params_                                — [num_v_heads, num_k_heads, head_k_dim, head_v_dim]
//
// Grid: one workgroup per (V head, V column). Workgroup size is 128, matching
// qwen35 head_k_dim. Columns are independent, so this exposes the V dimension
// instead of looping it serially inside one workgroup.

enable subgroups;

@group(0) @binding(0) var<storage, read>       q:        array<f32>;
@group(0) @binding(1) var<storage, read>       k:        array<f32>;
@group(0) @binding(2) var<storage, read>       v:        array<f32>;
@group(0) @binding(3) var<storage, read>       beta:     array<f32>;
@group(0) @binding(4) var<storage, read>       gate:     array<f32>;
@group(0) @binding(5) var<storage, read_write> state:    array<f32>;
@group(0) @binding(6) var<storage, read_write> y:        array<f32>;
@group(0) @binding(7) var<storage, read>       _params_: array<u32>;

var<workgroup> warp_sums: array<f32, 4>;

fn reduce_wg_128(x: f32, tid: u32) -> f32 {
    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let sum = subgroupAdd(x);
    if (lane == 0u) {
        warp_sums[warp_id] = sum;
    }
    workgroupBarrier();
    let total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nv      = _params_[0];  // num_v_heads
    let nk      = _params_[1];  // num_k_heads
    let dk      = _params_[2];  // head_k_dim
    let dv      = _params_[3];  // head_v_dim
    let head    = wid.x;        // v-head index
    let vi      = wid.y;        // v column within the head
    if (head >= nv || vi >= dv) { return; }
    let tid     = lid.x;

    let k_head_idx = head % nk;  // mapping v-head -> repeated k/q head

    let q_base = k_head_idx * dk;
    let k_base = k_head_idx * dk;
    let v_base = head * dv;
    let state_base = head * dk * dv;

    let bh = beta[head];
    let gh = exp(gate[head]);
    let q_scale = inverseSqrt(f32(dk));
    let ki = tid;
    var qv: f32 = 0.0;
    var kv: f32 = 0.0;
    if (ki < dk) {
        qv = q[q_base + ki] * q_scale;
        kv = k[k_base + ki];
    }

    var kv_partial: f32 = 0.0;
    if (ki < dk) {
        kv_partial = gh * state[state_base + ki * dv + vi] * kv;
    }
    let kv_dot = reduce_wg_128(kv_partial, ki);
    let delta = (v[v_base + vi] - kv_dot) * bh;

    var attn_partial: f32 = 0.0;
    if (ki < dk) {
        let idx = state_base + ki * dv + vi;
        let s_new = gh * state[idx] + kv * delta;
        state[idx] = s_new;
        attn_partial = s_new * qv;
    }
    let attn = reduce_wg_128(attn_partial, ki);
    if (ki == 0u) {
        y[v_base + vi] = attn;
    }
}
)WGSL";

// [ssm] delta_net_decode_x2
static const char* WGSL_DELTA_NET_DECODE_X2 = R"WGSL(
// DeltaNet decode step, two V columns per workgroup.
// Each column keeps the same 4-warp reduction shape as delta_net_decode.

@group(0) @binding(0) var<storage, read>       q:        array<f32>;
@group(0) @binding(1) var<storage, read>       k:        array<f32>;
@group(0) @binding(2) var<storage, read>       v:        array<f32>;
@group(0) @binding(3) var<storage, read>       beta:     array<f32>;
@group(0) @binding(4) var<storage, read>       gate:     array<f32>;
@group(0) @binding(5) var<storage, read_write> state:    array<f32>;
@group(0) @binding(6) var<storage, read_write> y:        array<f32>;
@group(0) @binding(7) var<storage, read>       _params_: array<u32>;

var<workgroup> reduce_scratch: array<f32, 256>;

fn reduce_col_128(x: f32, tid: u32, pair: u32) -> f32 {
    let local = tid & 127u;
    reduce_scratch[tid] = x;
    workgroupBarrier();
    for (var offset = 64u; offset > 0u; offset = offset / 2u) {
        if (local < offset) {
            reduce_scratch[tid] += reduce_scratch[tid + offset];
        }
        workgroupBarrier();
    }
    return reduce_scratch[pair * 128u];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nv      = _params_[0];
    let nk      = _params_[1];
    let dk      = _params_[2];
    let dv      = _params_[3];
    let head    = wid.x;
    let pair    = lid.x >> 7u;
    let tid     = lid.x & 127u;
    let vi      = wid.y * 2u + pair;
    let is_active = head < nv && vi < dv;

    let k_head_idx = head % nk;
    let q_base = k_head_idx * dk;
    let k_base = k_head_idx * dk;
    let v_base = head * dv;
    let state_base = head * dk * dv;

    let bh = select(0.0, beta[head], is_active);
    let gh = select(0.0, exp(gate[head]), is_active);
    let q_scale = inverseSqrt(f32(dk));
    let ki = tid;

    var qv = 0.0;
    var kv = 0.0;
    if (ki < dk && is_active) {
        qv = q[q_base + ki] * q_scale;
        kv = k[k_base + ki];
    }

    var kv_partial = 0.0;
    if (ki < dk && is_active) {
        kv_partial = gh * state[state_base + ki * dv + vi] * kv;
    }
    let kv_dot = reduce_col_128(kv_partial, lid.x, pair);
    let delta = select(0.0, (v[v_base + vi] - kv_dot) * bh, is_active);

    var attn_partial = 0.0;
    if (ki < dk && is_active) {
        let idx = state_base + ki * dv + vi;
        let s_new = gh * state[idx] + kv * delta;
        state[idx] = s_new;
        attn_partial = s_new * qv;
    }
    let attn = reduce_col_128(attn_partial, lid.x, pair);
    if (tid == 0u && is_active) {
        y[v_base + vi] = attn;
    }
}
)WGSL";

// [ssm] delta_net_scan_x2
static const char* WGSL_DELTA_NET_SCAN_X2 = R"WGSL(
enable subgroups;

// Qwen 3.5 DeltaNet prefill scan. A workgroup owns two V columns and advances
// them through all prompt rows, preserving the exact decode recurrence.
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> Beta: array<f32>;
@group(0) @binding(4) var<storage, read> Gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> State: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 8>;
fn reduce128(x:f32, lane128:u32, pair:u32)->f32 {
    let lane=lane128&31u; let warp=(lane128>>5u)+pair*4u;
    let s=subgroupAdd(x); if(lane==0u){sums[warp]=s;} workgroupBarrier();
    let b=pair*4u; let total=sums[b]+sums[b+1u]+sums[b+2u]+sums[b+3u];
    workgroupBarrier(); return total;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
    let nv=P[0]; let nk=P[1]; let dk=P[2]; let dv=P[3]; let T=P[4];
    let head=wid.x; let pair=lid.x>>7u; let ki=lid.x&127u;
    let vi=wid.y*2u+pair; let valid=head<nv&&vi<dv;
    let kh=head%nk; let sb=head*dk*dv; let qs=inverseSqrt(f32(dk));
    for(var t=0u;t<T;t++) {
        let qbase=(t*nk+kh)*dk; let vbase=(t*nv+head)*dv;
        let bh=select(0.0,Beta[t*nv+head],valid);
        let gh=select(0.0,exp(Gate[t*nv+head]),valid);
        let kv=select(0.0,K[qbase+ki],valid&&ki<dk);
        let qv=select(0.0,Q[qbase+ki]*qs,valid&&ki<dk);
        let idx=sb+ki*dv+vi;
        let old=select(0.0,State[idx],valid&&ki<dk);
        let pred=reduce128(gh*old*kv,ki,pair);
        let delta=select(0.0,(V[vbase+vi]-pred)*bh,valid);
        let sn=gh*old+kv*delta;
        if(valid&&ki<dk){State[idx]=sn;}
        let out=reduce128(sn*qv,ki,pair);
        if(ki==0u&&valid){Y[vbase+vi]=out;}
    }
}
)WGSL";

// Four-column DeltaNet prefill tile. Each 128-thread half owns two adjacent
// value columns, reuses Q/K loads for both, and retains recurrent state in
// registers until the complete prompt scan finishes.
static const char* WGSL_DELTA_NET_SCAN_X4 = R"WGSL(
enable subgroups;
@group(0) @binding(0)var<storage,read>Q:array<f32>;
@group(0) @binding(1)var<storage,read>K:array<f32>;
@group(0) @binding(2)var<storage,read>V:array<f32>;
@group(0) @binding(3)var<storage,read>Beta:array<f32>;
@group(0) @binding(4)var<storage,read>Gate:array<f32>;
@group(0) @binding(5)var<storage,read_write>State:array<f32>;
@group(0) @binding(6)var<storage,read_write>Y:array<f32>;
@group(0) @binding(7)var<storage,read>P:array<u32>;
var<workgroup>sums:array<f32,8>;
fn reduce128(x:f32,lane128:u32,pair:u32)->f32{let lane=lane128&31u;let warp=(lane128>>5u)+pair*4u;let s=subgroupAdd(x);if(lane==0u){sums[warp]=s;}workgroupBarrier();let b=pair*4u;let total=sums[b]+sums[b+1u]+sums[b+2u]+sums[b+3u];workgroupBarrier();return total;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>){
 let nv=P[0];let nk=P[1];let dk=P[2];let dv=P[3];let T=P[4];let head=wid.x;let pair=lid.x>>7u;let ki=lid.x&127u;
 let vi0=wid.y*4u+pair*2u;let vi1=vi0+1u;let valid0=head<nv&&vi0<dv;let valid1=head<nv&&vi1<dv;let svi0=min(vi0,dv-1u);let svi1=min(vi1,dv-1u);
 let kh=head%nk;let sb=head*dk*dv;let qs=inverseSqrt(f32(dk));let idx0=sb+ki*dv+svi0;let idx1=sb+ki*dv+svi1;
 var st0=select(0.0,State[idx0],valid0&&ki<dk);var st1=select(0.0,State[idx1],valid1&&ki<dk);
 for(var t=0u;t<T;t++){
  let qbase=(t*nk+kh)*dk;let vbase=(t*nv+head)*dv;let bh=select(0.0,Beta[t*nv+head],head<nv);let gh=select(0.0,exp(Gate[t*nv+head]),head<nv);
  let kv=select(0.0,K[qbase+ki],head<nv&&ki<dk);let qv=select(0.0,Q[qbase+ki]*qs,head<nv&&ki<dk);
  let pred0=reduce128(gh*st0*kv,ki,pair);let pred1=reduce128(gh*st1*kv,ki,pair);
  let d0=select(0.0,(V[vbase+svi0]-pred0)*bh,valid0);let d1=select(0.0,(V[vbase+svi1]-pred1)*bh,valid1);
  st0=gh*st0+kv*d0;st1=gh*st1+kv*d1;
  let out0=reduce128(st0*qv,ki,pair);let out1=reduce128(st1*qv,ki,pair);
  if(ki==0u&&valid0){Y[vbase+vi0]=out0;}if(ki==0u&&valid1){Y[vbase+vi1]=out1;}
 }
 if(valid0&&ki<dk){State[idx0]=st0;}if(valid1&&ki<dk){State[idx1]=st1;}
}
)WGSL";

// [ssm] dt_softplus
static const char* WGSL_DT_SOFTPLUS = R"WGSL(
// SSM dt softplus + bias — Mamba delta-time activation
//
// In Mamba's decode step, dt is computed as:
//   dt = softplus(linear(x) + dt_bias)
// where softplus(z) = log(1 + exp(z))
//
// Bindings:
//   0: dt_proj_out [d_inner]  — raw linear projection output
//   1: dt_bias     [d_inner]
//   2: dt_out      [d_inner]
//   3: _params_              — [d_inner]

@group(0) @binding(0) var<storage, read>       proj:    array<f32>;
@group(0) @binding(1) var<storage, read>       bias:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dt_out:  array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let c = gid.x;
    if (c >= d_inner) { return; }
    let z = proj[c] + bias[c];
    // softplus, numerically stable: max(z, 0) + log(1 + exp(-|z|))
    let abs_z = abs(z);
    dt_out[c] = max(z, 0.0) + log(1.0 + exp(-abs_z));
}
)WGSL";

// [ssm] qwen35_alpha_beta_gate
static const char* WGSL_QWEN35_ALPHA_BETA_GATE = R"WGSL(
// qwen35 DeltaNet scalar gates:
//   beta = sigmoid(beta_proj)
//   gate = softplus(alpha_proj + dt_bias) * ssm_a
//
// Bindings:
//   0: beta_proj  [num_v_heads]
//   1: alpha_proj [num_v_heads]
//   2: dt_bias    [num_v_heads]
//   3: ssm_a      [num_v_heads]
//   4: beta_out   [num_v_heads]
//   5: gate_out   [num_v_heads]
//   6: _params_   [num_v_heads, alpha_offset]

@group(0) @binding(0) var<storage, read>       beta_proj:  array<f32>;
@group(0) @binding(1) var<storage, read>       alpha_proj: array<f32>;
@group(0) @binding(2) var<storage, read>       dt_bias:    array<f32>;
@group(0) @binding(3) var<storage, read>       ssm_a:      array<f32>;
@group(0) @binding(4) var<storage, read_write> beta_out:   array<f32>;
@group(0) @binding(5) var<storage, read_write> gate_out:   array<f32>;
@group(0) @binding(6) var<storage, read>       _params_:   array<u32>;

fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = _params_[0];
    let alpha_offset = _params_[1];
    let i = gid.x;
    if (i >= n) { return; }
    let b = beta_proj[i];
    beta_out[i] = 1.0 / (1.0 + exp(-b));
    gate_out[i] = softplus(alpha_proj[alpha_offset + i] + dt_bias[i]) * ssm_a[i];
}
)WGSL";

// [ssm] qwen35_alpha_beta_gate_batched
static const char* WGSL_QWEN35_ALPHA_BETA_GATE_BATCHED = R"WGSL(
// Params [T,rank]. Raw rows contain beta projection followed by alpha.
@group(0) @binding(0) var<storage,read> Raw:array<f32>;
@group(0) @binding(1) var<storage,read> Dt:array<f32>;
@group(0) @binding(2) var<storage,read> A:array<f32>;
@group(0) @binding(3) var<storage,read_write> Beta:array<f32>;
@group(0) @binding(4) var<storage,read_write> Gate:array<f32>;
@group(0) @binding(5) var<storage,read> P:array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){let T=P[0];let R=P[1];let i=gid.x;if(i>=T*R){return;}let t=i/R;let j=i%R;let b=Raw[t*2u*R+j];let a=Raw[t*2u*R+R+j]+Dt[j];Beta[i]=1.0/(1.0+exp(-b));Gate[i]=(max(a,0.0)+log(1.0+exp(-abs(a))))*A[j];}
)WGSL";

// [ssm] qwen35_beta_alpha_gate_q8
static const char* WGSL_QWEN35_BETA_ALPHA_GATE_Q8 = R"WGSL(
// Fused qwen35 DeltaNet beta/alpha projection and scalar gates.
// Computes:
//   beta = sigmoid(X * W_beta^T)
//   gate = softplus(X * W_alpha^T + dt_bias) * ssm_a
//
// W_Q8 rows are packed as beta rows followed by alpha rows.
// Params: [K, rank, 0, 0]

@group(0) @binding(0) var<storage, read>       X_v4:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       W_Q8:      array<u32>;
@group(0) @binding(2) var<storage, read>       Scales:    array<u32>;
@group(0) @binding(3) var<storage, read>       DtBias:    array<f32>;
@group(0) @binding(4) var<storage, read>       SsmA:      array<f32>;
@group(0) @binding(5) var<storage, read_write> BetaOut:   array<f32>;
@group(0) @binding(6) var<storage, read_write> GateOut:   array<f32>;
@group(0) @binding(7) var<storage, read>       _params_:  array<u32>;

const TILE_N: u32 = 8u;
var<workgroup> reduce_scratch: array<f32, 256>;

fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let rank = _params_[1];
    let N = rank * 2u;

    let warp_id = tid / 32u;
    let lane = tid % 32u;
    let col = tile_col * TILE_N + warp_id;

    let n_strides = K / 256u;
    let stride_w = K / 4u;
    var acc: f32 = 0.0;

    if (col < N) {
        let w_base = col * stride_w;
        let n_blocks = K / 32u;
        let s_base = col * n_blocks;

        for (var g = 0u; g < n_strides; g = g + 1u) {
            let k_v4_base = g * 64u + lane * 2u;
            let xv0 = X_v4[k_v4_base];
            let xv1 = X_v4[k_v4_base + 1u];

            let w_off = w_base + g * 64u + lane * 2u;
            let pw0 = W_Q8[w_off];
            let pw1 = W_Q8[w_off + 1u];

            let wv0 = vec4<f32>(f32(extractBits(i32(pw0), 0u, 8u)),
                                f32(extractBits(i32(pw0), 8u, 8u)),
                                f32(extractBits(i32(pw0), 16u, 8u)),
                                f32(extractBits(i32(pw0), 24u, 8u)));
            let wv1 = vec4<f32>(f32(extractBits(i32(pw1), 0u, 8u)),
                                f32(extractBits(i32(pw1), 8u, 8u)),
                                f32(extractBits(i32(pw1), 16u, 8u)),
                                f32(extractBits(i32(pw1), 24u, 8u)));

            let block0 = g * 8u + (lane * 8u) / 32u;
            let block1 = g * 8u + (lane * 8u + 4u) / 32u;
            let sp0 = unpack2x16float(Scales[(s_base + block0) / 2u]);
            let scale0 = select(sp0.x, sp0.y, ((s_base + block0) & 1u) != 0u);
            let sp1 = unpack2x16float(Scales[(s_base + block1) / 2u]);
            let scale1 = select(sp1.x, sp1.y, ((s_base + block1) & 1u) != 0u);

            acc += dot(xv0, wv0) * scale0 + dot(xv1, wv1) * scale1;
        }
    }

    reduce_scratch[tid] = acc;
    workgroupBarrier();
    for (var offset = 16u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) { reduce_scratch[tid] += reduce_scratch[tid + offset]; }
        workgroupBarrier();
    }
    let sum = reduce_scratch[warp_id * 32u];
    if (lane == 0u && col < N) {
        if (col < rank) {
            BetaOut[col] = 1.0 / (1.0 + exp(-sum));
        } else {
            let j = col - rank;
            GateOut[j] = softplus(sum + DtBias[j]) * SsmA[j];
        }
    }
}
)WGSL";

// [ssm] qwen35_conv_scan_silu
static const char* WGSL_QWEN35_CONV_SCAN_SILU = R"WGSL(
// Batched causal depthwise convolution for Qwen 3.5 prefill. One invocation
// owns a channel and scans prompt rows in order, leaving State ready for decode.

@group(0) @binding(0) var<storage, read_write> State: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> P: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channels = P[0]; let convK = P[1]; let T = P[2];
    let c = gid.x;
    if (c >= channels) { return; }
    let base = c * convK;
    for (var t = 0u; t < T; t++) {
        var acc = Bias[c];
        for (var k = 0u; k + 1u < convK; k++) {
            let old = State[base + k + 1u];
            State[base + k] = old;
            acc += W[base + k] * old;
        }
        let newest = X[t * channels + c];
        State[base + convK - 1u] = newest;
        acc += W[base + convK - 1u] * newest;
        Y[t * channels + c] = acc / (1.0 + exp(-acc));
    }
}
)WGSL";

// [ssm] qwen35_conv_scan_split_l2
static const char* WGSL_QWEN35_CONV_SCAN_SPLIT_L2 = R"WGSL(
enable subgroups;

// Fused Qwen 3.5 prefill convolution scan, SiLU, Q/K split+L2 norm.
// Grid: (3, max(nk,nv), 1). Each workgroup scans T in causal order.
@group(0) @binding(0) var<storage,read_write> State:array<f32>;
@group(0) @binding(1) var<storage,read> X:array<f32>;
@group(0) @binding(2) var<storage,read> CW:array<f32>;
@group(0) @binding(3) var<storage,read> Bias:array<f32>;
@group(0) @binding(4) var<storage,read_write> Q:array<f32>;
@group(0) @binding(5) var<storage,read_write> K:array<f32>;
@group(0) @binding(6) var<storage,read_write> V:array<f32>;
@group(0) @binding(7) var<storage,read> P:array<u32>;
var<workgroup>sums:array<f32,4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id)wid:vec3<u32>,@builtin(local_invocation_id)lid:vec3<u32>){
 let nk=P[0];let nv=P[1];let dk=P[2];let dv=P[3];let convK=P[4];let eps=bitcast<f32>(P[5]);let T=P[6];
 let kind=wid.x;let h=wid.y;let d=lid.x;let dim=select(dk,dv,kind==2u);let heads=select(nk,nv,kind==2u);
 if(h>=heads){return;}let qsize=nk*dk;let coff=select(select(0u,qsize,kind==1u),2u*qsize,kind==2u);let channels=2u*qsize+nv*dv;let c=coff+h*dim+d;let sb=c*convK;
 for(var t=0u;t<T;t++){
  var a=Bias[c];for(var j=0u;j+1u<convK;j++){let old=State[sb+j+1u];State[sb+j]=old;a+=CW[sb+j]*old;}
  let newest=X[t*channels+c];State[sb+convK-1u]=newest;a+=CW[sb+convK-1u]*newest;let value=a/(1.0+exp(-a));
  if(kind==2u){V[(t*nv+h)*dv+d]=value;}else{let ss=subgroupAdd(value*value);if((d&31u)==0u){sums[d/32u]=ss;}workgroupBarrier();let inv=1.0/max(sqrt(sums[0]+sums[1]+sums[2]+sums[3]),eps);let out=(t*nk+h)*dk+d;if(kind==0u){Q[out]=value*inv;}else{K[out]=value*inv;}workgroupBarrier();}
 }
}
)WGSL";

// [ssm] qwen35_conv_update_silu
static const char* WGSL_QWEN35_CONV_UPDATE_SILU = R"WGSL(
// qwen35 SSM decode fused rolling-state update + depthwise conv + SiLU.
//
// Equivalent to:
//   conv_state_update(state, x)
//   y = conv1d_decode(state, weights, bias)
//   out = silu(y)
//
// State layout is [channels, K], oldest to newest.
// Bindings:
//   0: state   [channels * K] read_write
//   1: x       [channels] new sample
//   2: weights [channels * K]
//   3: bias    [channels]
//   4: out     [channels]
//   5: params  [channels, K]

@group(0) @binding(0) var<storage, read_write> state:   array<f32>;
@group(0) @binding(1) var<storage, read>       x:       array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read>       bias:    array<f32>;
@group(0) @binding(4) var<storage, read_write> out:     array<f32>;
@group(0) @binding(5) var<storage, read>       params:  array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channels = params[0];
    let K = params[1];
    let c = gid.x;
    if (c >= channels) { return; }

    let base = c * K;
    var acc = bias[c];

    for (var k = 0u; k + 1u < K; k = k + 1u) {
        let v = state[base + k + 1u];
        state[base + k] = v;
        acc = acc + weights[base + k] * v;
    }

    let newest = x[c];
    state[base + K - 1u] = newest;
    acc = acc + weights[base + K - 1u] * newest;

    out[c] = acc / (1.0 + exp(-acc));
}
)WGSL";

// [ssm] qwen35_norm_gated
static const char* WGSL_QWEN35_NORM_GATED = R"WGSL(
// Per-head RMSNorm over DeltaNet output, then gate with silu(z).
//
// Bindings:
//   0: y        [num_v_heads * head_v_dim]
//   1: norm_w   [head_v_dim]
//   2: z        [num_v_heads * head_v_dim]
//   3: out      [num_v_heads * head_v_dim]
//   4: _params_ [num_v_heads, head_v_dim, eps_bits]

@group(0) @binding(0) var<storage, read>       y:       array<f32>;
@group(0) @binding(1) var<storage, read>       norm_w:  array<f32>;
@group(0) @binding(2) var<storage, read>       z:       array<f32>;
@group(0) @binding(3) var<storage, read_write> out:     array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const WG: u32 = 128u;
var<workgroup> sums: array<f32, 128>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn reduce_wg(v: f32, tid: u32) -> f32 {
    sums[tid] = v;
    workgroupBarrier();
    for (var offset = 64u; offset > 0u; offset = offset / 2u) {
        if (tid < offset) { sums[tid] += sums[tid + offset]; }
        workgroupBarrier();
    }
    return sums[0];
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nv = _params_[0];
    let dv = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let h = wid.x;
    let tid = lid.x;
    if (h >= nv) { return; }

    var sum_sq: f32 = 0.0;
    for (var d = tid; d < dv; d = d + WG) {
        let v = y[h * dv + d];
        sum_sq = sum_sq + v * v;
    }
    let total = reduce_wg(sum_sq, tid);
    let scale = 1.0 / sqrt(total / f32(dv) + eps);
    for (var d = tid; d < dv; d = d + WG) {
        let idx = h * dv + d;
        out[idx] = y[idx] * scale * norm_w[d] * silu(z[idx]);
    }
}
)WGSL";

// [ssm] qwen35_norm_gated_batched
static const char* WGSL_QWEN35_NORM_GATED_BATCHED = R"WGSL(
enable subgroups;

// Grid: (num_v_heads, T, 1).
@group(0) @binding(0) var<storage, read> Y: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Z: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
var<workgroup> sums:array<f32,4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
    let nv=P[0];let dv=P[1];let eps=bitcast<f32>(P[2]);let T=P[3];
    let h=wid.x;let t=wid.y;let d=lid.x;if(h>=nv||t>=T){return;}
    let idx=(t*nv+h)*dv+d;let x=select(0.0,Y[idx],d<dv);
    let ss=subgroupAdd(x*x);if((d&31u)==0u){sums[d/32u]=ss;}workgroupBarrier();
    let scale=inverseSqrt((sums[0]+sums[1]+sums[2]+sums[3])/f32(dv)+eps);
    if(d<dv){let z=Z[idx];O[idx]=x*scale*W[d]*(z/(1.0+exp(-z)));}
}
)WGSL";

// [ssm] qwen35_split_qkv_l2
static const char* WGSL_QWEN35_SPLIT_QKV_L2 = R"WGSL(
// Split qwen35 convolved Q/K/V and L2-normalize Q and K per head.
//
// conv_out layout:
//   Q [num_k_heads * head_dim]
//   K [num_k_heads * head_dim]
//   V [num_v_heads * head_v_dim]
//
// Bindings:
//   0: conv_out [conv_channels]
//   1: q_out    [num_k_heads * head_dim]
//   2: k_out    [num_k_heads * head_dim]
//   3: v_out    [num_v_heads * head_v_dim]
//   4: _params_ [num_k_heads, num_v_heads, head_dim, head_v_dim, eps_bits]

@group(0) @binding(0) var<storage, read>       conv_out: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_out:    array<f32>;
@group(0) @binding(2) var<storage, read_write> k_out:    array<f32>;
@group(0) @binding(3) var<storage, read_write> v_out:    array<f32>;
@group(0) @binding(4) var<storage, read>       _params_: array<u32>;

const WG: u32 = 128u;
var<workgroup> sums: array<f32, 128>;

fn reduce_wg(v: f32, tid: u32) -> f32 {
    sums[tid] = v;
    workgroupBarrier();
    for (var offset = 64u; offset > 0u; offset = offset / 2u) {
        if (tid < offset) { sums[tid] += sums[tid + offset]; }
        workgroupBarrier();
    }
    return sums[0];
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nk = _params_[0];
    let nv = _params_[1];
    let dk = _params_[2];
    let dv = _params_[3];
    let eps = bitcast<f32>(_params_[4]);
    let tid = lid.x;

    let q_size = nk * dk;
    let k_base = q_size;
    let v_base = q_size * 2u;

    if (wid.x == 0u) {
        let h = wid.y;
        if (h >= nk) { return; }
        var sum_sq: f32 = 0.0;
        for (var d = tid; d < dk; d = d + WG) {
            let v = conv_out[h * dk + d];
            sum_sq = sum_sq + v * v;
        }
        let total = reduce_wg(sum_sq, tid);
        let rms = 1.0 / max(sqrt(total), eps);
        for (var d = tid; d < dk; d = d + WG) {
            q_out[h * dk + d] = conv_out[h * dk + d] * rms;
        }
    } else if (wid.x == 1u) {
        let h = wid.y;
        if (h >= nk) { return; }
        var sum_sq: f32 = 0.0;
        for (var d = tid; d < dk; d = d + WG) {
            let v = conv_out[k_base + h * dk + d];
            sum_sq = sum_sq + v * v;
        }
        let total = reduce_wg(sum_sq, tid);
        let rms = 1.0 / max(sqrt(total), eps);
        for (var d = tid; d < dk; d = d + WG) {
            k_out[h * dk + d] = conv_out[k_base + h * dk + d] * rms;
        }
    } else {
        let h = wid.y;
        if (h >= nv) { return; }
        for (var d = tid; d < dv; d = d + WG) {
            v_out[h * dv + d] = conv_out[v_base + h * dv + d];
        }
    }
}
)WGSL";

// [ssm] qwen35_split_qkv_l2_batched
static const char* WGSL_QWEN35_SPLIT_QKV_L2_BATCHED = R"WGSL(
enable subgroups;

// Batched split and per-head L2 normalization. Grid: (3, max(nk,nv), T).
@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nk=P[0]; let nv=P[1]; let dk=P[2]; let dv=P[3];
    let eps=bitcast<f32>(P[4]); let T=P[5];
    let kind=wid.x; let h=wid.y; let t=wid.z; let d=lid.x;
    if (t >= T) { return; }
    let qsize=nk*dk; let channels=2u*qsize+nv*dv;
    if (kind == 2u) {
        if (h >= nv) { return; }
        if (d < dv) { V[(t*nv+h)*dv+d] = C[t*channels+2u*qsize+h*dv+d]; }
        return;
    }
    if (h >= nk) { return; }
    let src=t*channels+kind*qsize+h*dk;
    let x=select(0.0,C[src+d],d<dk);
    let ss=subgroupAdd(x*x);
    if ((d&31u)==0u) { sums[d/32u]=ss; }
    workgroupBarrier();
    let inv=1.0/max(sqrt(sums[0]+sums[1]+sums[2]+sums[3]),eps);
    if (d < dk) {
        let dst=(t*nk+h)*dk+d;
        if (kind==0u) { Q[dst]=x*inv; } else { K[dst]=x*inv; }
    }
}
)WGSL";

// [ssm] selective_scan_decode
static const char* WGSL_SELECTIVE_SCAN_DECODE = R"WGSL(
// SSM selective scan — decode step
//
// Standard Mamba recurrence for a single decode token:
//   per (c in 0..d_inner, s in 0..d_state):
//     a   = exp(A_log[c, s]) * (-1)         // A_log is the stored log of -A
//     da  = dt[c] * a
//     decay = exp(da)
//     ddh = (decay - 1) / a * B[s] * x[c]
//     h[c, s] = decay * h[c, s] + ddh
//   per c:
//     y[c] = sum_s C[s] * h[c, s] + D[c] * x[c]
//
// NOTE: qwen35moe uses alpha/beta projections in place of standard B/C —
// host code needs to wire the right buffers. This kernel computes the
// recurrence assuming caller has provided the per-step B/C and dt.
//
// Bindings:
//   0: x        [d_inner]                  — input projection at this step
//   1: A_log    [d_inner * d_state]        — log of -A (negative-real init)
//   2: dt       [d_inner]                  — softplus(dt_proj(x) + dt_bias)
//   3: B        [d_state]                  — per-step B projection
//   4: C        [d_state]                  — per-step C projection
//   5: D        [d_inner]                  — skip-conn scale (may be zero buf)
//   6: h_state  [d_inner * d_state]        — recurrent state (read+update)
//   7: y_out    [d_inner]                  — output (write)
//   8: _params_                            — [d_inner, d_state]

@group(0) @binding(0) var<storage, read>       x_in:    array<f32>;
@group(0) @binding(1) var<storage, read>       A_log:   array<f32>;
@group(0) @binding(2) var<storage, read>       dt:      array<f32>;
@group(0) @binding(3) var<storage, read>       B:       array<f32>;
@group(0) @binding(4) var<storage, read>       C:       array<f32>;
@group(0) @binding(5) var<storage, read>       D:       array<f32>;
@group(0) @binding(6) var<storage, read_write> h_state: array<f32>;
@group(0) @binding(7) var<storage, read_write> y_out:   array<f32>;
@group(0) @binding(8) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let d_state = _params_[1];
    let c = gid.x;
    if (c >= d_inner) { return; }

    let xc = x_in[c];
    let dtc = dt[c];
    let Dc = D[c];

    var acc: f32 = Dc * xc;
    let base = c * d_state;
    for (var s: u32 = 0u; s < d_state; s = s + 1u) {
        let A_neg = -exp(A_log[base + s]);
        let da = dtc * A_neg;
        let decay = exp(da);
        // (decay - 1) / A_neg is numerically stable for small da
        let coef = select((decay - 1.0) / A_neg, dtc, abs(A_neg) < 1.0e-6);
        let ddh = coef * B[s] * xc;
        let h_new = decay * h_state[base + s] + ddh;
        h_state[base + s] = h_new;
        acc = acc + C[s] * h_new;
    }
    y_out[c] = acc;
}
)WGSL";

// ─── Registry ────────────────────────────────────────────────────────────

inline const std::unordered_map<std::string, ShaderInfo>& getEmbeddedKernels() {
    static const std::unordered_map<std::string, ShaderInfo> kernels = {
        {"matmul_q4_zp_batched_dp4a", {WGSL_MATMUL_Q4_ZP_BATCHED_DP4A, 6, false}},
        {"add_inplace_batched", {WGSL_ADD_INPLACE_BATCHED, 3, false}},
        {"add_rms_norm", {WGSL_ADD_RMS_NORM, 6, true}},
        {"add_rms_norm_batched", {WGSL_ADD_RMS_NORM_BATCHED, 6, false}},
        {"add_rms_norm_qwen_vulkan", {WGSL_ADD_RMS_NORM_QWEN_VULKAN, 6, false}},
        {"add_scaled", {WGSL_ADD_SCALED, 3, false}},
        {"argmax", {WGSL_ARGMAX, 4, false}},
        {"argmax_reduce", {WGSL_ARGMAX_REDUCE, 3, false}},
        {"bidirectional_attn", {WGSL_BIDIRECTIONAL_ATTN, 5, false}},
        {"binary_elementwise", {WGSL_BINARY_ELEMENTWISE, 4, false}},
        {"cast_f16_to_f32", {WGSL_CAST_F16_TO_F32, 3, false}},
        {"causal_attn", {WGSL_CAUSAL_ATTN, 5, false}},
        {"compute_offsets", {WGSL_COMPUTE_OFFSETS, 5, false}},
        {"concat_2d", {WGSL_CONCAT_2D, 4, false}},
        {"concat_2input", {WGSL_CONCAT_2INPUT, 4, false}},
        {"conv1d_decode", {WGSL_CONV1D_DECODE, 5, false}},
        {"conv2d", {WGSL_CONV2D, 5, false}},
        {"conv_state_update", {WGSL_CONV_STATE_UPDATE, 3, false}},
        {"conv_transpose2d", {WGSL_CONV_TRANSPOSE2D, 5, false}},
        {"copy_buffer", {WGSL_COPY_BUFFER, 3, false}},
        {"delta_net_decode", {WGSL_DELTA_NET_DECODE, 8, false}},
        {"delta_net_decode_x2", {WGSL_DELTA_NET_DECODE_X2, 8, false}},
        {"delta_net_scan_x2", {WGSL_DELTA_NET_SCAN_X2, 8, false}},
        {"delta_net_scan_x4", {WGSL_DELTA_NET_SCAN_X4, 8, false}},
        {"dt_softplus", {WGSL_DT_SOFTPLUS, 4, false}},
        {"embed_gather", {WGSL_EMBED_GATHER, 4, false}},
        {"equal_op", {WGSL_EQUAL_OP, 4, false}},
        {"expand", {WGSL_EXPAND, 3, false}},
        {"flash_attn_vulkan", {WGSL_FLASH_ATTN_VULKAN, 5, false}},
        {"fp16_gemm", {WGSL_FP16_GEMM, 5, false}},
        {"fp16_gemm_wide", {WGSL_FP16_GEMM_WIDE, 5, false}},
        {"fused_qknorm_rope", {WGSL_FUSED_QKNORM_ROPE, 9, true}},
        {"fused_qknorm_rope_batched", {WGSL_FUSED_QKNORM_ROPE_BATCHED, 9, false}},
        {"fused_rope", {WGSL_FUSED_ROPE, 9, true}},
        {"gate_residual_add", {WGSL_GATE_RESIDUAL_ADD, 4, false}},
        {"gated_output", {WGSL_GATED_OUTPUT, 4, false}},
        {"gated_output_batched", {WGSL_GATED_OUTPUT_BATCHED, 4, false}},
        {"gather", {WGSL_GATHER, 4, false}},
        {"gather_bq_q4", {WGSL_GATHER_BQ_Q4, 5, false}},
        {"gather_bq_q4_zp", {WGSL_GATHER_BQ_Q4_ZP, 6, false}},
        {"gather_bq_q8", {WGSL_GATHER_BQ_Q8, 5, false}},
        {"gather_elements", {WGSL_GATHER_ELEMENTS, 4, false}},
        {"gelu_mul", {WGSL_GELU_MUL, 4, false}},
        {"gelu_mul_batched", {WGSL_GELU_MUL_BATCHED, 3, false}},
        {"gelu_mul_fused", {WGSL_GELU_MUL_FUSED, 3, false}},
        {"gemm", {WGSL_GEMM, 5, false}},
        {"gemm_fp16_packed", {WGSL_GEMM_FP16_PACKED, 5, false}},
        {"gemm_fp16_packed_nosub", {WGSL_GEMM_FP16_PACKED_NOSUB, 5, false}},
        {"gemma_norm_add_batched", {WGSL_GEMMA_NORM_ADD_BATCHED, 5, false}},
        {"gemma_rope_batched", {WGSL_GEMMA_ROPE_BATCHED, 9, false}},
        {"gemma_sandwich_attn_batched", {WGSL_GEMMA_SANDWICH_ATTN_BATCHED, 7, false}},
        {"gptoss_gate", {WGSL_GPTOSS_GATE, 3, false}},
        {"gptoss_gate_batched", {WGSL_GPTOSS_GATE_BATCHED, 3, false}},
        {"gqa_chunked_pass1", {WGSL_GQA_CHUNKED_PASS1, 5, false}},
        {"gqa_chunked_pass2", {WGSL_GQA_CHUNKED_PASS2, 3, false}},
        {"gqa_decode", {WGSL_GQA_DECODE, 5, false}},
        {"gqa_decode_attn_sink", {WGSL_GQA_DECODE_ATTN_SINK, 6, false}},
        {"gqa_fused_attn", {WGSL_GQA_FUSED_ATTN, 5, false}},
        {"gqa_prefill", {WGSL_GQA_PREFILL, 5, false}},
        {"group_norm", {WGSL_GROUP_NORM, 5, false}},
        {"head_rmsnorm", {WGSL_HEAD_RMSNORM, 3, false}},
        {"head_rmsnorm_batched", {WGSL_HEAD_RMSNORM_BATCHED, 3, false}},
        {"instance_norm", {WGSL_INSTANCE_NORM, 5, false}},
        {"iq2s_matmul", {WGSL_IQ2S_MATMUL, 6, false}},
        {"iq2s_matmul_moe", {WGSL_IQ2S_MATMUL_MOE, 7, false}},
        {"iq3s_matmul", {WGSL_IQ3S_MATMUL, 6, false}},
        {"iq3s_matmul_moe", {WGSL_IQ3S_MATMUL_MOE, 7, false}},
        {"iq4xs_matmul", {WGSL_IQ4XS_MATMUL, 5, false}},
        {"iq4xs_matmul_moe", {WGSL_IQ4XS_MATMUL_MOE, 6, false}},
        {"kv_cache_append", {WGSL_KV_CACHE_APPEND, 4, false}},
        {"kv_cache_write", {WGSL_KV_CACHE_WRITE, 3, false}},
        {"kv_cache_write_batched", {WGSL_KV_CACHE_WRITE_BATCHED, 3, false}},
        {"layer_norm", {WGSL_LAYER_NORM, 5, false}},
        {"logit_softcap", {WGSL_LOGIT_SOFTCAP, 2, false}},
        {"matmul_f32", {WGSL_MATMUL, 4, false}},
        {"matmul_fp16_packed", {WGSL_MATMUL_FP16_PACKED, 5, false}},
        {"matmul_fp16_packed_nt", {WGSL_MATMUL_FP16_PACKED_NT, 4, false}},
        {"matmul_q4", {WGSL_MATMUL_Q4, 5, false}},
        {"matmul_q4_batched", {WGSL_MATMUL_Q4_BATCHED, 5, false}},
        {"matmul_q4_decode", {WGSL_MATMUL_Q4_DECODE, 5, false}},
        {"matmul_q4_decode_norm", {WGSL_MATMUL_Q4_DECODE_NORM, 7, false}},
        {"matmul_q4_decode_norm_wide", {WGSL_MATMUL_Q4_DECODE_NORM_WIDE, 7, false}},
        {"matmul_q4_decode_wide", {WGSL_MATMUL_Q4_DECODE_WIDE, 5, false}},
        {"matmul_q4_indirect", {WGSL_MATMUL_Q4_INDIRECT, 6, false}},
        {"matmul_q4_indirect_sub", {WGSL_MATMUL_Q4_INDIRECT_SUB, 6, false}},
        {"matmul_q4_indirect_sub_batched", {WGSL_MATMUL_Q4_INDIRECT_SUB_BATCHED, 6, false}},
        {"matmul_q4_indirect_wide_batched", {WGSL_MATMUL_Q4_INDIRECT_WIDE_BATCHED, 6, false}},
        {"matmul_q4_zp", {WGSL_MATMUL_Q4_ZP, 6, false}},
        {"matmul_q4_zp_sub", {WGSL_MATMUL_Q4_ZP_SUB, 6, false}},
        {"matmul_q4_zp_sub_prefill", {WGSL_MATMUL_Q4_ZP_SUB_PREFILL, 6, false}},
        {"matmul_q4_zp_wide", {WGSL_MATMUL_Q4_ZP_WIDE, 6, false}},
        {"matmul_q8_block32_subgroup", {WGSL_MATMUL_Q8_BLOCK32_SUBGROUP, 5, false}},
        {"mod_scale_shift", {WGSL_MOD_SCALE_SHIFT, 5, false}},
        {"moe_gate", {WGSL_MOE_GATE, 4, false}},
        {"moe_gate_batched", {WGSL_MOE_GATE_BATCHED, 4, false}},
        {"mul", {WGSL_MUL, 4, false}},
        {"mxfp4_matmul", {WGSL_MXFP4_MATMUL, 6, false}},
        {"norm_then_add", {WGSL_NORM_THEN_ADD, 4, false}},
        {"ple_combine", {WGSL_PLE_COMBINE, 3, false}},
        {"ple_combine_batched", {WGSL_PLE_COMBINE_BATCHED, 4, false}},
        {"ple_gelu_mul", {WGSL_PLE_GELU_MUL, 3, false}},
        {"ple_gelu_mul_batched", {WGSL_PLE_GELU_MUL_BATCHED, 3, false}},
        {"ple_norm_add_scale", {WGSL_PLE_NORM_ADD_SCALE, 4, false}},
        {"ple_slice_rms_norm", {WGSL_PLE_SLICE_RMS_NORM, 5, false}},
        {"q2k_matmul", {WGSL_Q2K_MATMUL, 5, false}},
        {"q3k_matmul", {WGSL_Q3K_MATMUL, 5, false}},
        {"q4_add_norm_decode", {WGSL_Q4_ADD_NORM_DECODE, 8, false}},
        {"q4_down_gelu_add_decode", {WGSL_Q4_DOWN_GELU_ADD_DECODE, 6, false}},
        {"q4_down_silu_add_decode", {WGSL_Q4_DOWN_SILU_ADD_DECODE, 6, false}},
        {"q4_gather_batched", {WGSL_Q4_GATHER_BATCHED, 5, false}},
        {"q4k_down_gelu", {WGSL_Q4K_DOWN_GELU, 5, false}},
        {"q4k_matmul", {WGSL_Q4K_MATMUL, 5, false}},
        {"q4k_matmul_prequant_dp4a", {WGSL_Q4K_MATMUL_PREQUANT_DP4A, 6, false}},
        {"q4k_matmul_prequant_dp4a_reduc16", {WGSL_Q4K_MATMUL_PREQUANT_DP4A_REDUC16, 6, false}},
        {"q4k_matmul_dp4a", {WGSL_Q4K_MATMUL_DP4A, 5, false}},
        {"q8_quantize_dp4a", {WGSL_Q8_QUANTIZE_DP4A, 4, false}},
        {"q8_quantize_batched_dp4a", {WGSL_Q8_QUANTIZE_BATCHED_DP4A, 4, false}},
        {"q4k_matmul_prequant_batched_dp4a", {WGSL_Q4K_MATMUL_PREQUANT_BATCHED_DP4A, 6, false}},
        {"q4k_matmul_128", {WGSL_Q4K_MATMUL_128, 5, false}},
        {"q4k_matmul_batched4", {WGSL_Q4K_MATMUL_BATCHED4, 5, false}},
        {"q4k_matmul_batched8", {WGSL_Q4K_MATMUL_BATCHED8, 5, false}},
        {"q4k_matmul_norm", {WGSL_Q4K_MATMUL_NORM, 7, false}},
        {"q5k_matmul", {WGSL_Q5K_MATMUL, 5, false}},
        {"q5k_matmul_batched4", {WGSL_Q5K_MATMUL_BATCHED4, 5, false}},
        {"q5k_matmul_prequant_batched_dp4a", {WGSL_Q5K_MATMUL_PREQUANT_BATCHED_DP4A, 6, false}},
        {"q5k_matmul_norm", {WGSL_Q5K_MATMUL_NORM, 7, false}},
        {"q6k_down_gelu", {WGSL_Q6K_DOWN_GELU, 5, false}},
        {"q6k_gather", {WGSL_Q6K_GATHER, 4, false}},
        {"q6k_gather_batched", {WGSL_Q6K_GATHER_BATCHED, 4, false}},
        {"q6k_matmul", {WGSL_Q6K_MATMUL, 5, false}},
        {"q6k_matmul_prequant_dp4a", {WGSL_Q6K_MATMUL_PREQUANT_DP4A, 6, false}},
        {"q6k_matmul_prequant_dp4a_reduc16", {WGSL_Q6K_MATMUL_PREQUANT_DP4A_REDUC16, 6, false}},
        {"q6k_matmul_batched4", {WGSL_Q6K_MATMUL_BATCHED4, 5, false}},
        {"q6k_matmul_norm", {WGSL_Q6K_MATMUL_NORM, 7, false}},
        {"q6k_matmul_wide", {WGSL_Q6K_MATMUL_WIDE, 5, false}},
        {"q8_down_silu_add", {WGSL_Q8_DOWN_SILU_ADD, 6, false}},
        {"q8_down_silu_add_batched", {WGSL_Q8_DOWN_SILU_ADD_BATCHED, 6, false}},
        {"q8_down_silu_add_d3d12", {WGSL_Q8_DOWN_SILU_ADD_D3D12, 6, false}},
        {"q8_down_silu_add_dp4a_d3d12", {WGSL_Q8_DOWN_SILU_ADD_DP4A_D3D12, 6, false}},
        {"q8_down_silu_add_tiled", {WGSL_Q8_DOWN_SILU_ADD_TILED, 6, false}},
        {"q8_down_silu_add_vulkan", {WGSL_Q8_DOWN_SILU_ADD_VULKAN, 6, false}},
        {"q8_gather_batched", {WGSL_Q8_GATHER_BATCHED, 5, false}},
        {"q8_matmul", {WGSL_Q8_MATMUL, 6, false}},
        {"q8_matmul_add", {WGSL_Q8_MATMUL_ADD, 6, false}},
        {"q8_matmul_add_batched", {WGSL_Q8_MATMUL_ADD_BATCHED, 6, false}},
        {"q8_matmul_add_fast", {WGSL_Q8_MATMUL_ADD_FAST, 6, false}},
        {"q8_matmul_add_lite", {WGSL_Q8_MATMUL_ADD_LITE, 6, false}},
        {"q8_matmul_add_norm", {WGSL_Q8_MATMUL_ADD_NORM, 8, false}},
        {"q8_matmul_add_smem", {WGSL_Q8_MATMUL_ADD_SMEM, 6, false}},
        {"q8_matmul_batched", {WGSL_Q8_MATMUL_BATCHED, 6, false}},
        {"q8_matmul_batched_dp4a", {WGSL_Q8_MATMUL_BATCHED_DP4A, 6, false}},
        {"q8_matmul_d3d12", {WGSL_Q8_MATMUL_D3D12, 6, false}},
        {"q8_matmul_decode_dp4a_d3d12", {WGSL_Q8_MATMUL_DECODE_DP4A_D3D12, 6, false}},
        {"q8_matmul_dp4a", {WGSL_Q8_MATMUL_DP4A, 6, false}},
        {"q8_matmul_dp4a_d3d12", {WGSL_Q8_MATMUL_DP4A_D3D12, 6, false}},
        {"q8_matmul_fast", {WGSL_Q8_MATMUL_FAST, 6, false}},
        {"q8_matmul_lite", {WGSL_Q8_MATMUL_LITE, 6, false}},
        {"q8_matmul_norm", {WGSL_Q8_MATMUL_NORM, 7, false}},
        {"q8_matmul_prequant_add_d3d12", {WGSL_Q8_MATMUL_PREQUANT_ADD_D3D12, 7, false}},
        {"q8_matmul_prequant_add_wide_d3d12", {WGSL_Q8_MATMUL_PREQUANT_ADD_WIDE_D3D12, 7, false}},
        {"q8_matmul_prequant_d3d12", {WGSL_Q8_MATMUL_PREQUANT_D3D12, 7, false}},
        {"q8_matmul_prequant_wide_d3d12", {WGSL_Q8_MATMUL_PREQUANT_WIDE_D3D12, 7, false}},
        {"q8_matmul_smem", {WGSL_Q8_MATMUL_SMEM, 6, false}},
        {"q8_matmul_tiled", {WGSL_Q8_MATMUL_TILED, 6, false}},
        {"q8_matmul_vec4", {WGSL_Q8_MATMUL_VEC4, 6, false}},
        {"q8_matmul_vulkan", {WGSL_Q8_MATMUL_VULKAN, 6, false}},
        {"qmoe_matmul_q4", {WGSL_QMOE_MATMUL_Q4, 6, false}},
        {"quantize_fp32_rows_d3d12", {WGSL_QUANTIZE_FP32_ROWS_D3D12, 4, false}},
        {"qwen35_alpha_beta_gate", {WGSL_QWEN35_ALPHA_BETA_GATE, 7, false}},
        {"qwen35_alpha_beta_gate_batched", {WGSL_QWEN35_ALPHA_BETA_GATE_BATCHED, 6, false}},
        {"qwen35_beta_alpha_gate_q8", {WGSL_QWEN35_BETA_ALPHA_GATE_Q8, 8, false}},
        {"qwen35_conv_scan_silu", {WGSL_QWEN35_CONV_SCAN_SILU, 6, false}},
        {"qwen35_conv_scan_split_l2", {WGSL_QWEN35_CONV_SCAN_SPLIT_L2, 8, false}},
        {"qwen35_conv_update_silu", {WGSL_QWEN35_CONV_UPDATE_SILU, 6, false}},
        {"qwen35_kv_cache_write", {WGSL_QWEN35_KV_CACHE_WRITE, 5, false}},
        {"qwen35_kv_cache_write_rope", {WGSL_QWEN35_KV_CACHE_WRITE_ROPE, 8, false}},
        {"qwen35_norm_gated", {WGSL_QWEN35_NORM_GATED, 5, false}},
        {"qwen35_norm_gated_batched", {WGSL_QWEN35_NORM_GATED_BATCHED, 5, false}},
        {"qwen35_rope_kv_batched", {WGSL_QWEN35_ROPE_KV_BATCHED, 9, false}},
        {"qwen35_rope_multi_partial", {WGSL_QWEN35_ROPE_MULTI_PARTIAL, 4, false}},
        {"qwen35_rope_q_to_qrot", {WGSL_QWEN35_ROPE_Q_TO_QROT, 5, false}},
        {"qwen35_split_qg_batched", {WGSL_QWEN35_SPLIT_QG_BATCHED, 4, false}},
        {"qwen35_split_qkv_l2", {WGSL_QWEN35_SPLIT_QKV_L2, 5, false}},
        {"qwen35_split_qkv_l2_batched", {WGSL_QWEN35_SPLIT_QKV_L2_BATCHED, 5, false}},
        {"resize_nearest", {WGSL_RESIZE_NEAREST, 3, false}},
        {"rms_norm", {WGSL_RMS_NORM, 5, true}},
        {"rms_norm_batched", {WGSL_RMS_NORM_BATCHED, 5, false}},
        {"rms_norm_qwen_vulkan", {WGSL_RMS_NORM_QWEN_VULKAN, 5, false}},
        {"rmsnorm", {WGSL_RMSNORM, 5, false}},
        {"rope_batched", {WGSL_ROPE_BATCHED, 9, false}},
        {"rope_batched_simple", {WGSL_ROPE_BATCHED_SIMPLE, 9, false}},
        {"rope_inplace", {WGSL_ROPE_INPLACE, 4, false}},
        {"rope_inplace_batched", {WGSL_ROPE_INPLACE_BATCHED, 4, false}},
        {"rope_multi", {WGSL_ROPE_MULTI, 3, false}},
        {"rope_multi_partial", {WGSL_ROPE_MULTI_PARTIAL, 3, false}},
        {"rotary_embedding", {WGSL_ROTARY_EMBEDDING, 6, false}},
        {"sandwich_norm_add_norm", {WGSL_SANDWICH_NORM_ADD_NORM, 6, false}},
        {"scale", {WGSL_SCALE, 2, false}},
        {"scale_by_buffer", {WGSL_SCALE_BY_BUFFER, 3, false}},
        {"scatter_elements", {WGSL_SCATTER_ELEMENTS, 5, false}},
        {"selective_scan_decode", {WGSL_SELECTIVE_SCAN_DECODE, 9, false}},
        {"sigmoid_gate_interleaved", {WGSL_SIGMOID_GATE_INTERLEAVED, 3, false}},
        {"silu", {WGSL_SILU, 3, false}},
        {"silu_mul_batched", {WGSL_SILU_MUL_BATCHED, 3, false}},
        {"silu_mul_fused", {WGSL_SILU_MUL_FUSED, 3, true}},
        {"silu_quantize_rows_d3d12", {WGSL_SILU_QUANTIZE_ROWS_D3D12, 4, false}},
        {"skip_rmsnorm", {WGSL_SKIP_RMSNORM, 6, false}},
        {"skip_rmsnorm_f16w_serial", {WGSL_SKIP_RMSNORM_F16W_SERIAL, 6, false}},
        {"slice", {WGSL_SLICE, 3, false}},
        {"softmax", {WGSL_SOFTMAX, 3, false}},
        {"split_copy", {WGSL_SPLIT_COPY, 3, false}},
        {"split_qg", {WGSL_SPLIT_QG, 4, false}},
        {"swiglu", {WGSL_SWIGLU, 3, false}},
        {"swiglu_batched", {WGSL_SWIGLU_BATCHED, 3, false}},
        {"topk", {WGSL_TOPK, 4, false}},
        {"topk_softmax", {WGSL_TOPK_SOFTMAX, 4, false}},
        {"transpose", {WGSL_TRANSPOSE, 3, false}},
        {"unary_elementwise", {WGSL_UNARY_ELEMENTWISE, 3, false}},
        {"weighted_accumulate_decode", {WGSL_WEIGHTED_ACCUMULATE_DECODE, 4, false}},
        {"weighted_add", {WGSL_WEIGHTED_ADD, 3, false}},
        {"weighted_add_indirect", {WGSL_WEIGHTED_ADD_INDIRECT, 4, false}},
        {"weighted_add_indirect_batched", {WGSL_WEIGHTED_ADD_INDIRECT_BATCHED, 4, false}},
        {"where_select", {WGSL_WHERE_SELECT, 5, false}},
        {"zero_init", {WGSL_ZERO_INIT, 2, false}},
    };
    return kernels;
}

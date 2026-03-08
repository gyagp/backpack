enable f16;
enable subgroups;

// Multi-query causal attention with fp16 KV cache + early exit.
// 4 query positions per workgroup. Each warp independently processes
// one query position. fp16 KV cache halves bandwidth.
// Uniform params enable data-dependent loop bound for early exit.
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
    pad1: u32,
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

    let q_base = q_idx * n_head_total * HD + head * HD;
    let q0 = Q[q_base + lane * HD_PER_THREAD];
    let q1 = Q[q_base + lane * HD_PER_THREAD + 1u];
    let q2 = Q[q_base + lane * HD_PER_THREAD + 2u];
    let q3 = Q[q_base + lane * HD_PER_THREAD + 3u];

    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var m_prev: f32 = neg_inf;
    var l_prev: f32 = 0.0;

    // Early exit: max causal position across all 4 warps in this WG.
    // Uniform because params is var<uniform> and q_block = wid.y (uniform).
    let max_causal = min(cache_offset + q_block * QUERIES_PER_WG + QUERIES_PER_WG, T_total);

    for (var t = 0u; t < max_causal; t = t + 1u) {
        let causal_valid = t <= q_abs_pos;

        let k_base = t * kv_stride + kv_off;
        let k_off = k_base + lane * HD_PER_THREAD;
        let k0 = select(0.0, f32(K_cache[k_off]), causal_valid);
        let k1 = select(0.0, f32(K_cache[k_off + 1u]), causal_valid);
        let k2 = select(0.0, f32(K_cache[k_off + 2u]), causal_valid);
        let k3 = select(0.0, f32(K_cache[k_off + 3u]), causal_valid);

        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let dot_qk = subgroupAdd(partial);
        let score = select(neg_inf, dot_qk * scale, causal_valid);

        let m_new = max(m_prev, score);
        let exp_prev = exp(m_prev - m_new);
        let exp_score = exp(score - m_new);
        let l_new = l_prev * exp_prev + exp_score;
        let rescale = l_prev * exp_prev / max(l_new, 1e-10);
        let w = exp_score / max(l_new, 1e-10);

        let v_off = k_base + lane * HD_PER_THREAD;
        let v0 = select(0.0, f32(V_cache[v_off]), causal_valid);
        let v1 = select(0.0, f32(V_cache[v_off + 1u]), causal_valid);
        let v2 = select(0.0, f32(V_cache[v_off + 2u]), causal_valid);
        let v3 = select(0.0, f32(V_cache[v_off + 3u]), causal_valid);

        acc0 = acc0 * rescale + v0 * w;
        acc1 = acc1 * rescale + v1 * w;
        acc2 = acc2 * rescale + v2 * w;
        acc3 = acc3 * rescale + v3 * w;

        m_prev = m_new;
        l_prev = l_new;
    }

    let out_base = q_idx * n_head_total * HD + head * HD;
    Out[out_base + lane * HD_PER_THREAD]      = acc0;
    Out[out_base + lane * HD_PER_THREAD + 1u] = acc1;
    Out[out_base + lane * HD_PER_THREAD + 2u] = acc2;
    Out[out_base + lane * HD_PER_THREAD + 3u] = acc3;
}

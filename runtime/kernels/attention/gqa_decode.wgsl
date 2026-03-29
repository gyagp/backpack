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

// Bidirectional multi-head attention (no causal mask)
// For DiT transformer and VAE self-attention.
// Uses online softmax (single-pass, no shared memory barriers).
//
// Q/K/V layout: [batch, seq_len, num_heads * head_dim]
// Each thread handles one output element (token, head, dim).
//
// Dispatch: (ceil(head_dim/128), num_heads, T_q)

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
            score += Q[q_base + dd] * K[k_base + dd];
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
        acc = acc * rescale + w * V[k_base + d];

        m_prev = m_new;
        l_prev = l_new;
    }

    Out[q_base + d] = acc;
}

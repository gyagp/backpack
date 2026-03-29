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

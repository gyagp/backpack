enable subgroups;
diagnostic(off, subgroup_uniformity);

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
                let v = Partials[base + 2u + lane * HD_PER_THREAD + e];
                acc[e] = acc[e] * rescale + v * w;
            }

            m_prev = m_new;
            l_prev = l_new;
        }
    }

    for (var e = 0u; e < HD_PER_THREAD; e++) {
        Out[head * HD + lane * HD_PER_THREAD + e] = acc[e];
    }
}

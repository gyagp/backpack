// @meta bindings=8
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

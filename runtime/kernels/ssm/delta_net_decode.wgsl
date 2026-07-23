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

    // Q/K heads are repeated in consecutive groups across value heads.
    let k_head_idx = head / max(1u, nv / nk);

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

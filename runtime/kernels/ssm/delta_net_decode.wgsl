// DeltaNet decode step — matrix-state recurrence for qwen35moe linear attention
//
// Per-head state is a matrix [head_k_dim, head_v_dim]. Per decode step:
//   state[h] = state[h] * decay[h]  +  beta[h] * outer(k[h], v[h])
//   y[h]     = q[h] @ state[h]                   // [head_k] · [head_k, head_v] = [head_v]
//
// Heads layout: num_v_heads outputs; K is shared in groups (num_k_heads <= num_v_heads).
// For qwen35moe: num_v_heads=32, num_k_heads=16, head_k_dim=128, head_v_dim=128.
//
// Bindings:
//   0: q       [num_v_heads * head_k_dim]      f32
//   1: k       [num_k_heads * head_k_dim]      f32  (shared groups of v_heads / k_heads)
//   2: v       [num_v_heads * head_v_dim]      f32
//   3: beta    [num_v_heads]                   f32
//   4: decay   [num_v_heads]                   f32  (already exp'd by host)
//   5: state   [num_v_heads * head_k_dim * head_v_dim]  f32  (read+write)
//   6: y       [num_v_heads * head_v_dim]      f32  (write)
//   7: _params_                                — [num_v_heads, num_k_heads, head_k_dim, head_v_dim]
//
// Workgroup: one workgroup per (head, v_dim_chunk). Threads cooperate on k_dim reduction.

@group(0) @binding(0) var<storage, read>       q:        array<f32>;
@group(0) @binding(1) var<storage, read>       k:        array<f32>;
@group(0) @binding(2) var<storage, read>       v:        array<f32>;
@group(0) @binding(3) var<storage, read>       beta:     array<f32>;
@group(0) @binding(4) var<storage, read>       decay:    array<f32>;
@group(0) @binding(5) var<storage, read_write> state:    array<f32>;
@group(0) @binding(6) var<storage, read_write> y:        array<f32>;
@group(0) @binding(7) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nv      = _params_[0];  // num_v_heads
    let nk      = _params_[1];  // num_k_heads
    let dk      = _params_[2];  // head_k_dim
    let dv      = _params_[3];  // head_v_dim
    let head    = wid.x;        // v-head index
    if (head >= nv) { return; }
    let tid     = lid.x;

    let k_head_idx = (head * nk) / nv;  // mapping v-head → k-head group

    let q_base = head * dk;
    let k_base = k_head_idx * dk;
    let v_base = head * dv;
    let state_base = head * dk * dv;

    let bh = beta[head];
    let dh = decay[head];

    // Phase 1: update state. Each thread handles one row of state (one k_dim index).
    // state[h, ki, vi] = state[h, ki, vi] * decay[h] + beta[h] * k[h, ki] * v[h, vi]
    // Workgroup_size=128 = dk for qwen35moe; iterate over dv inside each thread.
    let ki = tid;
    if (ki < dk) {
        let k_ki = k[k_base + ki];
        let bk = bh * k_ki;
        for (var vi: u32 = 0u; vi < dv; vi = vi + 1u) {
            let idx = state_base + ki * dv + vi;
            state[idx] = state[idx] * dh + bk * v[v_base + vi];
        }
    }
    workgroupBarrier();

    // Phase 2: y[h, vi] = sum over ki of q[h, ki] * state[h, ki, vi]
    // Re-use threads: each thread handles one vi (dv = dk = 128, so same threads).
    let vi = tid;
    if (vi < dv) {
        var acc: f32 = 0.0;
        for (var ki2: u32 = 0u; ki2 < dk; ki2 = ki2 + 1u) {
            acc = acc + q[q_base + ki2] * state[state_base + ki2 * dv + vi];
        }
        y[v_base + vi] = acc;
    }
}

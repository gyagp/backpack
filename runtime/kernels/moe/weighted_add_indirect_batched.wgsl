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

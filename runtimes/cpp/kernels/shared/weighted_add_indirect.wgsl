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

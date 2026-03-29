// Batched MoE gate — select top-k experts for T tokens in parallel.
// Input: router[T * num_experts] f32 (token-major)
// Output: expert_indices[T * k] u32, expert_weights[T * k] f32
// Params: [0]=num_experts, [1]=k, [2]=normalize, [3]=nTokens
// Dispatch: (1, nTokens, 1)

@group(0) @binding(0) var<storage, read> router: array<f32>;
@group(0) @binding(1) var<storage, read_write> expert_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let num_experts = _params_[0];
    let k = _params_[1];
    let normalize = _params_[2];
    let tok = wid.y;

    let r_base = tok * num_experts;
    let o_base = tok * k;

    var count = 0u;
    for (var e = 0u; e < num_experts; e++) {
        let v = router[r_base + e];
        if (v > -60000.0 && count < k) {
            expert_indices[o_base + count] = e;
            expert_weights[o_base + count] = exp(v);
            count++;
        }
    }
    for (var i = count; i < k; i++) {
        expert_indices[o_base + i] = 0u;
        expert_weights[o_base + i] = 0.0;
    }
    if (normalize != 0u && count > 0u) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < count; i++) { sum += expert_weights[o_base + i]; }
        if (sum > 0.0) {
            for (var i = 0u; i < count; i++) { expert_weights[o_base + i] /= sum; }
        }
    }
}

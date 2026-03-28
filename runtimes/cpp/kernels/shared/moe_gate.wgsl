// MoE gate — GPU expert selection for MoE.
// Scans router weights, finds active experts (value > -60000), applies exp+normalize.
// Input: router[num_experts] (ln(sigmoid) for active, -65504 for inactive)
// Output: expert_indices[k] u32, expert_weights[k] f32
// Params: [0]=num_experts, [1]=k, [2]=normalize
// Dispatch: (1, 1, 1)

${T_READ}

@group(0) @binding(0) var<storage, read> router: array<${T}>;
@group(0) @binding(1) var<storage, read_write> expert_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;
@group(0) @binding(3) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let num_experts = _params_[0];
    let k = _params_[1];
    let normalize = _params_[2];

    var count = 0u;
    for (var e = 0u; e < num_experts; e++) {
        let v = t_read(&router, e);
        if (v > -60000.0 && count < k) {
            expert_indices[count] = e;
            expert_weights[count] = exp(v);
            count++;
        }
    }

    // Zero out unused slots
    for (var i = count; i < k; i++) {
        expert_indices[i] = 0u;
        expert_weights[i] = 0.0;
    }

    // Normalize weights
    if (normalize != 0u && count > 0u) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < count; i++) {
            sum += expert_weights[i];
        }
        if (sum > 0.0) {
            for (var i = 0u; i < count; i++) {
                expert_weights[i] /= sum;
            }
        }
    }
}

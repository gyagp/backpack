// MoE top-k routing — find top-k of N router logits, then softmax-normalize
//
// For 256 experts top-8, this scans the router logits, finds the 8 highest,
// applies softmax over only those, and outputs indices + weights.
//
// Bindings:
//   0: router_logits [num_experts] f32
//   1: out_indices   [k] u32
//   2: out_weights   [k] f32
//   3: _params_     — [num_experts, k, normalize_flag]
//
// Single-workgroup design: 256 threads, parallel argmax with iterated removal.
// For top-8 of 256, that's 8 sequential argmax passes (each O(N) = 256 muls).

@group(0) @binding(0) var<storage, read>       logits:  array<f32>;
@group(0) @binding(1) var<storage, read_write> idx_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> wt_out:  array<f32>;
@group(0) @binding(3) var<storage, read>       _params_: array<u32>;

const MAX_K: u32 = 16u;
const NEG_INF: f32 = -3.4e38;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let num_experts = _params_[0];
    let k           = _params_[1];
    let normalize   = _params_[2];
    let tid = lid.x;
    if (tid != 0u) { return; }  // serial implementation — k is small, N=256

    // Iterated argmax with masking (k passes, each O(N))
    var selected: array<u32, MAX_K>;
    var values:   array<f32, MAX_K>;
    var mask:     array<bool, 1024>;  // assumes num_experts <= 1024 (fine for qwen35moe 256)
    for (var i: u32 = 0u; i < num_experts; i = i + 1u) {
        mask[i] = false;
    }
    for (var ki: u32 = 0u; ki < k; ki = ki + 1u) {
        var best_v: f32 = NEG_INF;
        var best_i: u32 = 0u;
        for (var e: u32 = 0u; e < num_experts; e = e + 1u) {
            if (!mask[e]) {
                let v = logits[e];
                if (v > best_v) { best_v = v; best_i = e; }
            }
        }
        selected[ki] = best_i;
        values[ki]   = best_v;
        mask[best_i] = true;
    }

    // Softmax over the k selected
    var max_v: f32 = values[0];
    for (var ki: u32 = 1u; ki < k; ki = ki + 1u) {
        if (values[ki] > max_v) { max_v = values[ki]; }
    }
    var sum_e: f32 = 0.0;
    for (var ki: u32 = 0u; ki < k; ki = ki + 1u) {
        values[ki] = exp(values[ki] - max_v);
        sum_e = sum_e + values[ki];
    }
    let inv = select(1.0, 1.0 / sum_e, normalize != 0u && sum_e > 0.0);
    for (var ki: u32 = 0u; ki < k; ki = ki + 1u) {
        idx_out[ki] = selected[ki];
        wt_out[ki]  = values[ki] * inv;
    }
}

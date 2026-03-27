// GroupNorm — Group Normalization (commonly used in VAE decoders)
// Y[n,c,h,w] = scale[c] * ((X[n,c,h,w] - mean[n,g]) / sqrt(var[n,g] + eps)) + bias[c]
// where g = c / (C / num_groups)
//
// One workgroup per (batch, group) pair. Each workgroup computes mean/var
// over all (channels_per_group * H * W) elements, then normalizes.
// Dispatch: (N * num_groups, 1, 1) where N = batch size
//
// Params: [C, HW, num_groups, eps_as_u32]

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> smem_sum: f32;
var<workgroup> smem_sq: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let num_groups = _params_[2];
    let eps = bitcast<f32>(_params_[3]);

    let group_idx = wid.x;
    let batch = group_idx / num_groups;
    let g = group_idx % num_groups;
    let cpg = C / num_groups;  // channels per group
    let group_size = cpg * HW;

    let tid = lid.x;

    // Parallel reduction for mean
    var local_sum: f32 = 0.0;
    for (var i = tid; i < group_size; i += 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        local_sum += X[batch * C * HW + c * HW + hw];
    }

    // Workgroup reduction (simple sequential for now)
    if (tid == 0u) { smem_sum = 0.0; smem_sq = 0.0; }
    workgroupBarrier();
    // Atomic add emulation via loop (WGSL doesn't have atomicAdd on f32)
    // Use sequential fallback: thread 0 does the full reduction
    if (tid == 0u) {
        var s: f32 = 0.0;
        var sq: f32 = 0.0;
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = X[batch * C * HW + c * HW + hw];
            s += v;
        }
        let mean = s / f32(group_size);
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = X[batch * C * HW + c * HW + hw] - mean;
            sq += v * v;
        }
        let inv_std = 1.0 / sqrt(sq / f32(group_size) + eps);
        smem_sum = mean;
        smem_sq = inv_std;
    }
    workgroupBarrier();

    let mean = smem_sum;
    let inv_std = smem_sq;

    // Normalize + scale + bias
    for (var i = tid; i < group_size; i += 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        let offset = batch * C * HW + c * HW + hw;
        let normed = (X[offset] - mean) * inv_std;
        Y[offset] = normed * Scale[c] + Bias[c];
    }
}

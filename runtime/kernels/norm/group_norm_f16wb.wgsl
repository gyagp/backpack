enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f16>;
@group(0) @binding(2) var<storage, read> Bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> smem_sum: f32;
var<workgroup> smem_sq: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let num_groups = _params_[2];
    let eps = bitcast<f32>(_params_[3]);
    let group_idx = wid.x;
    let batch = group_idx / num_groups;
    let g = group_idx % num_groups;
    let cpg = C / num_groups;
    let group_size = cpg * HW;
    let tid = lid.x;
    if (tid == 0u) { smem_sum = 0.0; smem_sq = 0.0; }
    workgroupBarrier();
    if (tid == 0u) {
        var s: f32 = 0.0;
        var sq: f32 = 0.0;
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            s += X[batch * C * HW + c * HW + hw];
        }
        let mean = s / f32(group_size);
        for (var i = 0u; i < group_size; i++) {
            let c = g * cpg + i / HW;
            let hw = i % HW;
            let v = X[batch * C * HW + c * HW + hw] - mean;
            sq += v * v;
        }
        smem_sum = mean;
        smem_sq = 1.0 / sqrt(sq / f32(group_size) + eps);
    }
    workgroupBarrier();
    let mean = smem_sum;
    let inv_std = smem_sq;
    for (var i = tid; i < group_size; i = i + 256u) {
        let c = g * cpg + i / HW;
        let hw = i % HW;
        let offset = batch * C * HW + c * HW + hw;
        let normed = (X[offset] - mean) * inv_std;
        Y[offset] = normed * f32(Scale[c]) + f32(Bias[c]);
    }
}

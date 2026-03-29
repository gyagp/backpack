enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f16>;
@group(0) @binding(2) var<storage, read> Bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let N = _params_[2];
    let eps = bitcast<f32>(_params_[3]);
    let lane = lid.x;
    let idx = wid.x;
    if (idx >= N * C) { return; }
    let n = idx / C;
    let c = idx % C;
    let base = n * C * HW + c * HW;
    var sum: f32 = 0.0;
    for (var i = lane; i < HW; i = i + 256u) { sum += X[base + i]; }
    partial[lane] = sum;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }
    let mean = partial[0] / f32(HW);
    var var_sum: f32 = 0.0;
    for (var i = lane; i < HW; i = i + 256u) {
        let d = X[base + i] - mean;
        var_sum += d * d;
    }
    partial[lane] = var_sum;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }
    let inv_std = 1.0 / sqrt(partial[0] / f32(HW) + eps);
    let s = f32(Scale[c]);
    let b = f32(Bias[c]);
    for (var i = lane; i < HW; i = i + 256u) {
        Y[base + i] = (X[base + i] - mean) * inv_std * s + b;
    }
}

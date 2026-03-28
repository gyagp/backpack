// Instance Normalization — per-channel per-sample normalization
// X layout: [N, C, H, W]
// Each thread handles one (batch, channel) pair.
//
// Dispatch: (ceil(N*C/256), 1, 1)

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> Scale: array<${T}>;
@group(0) @binding(2) var<storage, read> Bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let C = _params_[0];
    let HW = _params_[1];
    let N = _params_[2];
    let eps = bitcast<f32>(_params_[3]);

    let idx = gid.x;
    if (idx >= N * C) { return; }

    let n = idx / C;
    let c = idx % C;
    let base = n * C * HW + c * HW;

    var mean: f32 = 0.0;
    for (var i = 0u; i < HW; i++) { mean += t_read(&X, base + i); }
    mean /= f32(HW);

    var var_sum: f32 = 0.0;
    for (var i = 0u; i < HW; i++) {
        let d = t_read(&X, base + i) - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(HW) + eps);

    let s = t_read(&Scale, c);
    let b = t_read(&Bias, c);
    for (var i = 0u; i < HW; i++) {
        t_write(&Y, base + i, (t_read(&X, base + i) - mean) * inv_std * s + b);
    }
}

// LayerNormalization — mean + variance normalization
// Y = (X - mean) / sqrt(var + eps) * W + B
// Dispatch: (ceil(nRows/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);

    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;

    var mean: f32 = 0.0;
    for (var i = 0u; i < N; i++) { mean += X[base + i]; }
    mean = mean / f32(N);

    var var_sum: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let d = X[base + i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / f32(N) + eps);

    for (var i = 0u; i < N; i++) {
        Y[base + i] = (X[base + i] - mean) * inv_std * W[i] + B[i];
    }
}

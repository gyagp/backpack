enable f16;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Skip: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) {
        let v = X[base + i] + Skip[base + i];
        SkipOut[base + i] = v;
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / sqrt(sum_sq / f32(N) + eps);
    for (var i = 0u; i < N; i++) {
        Y[base + i] = SkipOut[base + i] * inv_rms * f32(W[i]);
    }
}

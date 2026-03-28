// SkipSimplifiedLayerNormalization — residual add + RMSNorm
// Computes: SkipOut = X + Skip, Y = RMSNorm(SkipOut) * W
// Dispatch: (ceil(nRows/256), 1, 1)
// Params: [0]=N (hidden dim), [1]=nRows, [2]=eps (bitcast<f32>)

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> Skip: array<${T}>;
@group(0) @binding(2) var<storage, read> W: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read_write> SkipOut: array<${T}>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let nRows = _params_[1];
    let eps = bitcast<f32>(_params_[2]);
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * N;

    // Pass 1: Residual add + sum of squares, write SkipOut in pairs
    var sum_sq: f32 = 0.0;
    let pairs = N / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) + t_read(&Skip, base + i0);
        let v1 = t_read(&X, base + i0 + 1u) + t_read(&Skip, base + i0 + 1u);
        sum_sq += v0 * v0 + v1 * v1;
        t_write2(&SkipOut, base + i0, v0, v1);
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        let v = t_read(&X, base + last) + t_read(&Skip, base + last);
        sum_sq += v * v;
        t_write2(&SkipOut, base + last, v, 0.0);
    }
    let inv_rms = 1.0 / sqrt(sum_sq / f32(N) + eps);

    // Pass 2: Recompute residual + apply norm weight, paired writes
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) + t_read(&Skip, base + i0);
        let v1 = t_read(&X, base + i0 + 1u) + t_read(&Skip, base + i0 + 1u);
        t_write2(&Y, base + i0, v0 * inv_rms * t_read(&W, i0), v1 * inv_rms * t_read(&W, i0 + 1u));
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        let v = t_read(&X, base + last) + t_read(&Skip, base + last);
        t_write2(&Y, base + last, v * inv_rms * t_read(&W, last), 0.0);
    }
}

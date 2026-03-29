// Softmax — numerically stable per-row softmax
// Dispatch: (ceil(nRows/256), 1, 1)
// Internal accumulation in f32 for numerical stability.
// Uses paired writes for fp16 correctness (u32 packing).

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nRows = _params_[0];
    let rowLen = _params_[1];
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * rowLen;

    // Find max (f32)
    var mx: f32 = -1e30;
    for (var i = 0u; i < rowLen; i++) {
        mx = max(mx, t_read(&X, base + i));
    }
    // Exp sum
    var expSum: f32 = 0.0;
    for (var i = 0u; i < rowLen; i++) {
        expSum += exp(t_read(&X, base + i) - mx);
    }
    let invSum = 1.0 / max(expSum, 1e-10);
    // Write output in pairs
    let pairs = rowLen / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = exp(t_read(&X, base + i0) - mx) * invSum;
        let v1 = exp(t_read(&X, base + i0 + 1u) - mx) * invSum;
        t_write2(&Y, base + i0, v0, v1);
    }
    if ((rowLen & 1u) != 0u) {
        let last = rowLen - 1u;
        t_write2(&Y, base + last, exp(t_read(&X, base + last) - mx) * invSum, 0.0);
    }
}

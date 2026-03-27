// Softmax — numerically stable per-row softmax
// Dispatch: (ceil(nRows/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nRows = _params_[0];
    let rowLen = _params_[1];
    let row = gid.x;
    if (row >= nRows) { return; }
    let base = row * rowLen;

    var maxVal: f32 = -1e30;
    for (var i = 0u; i < rowLen; i++) {
        maxVal = max(maxVal, X[base + i]);
    }
    var sumExp: f32 = 0.0;
    for (var i = 0u; i < rowLen; i++) {
        sumExp += exp(X[base + i] - maxVal);
    }
    let invSum = 1.0 / max(sumExp, 1e-9);
    for (var i = 0u; i < rowLen; i++) {
        Y[base + i] = exp(X[base + i] - maxVal) * invSum;
    }
}

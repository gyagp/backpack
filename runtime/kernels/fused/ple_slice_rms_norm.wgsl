// @meta bindings=5
// PLE: Per-slice RMSNorm — normalizes N slices of D elements each.
// Used to normalize the model projection output before combining with
// per-layer token embeddings.
//
// Each workgroup handles one slice of D elements.
// Dispatch: (N, 1, 1) where N = number of slices (nLayer)
//
// Bindings:
//   0: X (read) — [N*D] fp32, input (model projection output)
//   1: Y (read_write) — [N*D] fp32, output (normalized)
//   2: W (read) — [D] fp32, learned norm weights (shared across slices)
//   3: Rstd (read_write) — [N] fp32, reciprocal std per slice (unused but needed for bind group compat)
//   4: _params_ — {stride=D, N=D, eps_bits, 0}

enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

var<workgroup> wg_sum: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let D = _params_[1];  // slice dimension
    let eps = bitcast<f32>(_params_[2]);
    let slice = wid.x;
    let base = slice * D;
    let tid = lid.x;
    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Pass 1: sum of squares (strided for coalesced access)
    var ss: f32 = 0.0;
    var i = tid;
    for (; i < D; i = i + 256u) {
        let v = X[base + i];
        ss = ss + v * v;
    }

    // Warp reduce
    ss = ss + subgroupShuffleXor(ss, 16u);
    ss = ss + subgroupShuffleXor(ss, 8u);
    ss = ss + subgroupShuffleXor(ss, 4u);
    ss = ss + subgroupShuffleXor(ss, 2u);
    ss = ss + subgroupShuffleXor(ss, 1u);

    if (lane == 0u) { wg_sum[warp_id] = ss; }
    workgroupBarrier();

    // Reduce across 8 warps
    if (tid == 0u) {
        var total: f32 = 0.0;
        for (var w = 0u; w < 8u; w = w + 1u) { total = total + wg_sum[w]; }
        let rms = 1.0 / sqrt(total / f32(D) + eps);
        wg_sum[0] = rms;
    }
    workgroupBarrier();
    let rms = wg_sum[0];

    // Pass 2: normalize and apply weights
    i = tid;
    for (; i < D; i = i + 256u) {
        Y[base + i] = X[base + i] * rms * W[i];
    }
}

// @meta bindings=7
enable subgroups;

// Gemma sandwich attention epilogue, one workgroup per prompt row:
//   A = RMSNorm(A, post_attn_weight)
//   X += A
//   Y = RMSNorm(X, ffn_weight)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<storage, read> PostW: array<f32>;
@group(0) @binding(3) var<storage, read> NextW: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> Rstd: array<f32>;
@group(0) @binding(6) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 8>;

fn reduce_sum(v: f32, tid: u32) -> f32 {
    let lane = tid & 31u;
    let warp = tid / 32u;
    let ws = subgroupAdd(v);
    if (lane == 0u) { sums[warp] = ws; }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < 8u; i++) { total += sums[i]; }
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x; let tid = lid.x;
    let N = P[0]; let stride = P[1];
    let eps = bitcast<f32>(P[2]);
    let base = row * stride;

    var ss = 0.0;
    for (var i = tid; i < N; i += 256u) {
        let v = A[base + i]; ss += v * v;
    }
    let ar = inverseSqrt(reduce_sum(ss, tid) / f32(N) + eps);

    ss = 0.0;
    for (var i = tid; i < N; i += 256u) {
        let p = base + i;
        let v = X[p] + A[p] * ar * PostW[i];
        X[p] = v; ss += v * v;
    }
    let xr = inverseSqrt(reduce_sum(ss, tid) / f32(N) + eps);
    if (tid == 0u) { Rstd[row] = xr; }
    for (var i = tid; i < N; i += 256u) {
        let p = base + i;
        Y[p] = X[p] * xr * NextW[i];
    }
}

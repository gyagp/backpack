// @meta bindings=5
enable subgroups;

// Normalize each addend row, then add it to the residual in place:
//   X += RMSNorm(A, W)

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 8>;

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
    let lane = tid & 31u; let warp = tid / 32u;
    let ws = subgroupAdd(ss);
    if (lane == 0u) { sums[warp] = ws; }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < 8u; i++) { total += sums[i]; }
    let r = inverseSqrt(total / f32(N) + eps);
    if (tid == 0u) { Rstd[row] = r; }
    for (var i = tid; i < N; i += 256u) {
        let p = base + i;
        X[p] += A[p] * r * W[i];
    }
}

// @meta bindings=5
enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params {
    stride: i32,
    N: i32,
    eps: f32,
};
@group(0) @binding(4) var<storage, read> params: Params;

var<workgroup> warp_sums: array<f32, 8>;
var<workgroup> rstd_shared: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let N = u32(params.N);
    let base = row * u32(params.stride);

    var sum_sq = 0.0;
    for (var j = tid; j < N; j = j + 256u) {
        let v = X[base + j];
        sum_sq += v * v;
    }

    let warp_sum = subgroupAdd(sum_sq);
    if (lane == 0u) {
        warp_sums[warp] = warp_sum;
    }
    workgroupBarrier();

    let cross = select(0.0, warp_sums[min(lane, 7u)], warp == 0u && lane < 8u);
    let total = subgroupAdd(cross);
    if (tid == 0u) {
        rstd_shared = 1.0 / sqrt(total / f32(N) + params.eps);
        Rstd[row] = rstd_shared;
    }
    workgroupBarrier();

    let rstd = rstd_shared;
    for (var j = tid; j < N; j = j + 256u) {
        Y[base + j] = X[base + j] * rstd * W[j];
    }
}

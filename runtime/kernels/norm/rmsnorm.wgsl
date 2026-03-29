${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> W: array<${T}>;
@group(0) @binding(3) var<storage, read_write> Rstd: array<f32>;

struct Params { stride: i32, N: i32, eps: f32, };
@group(0) @binding(4) var<storage, read> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let N = u32(params.N);
    if (row >= u32(params.stride) / N) { return; }
    let base = row * N;
    var sum_sq: f32 = 0.0;
    for (var i = 0u; i < N; i++) { let v = t_read(&X, base + i); sum_sq += v * v; }
    let rms = sqrt(sum_sq / f32(N) + params.eps);
    let inv_rms = 1.0 / rms;
    // Write pairs to avoid fp16 u32 write races
    let pairs = N / 2u;
    for (var i = 0u; i < pairs; i++) {
        let i0 = i * 2u;
        let v0 = t_read(&X, base + i0) * inv_rms * t_read(&W, i0);
        let v1 = t_read(&X, base + i0 + 1u) * inv_rms * t_read(&W, i0 + 1u);
        t_write2(&Y, base + i0, v0, v1);
    }
    if ((N & 1u) != 0u) {
        let last = N - 1u;
        t_write2(&Y, base + last, t_read(&X, base + last) * inv_rms * t_read(&W, last), 0.0);
    }
    if (gid.x == 0u) { Rstd[row] = inv_rms; }
}

enable subgroups;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> _params_: array<u32>;

const TILE_N: u32 = 32u;
const COLS_PER_WARP: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tile_col = wid.y;
    let tid = lid.x;

    let K = _params_[0];
    let N = _params_[1];
    let x_base = row * K;

    let warp_id = tid / 32u;
    let lane = tid % 32u;

    // Each warp handles COLS_PER_WARP=4 columns
    let base_col = tile_col * TILE_N + warp_id * COLS_PER_WARP;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    let half_K = K / 2u;
    let stride = 128u;  // 32 lanes * 4 elements = 128
    var k = lane * 4u;

    for (; k + 3u < K; k = k + stride) {
        // Load X once, reuse for all 4 columns
        let x0 = X[x_base + k];
        let x1 = X[x_base + k + 1u];
        let x2 = X[x_base + k + 2u];
        let x3 = X[x_base + k + 3u];
        let xv = vec4<f32>(x0, x1, x2, x3);

        // Column 0
        if (base_col < N) {
            let w_base0 = base_col * half_K;
            let w01 = unpack2x16float(W[w_base0 + k / 2u]);
            let w23 = unpack2x16float(W[w_base0 + k / 2u + 1u]);
            acc0 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 1
        if (base_col + 1u < N) {
            let w_base1 = (base_col + 1u) * half_K;
            let w01 = unpack2x16float(W[w_base1 + k / 2u]);
            let w23 = unpack2x16float(W[w_base1 + k / 2u + 1u]);
            acc1 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 2
        if (base_col + 2u < N) {
            let w_base2 = (base_col + 2u) * half_K;
            let w01 = unpack2x16float(W[w_base2 + k / 2u]);
            let w23 = unpack2x16float(W[w_base2 + k / 2u + 1u]);
            acc2 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
        // Column 3
        if (base_col + 3u < N) {
            let w_base3 = (base_col + 3u) * half_K;
            let w01 = unpack2x16float(W[w_base3 + k / 2u]);
            let w23 = unpack2x16float(W[w_base3 + k / 2u + 1u]);
            acc3 += dot(xv, vec4<f32>(w01.x, w01.y, w23.x, w23.y));
        }
    }

    // Warp reduce
    let sum0 = subgroupAdd(acc0);
    let sum1 = subgroupAdd(acc1);
    let sum2 = subgroupAdd(acc2);
    let sum3 = subgroupAdd(acc3);

    if (lane == 0u) {
        if (base_col      < N) { Y[row * N + base_col]      = sum0 + Bias[base_col]; }
        if (base_col + 1u < N) { Y[row * N + base_col + 1u] = sum1 + Bias[base_col + 1u]; }
        if (base_col + 2u < N) { Y[row * N + base_col + 2u] = sum2 + Bias[base_col + 2u]; }
        if (base_col + 3u < N) { Y[row * N + base_col + 3u] = sum3 + Bias[base_col + 3u]; }
    }
}

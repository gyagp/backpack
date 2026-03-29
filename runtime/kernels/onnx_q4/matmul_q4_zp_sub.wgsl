// Q4 matmul with zero points + K-parallel subgroup reduction, TILE_N=8.
// Optimized for decode (M=1): 256 threads = 8 warps × 32 lanes.
// Each warp computes one output N, 32 lanes split blocks_per_col.
// Y[n] = sum_k A[k] * (dequant(B[n,k]) - zp) * scale
// Dispatch: (ceil(N/8), 1, 1) — M must be 1

enable subgroups;

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> A: array<${T}>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> ZeroPoints: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[1];
    let K = _params_[2];
    let blocks_per_col = K / 32u;

    let warp_id = lid.x / 32u;
    let lane = lid.x % 32u;
    let n = wid.x * 8u + warp_id;
    let valid = n < N;

    var acc: f32 = 0.0;
    let w_base = n * (K / 2u);

    if (valid) {
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let zp_byte_idx = scale_flat / 2u;
            let zp_byte = (ZeroPoints[zp_byte_idx / 4u] >> ((zp_byte_idx % 4u) * 8u)) & 0xFFu;
            let zp = f32(select(zp_byte & 0xFu, (zp_byte >> 4u) & 0xFu, (scale_flat & 1u) != 0u));

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = B[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                acc += t_read(&A, k0)      * (f32(b0 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 1u) * (f32(b0 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 2u) * (f32(b1 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 3u) * (f32(b1 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 4u) * (f32(b2 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 5u) * (f32(b2 >> 4u)  - zp) * scale;
                acc += t_read(&A, k0 + 6u) * (f32(b3 & 0xFu) - zp) * scale;
                acc += t_read(&A, k0 + 7u) * (f32(b3 >> 4u)  - zp) * scale;
            }
        }
    }

    // subgroupAdd requires subgroup-uniform control flow
    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        t_write(&Y, n, warp_sum);
    }
}

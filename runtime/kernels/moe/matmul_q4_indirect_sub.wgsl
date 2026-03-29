// Q4 matmul with K-parallel subgroup reduction and TILE_N=8.
// 256 threads = 8 warps × 32 lanes. Each warp computes one output element.
// 32 lanes split blocks_per_col, then reduce via subgroupAdd.
// Y[n] = sum_k X[k] * dequant(W[expert, n, k])
// Weights: W[num_experts, N, K/2] packed uint8
// Scales:  S[num_experts, N, blocks_per_col] packed fp16
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot
// Dispatch: (ceil(N/8), 1, 1)

enable subgroups;

${T_READ}
${T_WRITE}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];

    let warp_id = lid.x / 32u;
    let lane = lid.x % 32u;
    let n = wid.x * 8u + warp_id;
    let valid = n < N;

    let expert = select(0u, expert_indices[slot], valid);
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;
    let w_base = expert_w_offset + n * (K / 2u);

    var acc: f32 = 0.0;

    if (valid) {
        // Each lane processes blocks: lane, lane+32, lane+64, ...
        for (var blk = lane; blk < blocks_per_col; blk += 32u) {
            let scale_flat = expert_s_offset + n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let k_base = blk * 32u;
            let w_blk_base = w_base + k_base / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = W[w_blk_base / 4u + j];
                let k0 = k_base + j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;
                acc += t_read(&X, k0)      * (f32(b0 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 1u) * (f32(b0 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 2u) * (f32(b1 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 3u) * (f32(b1 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 4u) * (f32(b2 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 5u) * (f32(b2 >> 4u)  - 8.0) * scale;
                acc += t_read(&X, k0 + 6u) * (f32(b3 & 0xFu) - 8.0) * scale;
                acc += t_read(&X, k0 + 7u) * (f32(b3 >> 4u)  - 8.0) * scale;
            }
        }
    }

    // subgroupAdd requires subgroup-uniform control flow
    let warp_sum = subgroupAdd(acc);
    if (lane == 0u && valid) {
        t_write(&Y, n, warp_sum);
    }
}

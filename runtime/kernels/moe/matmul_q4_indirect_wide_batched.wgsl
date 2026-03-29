// Wide-tile batched Q4 matmul with expert indirection and shared-memory X reuse.
// 128 threads, TILE_N=128. Each thread owns one N-column.
// X tile [1 x 32] is cooperatively loaded into shared memory per Q4 block,
// then reused by all 128 threads for different N-column weight-activation dot products.
// Params: [0]=N, [1]=K, [2]=blocks_per_col, [3]=slot, [4]=k_val (topk k)
// Dispatch: (ceil(N/128), nTokens, 1)

enable subgroups;

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;
@group(0) @binding(5) var<storage, read> expert_indices: array<u32>;

var<workgroup> smem_x: array<f32, 32>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let N = _params_[0];
    let K = _params_[1];
    let blocks_per_col = _params_[2];
    let slot = _params_[3];
    let k_val = _params_[4];
    let tok = wid.y;

    let tid = lid.x;
    let n = wid.x * 128u + tid;
    let valid = n < N;

    let expert = expert_indices[tok * k_val + slot];
    let expert_w_offset = expert * N * (K / 2u);
    let expert_s_offset = expert * N * blocks_per_col;
    let w_base = select(0u, expert_w_offset + n * (K / 2u), valid);
    let x_base = tok * K;

    var acc: f32 = 0.0;

    for (var blk = 0u; blk < blocks_per_col; blk++) {
        let k_off = blk * 32u;

        // Cooperative X load: 128 threads load 32 elements
        if (tid < 32u) {
            smem_x[tid] = X[x_base + k_off + tid];
        }
        workgroupBarrier();

        if (valid) {
            let scale_flat = expert_s_offset + n * blocks_per_col + blk;
            let scale_u32 = Scales[scale_flat / 2u];
            let scale_half = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
            let scale = unpack2x16float(scale_half | (scale_half << 16u)).x;

            let w_blk_base = w_base + k_off / 2u;

            for (var j = 0u; j < 4u; j++) {
                let packed = W[w_blk_base / 4u + j];
                let kl = j * 8u;
                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                acc += smem_x[kl]      * (f32(b0 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 1u] * (f32(b0 >> 4u)  - 8.0) * scale
                     + smem_x[kl + 2u] * (f32(b1 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 3u] * (f32(b1 >> 4u)  - 8.0) * scale
                     + smem_x[kl + 4u] * (f32(b2 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 5u] * (f32(b2 >> 4u)  - 8.0) * scale
                     + smem_x[kl + 6u] * (f32(b3 & 0xFu) - 8.0) * scale
                     + smem_x[kl + 7u] * (f32(b3 >> 4u)  - 8.0) * scale;
            }
        }

        workgroupBarrier();
    }

    if (valid) {
        Y[tok * N + n] = acc;
    }
}

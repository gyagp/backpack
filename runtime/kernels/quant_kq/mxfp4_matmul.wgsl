@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_blocks: array<i32>;
@group(0) @binding(2) var<storage, read> W_scales: array<i32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

// FP4 E2M1 lookup table (indexed by 3-bit abs value)
// 0→0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
const FP4_LUT = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K = _params_[0];
    let N = _params_[1];
    let stride_blocks = _params_[2];
    let stride_scales = _params_[3];

    let col = gid.x;  // output column (N dimension)
    let row = wid.y;   // token index (T dimension)
    if (col >= N) { return; }

    let n_chunks = K / 32u;  // MXFP4 block size = 32
    let blocks_base = col * stride_blocks;
    let scales_base = col * stride_scales;
    let x_base = row * K;

    var acc: f32 = 0.0;

    for (var chunk_i = 0u; chunk_i < n_chunks; chunk_i++) {
        // Load E8M0 scale: 4 bytes packed per i32
        let scale_word = W_scales[scales_base + chunk_i / 4u];
        let scale_byte = (u32(scale_word) >> ((chunk_i % 4u) * 8u)) & 0xFFu;
        let scale = exp2(f32(scale_byte) - 127.0);

        let off = chunk_i * 32u;

        // Process 32 FP4 elements (4 i32 words × 8 nibbles each)
        for (var w_i = 0u; w_i < 4u; w_i++) {
            let word = u32(W_blocks[blocks_base + (off / 8u) + w_i]);

            // Unroll 8 nibbles per word
            for (var nib = 0u; nib < 8u; nib++) {
                let k = off + w_i * 8u + nib;
                let nibble = (word >> (nib * 4u)) & 0xFu;
                let sign = f32(nibble >> 3u);
                let abs_val = FP4_LUT[nibble & 7u];
                let w = abs_val * (1.0 - 2.0 * sign);
                acc += X[x_base + k] * w * scale;
            }
        }
    }

    Y[row * N + col] = acc + Bias[col];
}

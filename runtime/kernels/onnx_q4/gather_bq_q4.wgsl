// @meta bindings=5
// GatherBlockQuantized Q4 — ONNX Q4 embedding lookup
// Dequantizes Q4 packed weights at gathered indices.
// Weight: [V, K] where each element is 4 bits (2 per byte)
// Scale: [V, n_groups] fp16 packed
//
// Dispatch: (ceil(K/256), nIndices, 1)

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> Scales: array<u32>;
@group(0) @binding(2) var<storage, read> Indices: array<i32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nIdx = _params_[0];
    let K = _params_[1];
    let n_groups = _params_[2];
    let bs = _params_[3];
    let idx_i = gid.y;
    let k = gid.x;
    if (idx_i >= nIdx || k >= K) { return; }
    let vocab_idx = u32(Indices[idx_i]);
    let group = k / bs;
    let scale_flat = vocab_idx * n_groups + group;
    let scale_u32 = Scales[scale_flat / 2u];
    let scale_f16 = select(scale_u32 & 0xFFFFu, (scale_u32 >> 16u) & 0xFFFFu, (scale_flat & 1u) != 0u);
    let scale = unpack2x16float(scale_f16 | (scale_f16 << 16u)).x;
    let byte_flat = vocab_idx * (K / 2u) + k / 2u;
    let byte_u32 = W[byte_flat / 4u];
    let byte_val = (byte_u32 >> ((byte_flat % 4u) * 8u)) & 0xFFu;
    let nibble = select(byte_val & 0x0Fu, (byte_val >> 4u) & 0x0Fu, (k & 1u) != 0u);
    // UINT4 with default zero_point=8: dequant = (nibble - 8) * scale
    let centered = f32(i32(nibble)) - 8.0;
    Y[idx_i * K + k] = centered * scale;
}

// RotaryEmbedding — apply rotary position embeddings
// Supports both interleaved and non-interleaved modes.
//
// Dispatch: (ceil(total/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> PosIds: array<i32>;
@group(0) @binding(2) var<storage, read> CosCache: array<f32>;
@group(0) @binding(3) var<storage, read> SinCache: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = _params_[0];
    let head_dim = _params_[1];
    let interleaved = _params_[2];

    let idx = gid.x;
    if (idx >= total) { return; }

    let half = head_dim / 2u;
    let d = idx % head_dim;
    let pos_idx = idx / head_dim;
    let nPosIds = _params_[3];
    let pos = select(0, PosIds[pos_idx % nPosIds], nPosIds > 0u);

    if (interleaved != 0u) {
        let pair = d / 2u;
        let cos_val = CosCache[u32(pos) * half + pair];
        let sin_val = SinCache[u32(pos) * half + pair];
        let base = idx - d;
        if (d % 2u == 0u) {
            Y[idx] = X[idx] * cos_val - X[base + d + 1u] * sin_val;
        } else {
            Y[idx] = X[base + d - 1u] * sin_val + X[idx] * cos_val;
        }
    } else {
        if (d < half) {
            let cos_val = CosCache[u32(pos) * half + d];
            let sin_val = SinCache[u32(pos) * half + d];
            let base = idx - d;
            Y[idx] = X[idx] * cos_val - X[base + d + half] * sin_val;
        } else {
            let d2 = d - half;
            let cos_val = CosCache[u32(pos) * half + d2];
            let sin_val = SinCache[u32(pos) * half + d2];
            let base = idx - d;
            Y[idx] = X[base + d2] * sin_val + X[idx] * cos_val;
        }
    }
}

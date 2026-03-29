// Nearest-neighbor resize (upsampling)
// Dispatch: (ceil(N*C*H_out*W_out/512), 1, 1) — each thread handles 2 elements

${T_READ}
${T_WRITE2}

@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

fn compute_resize(idx: u32, N: u32, C: u32, H_in: u32, W_in: u32, H_out: u32, W_out: u32) -> f32 {
    let n = idx / (C * H_out * W_out);
    let c = (idx / (H_out * W_out)) % C;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;
    let ih = min(oh * H_in / H_out, H_in - 1u);
    let iw = min(ow * W_in / W_out, W_in - 1u);
    return t_read(&X, n * C * H_in * W_in + c * H_in * W_in + ih * W_in + iw);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let C = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let H_out = _params_[4];
    let W_out = _params_[5];

    let total = N * C * H_out * W_out;
    let base = gid.x * 2u;
    if (base >= total) { return; }

    let v0 = compute_resize(base, N, C, H_in, W_in, H_out, W_out);
    var v1: f32 = 0.0;
    if (base + 1u < total) {
        v1 = compute_resize(base + 1u, N, C, H_in, W_in, H_out, W_out);
    }
    t_write2(&Y, base, v0, v1);
}

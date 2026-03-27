// Nearest-neighbor resize (upsampling)
// Dispatch: (ceil(N*C*H_out*W_out/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = _params_[0];
    let C = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let H_out = _params_[4];
    let W_out = _params_[5];

    let total = N * C * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C * H_out * W_out);
    let c = (idx / (H_out * W_out)) % C;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ih = min(oh * H_in / H_out, H_in - 1u);
    let iw = min(ow * W_in / W_out, W_in - 1u);

    Y[idx] = X[n * C * H_in * W_in + c * H_in * W_in + ih * W_in + iw];
}

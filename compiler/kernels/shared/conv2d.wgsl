// Conv2D — 2D convolution with bias
// Y[n,co,oh,ow] = sum_{ci,kh,kw} X[n,ci,ih,iw] * W[co,ci,kh,kw] + bias[co]
//
// Supports padding, stride, dilation, and groups.
// Dispatch: (ceil(total_output/256), 1, 1)

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = _params_[0];
    let C_in = _params_[1];
    let H_in = _params_[2];
    let W_in = _params_[3];
    let C_out = _params_[4];
    let KH = _params_[5];
    let KW = _params_[6];
    let pad_h = _params_[7];
    let pad_w = _params_[8];
    let stride_h = _params_[9];
    let stride_w = _params_[10];
    let H_out = _params_[11];
    let W_out = _params_[12];
    let group = _params_[13];
    let dil_h = _params_[14];
    let dil_w = _params_[15];

    let total = batch * C_out * H_out * W_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let n = idx / (C_out * H_out * W_out);
    let co = (idx / (H_out * W_out)) % C_out;
    let oh = (idx / W_out) % H_out;
    let ow = idx % W_out;

    let ci_per_group = C_in / group;
    let co_per_group = C_out / group;
    let g = co / co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                let ih = i32(oh * stride_h + kh * dil_h) - i32(pad_h);
                let iw = i32(ow * stride_w + kw * dil_w) - i32(pad_w);
                if (ih >= 0 && ih < i32(H_in) && iw >= 0 && iw < i32(W_in)) {
                    let x_val = X[n * C_in * H_in * W_in + ci * H_in * W_in + u32(ih) * W_in + u32(iw)];
                    let w_val = W[co * ci_per_group * KH * KW + ci_local * KH * KW + kh * KW + kw];
                    acc += x_val * w_val;
                }
            }
        }
    }

    Y[idx] = acc + Bias[co];
}

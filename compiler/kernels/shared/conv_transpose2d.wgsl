// ConvTranspose2D — 2D transposed convolution with bias
// Y[n,co,oh,ow] = sum_{ci,kh,kw} X[n,ci,ih,iw] * W[ci,co,kh,kw] + bias[co]
//
// Note: weight layout is [C_in, C_out, KH, KW] (transposed from Conv)
//
// Supports padding, stride, output_padding, and groups.
// Dispatch: (ceil(total_output/256), 1, 1)
//
// Params: [batch, C_in, H_in, W_in, C_out, KH, KW, pad_h,
//          pad_w, stride_h, stride_w, H_out, W_out, group, out_pad_h, out_pad_w]

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
    let co_local = co % co_per_group;

    var acc: f32 = 0.0;
    for (var ci_local = 0u; ci_local < ci_per_group; ci_local++) {
        let ci = g * ci_per_group + ci_local;
        for (var kh = 0u; kh < KH; kh++) {
            for (var kw = 0u; kw < KW; kw++) {
                // In transposed conv, output position (oh, ow) is related to
                // input position by: oh = ih * stride_h - pad_h + kh
                // So: ih = (oh + pad_h - kh) / stride_h
                let oh_off = i32(oh) + i32(pad_h) - i32(kh);
                let ow_off = i32(ow) + i32(pad_w) - i32(kw);
                if (oh_off >= 0 && ow_off >= 0 &&
                    oh_off % i32(stride_h) == 0 && ow_off % i32(stride_w) == 0) {
                    let ih = u32(oh_off) / stride_h;
                    let iw = u32(ow_off) / stride_w;
                    if (ih < H_in && iw < W_in) {
                        let x_val = X[n * C_in * H_in * W_in + ci * H_in * W_in + ih * W_in + iw];
                        // Weight layout: [C_in, C_out_per_group, KH, KW]
                        let w_val = W[ci * co_per_group * KH * KW + co_local * KH * KW + kh * KW + kw];
                        acc += x_val * w_val;
                    }
                }
            }
        }
    }

    Y[idx] = acc + Bias[co];
}

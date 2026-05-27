// SSM selective scan — decode step
//
// Standard Mamba recurrence for a single decode token:
//   per (c in 0..d_inner, s in 0..d_state):
//     a   = exp(A_log[c, s]) * (-1)         // A_log is the stored log of -A
//     da  = dt[c] * a
//     decay = exp(da)
//     ddh = (decay - 1) / a * B[s] * x[c]
//     h[c, s] = decay * h[c, s] + ddh
//   per c:
//     y[c] = sum_s C[s] * h[c, s] + D[c] * x[c]
//
// NOTE: qwen35moe uses alpha/beta projections in place of standard B/C —
// host code needs to wire the right buffers. This kernel computes the
// recurrence assuming caller has provided the per-step B/C and dt.
//
// Bindings:
//   0: x        [d_inner]                  — input projection at this step
//   1: A_log    [d_inner * d_state]        — log of -A (negative-real init)
//   2: dt       [d_inner]                  — softplus(dt_proj(x) + dt_bias)
//   3: B        [d_state]                  — per-step B projection
//   4: C        [d_state]                  — per-step C projection
//   5: D        [d_inner]                  — skip-conn scale (may be zero buf)
//   6: h_state  [d_inner * d_state]        — recurrent state (read+update)
//   7: y_out    [d_inner]                  — output (write)
//   8: _params_                            — [d_inner, d_state]

@group(0) @binding(0) var<storage, read>       x_in:    array<f32>;
@group(0) @binding(1) var<storage, read>       A_log:   array<f32>;
@group(0) @binding(2) var<storage, read>       dt:      array<f32>;
@group(0) @binding(3) var<storage, read>       B:       array<f32>;
@group(0) @binding(4) var<storage, read>       C:       array<f32>;
@group(0) @binding(5) var<storage, read>       D:       array<f32>;
@group(0) @binding(6) var<storage, read_write> h_state: array<f32>;
@group(0) @binding(7) var<storage, read_write> y_out:   array<f32>;
@group(0) @binding(8) var<storage, read>       _params_: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_inner = _params_[0];
    let d_state = _params_[1];
    let c = gid.x;
    if (c >= d_inner) { return; }

    let xc = x_in[c];
    let dtc = dt[c];
    let Dc = D[c];

    var acc: f32 = Dc * xc;
    let base = c * d_state;
    for (var s: u32 = 0u; s < d_state; s = s + 1u) {
        let A_neg = -exp(A_log[base + s]);
        let da = dtc * A_neg;
        let decay = exp(da);
        // (decay - 1) / A_neg is numerically stable for small da
        let coef = select((decay - 1.0) / A_neg, dtc, abs(A_neg) < 1.0e-6);
        let ddh = coef * B[s] * xc;
        let h_new = decay * h_state[base + s] + ddh;
        h_state[base + s] = h_new;
        acc = acc + C[s] * h_new;
    }
    y_out[c] = acc;
}

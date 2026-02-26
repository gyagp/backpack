"""
Shared Triton kernels for WebGPU model inference.

All transformer models share these compute primitives:
  - Linear projections (single-pass and loop-based)
  - Normalization (LayerNorm, RMSNorm)
  - Activations (GELU, SiLU/SwiGLU)
  - Element-wise operations (add)
  - Causal attention (online softmax)

These kernels compile to WGSL and execute as WebGPU compute shaders.
"""
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Linear projection kernels
# ---------------------------------------------------------------------------

@triton.jit
def linear_kernel(X, W, Bias, Y, stride_x, stride_w, N,
                  BLOCK_K: tl.constexpr):
    """Y[row, col] = dot(X[row, :], W[col, :]) + Bias[col].

    Single-pass version for small K where BLOCK_K >= K.
    Grid: (T, N) — one output element per workgroup.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)
    ks = tl.arange(0, BLOCK_K)
    x = tl.load(X + row * stride_x + ks).to(tl.float32)
    w = tl.load(W + col * stride_w + ks).to(tl.float32)
    dot = tl.sum(x * w, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b)


@triton.jit
def linear_loop_kernel(X, W, Bias, Y, K, stride_x, stride_w, N,
                       BLOCK_K: tl.constexpr):
    """Loop-based linear projection for arbitrary K.

    Accumulates dot product across K-dimension chunks of size BLOCK_K,
    allowing any input dimension while keeping workgroup size small.
    Grid: (T, N) — one output element per workgroup.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)
    num_chunks = (K + BLOCK_K - 1) // BLOCK_K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K
        x = tl.load(X + row * stride_x + ks, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + col * stride_w + ks, mask=mask, other=0.0).to(tl.float32)
        acc += x * w
    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b)


@triton.jit
def linear_loop_fp16w_kernel(X, W, Bias, Y, K, stride_x, stride_w, N,
                              BLOCK_K: tl.constexpr):
    """Loop-based linear with fp16 weights, deferred reduction.

    Each thread accumulates products across all K-chunks independently.
    The tl.sum reduction runs ONCE after the loop instead of per-chunk,
    cutting workgroup barriers from ~360 to ~15 per output element.

    W is stored as fp16 to halve memory bandwidth.
    Grid: (T, N) — one output element per workgroup.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)
    num_chunks = (K + BLOCK_K - 1) // BLOCK_K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K
        x = tl.load(X + row * stride_x + ks, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + col * stride_w + ks, mask=mask, other=0.0).to(tl.float32)
        acc += x * w
    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b)


@triton.jit
def linear_q4_kernel(X, W_Q4, Scales, Zeros, Bias, Y,
                     K, stride_x, stride_w_q4, n_groups, N,
                     BLOCK_K: tl.constexpr):
    """INT4 fused matvec: dequantize packed INT4 weights on-the-fly.

    Reads packed 4-bit weights from GPU memory and dequantizes during
    the dot product, reducing memory bandwidth by ~4x vs fp16.

    W_Q4: packed i32 array, flat (N, K//8).  Each i32 holds 8 nibbles:
          bits [4*i : 4*i+3] = element at position 8*word + i  (i=0..7).
    Scales: fp16, flat (N, n_groups).  Per-group scale factor.
    Zeros:  fp16, flat (N, n_groups).  Per-group zero point.
    Dequantization: w = q * scale + zero,  q in {0..15}.

    BLOCK_K must equal the quantization group size (128).
    Grid: (T, N) — one output element per workgroup.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)

    w_base = col * stride_w_q4
    s_base = col * n_groups

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(n_groups):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K

        # Per-group scale and zero (scalar broadcast)
        scale = tl.load(Scales + s_base + chunk_i).to(tl.float32)
        zero = tl.load(Zeros + s_base + chunk_i).to(tl.float32)

        # Load X (fp32)
        x = tl.load(X + row * stride_x + ks, mask=mask, other=0.0).to(tl.float32)

        # Load packed INT4 from i32 array and extract nibble
        i32_idx = ks // 8
        bit_shift = (ks % 8) * 4
        word = tl.load(W_Q4 + w_base + i32_idx, mask=mask, other=0)
        q = (word >> bit_shift) & 0xF

        # Dequantize and accumulate
        w = q.to(tl.float32) * scale + zero
        acc += x * w

    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b)


@triton.jit
def linear_q4_add_kernel(X, W_Q4, Scales, Zeros, Bias, Y,
                          K, stride_x, stride_w_q4, n_groups, N,
                          BLOCK_K: tl.constexpr):
    """INT4 fused matvec with residual addition: Y[col] += dot + bias.

    Same as linear_q4_kernel but adds the result to Y instead of
    overwriting it.  Eliminates a separate add_inplace dispatch when
    the matmul output feeds directly into a residual stream.

    Grid: (T, N) — one output element per workgroup.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)

    w_base = col * stride_w_q4
    s_base = col * n_groups

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(n_groups):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K

        scale = tl.load(Scales + s_base + chunk_i).to(tl.float32)
        zero = tl.load(Zeros + s_base + chunk_i).to(tl.float32)

        x = tl.load(X + row * stride_x + ks, mask=mask, other=0.0).to(tl.float32)

        i32_idx = ks // 8
        bit_shift = (ks % 8) * 4
        word = tl.load(W_Q4 + w_base + i32_idx, mask=mask, other=0)
        q = (word >> bit_shift) & 0xF

        w = q.to(tl.float32) * scale + zero
        acc += x * w

    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    r = tl.load(Y + row * N + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b + r)


@triton.jit
def linear_q4_wide_kernel(X, W_Q4, Scales, Zeros, Bias, Y,
                           K, stride_x, stride_w_q4, n_groups, N,
                           grid_y,
                           BLOCK_K: tl.constexpr):
    """INT4 fused matvec with 2D->1D grid mapping for large N > 65535.

    Maps a 2D dispatch grid to a 1D column index:
      col = program_id(0) * grid_y + program_id(1)
    Workgroups with col >= N are discarded.  Used for lm_head where
    N = vocab_size can exceed the 65535 per-dimension dispatch limit.

    Same INT4 dequantization as linear_q4_kernel.
    T is fixed at 1 (decode only).
    Grid: (ceil(N/grid_y), grid_y).
    """
    col = tl.program_id(0) * grid_y + tl.program_id(1)
    valid = col < N

    w_base = col * stride_w_q4
    s_base = col * n_groups

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(n_groups):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K

        scale = tl.load(Scales + s_base + chunk_i, mask=valid, other=0.0).to(tl.float32)
        zero = tl.load(Zeros + s_base + chunk_i, mask=valid, other=0.0).to(tl.float32)

        x = tl.load(X + ks, mask=mask, other=0.0).to(tl.float32)

        i32_idx = ks // 8
        bit_shift = (ks % 8) * 4
        word = tl.load(W_Q4 + w_base + i32_idx, mask=mask & valid, other=0)
        q = (word >> bit_shift) & 0xF

        w = q.to(tl.float32) * scale + zero
        acc += x * w

    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col, mask=valid, other=0.0).to(tl.float32)
    tl.store(Y + col, dot + b, mask=valid)


@triton.jit
def linear_loop_fp16w_wide_kernel(X, W, Bias, Y, K, stride_x, stride_w, N,
                                   grid_y,
                                   BLOCK_K: tl.constexpr):
    """Linear projection with 2D->1D grid mapping for large N > 65535.

    Maps a 2D dispatch grid to a 1D column index:
      col = program_id(0) * grid_y + program_id(1)
    Workgroups with col >= N are discarded.  Used for lm_head where
    N = vocab_size can exceed the 65535 per-dimension dispatch limit.

    W is fp16, accumulation in fp32.  T is fixed at 1 (decode only).
    Grid: (ceil(N/grid_y), grid_y).
    """
    col = tl.program_id(0) * grid_y + tl.program_id(1)
    num_chunks = (K + BLOCK_K - 1) // BLOCK_K
    valid = col < N

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)
        mask = ks < K
        x = tl.load(X + ks, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + col * stride_w + ks, mask=mask & valid, other=0.0).to(tl.float32)
        acc += x * w
    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col, mask=valid, other=0.0).to(tl.float32)
    tl.store(Y + col, dot + b, mask=valid)


@triton.jit
def linear_Wt_fp16_kernel(X, W_T, Bias, Y, K, N, stride_x,
                           BLOCK_N: tl.constexpr):
    """Barrier-free linear with transposed fp16 weights.

    W_T has layout (K, N) row-major — each row of K is contiguous across
    the N dimension.  Each thread independently accumulates one dot
    product over the K dimension, so there is NO inter-thread reduction
    and therefore NO workgroup barriers in the inner loop.

    The loop is manually unrolled by 4 to reduce iteration count and
    increase memory-level parallelism (4 W loads issued per iteration).
    K must be divisible by 4.

    Memory access pattern:
      X: scalar broadcast (same element for all threads per k)
      W_T: coalesced reads (adjacent threads read adjacent fp16 values)

    Grid: (T, ceil(N / BLOCK_N))
    """
    row = tl.program_id(0)
    col_group = tl.program_id(1)
    cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = cols < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    num_chunks = K // 4
    for chunk_i in range(num_chunks):
        k = chunk_i * 4
        x0 = tl.load(X + row * stride_x + k)
        x1 = tl.load(X + row * stride_x + k + 1)
        x2 = tl.load(X + row * stride_x + k + 2)
        x3 = tl.load(X + row * stride_x + k + 3)
        w0 = tl.load(W_T + k * N + cols, mask=valid, other=0.0)
        w1 = tl.load(W_T + (k + 1) * N + cols, mask=valid, other=0.0)
        w2 = tl.load(W_T + (k + 2) * N + cols, mask=valid, other=0.0)
        w3 = tl.load(W_T + (k + 3) * N + cols, mask=valid, other=0.0)
        acc += (x0 * w0.to(tl.float32) + x1 * w1.to(tl.float32)
                + x2 * w2.to(tl.float32) + x3 * w3.to(tl.float32))

    b = tl.load(Bias + cols, mask=valid, other=0.0).to(tl.float32)
    tl.store(Y + row * N + cols, acc + b, mask=valid)


# ---------------------------------------------------------------------------
# Normalization kernels
# ---------------------------------------------------------------------------

@triton.jit
def layer_norm_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps,
                      BLOCK_SIZE: tl.constexpr):
    """Fused LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b.

    Single-pass version for small N where BLOCK_SIZE >= N (power of 2).
    Used by GPT-2 and other models with LayerNorm.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xmean = x - mean
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xmean * rstd
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, y * w + b, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


@triton.jit
def layer_norm_loop_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps,
                           BLOCK: tl.constexpr):
    """Loop-based LayerNorm for arbitrary N.

    Three passes: (1) sum for mean, (2) sum of squared diffs for variance,
    (3) normalize and store.
    Uses deferred reduction: accumulates element-wise across chunks,
    then reduces once per pass (15 barriers instead of num_chunks×15).
    """
    row = tl.program_id(0)
    num_chunks = (N + BLOCK - 1) // BLOCK

    _sum_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        _sum_acc += x
    mean = tl.sum(_sum_acc, axis=0) / N

    _var_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        diff = tl.where(mask, diff, 0.0)
        _var_acc += diff * diff
    var = tl.sum(_var_acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + row * stride + cols, y, mask=mask)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


@triton.jit
def rms_norm_kernel(X, Y, W, Rstd, stride, N, eps,
                    BLOCK_SIZE: tl.constexpr):
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * w.

    Single-pass version for small N where BLOCK_SIZE >= N (power of 2).
    Unlike LayerNorm, RMSNorm has no mean subtraction and no bias.
    Used by LLaMA-family models (SmolLM2, Gemma, Phi, Qwen, etc.).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    tl.store(Y + row * stride + cols, x * rstd * w, mask=mask)
    tl.store(Rstd + row, rstd)


@triton.jit
def rms_norm_loop_kernel(X, Y, W, Rstd, stride, N, eps,
                         BLOCK: tl.constexpr):
    """Loop-based RMSNorm for arbitrary N.

    Two passes: (1) sum of squares for RMS, (2) normalize and store.
    Uses deferred reduction: accumulates element-wise across chunks,
    then reduces once at the end (15 barriers instead of num_chunks×15).
    """
    row = tl.program_id(0)
    num_chunks = (N + BLOCK - 1) // BLOCK

    _ss_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        _ss_acc += x * x
    ms = tl.sum(_ss_acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        tl.store(Y + row * stride + cols, x * rstd * w, mask=mask)

    tl.store(Rstd + row, rstd)


@triton.jit
def add_rms_norm_loop_kernel(X, Residual, Y, W, Rstd, stride, N, eps,
                              BLOCK: tl.constexpr):
    """Fused residual add + RMSNorm: x += residual; y = rms_norm(x) * w.

    Combines add_inplace + rms_norm_loop into one kernel.
    Reads X and Residual, writes updated X and normalized Y.
    Uses deferred reduction for minimal barriers.
    Grid: (T,) — one workgroup per row.
    """
    row = tl.program_id(0)
    num_chunks = (N + BLOCK - 1) // BLOCK

    # Pass 1: add residual to X, accumulate sum of squares
    _ss_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(Residual + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(X + row * stride + cols, x, mask=mask)
        _ss_acc += x * x
    ms = tl.sum(_ss_acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    # Pass 2: normalize and store
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        tl.store(Y + row * stride + cols, x * rstd * w, mask=mask)

    tl.store(Rstd + row, rstd)


# ---------------------------------------------------------------------------
# Activation kernels
# ---------------------------------------------------------------------------

@triton.jit
def gelu_kernel(X, Y, N,
                BLOCK: tl.constexpr):
    """GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).

    Uses exp-based tanh to avoid needing a tanh intrinsic:
    tanh(z) = 1 - 2 / (exp(2z) + 1)

    Used by GPT-2 and other older transformer models.
    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    tanh_val = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def silu_mul_kernel(Gate, Up, Out, N,
                    BLOCK: tl.constexpr):
    """Fused SwiGLU activation: Out = SiLU(Gate) * Up.

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
    Used by LLaMA-family models (SmolLM2, Phi, Qwen, etc.).
    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    gate = tl.load(Gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(Out + offs, silu * up, mask=mask)


@triton.jit
def silu_mul_fused_rows_kernel(GateUp, Out, N,
                               BLOCK: tl.constexpr):
    """Fused SwiGLU from concatenated [gate|up] buffer, multi-row.

    Input GateUp has shape (T, 2*N) with row-major layout.
    Output is (T, N).
    Grid: (T, ceil(N / BLOCK)).
    """
    row = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = row * 2 * N
    gate = tl.load(GateUp + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp + base + N + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(Out + row * N + offs, silu * up, mask=mask)


@triton.jit
def silu_mul_fused_kernel(GateUp, Out, N,
                          BLOCK: tl.constexpr):
    """Fused SwiGLU from concatenated [gate|up] buffer.

    Input GateUp has shape (T, 2*N) where first N columns are gate,
    last N columns are up. Output is (T, N).
    Eliminates the need to split the buffer on CPU between gate_up
    projection and SiLU activation.
    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    gate = tl.load(GateUp + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp + N + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(Out + offs, silu * up, mask=mask)


@triton.jit
def gelu_mul_kernel(Gate, Up, Out, N,
                    BLOCK: tl.constexpr):
    """Fused GeGLU activation: Out = GELU(Gate) * Up.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
    Uses exp-based tanh to avoid needing a tanh intrinsic.
    Used by Gemma models (GeGLU MLP).
    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    gate = tl.load(Gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608 * (gate + 0.044715 * gate * gate * gate)
    tanh_val = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    gelu = 0.5 * gate * (1.0 + tanh_val)
    tl.store(Out + offs, gelu * up, mask=mask)


@triton.jit
def silu_kernel(X, Y, N,
                BLOCK: tl.constexpr):
    """SiLU (Swish) activation: Y = X * sigmoid(X) = X / (1 + exp(-X)).

    Standalone SiLU without multiplication.
    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, x / (1.0 + tl.exp(-x)), mask=mask)


@triton.jit
def sigmoid_kernel(X, Y, N,
                   BLOCK: tl.constexpr):
    """Sigmoid activation: Y = 1 / (1 + exp(-X)).

    Used by SAM mask prediction and diffusion models.
    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, 1.0 / (1.0 + tl.exp(-x)), mask=mask)


@triton.jit
def mul_kernel(X, Y, Out, N,
               BLOCK: tl.constexpr):
    """Element-wise multiply: Out = X * Y.

    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out + offs, x * y, mask=mask)


# ---------------------------------------------------------------------------
# Element-wise kernels
# ---------------------------------------------------------------------------

@triton.jit
def add_kernel(X, Y, Out, N,
               BLOCK: tl.constexpr):
    """Element-wise vector add: Out = X + Y.

    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out + offs, x + y, mask=mask)


@triton.jit
def add_inplace_kernel(X, Y, N,
                       BLOCK: tl.constexpr):
    """In-place vector add: X[i] += Y[i].

    Grid: (ceil(N / BLOCK),) — each workgroup processes BLOCK elements.
    X is modified in-place, avoiding extra buffer allocation.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(X + offs, x + y, mask=mask)


@triton.jit
def mod_scale_shift_kernel(X, Scale, Shift, Out, D, N,
                            BLOCK: tl.constexpr):
    """Modulation: Out[i] = (1 + Scale[i % D]) * X[i] + Shift[i % D].

    X is (T, D) flattened; Scale, Shift are (D,) broadcast over rows.
    N = T * D total elements. D is the hidden dimension.
    Grid: (ceil(N / BLOCK),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    mod_offs = offs % D
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(Scale + mod_offs, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(Shift + mod_offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out + offs, (1.0 + s) * x + sh, mask=mask)


@triton.jit
def gate_residual_add_kernel(Residual, Gate, X, D, N,
                              BLOCK: tl.constexpr):
    """Fused gated residual: Residual[i] += Gate[i % D] * X[i].

    Residual is (T, D) flattened; Gate is (D,) broadcast over rows.
    X is (T, D) flattened. N = T * D.
    In-place update of Residual.
    Grid: (ceil(N / BLOCK),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    mod_offs = offs % D
    r = tl.load(Residual + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate + mod_offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Residual + offs, r + g * x, mask=mask)


@triton.jit
def concat_2d_kernel(A, B, Out, N_a, N_total,
                     BLOCK: tl.constexpr):
    """Concatenate A and B along axis 0 (row concat) on GPU.

    A: first N_a elements; B: remaining (N_total - N_a) elements.
    Out: N_total elements. All are flattened 1-D views.
    Grid: (ceil(N_total / BLOCK),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total
    # For offs < N_a, read from A; for offs >= N_a, read from B[offs - N_a]
    # Load from both and select to avoid boolean negation issues
    a_val = tl.load(A + offs, mask=(offs < N_a), other=0.0).to(tl.float32)
    b_offs = offs - N_a
    b_val = tl.load(B + b_offs, mask=(offs >= N_a) & mask, other=0.0).to(tl.float32)
    val = tl.where(offs < N_a, a_val, b_val)
    tl.store(Out + offs, val, mask=mask)


@triton.jit
def split_copy_kernel(Src, Dst, src_offset, N,
                      BLOCK: tl.constexpr):
    """Copy N elements from Src[src_offset:src_offset+N] into Dst[0:N].

    Used to extract a slice from a GPU buffer without readback.
    Grid: (ceil(N / BLOCK),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    val = tl.load(Src + src_offset + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Dst + offs, val, mask=mask)


# ---------------------------------------------------------------------------
# Attention kernels
# ---------------------------------------------------------------------------

@triton.jit
def causal_attn_kernel(Q, K, V, Out,
                       stride_q, stride_k, stride_v, stride_o,
                       seq_len, scale, neg_inf,
                       BLOCK_HD: tl.constexpr):
    """Single-head causal attention with online softmax.

    Flash-attention style: processes one query position per workgroup,
    iterating over all K/V positions ≤ query position (causal mask).
    Uses online softmax to avoid materializing the full attention matrix.

    Grid: (T,) where T = number of query positions.

    Note: neg_inf is passed as a parameter (not a compile-time constant)
    because the WGSL backend drops constants in `const + tl.zeros([1], ...)`
    expressions, wrongly initializing the loop-carried m_prev to zero.
    """
    q_pos = tl.program_id(0)
    hd = tl.arange(0, BLOCK_HD)
    q = tl.load(Q + q_pos * stride_q + hd).to(tl.float32)
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    m_prev = neg_inf + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)
    for k_pos in range(q_pos + 1):
        k = tl.load(K + k_pos * stride_k + hd).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_score = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_score
        v = tl.load(V + k_pos * stride_v + hd).to(tl.float32)
        acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
        m_prev = m_new
        l_prev = l_new
    tl.store(Out + q_pos * stride_o + hd, acc)


@triton.jit
def causal_attn_multihead_kernel(Q, K, V, Out,
                                 stride_q_t, stride_q_h,
                                 stride_k_t, stride_k_h,
                                 stride_v_t, stride_v_h,
                                 stride_o_t, stride_o_h,
                                 n_rep, scale, neg_inf,
                                 BLOCK_HD: tl.constexpr):
    """Multi-head causal attention with online softmax and GQA.

    Processes all query heads in a single dispatch.
    Grid: (T, n_head) — one workgroup per (position, head).
    KV heads are shared via GQA: kv_head = q_head // n_rep.
    """
    q_pos = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // n_rep
    hd = tl.arange(0, BLOCK_HD)
    q = tl.load(Q + q_pos * stride_q_t + head * stride_q_h + hd).to(tl.float32)
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    m_prev = neg_inf + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)
    for k_pos in range(q_pos + 1):
        k = tl.load(K + k_pos * stride_k_t + kv_head * stride_k_h + hd).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_score = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_score
        v = tl.load(V + k_pos * stride_v_t + kv_head * stride_v_h + hd).to(tl.float32)
        acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
        m_prev = m_new
        l_prev = l_new
    tl.store(Out + q_pos * stride_o_t + head * stride_o_h + hd, acc)


@triton.jit
def full_attn_kernel(Q, K, V, Out,
                     stride_q, stride_k, stride_v, stride_o,
                     seq_len, scale, neg_inf,
                     BLOCK_HD: tl.constexpr):
    """Single-head non-causal (full) attention with online softmax.

    Same as causal_attn_kernel but attends to ALL positions (no mask).
    Used by vision models (SAM, ViT) and diffusion model encoders.

    Grid: (T,) where T = number of query positions.
    """
    q_pos = tl.program_id(0)
    hd = tl.arange(0, BLOCK_HD)
    q = tl.load(Q + q_pos * stride_q + hd).to(tl.float32)
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    m_prev = neg_inf + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)
    for k_pos in range(seq_len):
        k = tl.load(K + k_pos * stride_k + hd).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_score = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_score
        v = tl.load(V + k_pos * stride_v + hd).to(tl.float32)
        acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
        m_prev = m_new
        l_prev = l_new
    tl.store(Out + q_pos * stride_o + hd, acc)


@triton.jit
def full_attn_multihead_kernel(Q, K, V, Out,
                               stride_q_t, stride_q_h,
                               stride_k_t, stride_k_h,
                               stride_v_t, stride_v_h,
                               stride_o_t, stride_o_h,
                               seq_len, scale, neg_inf,
                               BLOCK_HD: tl.constexpr):
    """Multi-head non-causal (full) attention with online softmax.

    Processes all query heads in a single dispatch.
    Grid: (T, n_head) — one workgroup per (position, head).
    No causal mask — attends to ALL positions.
    Used by diffusion transformer (DiT) blocks.
    """
    q_pos = tl.program_id(0)
    head = tl.program_id(1)
    hd = tl.arange(0, BLOCK_HD)
    q = tl.load(Q + q_pos * stride_q_t + head * stride_q_h + hd).to(tl.float32)
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    m_prev = neg_inf + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)
    for k_pos in range(seq_len):
        k = tl.load(K + k_pos * stride_k_t + head * stride_k_h + hd).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_score = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_score
        v = tl.load(V + k_pos * stride_v_t + head * stride_v_h + hd).to(tl.float32)
        acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
        m_prev = m_new
        l_prev = l_new
    tl.store(Out + q_pos * stride_o_t + head * stride_o_h + hd, acc)


@triton.jit
def gqa_decode_attn_kernel(Q, K_cache, V_cache, Out,
                           kv_stride, n_rep,
                           T_total, scale, neg_inf,
                           BLOCK_HD: tl.constexpr):
    """GQA decode attention for T=1 — one workgroup per Q head.

    Computes single-token attention against the full KV cache with
    grouped query attention (GQA): multiple Q heads share one KV head.

    Layout:
      Q:       (n_head * HD,) contiguous — all Q heads for the new token
      K_cache: (MAX_SEQ, n_kv_heads, HD) contiguous — only [0:T_total] valid
      V_cache: same layout as K_cache
      Out:     (n_head * HD,) contiguous

    Grid: (n_head,) — each workgroup handles one Q head.
    kv_stride = n_kv_heads * HD (stride between time steps in KV cache).
    n_rep = n_head // n_kv_heads (GQA repetition factor).
    """
    head = tl.program_id(0)
    kv_head = head // n_rep
    hd = tl.arange(0, BLOCK_HD)

    q = tl.load(Q + head * BLOCK_HD + hd).to(tl.float32)

    # Online softmax over T_total KV positions
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    m_prev = neg_inf + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)

    kv_off = kv_head * BLOCK_HD  # offset for this KV head within each time step

    for t in range(T_total):
        k = tl.load(K_cache + t * kv_stride + kv_off + hd).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_score = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_score
        v = tl.load(V_cache + t * kv_stride + kv_off + hd).to(tl.float32)
        acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
        m_prev = m_new
        l_prev = l_new

    tl.store(Out + head * BLOCK_HD + hd, acc)


@triton.jit
def partial_rope_decode_kernel(X, Y, CosTable, SinTable,
                               src_offset, pos, half_rot,
                               BLOCK_HD: tl.constexpr):
    """Apply partial RoPE to n_heads vectors for T=1 decode.

    Reads n_heads vectors from X starting at src_offset.
    Applies partial RoPE (first rotary_dim = 2*half_rot elements rotated,
    remaining pass through unchanged).
    Cos/Sin are looked up from pre-computed tables at position `pos`.

    Grid: (n_heads,) — one workgroup per head.
    """
    head = tl.program_id(0)
    hd = tl.arange(0, BLOCK_HD)
    src = src_offset + head * BLOCK_HD

    x = tl.load(X + src + hd).to(tl.float32)

    # cos/sin index: both halves of the rotary part map to [0, half_rot)
    cs_idx = hd % half_rot
    is_rotary = hd < half_rot * 2
    cos_v = tl.load(CosTable + pos * half_rot + cs_idx,
                    mask=is_rotary, other=1.0).to(tl.float32)
    sin_v = tl.load(SinTable + pos * half_rot + cs_idx,
                    mask=is_rotary, other=0.0).to(tl.float32)

    # Partner element for rotation:
    #   [0:half_rot)     → partner at [half_rot:2*half_rot)
    #   [half_rot:2*hr)  → partner at [0:half_rot)
    #   [2*hr:HD)        → unused (clamped, masked out)
    partner_idx = tl.where(hd < half_rot, hd + half_rot, hd - half_rot)
    partner_idx = tl.where(is_rotary, partner_idx, hd)
    partner = tl.load(X + src + partner_idx).to(tl.float32)

    # Rotation: first half gets -sin, second half gets +sin
    sign = tl.where(hd < half_rot, -1.0, 1.0)
    rotated = x * cos_v + sign * partner * sin_v
    y = tl.where(is_rotary, rotated, x)

    tl.store(Y + head * BLOCK_HD + hd, y)


@triton.jit
def rope_kv_scatter_kernel(QKV, K_cache, V_cache,
                           CosTable, SinTable,
                           q_size, kv_size, pos, half_rot,
                           cache_offset,
                           BLOCK_HD: tl.constexpr):
    """Apply partial RoPE to K and scatter K+V into GPU KV cache.

    For T=1 decode: reads K from QKV[q_size + kv_head*HD] and
    V from QKV[q_size + kv_size + kv_head*HD], applies RoPE to K,
    then writes K_rot and V at cache_offset in the KV cache buffers.

    Grid: (n_kv_heads,) — one workgroup per KV head.
    """
    kv_head = tl.program_id(0)
    hd = tl.arange(0, BLOCK_HD)

    # K: apply RoPE
    k_src = q_size + kv_head * BLOCK_HD
    k = tl.load(QKV + k_src + hd).to(tl.float32)

    cs_idx = hd % half_rot
    is_rotary = hd < half_rot * 2
    cos_v = tl.load(CosTable + pos * half_rot + cs_idx,
                    mask=is_rotary, other=1.0).to(tl.float32)
    sin_v = tl.load(SinTable + pos * half_rot + cs_idx,
                    mask=is_rotary, other=0.0).to(tl.float32)

    partner_idx = tl.where(hd < half_rot, hd + half_rot, hd - half_rot)
    partner_idx = tl.where(is_rotary, partner_idx, hd)
    partner = tl.load(QKV + k_src + partner_idx).to(tl.float32)

    sign = tl.where(hd < half_rot, -1.0, 1.0)
    rotated = k * cos_v + sign * partner * sin_v
    k_rot = tl.where(is_rotary, rotated, k)

    # Write K_rot to cache at position
    tl.store(K_cache + cache_offset + kv_head * BLOCK_HD + hd, k_rot)

    # V: straight copy to cache
    v_src = q_size + kv_size + kv_head * BLOCK_HD
    v = tl.load(QKV + v_src + hd).to(tl.float32)
    tl.store(V_cache + cache_offset + kv_head * BLOCK_HD + hd, v)


@triton.jit
def fused_rope_qkv_kernel(QKV, Q_out, K_cache, V_cache,
                           CosTable, SinTable,
                           n_head, q_size, kv_size, pos, half_rot,
                           cache_offset,
                           BLOCK_HD: tl.constexpr):
    """Fused RoPE for Q heads + RoPE K + scatter KV in one dispatch.

    Grid: (n_head + n_kv_heads,)
      - pid < n_head:      apply RoPE to Q head, write to Q_out
      - pid >= n_head:      apply RoPE to K head, scatter K+V to cache

    Both paths share the same RoPE logic (cos/sin lookup + rotation).
    """
    pid = tl.program_id(0)
    hd = tl.arange(0, BLOCK_HD)

    # cos/sin (shared by Q and K RoPE)
    cs_idx = hd % half_rot
    is_rotary = hd < half_rot * 2
    cos_v = tl.load(CosTable + pos * half_rot + cs_idx,
                    mask=is_rotary, other=1.0).to(tl.float32)
    sin_v = tl.load(SinTable + pos * half_rot + cs_idx,
                    mask=is_rotary, other=0.0).to(tl.float32)
    sign = tl.where(hd < half_rot, -1.0, 1.0)
    partner_hd = tl.where(hd < half_rot, hd + half_rot, hd - half_rot)
    partner_hd = tl.where(is_rotary, partner_hd, hd)

    is_q = pid < n_head
    # Q heads: read from QKV[pid * HD]
    # K heads: read from QKV[q_size + (pid - n_head) * HD]
    kv_head = pid - n_head
    src = tl.where(is_q, pid * BLOCK_HD, q_size + kv_head * BLOCK_HD)
    x = tl.load(QKV + src + hd).to(tl.float32)
    partner = tl.load(QKV + src + partner_hd).to(tl.float32)

    rotated = x * cos_v + sign * partner * sin_v
    y = tl.where(is_rotary, rotated, x)

    # Q heads: write to Q_out[pid * HD]
    tl.store(Q_out + pid * BLOCK_HD + hd, y, mask=is_q)
    # K heads: write to K_cache[cache_offset + kv_head * HD]
    kv_mask = is_q == 0  # True for KV workgroups
    tl.store(K_cache + cache_offset + kv_head * BLOCK_HD + hd, y,
             mask=kv_mask)

    # V: straight copy to cache (only for KV workgroups)
    v_src = q_size + kv_size + kv_head * BLOCK_HD
    v = tl.load(QKV + v_src + hd, mask=kv_mask, other=0.0).to(tl.float32)
    tl.store(V_cache + cache_offset + kv_head * BLOCK_HD + hd, v,
             mask=kv_mask)


@triton.jit
def partial_rope_prefill_kernel(X, Y, Cos, Sin,
                                x_offset, x_stride_t,
                                y_stride_t, half_rot,
                                BLOCK_HD: tl.constexpr):
    """Apply partial RoPE to multi-token, multi-head input.

    Reads heads from X at x_offset with x_stride_t between tokens.
    Writes to Y with y_stride_t between tokens.
    Cos, Sin: pre-built per-token tables (T * half_rot) flat.
    Grid: (T, n_heads)
    """
    t = tl.program_id(0)
    head = tl.program_id(1)
    hd = tl.arange(0, BLOCK_HD)

    src = x_offset + t * x_stride_t + head * BLOCK_HD
    x = tl.load(X + src + hd).to(tl.float32)

    cs_idx = hd % half_rot
    is_rotary = hd < half_rot * 2
    cos_v = tl.load(Cos + t * half_rot + cs_idx,
                    mask=is_rotary, other=1.0).to(tl.float32)
    sin_v = tl.load(Sin + t * half_rot + cs_idx,
                    mask=is_rotary, other=0.0).to(tl.float32)

    partner_idx = tl.where(hd < half_rot, hd + half_rot, hd - half_rot)
    partner_idx = tl.where(is_rotary, partner_idx, hd)
    partner = tl.load(X + src + partner_idx).to(tl.float32)

    sign = tl.where(hd < half_rot, -1.0, 1.0)
    rotated = x * cos_v + sign * partner * sin_v
    y = tl.where(is_rotary, rotated, x)

    tl.store(Y + t * y_stride_t + head * BLOCK_HD + hd, y)


@triton.jit
def rope_kv_scatter_prefill_kernel(QKV, K_cache, V_cache,
                                   Cos, Sin,
                                   q_size, kv_size, qkv_stride_t,
                                   cache_stride_t, half_rot,
                                   BLOCK_HD: tl.constexpr):
    """Apply RoPE to K and scatter K+V to GPU KV cache, multi-token.

    Reads K and V from QKV buffer, applies partial RoPE to K,
    writes K_rot and V to cache buffers starting at position 0.
    Grid: (T, n_kv_heads)
    """
    t = tl.program_id(0)
    kv_head = tl.program_id(1)
    hd = tl.arange(0, BLOCK_HD)

    # K: apply RoPE
    k_src = t * qkv_stride_t + q_size + kv_head * BLOCK_HD
    k = tl.load(QKV + k_src + hd).to(tl.float32)

    cs_idx = hd % half_rot
    is_rotary = hd < half_rot * 2
    cos_v = tl.load(Cos + t * half_rot + cs_idx,
                    mask=is_rotary, other=1.0).to(tl.float32)
    sin_v = tl.load(Sin + t * half_rot + cs_idx,
                    mask=is_rotary, other=0.0).to(tl.float32)

    partner_idx = tl.where(hd < half_rot, hd + half_rot, hd - half_rot)
    partner_idx = tl.where(is_rotary, partner_idx, hd)
    partner = tl.load(QKV + k_src + partner_idx).to(tl.float32)

    sign = tl.where(hd < half_rot, -1.0, 1.0)
    rotated = k * cos_v + sign * partner * sin_v
    k_rot = tl.where(is_rotary, rotated, k)

    tl.store(K_cache + t * cache_stride_t + kv_head * BLOCK_HD + hd, k_rot)

    # V: straight copy to cache
    v_src = t * qkv_stride_t + q_size + kv_size + kv_head * BLOCK_HD
    v = tl.load(QKV + v_src + hd).to(tl.float32)
    tl.store(V_cache + t * cache_stride_t + kv_head * BLOCK_HD + hd, v)


@triton.jit
def qk_norm_rope_kernel(QKV, Q_out, K_out, V_out,
                         NormQ, NormK, Cos, Sin,
                         n_head, stride_t, eps,
                         BLOCK_HD: tl.constexpr):
    """Fused per-head RMSNorm on Q/K + interleaved-pair RoPE, V passthrough.

    Reads from flat QKV buffer laid out as (T, 3*n_head, HD).
    Grid: (T, n_head).

    QKV per token: [q_h0..q_hN, k_h0..k_hN, v_h0..v_hN], each HD wide.

    RoPE uses FLUX interleaved-pair convention:
      partner = hd ^ 1 (swap even/odd within pairs)
      sign = -1 for even, +1 for odd
      out = normed * cos + sign * normed_partner * sin
    cos, sin: (T, HD) row-major.
    """
    t = tl.program_id(0)
    head = tl.program_id(1)
    hd = tl.arange(0, BLOCK_HD)

    nhd = n_head * BLOCK_HD  # stride between q/k/v sections per token
    partner = hd ^ 1  # swap even/odd: 0<->1, 2<->3, ...

    # --- load Q, K, V and partners for this (t, head) ---
    base = t * stride_t + head * BLOCK_HD
    q = tl.load(QKV + base + hd).to(tl.float32)
    q_part = tl.load(QKV + base + partner).to(tl.float32)
    k = tl.load(QKV + base + nhd + hd).to(tl.float32)
    k_part = tl.load(QKV + base + nhd + partner).to(tl.float32)
    v = tl.load(QKV + base + 2 * nhd + hd).to(tl.float32)

    # --- RMSNorm on Q ---
    q_mean = tl.sum(q * q, axis=0) / BLOCK_HD
    q_rstd = 1.0 / tl.sqrt(q_mean + eps)
    wq = tl.load(NormQ + hd).to(tl.float32)
    wq_part = tl.load(NormQ + partner).to(tl.float32)
    q_n = q * q_rstd * wq
    q_n_part = q_part * q_rstd * wq_part

    # --- RMSNorm on K ---
    k_mean = tl.sum(k * k, axis=0) / BLOCK_HD
    k_rstd = 1.0 / tl.sqrt(k_mean + eps)
    wk = tl.load(NormK + hd).to(tl.float32)
    wk_part = tl.load(NormK + partner).to(tl.float32)
    k_n = k * k_rstd * wk
    k_n_part = k_part * k_rstd * wk_part

    # --- RoPE (interleaved-pair: [-imag, real] pattern) ---
    cos_v = tl.load(Cos + t * BLOCK_HD + hd).to(tl.float32)
    sin_v = tl.load(Sin + t * BLOCK_HD + hd).to(tl.float32)
    sign = tl.where(hd % 2 == 0, -1.0, 1.0)

    q_out = q_n * cos_v + sign * q_n_part * sin_v
    k_out = k_n * cos_v + sign * k_n_part * sin_v

    # --- write (T, n_head, HD) layout ---
    out_base = t * n_head * BLOCK_HD + head * BLOCK_HD
    tl.store(Q_out + out_base + hd, q_out)
    tl.store(K_out + out_base + hd, k_out)
    tl.store(V_out + out_base + hd, v)


@triton.jit
def group_norm_kernel(X, Y, W, B, Rstd, stride, N, num_groups, eps,
                      BLOCK: tl.constexpr):
    """GroupNorm: normalize within each group of channels.

    Each workgroup processes one (sample, group) pair.
    Grid: (num_rows * num_groups,) where num_rows is batch*spatial.
    N is the number of channels per group.

    Used by UNet (SDXL) and diffusion models.
    """
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups
    group_off = group * N
    num_chunks = (N + BLOCK - 1) // BLOCK

    # Compute mean
    _sum = tl.zeros([1], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + group_off + cols,
                     mask=mask, other=0.0).to(tl.float32)
        _sum += tl.sum(x, axis=0)
    mean = tl.sum(_sum, axis=0) / N

    # Compute variance
    _var = tl.zeros([1], dtype=tl.float32)
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + group_off + cols,
                     mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        diff = tl.where(mask, diff, 0.0)
        _var += tl.sum(diff * diff, axis=0)
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for chunk_i in range(num_chunks):
        off = chunk_i * BLOCK
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + group_off + cols,
                     mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + group_off + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + group_off + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + row * stride + group_off + cols, y, mask=mask)

    tl.store(Rstd + pid, rstd)


# ---------------------------------------------------------------------------
# MXFP4 fused matmul kernel (for GPT-OSS MoE experts)
# ---------------------------------------------------------------------------

@triton.jit
def linear_mxfp4_kernel(X, W_blocks, W_scales, Bias, Y,
                         K, stride_x, stride_blocks, stride_scales, N,
                         BLOCK_K: tl.constexpr):
    """MXFP4 fused matvec: dequantize MX FP4 weights on-the-fly.

    W_blocks: packed i32, flat (N, K//8).  Each i32 = 4 bytes = 8 FP4 nibbles.
              nibble n at bit_shift = (n % 8) * 4.
    W_scales: packed i32, flat (N, ceil(K/128)).  4 E8M0 bytes packed per i32.
              E8M0: scale = 2^(byte_val - 127).
              One scale per 32 FP4 elements (MXFP4 block size).
    Bias:     fp32, flat (N,).
    BLOCK_K:  must be 32 (MXFP4 block size).

    FP4 E2M1 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}

    Grid: (T, N) — one output element per workgroup.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)

    n_chunks = K // BLOCK_K
    blocks_base = col * stride_blocks
    scales_base = col * stride_scales

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for chunk_i in range(n_chunks):
        off = chunk_i * BLOCK_K
        ks = off + tl.arange(0, BLOCK_K)

        # Load X (fp32)
        x = tl.load(X + row * stride_x + ks).to(tl.float32)

        # Load packed i32 words and extract FP4 nibble
        word = tl.load(W_blocks + blocks_base + ks // 8)
        bit_shift = (ks % 8) * 4
        nibble = (word >> bit_shift) & 0xF

        # FP4 E2M1 decode
        sign_bit = (nibble >> 3) & 1
        abs_nib = nibble & 0x7
        exp = abs_nib >> 1
        mant = abs_nib & 1

        pow2 = tl.where(exp == 1, 0.5,
               tl.where(exp == 2, 1.0,
               tl.where(exp == 3, 2.0, 0.5)))
        abs_val = tl.where(exp > 0,
                           (2.0 + mant.to(tl.float32)) * pow2,
                           mant.to(tl.float32) * 0.5)
        w = abs_val * (1.0 - 2.0 * sign_bit.to(tl.float32))

        # Load E8M0 scale from packed i32 (4 bytes per word)
        scale_word = tl.load(W_scales + scales_base + chunk_i // 4)
        scale_byte = (scale_word >> ((chunk_i % 4) * 8)) & 0xFF
        scale = tl.exp((scale_byte.to(tl.float32) - 127.0) * 0.6931471805599453)

        acc += x * w * scale

    dot = tl.sum(acc, axis=0)
    b = tl.load(Bias + col).to(tl.float32)
    tl.store(Y + row * N + col, dot + b)


# ---------------------------------------------------------------------------
# GPT-OSS gated activation kernel
# ---------------------------------------------------------------------------

@triton.jit
def gqa_decode_attn_sink_kernel(Q, K_cache, V_cache, Sinks, Out,
                                 kv_stride, kv_start, n_rep,
                                 T_win, scale, neg_inf,
                                 BLOCK_HD: tl.constexpr):
    """GQA decode attention with attention sinks and sliding window.

    Like gqa_decode_attn_kernel but adds:
    - Sliding window: only attends to KV positions [kv_start, kv_start+T_win)
    - Attention sinks: per-head learnable logit that competes in softmax
      but contributes no V — it absorbs probability mass.

    Layout:
      Q:       (n_head * HD,) contiguous
      K_cache: (MAX_SEQ, n_kv_heads, HD) contiguous
      V_cache: same layout
      Sinks:   (n_head,) float32 — learnable sink logits
      Out:     (n_head * HD,) contiguous

    Grid: (n_head,)
    """
    head = tl.program_id(0)
    kv_head = head // n_rep
    hd = tl.arange(0, BLOCK_HD)

    q = tl.load(Q + head * BLOCK_HD + hd).to(tl.float32)

    # Online softmax over T_win KV positions
    acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
    m_prev = neg_inf + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)

    kv_off = kv_head * BLOCK_HD

    for t in range(T_win):
        k = tl.load(K_cache + (kv_start + t) * kv_stride + kv_off + hd).to(tl.float32)
        score = tl.sum(q * k, axis=0) * scale
        m_new = tl.maximum(m_prev, score)
        exp_prev = tl.exp(m_prev - m_new)
        exp_score = tl.exp(score - m_new)
        l_new = l_prev * exp_prev + exp_score
        v = tl.load(V_cache + (kv_start + t) * kv_stride + kv_off + hd).to(tl.float32)
        acc = acc * (l_prev * exp_prev / l_new) + v * (exp_score / l_new)
        m_prev = m_new
        l_prev = l_new

    # Incorporate sink logit (steals probability mass, no V contribution)
    sink_logit = tl.load(Sinks + head).to(tl.float32)
    m_new = tl.maximum(m_prev, sink_logit)
    exp_prev = tl.exp(m_prev - m_new)
    exp_sink = tl.exp(sink_logit - m_new)
    l_new = l_prev * exp_prev + exp_sink
    # Re-weight accumulated V (sink contributes no V vector)
    acc = acc * (l_prev * exp_prev / l_new)

    tl.store(Out + head * BLOCK_HD + hd, acc)


@triton.jit
def add_scaled_kernel(Acc, X, alpha, N, BLOCK: tl.constexpr):
    """Acc += alpha * X  (AXPY operation for weighted accumulation).

    Grid: (ceil(N/BLOCK),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(Acc + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Acc + offs, a + alpha * x, mask=mask)


@triton.jit
def gptoss_gate_kernel(X, Y, N, BLOCK: tl.constexpr):
    """GPT-OSS gated activation with interleaved gate/up layout.

    Input X: (T, 2*N) with gate = X[..., ::2], up = X[..., 1::2]
    Output Y: (T, N)

    gate = clamp(gate, max=7.0)
    up = clamp(up, min=-7.0, max=7.0)
    glu = gate * sigmoid(gate * 1.702)
    y = (up + 1) * glu

    Grid: (ceil(N/BLOCK),) for T=1
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    gate = tl.load(X + offs * 2, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)

    # Clamp
    gate = tl.where(gate > 7.0, 7.0, gate)
    up = tl.where(up > 7.0, 7.0, tl.where(up < -7.0, -7.0, up))

    # Gated activation: gate * sigmoid(gate * alpha) * (up + 1)
    glu = gate * (1.0 / (1.0 + tl.exp(-gate * 1.702)))
    y = (up + 1.0) * glu

    tl.store(Y + offs, y, mask=mask)

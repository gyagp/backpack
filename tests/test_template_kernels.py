#!/usr/bin/env python3
"""Unit tests for dtype-polymorphic template WGSL kernels.

Tests each template kernel (containing ${T} markers) in both f32 and f16 modes.
f32 tests validate correctness against numpy.
f16 tests validate that the packed u32 storage path produces correct results
with relaxed tolerance (fp16 precision).

Usage:
    python -m pytest tests/test_template_kernels.py -v
    python -m pytest tests/test_template_kernels.py -k "test_binary" -v
    python -m pytest tests/test_template_kernels.py -k "f16" -v
"""
import os, sys, math
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from triton.backends.webgpu.dawn_runner import DawnRunner, BufferBinding

KERNEL_DIR = os.path.join(_ROOT, "runtime", "kernels")

# ── Template instantiation (mirrors wgsl_template.h) ─────────────────────────

T_READ_F32 = """
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}
"""
T_READ_RW_F32 = """
fn t_read_rw(buf: ptr<storage, array<f32>, read_write>, idx: u32) -> f32 {
    return (*buf)[idx];
}
"""
T_WRITE_F32 = """
fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}
"""
T_WRITE2_F32 = """
fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}
"""

T_READ_F16 = """
fn t_read(buf: ptr<storage, array<u32>, read>, idx: u32) -> f32 {
    let packed = (*buf)[idx / 2u];
    let pair = unpack2x16float(packed);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
"""
T_READ_RW_F16 = """
fn t_read_rw(buf: ptr<storage, array<u32>, read_write>, idx: u32) -> f32 {
    let packed = (*buf)[idx / 2u];
    let pair = unpack2x16float(packed);
    return select(pair.x, pair.y, (idx & 1u) != 0u);
}
"""
T_WRITE_F16 = """
fn t_write(buf: ptr<storage, array<u32>, read_write>, idx: u32, val: f32) {
    let u32_idx = idx / 2u;
    let existing = (*buf)[u32_idx];
    let pair = unpack2x16float(existing);
    if ((idx & 1u) == 0u) {
        (*buf)[u32_idx] = pack2x16float(vec2<f32>(val, pair.y));
    } else {
        (*buf)[u32_idx] = pack2x16float(vec2<f32>(pair.x, val));
    }
}
"""
T_WRITE2_F16 = """
fn t_write2(buf: ptr<storage, array<u32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx / 2u] = pack2x16float(vec2<f32>(v0, v1));
}
"""


def instantiate_f32(src: str) -> str:
    s = src.replace("${T}", "f32")
    s = s.replace("${T_DTYPE}", "f32")
    s = s.replace("${T_BYTES}", "4")
    s = s.replace("${T_READ}", T_READ_F32)
    s = s.replace("${T_READ_RW}", T_READ_RW_F32)
    s = s.replace("${T_WRITE2}", T_WRITE2_F32)
    s = s.replace("${T_WRITE}", T_WRITE_F32)
    return s


def instantiate_f16(src: str) -> str:
    s = src.replace("${T}", "u32")
    s = s.replace("${T_DTYPE}", "f16")
    s = s.replace("${T_BYTES}", "2")
    s = s.replace("${T_READ}", T_READ_F16)
    s = s.replace("${T_READ_RW}", T_READ_RW_F16)
    s = s.replace("${T_WRITE2}", T_WRITE2_F16)
    s = s.replace("${T_WRITE}", T_WRITE_F16)
    return s


# ── fp16 packing helpers ─────────────────────────────────────────────────────

def pack_f16(arr_f32: np.ndarray) -> np.ndarray:
    """Pack f32 array into u32 array with 2 fp16 values per u32.
    Mimics WGSL pack2x16float: u32 = low_f16 | (high_f16 << 16)."""
    f16 = arr_f32.astype(np.float16)
    if len(f16) % 2:
        f16 = np.append(f16, np.float16(0))
    u16 = f16.view(np.uint16)
    return (u16[0::2].astype(np.uint32) | (u16[1::2].astype(np.uint32) << 16))


def unpack_f16(arr_u32: np.ndarray, count: int) -> np.ndarray:
    """Unpack u32 array to f32, interpreting each u32 as 2 fp16 values.
    Mimics WGSL unpack2x16float."""
    lo = (arr_u32 & 0xFFFF).astype(np.uint16)
    hi = ((arr_u32 >> 16) & 0xFFFF).astype(np.uint16)
    result = np.empty(len(arr_u32) * 2, dtype=np.float16)
    result[0::2] = lo.view(np.float16)
    result[1::2] = hi.view(np.float16)
    return result[:count].astype(np.float32)


# ── Shared helpers ────────────────────────────────────────────────────────────

_runner = None

def get_runner() -> DawnRunner:
    global _runner
    if _runner is None:
        _runner = DawnRunner()
    return _runner


def load_template(name: str) -> str:
    path = os.path.join(KERNEL_DIR, "shared", name + ".wgsl")
    with open(path) as f:
        return f.read()


def bb(binding, name, elem_type="f32", access="read"):
    return BufferBinding(binding=binding, name=name,
                         elem_type=elem_type, access=access)


def make_params(*vals):
    return np.array(vals, dtype=np.uint32)


def f32_as_u32(x):
    return np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]


def ceil_div(a, b):
    return (a + b - 1) // b


def dispatch(wgsl, bindings, buffers, grid, workgroup_size=256):
    return get_runner().run_kernel(
        wgsl_code=wgsl,
        buffer_bindings=bindings,
        param_fields=[],
        workgroup_size=workgroup_size,
        grid=grid,
        buffers=buffers,
    )


# ── Binary elementwise ────────────────────────────────────────────────────────

@pytest.mark.parametrize("op_code,name,ref_fn", [
    (0, "add", lambda a, b: a + b),
    (1, "sub", lambda a, b: a - b),
    (2, "mul", lambda a, b: a * b),
    (3, "div", lambda a, b: a / b),
], ids=["add", "sub", "mul", "div"])
def test_binary_elementwise_f32(op_code, name, ref_fn):
    tmpl = load_template("binary_elementwise")
    wgsl = instantiate_f32(tmpl)
    N = 64
    np.random.seed(42)
    A = np.random.randn(N).astype(np.float32)
    B = np.random.uniform(0.5, 2.0, N).astype(np.float32)
    params = make_params(N, op_code, N, N)  # N, op, A_N, B_N

    out = dispatch(wgsl,
        [bb(0, "A"), bb(1, "B"), bb(2, "C", access="read_write"),
         bb(3, "_params_", "u32")],
        {"A": A, "B": B, "C": np.zeros(N, np.float32), "_params_": params},
        grid=(ceil_div(N, 512), 1, 1))

    np.testing.assert_allclose(out["C"], ref_fn(A, B), atol=1e-5,
                               err_msg=f"binary {name} f32")


@pytest.mark.parametrize("op_code,name,ref_fn", [
    (0, "add", lambda a, b: a + b),
    (2, "mul", lambda a, b: a * b),
], ids=["add", "mul"])
def test_binary_elementwise_f16(op_code, name, ref_fn):
    tmpl = load_template("binary_elementwise")
    wgsl = instantiate_f16(tmpl)
    N = 64
    np.random.seed(42)
    A = np.random.uniform(-2.0, 2.0, N).astype(np.float32)
    B = np.random.uniform(0.5, 2.0, N).astype(np.float32)
    params = make_params(N, op_code, N, N)
    n_u32 = ceil_div(N, 2)

    out = dispatch(wgsl,
        [bb(0, "A", "u32"), bb(1, "B", "u32"), bb(2, "C", "u32", "read_write"),
         bb(3, "_params_", "u32")],
        {"A": pack_f16(A), "B": pack_f16(B),
         "C": np.zeros(n_u32, np.uint32), "_params_": params},
        grid=(ceil_div(N, 512), 1, 1))

    result = unpack_f16(out["C"], N)
    # Reference in fp16 precision
    A16 = A.astype(np.float16).astype(np.float32)
    B16 = B.astype(np.float16).astype(np.float32)
    expected = ref_fn(A16, B16)
    np.testing.assert_allclose(result, expected, atol=0.05, rtol=0.01,
                               err_msg=f"binary {name} f16")


def test_binary_broadcast_f32():
    tmpl = load_template("binary_elementwise")
    wgsl = instantiate_f32(tmpl)
    N = 16
    A = np.arange(N, dtype=np.float32)
    B = np.array([10.0], dtype=np.float32)
    params = make_params(N, 0, N, 1)  # N, op=add, A_N=N, B_N=1

    out = dispatch(wgsl,
        [bb(0, "A"), bb(1, "B"), bb(2, "C", access="read_write"),
         bb(3, "_params_", "u32")],
        {"A": A, "B": B, "C": np.zeros(N, np.float32), "_params_": params},
        grid=(ceil_div(N, 512), 1, 1))

    np.testing.assert_allclose(out["C"], A + 10.0, atol=1e-6)


# ── Unary elementwise ─────────────────────────────────────────────────────────

def ref_sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def ref_silu(x): return x * ref_sigmoid(x)
def ref_gelu(x): return 0.5 * x * (1 + np.vectorize(math.erf)(x * 0.7071067811865476))
def ref_softplus(x): return np.log(1 + np.exp(x))

UNARY_OPS = [
    (0, "sigmoid", ref_sigmoid),
    (1, "tanh", np.tanh),
    (2, "neg", lambda x: -x),
    (10, "relu", lambda x: np.maximum(x, 0)),
    (11, "exp", np.exp),
    (13, "abs", np.abs),
]

@pytest.mark.parametrize("op_code,name,ref_fn", UNARY_OPS,
                         ids=[t[1] for t in UNARY_OPS])
def test_unary_elementwise_f32(op_code, name, ref_fn):
    tmpl = load_template("unary_elementwise")
    wgsl = instantiate_f32(tmpl)
    N = 64
    np.random.seed(42)
    if name == "exp":
        A = np.random.uniform(-2.0, 2.0, N).astype(np.float32)
    else:
        A = np.random.randn(N).astype(np.float32)
    params = make_params(N, op_code)

    out = dispatch(wgsl,
        [bb(0, "A"), bb(1, "C", access="read_write"), bb(2, "_params_", "u32")],
        {"A": A, "C": np.zeros(N, np.float32), "_params_": params},
        grid=(ceil_div(N, 512), 1, 1))

    np.testing.assert_allclose(out["C"], ref_fn(A), atol=2e-5, rtol=1e-4,
                               err_msg=f"unary {name} f32")


@pytest.mark.parametrize("op_code,name,ref_fn", [
    (0, "sigmoid", ref_sigmoid),
    (10, "relu", lambda x: np.maximum(x, 0)),
], ids=["sigmoid", "relu"])
def test_unary_elementwise_f16(op_code, name, ref_fn):
    tmpl = load_template("unary_elementwise")
    wgsl = instantiate_f16(tmpl)
    N = 64
    np.random.seed(42)
    A = np.random.uniform(-2.0, 2.0, N).astype(np.float32)
    params = make_params(N, op_code)
    n_u32 = ceil_div(N, 2)

    out = dispatch(wgsl,
        [bb(0, "A", "u32"), bb(1, "C", "u32", "read_write"),
         bb(2, "_params_", "u32")],
        {"A": pack_f16(A), "C": np.zeros(n_u32, np.uint32), "_params_": params},
        grid=(ceil_div(N, 512), 1, 1))

    result = unpack_f16(out["C"], N)
    A16 = A.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(result, ref_fn(A16), atol=0.05, rtol=0.01,
                               err_msg=f"unary {name} f16")


# ── Softmax ───────────────────────────────────────────────────────────────────

def test_softmax_f32():
    tmpl = load_template("softmax")
    wgsl = instantiate_f32(tmpl)
    rows, cols = 4, 8
    np.random.seed(42)
    X = np.random.randn(rows, cols).astype(np.float32)
    params = make_params(rows, cols)

    out = dispatch(wgsl,
        [bb(0, "X"), bb(1, "Y", access="read_write"), bb(2, "_params_", "u32")],
        {"X": X.ravel(), "Y": np.zeros(rows * cols, np.float32),
         "_params_": params},
        grid=(ceil_div(rows, 256), 1, 1))

    e = np.exp(X - X.max(axis=1, keepdims=True))
    expected = (e / e.sum(axis=1, keepdims=True)).ravel()
    np.testing.assert_allclose(out["Y"], expected, atol=1e-5)


def test_softmax_f16():
    tmpl = load_template("softmax")
    wgsl = instantiate_f16(tmpl)
    rows, cols = 4, 8
    np.random.seed(42)
    X = np.random.randn(rows, cols).astype(np.float32)
    total = rows * cols
    n_u32 = ceil_div(total, 2)
    params = make_params(rows, cols)

    out = dispatch(wgsl,
        [bb(0, "X", "u32"), bb(1, "Y", "u32", "read_write"),
         bb(2, "_params_", "u32")],
        {"X": pack_f16(X.ravel()), "Y": np.zeros(n_u32, np.uint32),
         "_params_": params},
        grid=(ceil_div(rows, 256), 1, 1))

    result = unpack_f16(out["Y"], total).reshape(rows, cols)
    # Reference in fp16 precision
    X16 = X.astype(np.float16).astype(np.float32)
    e = np.exp(X16 - X16.max(axis=1, keepdims=True))
    expected = e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result, expected, atol=0.05, rtol=0.02)


# ── Layer norm ────────────────────────────────────────────────────────────────

def test_layer_norm_f32():
    tmpl = load_template("layer_norm")
    wgsl = instantiate_f32(tmpl)
    nRows, N = 3, 8
    np.random.seed(42)
    X = np.random.randn(nRows, N).astype(np.float32)
    W = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)
    eps = 1e-5
    params = make_params(N, nRows, f32_as_u32(eps))

    out = dispatch(wgsl,
        [bb(0, "X"), bb(1, "W"), bb(2, "B"),
         bb(3, "Y", access="read_write"), bb(4, "_params_", "u32")],
        {"X": X.ravel(), "W": W, "B": B,
         "Y": np.zeros(nRows * N, np.float32), "_params_": params},
        grid=(ceil_div(nRows, 256), 1, 1))

    # Reference
    mean = X.mean(axis=1, keepdims=True)
    var = X.var(axis=1, keepdims=True)
    expected = ((X - mean) / np.sqrt(var + eps) * W + B).ravel()
    np.testing.assert_allclose(out["Y"], expected, atol=1e-4)


def test_layer_norm_f16():
    tmpl = load_template("layer_norm")
    wgsl = instantiate_f16(tmpl)
    nRows, N = 2, 8
    np.random.seed(42)
    X = np.random.uniform(-1.0, 1.0, (nRows, N)).astype(np.float32)
    W = np.ones(N, dtype=np.float32)
    B = np.zeros(N, dtype=np.float32)
    eps = 1e-5
    total = nRows * N
    n_u32 = ceil_div(total, 2)
    n_u32_n = ceil_div(N, 2)
    params = make_params(N, nRows, f32_as_u32(eps))

    out = dispatch(wgsl,
        [bb(0, "X", "u32"), bb(1, "W", "u32"), bb(2, "B", "u32"),
         bb(3, "Y", "u32", "read_write"), bb(4, "_params_", "u32")],
        {"X": pack_f16(X.ravel()), "W": pack_f16(W), "B": pack_f16(B),
         "Y": np.zeros(n_u32, np.uint32), "_params_": params},
        grid=(ceil_div(nRows, 256), 1, 1))

    result = unpack_f16(out["Y"], total).reshape(nRows, N)
    X16 = X.astype(np.float16).astype(np.float32)
    mean = X16.mean(axis=1, keepdims=True)
    var = X16.var(axis=1, keepdims=True)
    expected = (X16 - mean) / np.sqrt(var + eps)
    np.testing.assert_allclose(result, expected, atol=0.1, rtol=0.05)


# ── Scale ─────────────────────────────────────────────────────────────────────

def test_scale_f32():
    tmpl = load_template("scale")
    wgsl = instantiate_f32(tmpl)
    N = 32
    data = np.arange(N, dtype=np.float32) + 1.0
    scale_val = 0.5
    params = make_params(N, f32_as_u32(scale_val))

    out = dispatch(wgsl,
        [bb(0, "data", access="read_write"), bb(1, "params", "u32")],
        {"data": data.copy(), "params": params},
        grid=(ceil_div(N, 512), 1, 1))

    np.testing.assert_allclose(out["data"], data * scale_val, atol=1e-6)


def test_scale_f16():
    tmpl = load_template("scale")
    wgsl = instantiate_f16(tmpl)
    N = 32
    data = (np.arange(N, dtype=np.float32) + 1.0) * 0.1  # small range for f16
    scale_val = 0.5
    n_u32 = ceil_div(N, 2)
    params = make_params(N, f32_as_u32(scale_val))

    out = dispatch(wgsl,
        [bb(0, "data", "u32", "read_write"), bb(1, "params", "u32")],
        {"data": pack_f16(data), "params": params},
        grid=(ceil_div(N, 512), 1, 1))

    result = unpack_f16(out["data"], N)
    data16 = data.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(result, data16 * scale_val, atol=0.01)


# ── Expand ────────────────────────────────────────────────────────────────────

def test_expand_f32():
    tmpl = load_template("expand")
    wgsl = instantiate_f32(tmpl)
    X = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    total = 12  # [1,4] → [3,4]
    out_strides = [4, 1]
    in_dims = [1, 4]
    in_strides = [4, 1]
    params = make_params(total, 2, 0, 0, *out_strides, *in_dims, *in_strides)

    out = dispatch(wgsl,
        [bb(0, "X"), bb(1, "Y", access="read_write"), bb(2, "_params_", "u32")],
        {"X": X, "Y": np.zeros(total, np.float32), "_params_": params},
        grid=(ceil_div(total, 512), 1, 1))

    np.testing.assert_allclose(out["Y"], np.tile(X, 3), atol=1e-6)


def test_expand_f16():
    tmpl = load_template("expand")
    wgsl = instantiate_f16(tmpl)
    X = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    total = 12
    out_strides = [4, 1]
    in_dims = [1, 4]
    in_strides = [4, 1]
    params = make_params(total, 2, 0, 0, *out_strides, *in_dims, *in_strides)
    n_u32_out = ceil_div(total, 2)
    n_u32_in = ceil_div(4, 2)

    out = dispatch(wgsl,
        [bb(0, "X", "u32"), bb(1, "Y", "u32", "read_write"),
         bb(2, "_params_", "u32")],
        {"X": pack_f16(X), "Y": np.zeros(n_u32_out, np.uint32),
         "_params_": params},
        grid=(ceil_div(total, 512), 1, 1))

    result = unpack_f16(out["Y"], total)
    np.testing.assert_allclose(result, np.tile(X, 3), atol=0.01)


# ── Resize nearest ───────────────────────────────────────────────────────────

def test_resize_nearest_f32():
    tmpl = load_template("resize_nearest")
    wgsl = instantiate_f32(tmpl)
    N, C, H_in, W_in = 1, 1, 2, 2
    H_out, W_out = 4, 4
    X = np.array([1, 2, 3, 4], dtype=np.float32)
    total = N * C * H_out * W_out
    params = make_params(N, C, H_in, W_in, H_out, W_out, 0, 0)

    out = dispatch(wgsl,
        [bb(0, "X"), bb(1, "Y", access="read_write"), bb(2, "_params_", "u32")],
        {"X": X, "Y": np.zeros(total, np.float32), "_params_": params},
        grid=(ceil_div(total, 512), 1, 1))

    expected = np.array([1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4], dtype=np.float32)
    np.testing.assert_allclose(out["Y"], expected, atol=1e-6)


def test_resize_nearest_f16():
    tmpl = load_template("resize_nearest")
    wgsl = instantiate_f16(tmpl)
    N, C, H_in, W_in = 1, 1, 2, 2
    H_out, W_out = 4, 4
    X = np.array([1, 2, 3, 4], dtype=np.float32)
    total = N * C * H_out * W_out
    params = make_params(N, C, H_in, W_in, H_out, W_out, 0, 0)
    n_u32_out = ceil_div(total, 2)

    out = dispatch(wgsl,
        [bb(0, "X", "u32"), bb(1, "Y", "u32", "read_write"),
         bb(2, "_params_", "u32")],
        {"X": pack_f16(X), "Y": np.zeros(n_u32_out, np.uint32),
         "_params_": params},
        grid=(ceil_div(total, 512), 1, 1))

    result = unpack_f16(out["Y"], total)
    expected = np.array([1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=0.01)


# ── Where select ──────────────────────────────────────────────────────────────

def test_where_select_f32():
    tmpl = load_template("where_select")
    wgsl = instantiate_f32(tmpl)
    N = 8
    cond_bytes = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    cond_u32 = cond_bytes.view(np.uint32)
    A = np.ones(N, np.float32) * 10.0
    B = np.ones(N, np.float32) * 20.0
    params = make_params(N, N, N, N)  # N, N_cond, N_x, N_y

    out = dispatch(wgsl,
        [bb(0, "Cond", "u32"), bb(1, "X"), bb(2, "Y"),
         bb(3, "Out", access="read_write"), bb(4, "_params_", "u32")],
        {"Cond": cond_u32, "X": A, "Y": B,
         "Out": np.zeros(N, np.float32), "_params_": params},
        grid=(ceil_div(N, 512), 1, 1))

    expected = np.where(cond_bytes > 0, A, B)
    np.testing.assert_allclose(out["Out"], expected, atol=1e-6)


def test_where_select_f16():
    tmpl = load_template("where_select")
    wgsl = instantiate_f16(tmpl)
    N = 8
    cond_bytes = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    cond_u32 = cond_bytes.view(np.uint32)
    A = np.ones(N, np.float32) * 10.0
    B = np.ones(N, np.float32) * 20.0
    params = make_params(N, N, N, N)
    n_u32 = ceil_div(N, 2)

    out = dispatch(wgsl,
        [bb(0, "Cond", "u32"), bb(1, "X", "u32"), bb(2, "Y", "u32"),
         bb(3, "Out", "u32", "read_write"), bb(4, "_params_", "u32")],
        {"Cond": cond_u32, "X": pack_f16(A), "Y": pack_f16(B),
         "Out": np.zeros(n_u32, np.uint32), "_params_": params},
        grid=(ceil_div(N, 512), 1, 1))

    result = unpack_f16(out["Out"], N)
    expected = np.where(cond_bytes > 0, A, B)
    np.testing.assert_allclose(result, expected, atol=0.01)


# ── Embed gather ──────────────────────────────────────────────────────────────

def test_embed_gather_f32():
    tmpl = load_template("embed_gather")
    wgsl = instantiate_f32(tmpl)
    vocab, E = 10, 16
    np.random.seed(42)
    table = np.random.randn(vocab * E).astype(np.float32)
    token_id = 3
    params = make_params(E)

    out = dispatch(wgsl,
        [bb(0, "EmbeddingTable"), bb(1, "TokenId", "i32"),
         bb(2, "X", access="read_write"), bb(3, "_params_", "u32")],
        {"EmbeddingTable": table,
         "TokenId": np.array([token_id], dtype=np.int32),
         "X": np.zeros(E, np.float32), "_params_": params},
        grid=(ceil_div(E, 512), 1, 1))

    expected = table[token_id * E:(token_id + 1) * E]
    np.testing.assert_allclose(out["X"], expected, atol=1e-6)


def test_embed_gather_f16():
    tmpl = load_template("embed_gather")
    wgsl = instantiate_f16(tmpl)
    vocab, E = 10, 16
    np.random.seed(42)
    table = np.random.randn(vocab * E).astype(np.float32)
    token_id = 3
    params = make_params(E)
    n_u32_table = ceil_div(vocab * E, 2)
    n_u32_out = ceil_div(E, 2)

    out = dispatch(wgsl,
        [bb(0, "EmbeddingTable", "u32"), bb(1, "TokenId", "i32"),
         bb(2, "X", "u32", "read_write"), bb(3, "_params_", "u32")],
        {"EmbeddingTable": pack_f16(table),
         "TokenId": np.array([token_id], dtype=np.int32),
         "X": np.zeros(n_u32_out, np.uint32), "_params_": params},
        grid=(ceil_div(E, 512), 1, 1))

    result = unpack_f16(out["X"], E)
    expected = table[token_id * E:(token_id + 1) * E].astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(result, expected, atol=0.01)


# ── MatMul f32 ────────────────────────────────────────────────────────────────

def test_matmul_f32():
    tmpl = load_template("matmul")
    wgsl = instantiate_f32(tmpl)
    M, N, K = 4, 8, 16
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    params = make_params(M, N, K)

    out = dispatch(wgsl,
        [bb(0, "A"), bb(1, "B"), bb(2, "C", access="read_write"),
         bb(3, "_params_", "u32")],
        {"A": A.ravel(), "B": B.ravel(),
         "C": np.zeros(M * N, np.float32), "_params_": params},
        grid=(ceil_div(N, 32), ceil_div(M, 16), 1),
        workgroup_size=(16 * 16))

    np.testing.assert_allclose(out["C"], (A @ B).ravel(), atol=1e-3)


def test_matmul_f16():
    tmpl = load_template("matmul")
    wgsl = instantiate_f16(tmpl)
    M, N, K = 4, 8, 16
    np.random.seed(42)
    A = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    params = make_params(M, N, K)
    n_u32_a = ceil_div(M * K, 2)
    n_u32_b = ceil_div(K * N, 2)
    n_u32_c = ceil_div(M * N, 2)

    out = dispatch(wgsl,
        [bb(0, "A", "u32"), bb(1, "B", "u32"),
         bb(2, "C", "u32", "read_write"), bb(3, "_params_", "u32")],
        {"A": pack_f16(A.ravel()), "B": pack_f16(B.ravel()),
         "C": np.zeros(n_u32_c, np.uint32), "_params_": params},
        grid=(ceil_div(N, 32), ceil_div(M, 16), 1),
        workgroup_size=(16 * 16))

    result = unpack_f16(out["C"], M * N).reshape(M, N)
    A16 = A.astype(np.float16).astype(np.float32)
    B16 = B.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(result, A16 @ B16, atol=0.5, rtol=0.1)


# ── GeMM ──────────────────────────────────────────────────────────────────────

def test_gemm_f32():
    tmpl = load_template("gemm")
    wgsl = instantiate_f32(tmpl)
    M, N, K = 4, 8, 16
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)  # B is [N,K], transposed
    Bias = np.random.randn(N).astype(np.float32)
    params = make_params(M, N, K, 1)  # M, N, K, hasBias

    out = dispatch(wgsl,
        [bb(0, "A"), bb(1, "B"), bb(2, "Bias"),
         bb(3, "Y", access="read_write"), bb(4, "_params_", "u32")],
        {"A": A.ravel(), "B": B.ravel(), "Bias": Bias,
         "Y": np.zeros(M * N, np.float32), "_params_": params},
        grid=(ceil_div(N, 16), ceil_div(M, 16), 1),
        workgroup_size=(16 * 16))

    expected = (A @ B.T + Bias).ravel()
    np.testing.assert_allclose(out["Y"], expected, atol=1e-3)


# ── Conv2D ────────────────────────────────────────────────────────────────────

def test_conv2d_f32():
    tmpl = load_template("conv2d")
    wgsl = instantiate_f32(tmpl)
    batch, C_in, H_in, W_in = 1, 1, 4, 4
    C_out, KH, KW = 1, 3, 3
    H_out = H_in - KH + 1  # 2
    W_out = W_in - KW + 1  # 2

    np.random.seed(42)
    X = np.random.randn(batch * C_in * H_in * W_in).astype(np.float32)
    W = np.random.randn(C_out * C_in * KH * KW).astype(np.float32)
    bias = np.array([0.5], dtype=np.float32)
    total = batch * C_out * H_out * W_out

    params = make_params(batch, C_in, H_in, W_in, C_out, KH, KW,
                         0, 0, 1, 1, H_out, W_out, 1, 1, 1)

    out = dispatch(wgsl,
        [bb(0, "X"), bb(1, "W"), bb(2, "Bias"),
         bb(3, "Y", access="read_write"), bb(4, "_params_", "u32")],
        {"X": X, "W": W, "Bias": bias,
         "Y": np.zeros(total, np.float32), "_params_": params},
        grid=(ceil_div(total, 256), 1, 1))

    # Numpy reference
    X_4d = X.reshape(batch, C_in, H_in, W_in)
    W_4d = W.reshape(C_out, C_in, KH, KW)
    expected = np.zeros((batch, C_out, H_out, W_out), np.float32)
    for co in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                val = 0.0
                for ci in range(C_in):
                    for kh in range(KH):
                        for kw in range(KW):
                            val += X_4d[0, ci, oh + kh, ow + kw] * W_4d[co, ci, kh, kw]
                expected[0, co, oh, ow] = val + bias[co]

    np.testing.assert_allclose(out["Y"], expected.ravel(), atol=1e-4)


# ── Template instantiation correctness ───────────────────────────────────────

@pytest.mark.parametrize("name", [
    "binary_elementwise", "unary_elementwise", "softmax", "layer_norm",
    "expand", "scale", "resize_nearest", "where_select", "embed_gather",
    "matmul", "conv2d", "gemm",
])
def test_template_has_markers(name):
    """Verify all template .wgsl files contain ${T} markers."""
    tmpl = load_template(name)
    assert "${T}" in tmpl, f"{name}.wgsl should have ${{T}} markers"
    assert "${T_READ}" in tmpl or "${T_WRITE}" in tmpl or "${T_WRITE2}" in tmpl


@pytest.mark.parametrize("name", [
    "binary_elementwise", "unary_elementwise", "softmax", "layer_norm",
    "expand", "scale", "resize_nearest", "where_select", "embed_gather",
    "matmul", "conv2d", "gemm",
])
def test_f32_instantiation_valid(name):
    """Verify f32 instantiation produces valid WGSL (no leftover markers)."""
    tmpl = load_template(name)
    f32 = instantiate_f32(tmpl)
    assert "${" not in f32, f"f32 instantiation of {name} has leftover markers"
    assert "array<f32>" in f32


@pytest.mark.parametrize("name", [
    "binary_elementwise", "unary_elementwise", "softmax", "layer_norm",
    "expand", "scale", "resize_nearest", "where_select", "embed_gather",
    "matmul", "conv2d", "gemm",
])
def test_f16_instantiation_valid(name):
    """Verify f16 instantiation produces valid WGSL (no leftover markers)."""
    tmpl = load_template(name)
    f16 = instantiate_f16(tmpl)
    assert "${" not in f16, f"f16 instantiation of {name} has leftover markers"
    assert "array<u32>" in f16
    assert "unpack2x16float" in f16


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

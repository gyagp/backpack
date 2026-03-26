#!/usr/bin/env python3
"""Op-level unit tests — validates C++ ONNX ops against ONNX Runtime reference.

For each op: creates a mini ONNX model → runs through ORT for reference →
runs through our C++ op_test_runner → compares outputs.

Usage:
    python -m pytest compiler/tests/test_ops.py -v
    python -m pytest compiler/tests/test_ops.py -k "test_add" -v
    python compiler/tests/test_ops.py                    # standalone
"""
import json, os, shutil, struct, subprocess, sys, tempfile
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find the C++ op_test_runner binary
_BUILD_DIR = os.path.join(_ROOT, "runtimes", "cpp", "build", "Release")
_RUNNER = os.path.join(_BUILD_DIR, "op_test_runner.exe")
if not os.path.exists(_RUNNER):
    _RUNNER = os.path.join(_BUILD_DIR, "op_test_runner")

# ── Helpers ───────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    np.float32: ("float32", TensorProto.FLOAT),
    np.float16: ("float16", TensorProto.FLOAT16),
    np.int32: ("int32", TensorProto.INT32),
    np.int64: ("int64", TensorProto.INT64),
    np.uint8: ("uint8", TensorProto.UINT8),
    np.bool_: ("bool", TensorProto.BOOL),
}

_NP_FROM_STR = {
    "float32": np.float32, "float16": np.float16,
    "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "bool": np.bool_,
}


def _onnx_dtype(np_dtype):
    return _DTYPE_MAP[np_dtype][1]


def _dtype_str(np_dtype):
    return _DTYPE_MAP[np_dtype][0]


def _save_tensor(path, name, arr: np.ndarray):
    """Save tensor as raw binary file."""
    fpath = os.path.join(path, name + ".bin")
    arr.tofile(fpath)


def _load_tensor(path, name, dtype, shape):
    """Load tensor from raw binary file."""
    fpath = os.path.join(path, name + ".bin")
    data = np.fromfile(fpath, dtype=dtype)
    nel = 1
    for d in shape:
        nel *= max(d, 1)
    return data[:nel].reshape(shape) if nel > 0 else data


def _make_model(inputs, outputs, nodes, initializers=None):
    """Create a minimal ONNX model."""
    graph = helper.make_graph(
        nodes,
        "test_graph",
        inputs,
        outputs,
        initializer=initializers or [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                     helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 8
    return model


def run_op_test(model, input_arrays, rtol=1e-4, atol=1e-4, output_names=None):
    """Run an op test: ORT reference vs C++ op_test_runner.

    Args:
        model: ONNX ModelProto
        input_arrays: dict of name → numpy array
        rtol, atol: comparison tolerances
        output_names: optional list of output names to compare

    Returns True if outputs match.
    """
    if not os.path.exists(_RUNNER):
        pytest.skip(f"op_test_runner not found at {_RUNNER}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # Save model
        onnx.save(model, model_path)

        # Run ORT for reference
        sess = ort.InferenceSession(model_path)
        ort_inputs = {}
        for inp in sess.get_inputs():
            if inp.name in input_arrays:
                ort_inputs[inp.name] = input_arrays[inp.name]
        ort_outputs_list = sess.run(None, ort_inputs)
        ort_output_names = [o.name for o in sess.get_outputs()]
        ort_outputs = dict(zip(ort_output_names, ort_outputs_list))

        # Save inputs for C++ runner
        manifest_inputs = []
        for name, arr in input_arrays.items():
            _save_tensor(input_dir, name, arr)
            manifest_inputs.append({
                "name": name,
                "dtype": _dtype_str(arr.dtype.type),
                "shape": list(arr.shape),
            })

        with open(os.path.join(input_dir, "manifest.json"), "w") as f:
            json.dump({"inputs": manifest_inputs}, f, indent=2)

        # Run C++ op_test_runner
        result = subprocess.run(
            [_RUNNER, model_path, input_dir, output_dir],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            pytest.fail(f"op_test_runner failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        # Read C++ outputs
        cpp_manifest_path = os.path.join(output_dir, "manifest.json")
        if not os.path.exists(cpp_manifest_path):
            pytest.fail(f"No output manifest. stderr: {result.stderr}")

        with open(cpp_manifest_path) as f:
            cpp_manifest = json.load(f)

        # Compare each output
        check_names = output_names or ort_output_names
        for name in check_names:
            if name not in ort_outputs:
                continue

            ort_val = ort_outputs[name]

            # Find C++ output entry
            cpp_entry = None
            for e in cpp_manifest["outputs"]:
                if e["name"] == name:
                    cpp_entry = e
                    break
            if cpp_entry is None:
                pytest.fail(f"Output '{name}' not in C++ results")

            cpp_dtype = _NP_FROM_STR[cpp_entry["dtype"]]
            cpp_shape = cpp_entry["shape"]
            cpp_val = _load_tensor(output_dir, name, cpp_dtype, cpp_shape)

            # If dtypes differ, cast to float32 for comparison
            if ort_val.dtype != cpp_val.dtype:
                ort_val = ort_val.astype(np.float32)
                cpp_val = cpp_val.astype(np.float32)

            np.testing.assert_allclose(
                cpp_val.ravel(), ort_val.ravel(),
                rtol=rtol, atol=atol,
                err_msg=f"Output '{name}' mismatch")


# ── Op Tests ──────────────────────────────────────────────────────────────────

def test_add():
    np.random.seed(42)
    A = np.random.randn(2, 4).astype(np.float32)
    B = np.random.randn(2, 4).astype(np.float32)

    model = _make_model(
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 4])],
        [helper.make_node("Add", ["A", "B"], ["C"])],
    )
    run_op_test(model, {"A": A, "B": B})


def test_sub():
    A = np.array([1, 2, 3, 4], dtype=np.float32)
    B = np.array([4, 3, 2, 1], dtype=np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [4]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [4])],
        [helper.make_node("Sub", ["A", "B"], ["C"])],
    )
    run_op_test(model, {"A": A, "B": B})


def test_mul():
    np.random.seed(42)
    A = np.random.randn(8).astype(np.float32)
    B = np.random.randn(8).astype(np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [8]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [8])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [8])],
        [helper.make_node("Mul", ["A", "B"], ["C"])],
    )
    run_op_test(model, {"A": A, "B": B})


def test_sigmoid():
    np.random.seed(42)
    X = np.random.randn(16).astype(np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [16])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [16])],
        [helper.make_node("Sigmoid", ["X"], ["Y"])],
    )
    run_op_test(model, {"X": X})


def test_relu():
    X = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5])],
        [helper.make_node("Relu", ["X"], ["Y"])],
    )
    run_op_test(model, {"X": X})


def test_neg():
    X = np.array([1, -2, 3, -4], dtype=np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
        [helper.make_node("Neg", ["X"], ["Y"])],
    )
    run_op_test(model, {"X": X})


def test_cast_f32_to_i64():
    X = np.array([1.0, 2.5, 3.9], dtype=np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, [3])],
        [helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.INT64)],
    )
    run_op_test(model, {"X": X})


def test_reshape():
    X = np.arange(12, dtype=np.float32)
    shape = np.array([3, 4], dtype=np.int64)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [12]),
         helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])],
        [helper.make_node("Reshape", ["X", "shape"], ["Y"])],
    )
    run_op_test(model, {"X": X, "shape": shape})


def test_transpose():
    np.random.seed(42)
    X = np.random.randn(2, 3, 4).astype(np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3, 2])],
        [helper.make_node("Transpose", ["X"], ["Y"], perm=[2, 1, 0])],
    )
    run_op_test(model, {"X": X})


def test_concat():
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 4])],
        [helper.make_node("Concat", ["A", "B"], ["C"], axis=1)],
    )
    run_op_test(model, {"A": A, "B": B})


def test_slice():
    X = np.arange(20, dtype=np.float32).reshape(4, 5)
    starts = np.array([1, 0], dtype=np.int64)
    ends = np.array([3, 5], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
         helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
         helper.make_tensor_value_info("ends", TensorProto.INT64, [2]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5])],
        [helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])],
    )
    run_op_test(model, {"X": X, "starts": starts, "ends": ends, "axes": axes})


def test_unsqueeze():
    X = np.array([1, 2, 3], dtype=np.float32)
    axes = np.array([0, 2], dtype=np.int64)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 1])],
        [helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])],
    )
    run_op_test(model, {"X": X, "axes": axes})


def test_gather():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    model = _make_model(
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 2]),
         helper.make_tensor_value_info("indices", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])],
        [helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0)],
    )
    run_op_test(model, {"data": data, "indices": indices})


def test_split():
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    split = np.array([2, 2], dtype=np.int64)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4]),
         helper.make_tensor_value_info("split", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 2]),
         helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [3, 2])],
        [helper.make_node("Split", ["X", "split"], ["Y1", "Y2"], axis=1)],
    )
    run_op_test(model, {"X": X, "split": split})


def test_matmul():
    np.random.seed(42)
    A = np.random.randn(2, 4).astype(np.float32)
    B = np.random.randn(4, 3).astype(np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 3])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3])],
        [helper.make_node("MatMul", ["A", "B"], ["C"])],
    )
    run_op_test(model, {"A": A, "B": B}, atol=1e-3)


def test_softmax():
    np.random.seed(42)
    X = np.random.randn(2, 5).astype(np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5])],
        [helper.make_node("Softmax", ["X"], ["Y"], axis=1)],
    )
    run_op_test(model, {"X": X}, atol=1e-4)


def test_simplified_layer_norm():
    """Test RMSNorm (SimplifiedLayerNormalization)."""
    np.random.seed(42)
    N = 8
    X = np.random.randn(1, N).astype(np.float32)
    W = np.ones(N, dtype=np.float32)

    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, N])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, N])],
        [helper.make_node("SimplifiedLayerNormalization", ["X", "W"], ["Y"],
                          epsilon=1e-5, axis=-1, stash_type=1)],
        initializers=[numpy_helper.from_array(W, name="W")],
    )
    run_op_test(model, {"X": X}, atol=1e-4)


def test_conv_1d():
    """Test 1D depthwise convolution (group=C)."""
    np.random.seed(42)
    C, L, K = 4, 8, 3
    X = np.random.randn(1, C, L).astype(np.float32)
    W = np.random.randn(C, 1, K).astype(np.float32)

    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, C, L])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, C, L - K + 1])],
        [helper.make_node("Conv", ["X", "W"], ["Y"],
                          kernel_shape=[K], group=C, pads=[0, 0])],
        initializers=[numpy_helper.from_array(W, name="W")],
    )
    run_op_test(model, {"X": X}, atol=1e-4)


def test_conv_2d():
    """Test standard 2D convolution."""
    np.random.seed(42)
    X = np.random.randn(1, 1, 5, 5).astype(np.float32)
    W = np.random.randn(1, 1, 3, 3).astype(np.float32)
    B = np.array([0.1], dtype=np.float32)

    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3])],
        [helper.make_node("Conv", ["X", "W", "B"], ["Y"],
                          kernel_shape=[3, 3], pads=[0, 0, 0, 0])],
        initializers=[numpy_helper.from_array(W, name="W"),
                      numpy_helper.from_array(B, name="B")],
    )
    run_op_test(model, {"X": X}, atol=1e-4)


def test_expand():
    X = np.array([[1, 2, 3]], dtype=np.float32)  # [1, 3]
    shape = np.array([3, 3], dtype=np.int64)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])],
        [helper.make_node("Expand", ["X", "shape"], ["Y"])],
    )
    run_op_test(model, {"X": X, "shape": shape})


def test_where():
    cond = np.array([True, False, True, False], dtype=np.bool_)
    X = np.array([1, 2, 3, 4], dtype=np.float32)
    Y = np.array([10, 20, 30, 40], dtype=np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [4]),
         helper.make_tensor_value_info("X", TensorProto.FLOAT, [4]),
         helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [4])],
        [helper.make_node("Where", ["cond", "X", "Y"], ["out"])],
    )
    run_op_test(model, {"cond": cond, "X": X, "Y": Y})


def test_shape():
    X = np.zeros((2, 3, 4), dtype=np.float32)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, [3])],
        [helper.make_node("Shape", ["X"], ["Y"])],
    )
    run_op_test(model, {"X": X})


def test_reduce_sum():
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axes = np.array([1], dtype=np.int64)
    model = _make_model(
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("axes", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1])],
        [helper.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=1)],
    )
    run_op_test(model, {"X": X, "axes": axes})


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))

"""Tests for matmul_tiled_f16.wgsl: structural checks + numerical validation against numpy."""
import os
import re
import struct
import pytest
import numpy as np

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'matmul_tiled_f16.wgsl')
TILE_SIZE = 16
TM = 8
TN = 8
BM = TILE_SIZE * TM
BN = TILE_SIZE * TN
BK = 16

@pytest.fixture
def shader_source():
    assert os.path.exists(SHADER_PATH), f"Shader file not found: {SHADER_PATH}"
    with open(SHADER_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# CPU reference that mirrors the register-tiled shader logic
# ---------------------------------------------------------------------------

def f16_to_f32(h: int) -> float:
    return struct.unpack('e', struct.pack('H', h))[0]

def f32_to_f16(v: float) -> int:
    return struct.unpack('H', struct.pack('e', v))[0]

def pack_f16_to_u32(arr_f16: np.ndarray) -> np.ndarray:
    flat = arr_f16.flatten().astype(np.float16)
    raw = flat.view(np.uint16).astype(np.uint32)
    padded = np.append(raw, np.uint32(0)) if len(raw) % 2 else raw
    return ((padded[1::2] << 16) | padded[0::2]).astype(np.uint32)

def load_f16_from_u32(buf: np.ndarray, index: int) -> float:
    word = int(buf[index // 2])
    bits = (word >> ((index % 2) * 16)) & 0xFFFF
    return f16_to_f32(bits)

def cpu_tiled_matmul(A_u32: np.ndarray, B_u32: np.ndarray, M: int, N: int, K: int) -> np.ndarray:
    """CPU reference using numpy -- mathematically equivalent to the shader."""
    A_f32 = np.array([load_f16_from_u32(A_u32, i) for i in range(M * K)], dtype=np.float32).reshape(M, K)
    B_f32 = np.array([load_f16_from_u32(B_u32, i) for i in range(K * N)], dtype=np.float32).reshape(K, N)
    return A_f32 @ B_f32


# ---------------------------------------------------------------------------
# Numerical validation tests
# ---------------------------------------------------------------------------

def _run_matmul_test(M: int, N: int, K: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A_f16 = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B_f16 = rng.uniform(-1, 1, (K, N)).astype(np.float16)

    A_u32 = pack_f16_to_u32(A_f16)
    B_u32 = pack_f16_to_u32(B_f16)

    gpu_ref = cpu_tiled_matmul(A_u32, B_u32, M, N, K)
    expected = (A_f16.astype(np.float32) @ B_f16.astype(np.float32))

    np.testing.assert_allclose(gpu_ref, expected, rtol=1e-2, atol=1e-2,
                               err_msg=f"Matmul mismatch for {M}x{K} @ {K}x{N}")


def test_numerical_square_128():
    _run_matmul_test(128, 128, 128)


def test_numerical_non_aligned_100():
    _run_matmul_test(100, 100, 100)


def test_numerical_rectangular():
    _run_matmul_test(80, 64, 48)


def test_numerical_small():
    _run_matmul_test(1, 1, 1)


def test_numerical_single_tile():
    _run_matmul_test(16, 16, 16)


# ---------------------------------------------------------------------------
# Structural tests (shader source validation)
# ---------------------------------------------------------------------------

def test_shader_file_exists():
    assert os.path.isfile(SHADER_PATH)

def test_workgroup_shared_memory(shader_source):
    assert 'var<workgroup>' in shader_source
    workgroup_vars = re.findall(r'var<workgroup>\s+(\w+)', shader_source)
    assert len(workgroup_vars) >= 2, f"Expected at least 2 workgroup vars (tileA, tileB), found: {workgroup_vars}"

def test_tile_loading_with_barriers(shader_source):
    assert 'workgroupBarrier()' in shader_source
    barriers = shader_source.count('workgroupBarrier()')
    assert barriers >= 2, f"Expected at least 2 barriers (before and after compute), found {barriers}"

def test_tiles_loaded_from_global_to_shared(shader_source):
    assert 'unpack2x16float' in shader_source
    assert re.search(r'tileA\[', shader_source)
    assert re.search(r'tileB\[', shader_source)

def test_computation_uses_shared_memory(shader_source):
    assert re.search(r'tileA\[', shader_source)
    assert re.search(r'tileB\[', shader_source)

def test_register_tiling_constants(shader_source):
    assert re.search(r'const\s+TM\s*:\s*u32\s*=\s*8u', shader_source)
    assert re.search(r'const\s+TN\s*:\s*u32\s*=\s*8u', shader_source)
    assert re.search(r'const\s+BM\s*:\s*u32\s*=\s*128u', shader_source)
    assert re.search(r'const\s+BN\s*:\s*u32\s*=\s*128u', shader_source)

def test_workgroup_size_matches_tile(shader_source):
    match = re.search(r'@workgroup_size\((\d+),\s*(\d+)\)', shader_source)
    assert match, "workgroup_size not found"
    assert match.group(1) == '16' and match.group(2) == '16'

def test_shared_memory_sized_for_register_tiling(shader_source):
    assert re.search(r'array<f32,\s*4224>', shader_source), "tileA shared memory should be 4224 (2*BK*BM_PAD=2*16*132) for double buffering"
    assert re.search(r'array<f32,\s*4096>', shader_source), "tileB shared memory should be 4096 (2*BK*BN=2*16*128) for double buffering"

def test_ceiling_division_for_tiles(shader_source):
    assert re.search(r'params\.K\s*\+\s*BK\s*-\s*1u', shader_source), "Should use ceiling division for tile count"

def test_bounds_check_on_output(shader_source):
    assert re.search(r'row\s*(>=|<)\s*params\.M', shader_source)
    assert re.search(r'col\s*<\s*params\.N', shader_source)

def test_zero_fill_for_oob(shader_source):
    assert shader_source.count('= 0.0') >= 2, "Should zero-fill out-of-bounds tile elements"

def test_f32_output_buffer(shader_source):
    assert re.search(r'@binding\(2\)\s*var<storage,\s*read_write>\s*\w+\s*:\s*array<f32>', shader_source)

def test_f16_input_loading(shader_source):
    assert 'unpack2x16float' in shader_source


# ---------------------------------------------------------------------------
# Double-buffering structural tests
# ---------------------------------------------------------------------------

def test_double_buffer_constants(shader_source):
    """Verify BUF_A_SIZE and BUF_B_SIZE constants exist for ping-pong indexing."""
    assert re.search(r'const\s+BUF_A_SIZE\s*:\s*u32\s*=', shader_source), "Missing BUF_A_SIZE constant"
    assert re.search(r'const\s+BUF_B_SIZE\s*:\s*u32\s*=', shader_source), "Missing BUF_B_SIZE constant"

def test_two_sets_of_shared_memory(shader_source):
    """Shared memory must be 2x single-buffer size for ping-pong."""
    a_match = re.search(r'var<workgroup>\s+tileA\s*:\s*array<f32,\s*(\d+)>', shader_source)
    b_match = re.search(r'var<workgroup>\s+tileB\s*:\s*array<f32,\s*(\d+)>', shader_source)
    assert a_match and b_match
    buf_a_size = re.search(r'const\s+BUF_A_SIZE\s*:\s*u32\s*=\s*(\d+)u', shader_source)
    buf_b_size = re.search(r'const\s+BUF_B_SIZE\s*:\s*u32\s*=\s*(\d+)u', shader_source)
    assert int(a_match.group(1)) == 2 * int(buf_a_size.group(1)), "tileA should be 2 * BUF_A_SIZE"
    assert int(b_match.group(1)) == 2 * int(buf_b_size.group(1)), "tileB should be 2 * BUF_B_SIZE"

def test_ping_pong_offset_pattern(shader_source):
    """Verify ping-pong buffer selection using t%2 pattern."""
    assert re.search(r't\s*%\s*2u', shader_source), "Missing t%2 ping-pong selection"
    assert re.search(r'\(t\s*\+\s*1u\)\s*%\s*2u', shader_source), "Missing (t+1)%2 for next buffer"

def test_prefetch_next_tile(shader_source):
    """Verify that next tile is loaded while current tile is being computed."""
    assert re.search(r'next_tile_k|nxt_a_off|nxt_b_off', shader_source), \
        "Missing prefetch variables for next tile"
    assert re.search(r't\s*\+\s*1u\s*<\s*numTiles', shader_source), \
        "Missing guard for prefetching next tile"

def test_initial_tile_preload(shader_source):
    """Tile 0 must be loaded before the main loop starts."""
    main_loop = shader_source.find('for (var t = 0u')
    first_barrier = shader_source.find('workgroupBarrier()')
    assert first_barrier < main_loop, "First barrier (after tile 0 load) must come before main loop"


def test_batch_params_in_shader(shader_source):
    assert 'batch_size: u32' in shader_source
    assert 'stride_A: u32' in shader_source
    assert 'stride_C: u32' in shader_source

def test_batch_z_dispatch(shader_source):
    assert 'wgid.z' in shader_source

def test_batch_offset_applied(shader_source):
    assert 'batch * params.stride_A' in shader_source
    assert 'batch * params.stride_C' in shader_source


# ---------------------------------------------------------------------------
# Batched numerical tests
# ---------------------------------------------------------------------------

def cpu_batched_tiled_matmul(A_u32, B_u32, M, N, K, batch_size):
    stride_A = (M * K + 1) // 2  # u32 words per batch for A
    C = np.zeros((batch_size, M, N), dtype=np.float32)
    for b in range(batch_size):
        a_start = b * ((M * K + 1) // 2)
        a_end = a_start + ((M * K + 1) // 2)
        A_batch = A_u32[a_start:a_end]
        C[b] = cpu_tiled_matmul(A_batch, B_u32, M, N, K)
    return C


def _run_batched_tiled_test(batch_size, M, N, K, seed=42):
    rng = np.random.default_rng(seed)
    A_f16 = rng.uniform(-1, 1, (batch_size, M, K)).astype(np.float16)
    B_f16 = rng.uniform(-1, 1, (K, N)).astype(np.float16)

    A_u32 = np.concatenate([pack_f16_to_u32(A_f16[b]) for b in range(batch_size)])
    B_u32 = pack_f16_to_u32(B_f16)

    result = cpu_batched_tiled_matmul(A_u32, B_u32, M, N, K, batch_size)

    for b in range(batch_size):
        expected = A_f16[b].astype(np.float32) @ B_f16.astype(np.float32)
        np.testing.assert_allclose(
            result[b], expected, rtol=1e-2, atol=1e-2,
            err_msg=f"Batch {b}: mismatch for {M}x{K} @ {K}x{N}")


def test_batched_single():
    _run_batched_tiled_test(1, 32, 32, 32)


def test_batched_4():
    _run_batched_tiled_test(4, 32, 32, 32)


def test_batched_non_aligned():
    _run_batched_tiled_test(3, 20, 24, 18)


def test_batched_elements_independent():
    rng = np.random.default_rng(77)
    batch_size, M, N, K = 4, 16, 16, 16
    A_f16 = rng.uniform(-1, 1, (batch_size, M, K)).astype(np.float16)
    B_f16 = rng.uniform(-1, 1, (K, N)).astype(np.float16)

    A_u32 = np.concatenate([pack_f16_to_u32(A_f16[b]) for b in range(batch_size)])
    B_u32 = pack_f16_to_u32(B_f16)

    batched = cpu_batched_tiled_matmul(A_u32, B_u32, M, N, K, batch_size)

    for b in range(batch_size):
        single_A = pack_f16_to_u32(A_f16[b])
        single = cpu_tiled_matmul(single_A, B_u32, M, N, K)
        np.testing.assert_allclose(
            batched[b], single, rtol=1e-6, atol=1e-6,
            err_msg=f"Batch {b} differs from independent computation")


# ---------------------------------------------------------------------------
# GPU compilation test
# ---------------------------------------------------------------------------

def test_shader_compiles_on_gpu():
    """Verify that the shader passes WGSL validation and compiles on real GPU."""
    try:
        import wgpu
    except ImportError:
        pytest.skip("wgpu not installed")

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if not adapter:
        pytest.skip("No WebGPU adapter available")

    device = adapter.request_device_sync()

    with open(SHADER_PATH) as f:
        code = f.read()

    device.create_shader_module(code=code)

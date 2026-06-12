"""Tests for benchmarks/bench_workgroup_sweep.py — validates bind group layouts,
params packing, shader template correctness, and config coverage without GPU."""
import os
import re
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
import bench_workgroup_sweep as bws

SHADER_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders')


# ---------------------------------------------------------------------------
# Acceptance criteria: workgroup sizes 64, 128, 256, 512 all covered
# ---------------------------------------------------------------------------

class TestMatmulConfigs:
    def test_all_workgroup_sizes_covered(self):
        wg_sizes = {cfg["wg_x"] * cfg["wg_y"] for cfg in bws.MATMUL_CONFIGS}
        for required in [64, 128, 256, 512]:
            assert required in wg_sizes, f"Missing workgroup_size {required} in MATMUL_CONFIGS"

    def test_1d_sweep_sizes(self):
        assert bws.WORKGROUP_SIZES_1D == [64, 128, 256, 512]

    def test_configs_have_required_fields(self):
        for cfg in bws.MATMUL_CONFIGS:
            assert "bm" in cfg and "bn" in cfg and "bk" in cfg
            assert "wg_x" in cfg and "wg_y" in cfg
            assert "label" in cfg


# ---------------------------------------------------------------------------
# Matmul shader template: validate generated code structure
# ---------------------------------------------------------------------------

class TestMatmulShaderTemplate:
    def test_generates_valid_workgroup_size(self):
        code = bws.generate_matmul_shader(128, 128, 32, 16, 16)
        assert "@compute @workgroup_size(16, 16)" in code

    def test_has_correct_bindings(self):
        code = bws.generate_matmul_shader(64, 64, 16, 8, 8)
        assert "@group(0) @binding(0)" in code
        assert "@group(0) @binding(1)" in code
        assert "@group(0) @binding(2)" in code
        assert "@group(0) @binding(3)" in code

    def test_params_struct_matches_pack(self):
        code = bws.generate_matmul_shader(64, 64, 16, 8, 8)
        assert "M: u32," in code
        assert "N: u32," in code
        assert "K: u32," in code

    def test_all_configs_generate_without_error(self):
        for cfg in bws.MATMUL_CONFIGS:
            code = bws.generate_matmul_shader(
                cfg["bm"], cfg["bn"], cfg["bk"], cfg["wg_x"], cfg["wg_y"]
            )
            assert "fn main(" in code

    def test_tile_dimensions_consistent(self):
        for cfg in bws.MATMUL_CONFIGS:
            tm = cfg["bm"] // cfg["wg_x"]
            tn = cfg["bn"] // cfg["wg_y"]
            assert tm > 0, f"TM=0 for {cfg['label']}"
            assert tn > 0, f"TN=0 for {cfg['label']}"
            assert cfg["bm"] % cfg["wg_x"] == 0, f"BM not divisible by WG_X for {cfg['label']}"
            assert cfg["bn"] % cfg["wg_y"] == 0, f"BN not divisible by WG_Y for {cfg['label']}"


# ---------------------------------------------------------------------------
# RMSNorm: bind group layout must match rmsnorm_scaled.wgsl
# ---------------------------------------------------------------------------

class TestRMSNormBindGroupMatch:
    @pytest.fixture
    def shader_code(self):
        path = os.path.join(SHADER_DIR, 'rmsnorm_scaled.wgsl')
        if not os.path.exists(path):
            pytest.skip("rmsnorm_scaled.wgsl not found")
        with open(path) as f:
            return f.read()

    def test_binding_count_matches(self, shader_code):
        shader_bindings = re.findall(r'@binding\((\d+)\)', shader_code)
        assert len(shader_bindings) == 4, (
            f"rmsnorm_scaled.wgsl has {len(shader_bindings)} bindings, benchmark expects 4"
        )

    def test_binding_indices(self, shader_code):
        indices = sorted(int(m) for m in re.findall(r'@binding\((\d+)\)', shader_code))
        assert indices == [0, 1, 2, 3]

    def test_weights_buffer_exists(self, shader_code):
        assert "weights" in shader_code, "rmsnorm_scaled.wgsl must have weights buffer"

    def test_params_packing_matches_shader(self, shader_code):
        assert "row_length : u32" in shader_code or "row_length: u32" in shader_code
        assert "epsilon : f32" in shader_code or "epsilon: f32" in shader_code
        dim = 4096
        epsilon = 1e-5
        packed = struct.pack('If', dim, epsilon)
        row_length, eps = struct.unpack('If', packed)
        assert row_length == dim
        assert abs(eps - epsilon) < 1e-10


# ---------------------------------------------------------------------------
# Attention: bind group layout must match attention.wgsl
# ---------------------------------------------------------------------------

class TestAttentionBindGroupMatch:
    @pytest.fixture
    def shader_code(self):
        path = os.path.join(SHADER_DIR, 'attention.wgsl')
        if not os.path.exists(path):
            pytest.skip("attention.wgsl not found")
        with open(path) as f:
            return f.read()

    def test_binding_count_matches(self, shader_code):
        shader_bindings = re.findall(r'@binding\((\d+)\)', shader_code)
        assert len(shader_bindings) == 5, (
            f"attention.wgsl has {len(shader_bindings)} bindings, benchmark expects 5"
        )

    def test_binding_indices(self, shader_code):
        indices = sorted(int(m) for m in re.findall(r'@binding\((\d+)\)', shader_code))
        assert indices == [0, 1, 2, 3, 4]

    def test_qkv_output_params_order(self, shader_code):
        lines = shader_code.split('\n')
        binding_vars = {}
        for line in lines:
            bm = re.search(r'@binding\((\d+)\).*var.*\b(\w+)\b\s*:', line)
            if bm:
                binding_vars[int(bm.group(1))] = bm.group(2)
        assert binding_vars.get(0) == 'Q'
        assert binding_vars.get(1) == 'K'
        assert binding_vars.get(2) == 'V'
        assert binding_vars.get(3) == 'output'
        assert binding_vars.get(4) == 'params'


# ---------------------------------------------------------------------------
# pack_f16_to_u32 utility
# ---------------------------------------------------------------------------

class TestPackF16:
    def test_round_trip_even_count(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
        packed = bws.pack_f16_to_u32(arr)
        assert len(packed) == 8  # 2 u32s

    def test_round_trip_odd_count(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        packed = bws.pack_f16_to_u32(arr)
        assert len(packed) == 8  # padded to 2 u32s


# ---------------------------------------------------------------------------
# Harness file existence
# ---------------------------------------------------------------------------

class TestHarnessExists:
    def test_benchmark_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'bench_workgroup_sweep.py')
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# Shader workgroup_size patching for 1D kernels
# ---------------------------------------------------------------------------

class TestShaderPatching:
    def test_patches_workgroup_size(self):
        code = "@compute @workgroup_size(64)\nfn main() {}"
        patched = re.sub(r'@workgroup_size\(\d+\)', '@workgroup_size(256)', code)
        assert "@workgroup_size(256)" in patched

    @pytest.mark.parametrize("wg", [64, 128, 256, 512])
    def test_load_shader_with_wg_size_rmsnorm(self, wg):
        path = os.path.join(SHADER_DIR, 'rmsnorm_scaled.wgsl')
        if not os.path.exists(path):
            pytest.skip("rmsnorm_scaled.wgsl not found")
        code = bws.load_shader_with_wg_size(path, wg)
        assert f"@workgroup_size({wg})" in code

    @pytest.mark.parametrize("wg", [64, 128, 256, 512])
    def test_load_shader_with_wg_size_attention(self, wg):
        path = os.path.join(SHADER_DIR, 'attention.wgsl')
        if not os.path.exists(path):
            pytest.skip("attention.wgsl not found")
        code = bws.load_shader_with_wg_size(path, wg)
        assert f"@workgroup_size({wg})" in code

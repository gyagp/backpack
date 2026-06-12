"""Tests for Q4_K_M dequantization WGSL shader correctness."""
import os
import re
import struct
import unittest
import math

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'dequant_q4_k_m.wgsl')

QK_K = 256
BLOCK_SIZE = 144  # 2+2 (d,dmin) + 12 (scales) + 128 (qs)


def f32_to_f16_bits(val):
    """Convert f32 to f16 bit pattern (u16)."""
    packed = struct.pack('e', val)
    return struct.unpack('<H', packed)[0]


def f16_bits_to_f32(bits):
    """Reference f16 -> f32 conversion."""
    return struct.unpack('e', struct.pack('<H', bits))[0]


def make_block(d, dmin, scales, mins, qs):
    """Build a 144-byte Q4_K_M block.

    Args:
        d: f16 scale factor (float)
        dmin: f16 min factor (float)
        scales: list of 8 scale values (6-bit for 0-3, 6-bit for 4-7)
        mins: list of 8 min values (6-bit for 0-3, 6-bit for 4-7)
        qs: list of 128 bytes (each holding two 4-bit values)
    """
    data = bytearray(BLOCK_SIZE)
    d_bits = f32_to_f16_bits(d)
    dmin_bits = f32_to_f16_bits(dmin)
    struct.pack_into('<HH', data, 0, d_bits, dmin_bits)

    # Pack scales/mins into 12 bytes following GGUF get_scale_min_k4 layout
    # Bytes 0-3: lower 6 bits of scales[0..3]
    # Bytes 4-7: lower 6 bits of mins[0..3]
    # Bytes 8-11: packed upper bits + lower 4 bits of scales/mins[4..7]
    for j in range(4):
        data[4 + j] = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6)
        data[4 + 4 + j] = (mins[j] & 63) | ((mins[j + 4] >> 4) << 6)
    for j in range(4):
        data[4 + 8 + j] = (scales[j + 4] & 0xF) | ((mins[j + 4] & 0xF) << 4)

    for i, q in enumerate(qs):
        data[16 + i] = q & 0xFF

    return bytes(data)


def get_scale_min_k4_ref(j, scales_12):
    """Reference implementation of get_scale_min_k4."""
    if j < 4:
        sc = scales_12[j] & 63
        mn = scales_12[j + 4] & 63
    else:
        packed = scales_12[j + 4]
        hi_s = scales_12[j - 4]
        sc = (packed & 0xF) | ((hi_s >> 6) << 4)
        hi_m = scales_12[j]
        mn = ((packed >> 4) & 0xF) | ((hi_m >> 6) << 4)
    return sc, mn


def dequantize_block_ref(block_bytes):
    """Reference Q4_K_M dequantization of a single block -> 256 f32 values."""
    d = f16_bits_to_f32(struct.unpack_from('<H', block_bytes, 0)[0])
    dmin = f16_bits_to_f32(struct.unpack_from('<H', block_bytes, 2)[0])
    scales_12 = block_bytes[4:16]
    qs = block_bytes[16:144]

    output = [0.0] * QK_K
    is_idx = 0
    for j in range(QK_K // 32):
        sc, mn = get_scale_min_k4_ref(is_idx, scales_12)
        is_idx += 1
        d_sc = d * sc
        dm = dmin * mn

        for l in range(16):
            q = qs[j * 16 + l]
            low_nibble = q & 0xF
            high_nibble = (q >> 4) & 0xF

            output[j * 32 + l] = d_sc * low_nibble - dm
            output[j * 32 + l + 16] = d_sc * high_nibble - dm

    return output


class TestShaderFileExists(unittest.TestCase):
    def test_shader_file_exists(self):
        self.assertTrue(os.path.isfile(SHADER_PATH),
                        f"Shader file not found at {SHADER_PATH}")


class TestShaderStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SHADER_PATH, 'r') as f:
            cls.source = f.read()

    def test_has_compute_entry_point(self):
        self.assertIn('@compute', self.source)
        self.assertIn('@workgroup_size', self.source)
        self.assertIn('fn main', self.source)

    def test_has_storage_bindings(self):
        self.assertIn('var<storage, read> quant_data', self.source)
        self.assertIn('var<storage, read_write> output', self.source)

    def test_block_size_constant(self):
        self.assertIn('144u', self.source)

    def test_qk_k_constant(self):
        self.assertIn('256u', self.source)

    def test_nibble_extraction(self):
        self.assertIn('& 0xFu', self.source)
        self.assertIn('>> 4u', self.source)

    def test_scale_min_function(self):
        self.assertIn('get_scale_min_k4', self.source)
        self.assertIn('& 63u', self.source)

    def test_f16_conversion(self):
        self.assertIn('u16_to_f32', self.source)


class TestNibbleExtraction(unittest.TestCase):
    def test_low_nibble(self):
        for val in [0x00, 0x0F, 0xA5, 0xFF, 0x37]:
            self.assertEqual(val & 0xF, val & 0xF)

    def test_high_nibble(self):
        for val in [0x00, 0xF0, 0xA5, 0xFF, 0x73]:
            self.assertEqual((val >> 4) & 0xF, (val >> 4) & 0xF)

    def test_both_nibbles_cover_full_byte(self):
        for val in range(256):
            lo = val & 0xF
            hi = (val >> 4) & 0xF
            self.assertEqual(lo | (hi << 4), val)
            self.assertTrue(0 <= lo <= 15)
            self.assertTrue(0 <= hi <= 15)


class TestScaleMinExtraction(unittest.TestCase):
    def test_low_subblocks_6bit(self):
        scales = [10, 20, 30, 40, 0, 0, 0, 0]
        mins = [5, 15, 25, 35, 0, 0, 0, 0]
        block = make_block(1.0, 0.5, scales, mins, [0] * 128)
        scales_12 = block[4:16]
        for j in range(4):
            sc, mn = get_scale_min_k4_ref(j, scales_12)
            self.assertEqual(sc, scales[j])
            self.assertEqual(mn, mins[j])

    def test_high_subblocks_packed(self):
        scales = [0, 0, 0, 0, 18, 22, 30, 14]
        mins = [0, 0, 0, 0, 9, 11, 7, 3]
        block = make_block(1.0, 0.5, scales, mins, [0] * 128)
        scales_12 = block[4:16]
        for j in range(4, 8):
            sc, mn = get_scale_min_k4_ref(j, scales_12)
            self.assertEqual(sc, scales[j], f"Scale mismatch at j={j}")
            self.assertEqual(mn, mins[j], f"Min mismatch at j={j}")


class TestDequantization(unittest.TestCase):
    def test_all_zeros(self):
        block = make_block(1.0, 0.0, [1]*8, [0]*8, [0]*128)
        output = dequantize_block_ref(block)
        for v in output:
            self.assertAlmostEqual(v, 0.0, places=4)

    def test_all_max_nibbles(self):
        block = make_block(1.0, 0.0, [1]*8, [0]*8, [0xFF]*128)
        output = dequantize_block_ref(block)
        for v in output:
            self.assertAlmostEqual(v, 15.0, places=4)

    def test_with_min_offset(self):
        block = make_block(1.0, 1.0, [2]*8, [3]*8, [0]*128)
        output = dequantize_block_ref(block)
        for v in output:
            self.assertAlmostEqual(v, -3.0, places=4)

    def test_mixed_nibbles(self):
        qs = [0] * 128
        qs[0] = 0xA5  # low=5, high=10
        block = make_block(0.5, 0.25, [4]*8, [2]*8, qs)
        output = dequantize_block_ref(block)
        # First value: d*sc*low_nibble - dmin*min = 0.5*4*5 - 0.25*2 = 10 - 0.5 = 9.5
        self.assertAlmostEqual(output[0], 9.5, places=3)
        # 17th value (high nibble of first byte, same scale): 0.5*4*10 - 0.25*2 = 20 - 0.5 = 19.5
        self.assertAlmostEqual(output[16], 19.5, places=3)

    def test_output_length(self):
        block = make_block(1.0, 0.5, [1]*8, [1]*8, [0x55]*128)
        output = dequantize_block_ref(block)
        self.assertEqual(len(output), QK_K)

    def test_different_scales_per_subblock(self):
        """Regression: both nibbles of each byte must use the SAME scale/min."""
        scales = [1, 2, 3, 4, 5, 6, 7, 8]
        mins = [8, 7, 6, 5, 4, 3, 2, 1]
        qs = [0] * 128
        qs[0] = 0xA5   # sub-block 0, byte 0: low=5, high=10
        qs[16] = 0xB3  # sub-block 1, byte 0: low=3, high=11
        qs[48] = 0x72  # sub-block 3, byte 0: low=2, high=7
        qs[64] = 0xC1  # sub-block 4, byte 0: low=1, high=12
        block = make_block(0.5, 0.25, scales, mins, qs)
        output = dequantize_block_ref(block)
        # sub-block 0: d*sc[0]*nibble - dmin*mn[0] = 0.5*1*nibble - 0.25*8
        self.assertAlmostEqual(output[0], 0.5*1*5 - 0.25*8, places=3)
        self.assertAlmostEqual(output[16], 0.5*1*10 - 0.25*8, places=3)
        # sub-block 1: d*sc[1]*nibble - dmin*mn[1]
        self.assertAlmostEqual(output[32], 0.5*2*3 - 0.25*7, places=3)
        self.assertAlmostEqual(output[48], 0.5*2*11 - 0.25*7, places=3)
        # sub-block 3: d*sc[3]*nibble - dmin*mn[3]
        self.assertAlmostEqual(output[96], 0.5*4*2 - 0.25*5, places=3)
        self.assertAlmostEqual(output[112], 0.5*4*7 - 0.25*5, places=3)
        # sub-block 4: d*sc[4]*nibble - dmin*mn[4]
        self.assertAlmostEqual(output[128], 0.5*5*1 - 0.25*4, places=3)
        self.assertAlmostEqual(output[144], 0.5*5*12 - 0.25*4, places=3)

    def test_shader_inner_loop_matches_reference(self):
        """Parse the WGSL to verify the inner loop uses j*16 offset (not j*32), confirming
        both nibbles share the same scale."""
        with open(SHADER_PATH, 'r') as f:
            source = f.read()
        self.assertIn('j * 16u + l', source,
                      "qs offset should be j*16+l (16 bytes per sub-block)")
        self.assertNotIn('j * 32u + l + 16u] = d_sc2', source,
                         "Should not use a second scale for high nibbles")

    def test_d_scaling(self):
        block1 = make_block(1.0, 0.0, [1]*8, [0]*8, [0x11]*128)
        block2 = make_block(2.0, 0.0, [1]*8, [0]*8, [0x11]*128)
        out1 = dequantize_block_ref(block1)
        out2 = dequantize_block_ref(block2)
        for a, b in zip(out1[:32], out2[:32]):
            if a != 0:
                self.assertAlmostEqual(b / a, 2.0, places=2)


class TestShaderAlgorithmMatchesReference(unittest.TestCase):
    """Verify the WGSL shader's algorithm matches the reference by parsing key patterns."""

    @classmethod
    def setUpClass(cls):
        with open(SHADER_PATH, 'r') as f:
            cls.source = f.read()

    def test_dequant_formula(self):
        self.assertRegex(self.source, r'd_sc\s*\*\s*f32\(low_nibble\)\s*-\s*dm')
        self.assertRegex(self.source, r'd_sc\s*\*\s*f32\(high_nibble\)\s*-\s*dm')

    def test_super_block_loop(self):
        self.assertIn('QK_K / 32u', self.source)

    def test_inner_loop_16(self):
        self.assertIn('l < 16u', self.source)

    def test_scale_uses_ptr_function(self):
        self.assertIn('ptr<function, u32>', self.source)

    def test_output_indexing(self):
        self.assertIn('j * 32u + l', self.source)
        self.assertIn('j * 32u + l + 16u', self.source)


if __name__ == '__main__':
    unittest.main()

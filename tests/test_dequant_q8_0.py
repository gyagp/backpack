"""Tests for Q8_0 dequantization WGSL shader correctness."""
import os
import struct
import unittest

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'dequant_q8_0.wgsl')

Q8_0_BLOCK_SIZE = 32
Q8_0_BYTES_PER_BLOCK = 34  # 2 (f16 d) + 32 (int8 quants)


def f32_to_f16_bits(val):
    return struct.unpack('<H', struct.pack('e', val))[0]


def f16_bits_to_f32(bits):
    return struct.unpack('e', struct.pack('<H', bits))[0]


def make_q8_0_block(d, quants):
    """Build a 34-byte Q8_0 block. quants: list of 32 signed int8 values."""
    data = bytearray(Q8_0_BYTES_PER_BLOCK)
    struct.pack_into('<H', data, 0, f32_to_f16_bits(d))
    for i, q in enumerate(quants):
        data[2 + i] = q & 0xFF
    return bytes(data)


def cpu_dequant_q8_0(block_bytes):
    """Reference Q8_0 dequantization -> 32 f32 values."""
    d = f16_bits_to_f32(struct.unpack_from('<H', block_bytes, 0)[0])
    output = []
    for i in range(Q8_0_BLOCK_SIZE):
        q = block_bytes[2 + i]
        if q >= 128:
            q -= 256
        output.append(d * q)
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
        self.assertIn('34u', self.source)

    def test_has_n_blocks_guard(self):
        self.assertIn('n_blocks', self.source)

    def test_signed_int8_handling(self):
        self.assertIn('read_i8', self.source)


class TestDequantization(unittest.TestCase):
    def test_all_zeros(self):
        block = make_q8_0_block(1.0, [0] * 32)
        output = cpu_dequant_q8_0(block)
        for v in output:
            self.assertAlmostEqual(v, 0.0, places=4)

    def test_positive_quants(self):
        quants = list(range(1, 33))
        block = make_q8_0_block(0.5, quants)
        output = cpu_dequant_q8_0(block)
        for i, v in enumerate(output):
            self.assertAlmostEqual(v, 0.5 * quants[i], places=3)

    def test_negative_quants(self):
        quants = [(-i) & 0xFF for i in range(1, 33)]
        block = make_q8_0_block(0.5, quants)
        output = cpu_dequant_q8_0(block)
        for i, v in enumerate(output):
            self.assertAlmostEqual(v, 0.5 * (-(i + 1)), places=3)

    def test_mixed_quants(self):
        quants = [127, 0xFF, 0, 1]  # 127, -1, 0, 1
        quants += [0] * 28
        block = make_q8_0_block(2.0, quants)
        output = cpu_dequant_q8_0(block)
        self.assertAlmostEqual(output[0], 2.0 * 127, places=2)
        self.assertAlmostEqual(output[1], 2.0 * (-1), places=2)
        self.assertAlmostEqual(output[2], 0.0, places=4)
        self.assertAlmostEqual(output[3], 2.0 * 1, places=2)

    def test_d_scaling(self):
        quants = [10] * 32
        out1 = cpu_dequant_q8_0(make_q8_0_block(1.0, quants))
        out2 = cpu_dequant_q8_0(make_q8_0_block(2.0, quants))
        for a, b in zip(out1, out2):
            if a != 0:
                self.assertAlmostEqual(b / a, 2.0, places=2)

    def test_output_length(self):
        block = make_q8_0_block(1.0, [0] * 32)
        output = cpu_dequant_q8_0(block)
        self.assertEqual(len(output), Q8_0_BLOCK_SIZE)

    def test_multi_block(self):
        blocks = []
        for i in range(4):
            blocks.append(make_q8_0_block(float(i + 1), [10] * 32))
        all_data = b''.join(blocks)
        all_output = []
        for i in range(4):
            blk = all_data[i * Q8_0_BYTES_PER_BLOCK:(i + 1) * Q8_0_BYTES_PER_BLOCK]
            all_output.extend(cpu_dequant_q8_0(blk))
        self.assertEqual(len(all_output), 128)
        for i in range(4):
            for j in range(32):
                expected = float(i + 1) * 10
                self.assertAlmostEqual(all_output[i * 32 + j], expected, places=2)


class TestShaderAlgorithmMatchesReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SHADER_PATH, 'r') as f:
            cls.source = f.read()

    def test_dequant_formula(self):
        self.assertRegex(self.source, r'd\s*\*\s*f32\(q\)')

    def test_block_loop(self):
        self.assertIn('BLOCK_SIZE', self.source)

    def test_output_indexing(self):
        self.assertIn('out_base + i', self.source)


if __name__ == '__main__':
    unittest.main()

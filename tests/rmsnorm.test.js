import { readFileSync, existsSync } from 'fs';
import { describe, it, expect, beforeAll } from 'vitest';

describe('rmsnorm.wgsl', () => {
  const shaderPath = 'src/shaders/rmsnorm.wgsl';

  it('file exists', () => {
    expect(existsSync(shaderPath)).toBe(true);
  });

  describe('shader content', () => {
    let source;

    beforeAll(() => {
      source = readFileSync(shaderPath, 'utf-8');
    });

    it('has input storage buffer of array<f32>', () => {
      expect(source).toMatch(/var<storage,\s*read>\s*\w+\s*:\s*array<f32>/);
    });

    it('has output storage buffer of array<f32>', () => {
      expect(source).toMatch(/var<storage,\s*read_write>\s*\w+\s*:\s*array<f32>/);
    });

    it('has uniform params with row_length and epsilon', () => {
      expect(source).toMatch(/row_length\s*:\s*u32/);
      expect(source).toMatch(/epsilon\s*:\s*f32/);
      expect(source).toMatch(/var<uniform>\s*\w+\s*:\s*Params/);
    });

    it('computes sum of squares for RMS calculation', () => {
      expect(source).toContain('sum_sq');
      expect(source).toMatch(/v\s*\*\s*v/);
    });

    it('computes inverse RMS with epsilon', () => {
      expect(source).toMatch(/sqrt\s*\(/);
      expect(source).toContain('epsilon');
      expect(source).toMatch(/1\.0\s*\/\s*rms/);
    });

    it('multiplies input by inverse RMS to produce output', () => {
      expect(source).toMatch(/output\[.*\]\s*=\s*input\[.*\]\s*\*\s*inv_rms/);
    });

    it('is a compute shader with workgroup_size', () => {
      expect(source).toMatch(/@compute\s+@workgroup_size\(\d+\)/);
    });

    it('has bounds checking', () => {
      expect(source).toContain('arrayLength');
    });
  });
});

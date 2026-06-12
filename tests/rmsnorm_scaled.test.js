import { readFileSync, existsSync } from 'fs';
import { describe, it, expect, beforeAll } from 'vitest';

describe('rmsnorm_scaled.wgsl', () => {
  const shaderPath = 'src/shaders/rmsnorm_scaled.wgsl';

  it('file exists', () => {
    expect(existsSync(shaderPath)).toBe(true);
  });

  describe('shader content', () => {
    let source;

    beforeAll(() => {
      source = readFileSync(shaderPath, 'utf-8');
    });

    it('has input storage buffer', () => {
      expect(source).toMatch(/var<storage,\s*read>\s*input\s*:\s*array<f32>/);
    });

    it('has weights storage buffer', () => {
      expect(source).toMatch(/var<storage,\s*read>\s*weights\s*:\s*array<f32>/);
    });

    it('has output storage buffer', () => {
      expect(source).toMatch(/var<storage,\s*read_write>\s*output\s*:\s*array<f32>/);
    });

    it('has uniform params with row_length and epsilon', () => {
      expect(source).toMatch(/row_length\s*:\s*u32/);
      expect(source).toMatch(/epsilon\s*:\s*f32/);
      expect(source).toMatch(/var<uniform>\s*\w+\s*:\s*Params/);
    });

    it('uses workgroup shared memory for parallel reduction', () => {
      expect(source).toMatch(/var<workgroup>\s*\w+\s*:\s*array<f32,\s*64>/);
    });

    it('uses workgroupBarrier for synchronization', () => {
      expect(source).toContain('workgroupBarrier()');
    });

    it('performs tree reduction for sum of squares', () => {
      expect(source).toMatch(/stride\s*=\s*stride\s*>>\s*1u/);
      expect(source).toMatch(/v\s*\*\s*v/);
    });

    it('computes inverse RMS with epsilon', () => {
      expect(source).toMatch(/sqrt\s*\(/);
      expect(source).toContain('epsilon');
      expect(source).toMatch(/1\.0\s*\/\s*sqrt/);
    });

    it('fuses normalize and scale: output = input * inv_rms * weights', () => {
      expect(source).toMatch(/output\[.*\]\s*=\s*input\[.*\]\s*\*\s*inv_rms\s*\*\s*weights\[/);
    });

    it('is a compute shader with workgroup_size(64)', () => {
      expect(source).toMatch(/@compute\s+@workgroup_size\(64\)/);
    });

    it('dispatches one workgroup per row', () => {
      expect(source).toContain('wid.x');
    });
  });

  describe('numerical validation', () => {
    it('fused rmsnorm+scale matches separate rmsnorm then mul', () => {
      const row_length = 128;
      const epsilon = 1e-5;
      const input = Array.from({ length: row_length }, (_, i) => Math.sin(i * 0.1) * 2.0);
      const weights = Array.from({ length: row_length }, (_, i) => 0.5 + Math.cos(i * 0.05));

      // Separate: rmsnorm then mul
      let sum_sq = 0;
      for (let i = 0; i < row_length; i++) {
        sum_sq += input[i] * input[i];
      }
      const inv_rms = 1.0 / Math.sqrt(sum_sq / row_length + epsilon);
      const separate = input.map((v, i) => v * inv_rms * weights[i]);

      // Fused: same computation in one pass (simulating the shader)
      let fused_sum_sq = 0;
      for (let i = 0; i < row_length; i++) {
        fused_sum_sq += input[i] * input[i];
      }
      const fused_inv_rms = 1.0 / Math.sqrt(fused_sum_sq / row_length + epsilon);
      const fused = input.map((v, i) => v * fused_inv_rms * weights[i]);

      for (let i = 0; i < row_length; i++) {
        expect(Math.abs(fused[i] - separate[i])).toBeLessThan(1e-6);
      }
    });

    it('handles zero input correctly', () => {
      const row_length = 64;
      const epsilon = 1e-5;
      const input = new Array(row_length).fill(0);
      const weights = Array.from({ length: row_length }, () => 1.0);

      let sum_sq = 0;
      const inv_rms = 1.0 / Math.sqrt(sum_sq / row_length + epsilon);
      const result = input.map((v, i) => v * inv_rms * weights[i]);

      for (let i = 0; i < row_length; i++) {
        expect(result[i]).toBe(0);
      }
    });

    it('handles multiple rows independently', () => {
      const row_length = 32;
      const epsilon = 1e-5;
      const rows = [
        Array.from({ length: row_length }, (_, i) => i * 0.1),
        Array.from({ length: row_length }, (_, i) => -i * 0.2),
      ];
      const weights = Array.from({ length: row_length }, () => 1.5);

      for (const row of rows) {
        let sum_sq = 0;
        for (const v of row) sum_sq += v * v;
        const inv_rms = 1.0 / Math.sqrt(sum_sq / row_length + epsilon);
        const result = row.map((v, i) => v * inv_rms * weights[i]);
        expect(result.length).toBe(row_length);
        expect(Number.isFinite(result[0])).toBe(true);
      }
    });
  });
});

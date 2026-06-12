import { readFileSync, existsSync } from 'fs';
import { describe, it, expect, beforeAll } from 'vitest';

describe('fused_qkv.wgsl', () => {
  const shaderPath = 'src/shaders/fused_qkv.wgsl';

  it('file exists', () => {
    expect(existsSync(shaderPath)).toBe(true);
  });

  describe('shader content', () => {
    let source;

    beforeAll(() => {
      source = readFileSync(shaderPath, 'utf-8');
    });

    it('has input storage buffer for packed f16', () => {
      expect(source).toMatch(/var<storage,\s*read>\s*input\s*:\s*array<u32>/);
    });

    it('has three weight buffers Wq, Wk, Wv', () => {
      expect(source).toMatch(/var<storage,\s*read>\s*Wq\s*:\s*array<u32>/);
      expect(source).toMatch(/var<storage,\s*read>\s*Wk\s*:\s*array<u32>/);
      expect(source).toMatch(/var<storage,\s*read>\s*Wv\s*:\s*array<u32>/);
    });

    it('has three output buffers outQ, outK, outV as f32', () => {
      expect(source).toMatch(/var<storage,\s*read_write>\s*outQ\s*:\s*array<f32>/);
      expect(source).toMatch(/var<storage,\s*read_write>\s*outK\s*:\s*array<f32>/);
      expect(source).toMatch(/var<storage,\s*read_write>\s*outV\s*:\s*array<f32>/);
    });

    it('has uniform params with M, N, K fields', () => {
      expect(source).toMatch(/M\s*:\s*u32/);
      expect(source).toMatch(/N\s*:\s*u32/);
      expect(source).toMatch(/K\s*:\s*u32/);
      expect(source).toMatch(/var<uniform>\s*params\s*:\s*Params/);
    });

    it('has 8 bindings (input, Wq, Wk, Wv, outQ, outK, outV, params)', () => {
      for (let i = 0; i <= 7; i++) {
        expect(source).toContain(`@binding(${i})`);
      }
    });

    it('uses workgroup shared memory for tiling', () => {
      expect(source).toMatch(/var<workgroup>\s*tileA/);
      expect(source).toMatch(/var<workgroup>\s*tileB/);
    });

    it('uses workgroupBarrier for synchronization', () => {
      expect(source).toContain('workgroupBarrier()');
    });

    it('accumulates into three separate accumulators', () => {
      expect(source).toContain('accQ');
      expect(source).toContain('accK');
      expect(source).toContain('accV');
    });

    it('writes to all three outputs', () => {
      expect(source).toMatch(/outQ\[.*\]\s*=\s*accQ/);
      expect(source).toMatch(/outK\[.*\]\s*=\s*accK/);
      expect(source).toMatch(/outV\[.*\]\s*=\s*accV/);
    });

    it('is a compute shader with workgroup_size(16, 16)', () => {
      expect(source).toMatch(/@compute\s+@workgroup_size\(16,\s*16\)/);
    });

    it('reuses tileA across Q, K, V projections (loaded once per tile iteration)', () => {
      const loadTileACalls = (source.match(/load_tile_a_fast/g) || []).length;
      // tileA loaded once per tile, but accumulate called 3 times
      const accumulateCallSites = (source.match(/accumulate\(a_base, b_base, &acc/g) || []).length;
      expect(accumulateCallSites).toBe(3);
      expect(loadTileACalls).toBeGreaterThanOrEqual(1);
    });
  });
});

describe('transformer_layer.h fused_qkv wiring', () => {
  const filePath = 'src/transformer_layer.h';

  it('file exists', () => {
    expect(existsSync(filePath)).toBe(true);
  });

  describe('source content', () => {
    let source;

    beforeAll(() => {
      source = readFileSync(filePath, 'utf-8');
    });

    it('TransformerPipelines has fused_qkv pipeline', () => {
      expect(source).toMatch(/ComputePipeline\s+fused_qkv/);
    });

    it('loads fused_qkv.wgsl shader', () => {
      expect(source).toContain('fused_qkv.wgsl');
    });

    it('has encode_fused_qkv helper function', () => {
      expect(source).toMatch(/encode_fused_qkv\s*\(/);
    });

    it('encode_fused_qkv takes input, Wq, Wk, Wv, outQ, outK, outV', () => {
      expect(source).toMatch(/encode_fused_qkv\([^)]*input_f16[^)]*Wq[^)]*Wk[^)]*Wv[^)]*outQ[^)]*outK[^)]*outV/s);
    });

    it('encode_fused_qkv creates 8 bind group entries', () => {
      expect(source).toMatch(/entries\[8\]/);
    });

    it('transformer_layer_forward calls encode_fused_qkv when q_dim == kv_dim', () => {
      expect(source).toMatch(/if\s*\(\s*q_dim\s*==\s*kv_dim\s*\)/);
      expect(source).toContain('encode_fused_qkv');
    });

    it('falls back to 3 separate matmuls when q_dim != kv_dim', () => {
      expect(source).toContain('} else {');
      const elseBlock = source.slice(source.indexOf('} else {'));
      expect(elseBlock).toContain('dispatch_matmul_tiled_f16_enc');
    });

    it('fused path uses one dispatch instead of three', () => {
      // The fused_qkv call should appear once in the if-branch
      const ifBlock = source.slice(
        source.indexOf('if (q_dim == kv_dim)'),
        source.indexOf('} else {')
      );
      const fusedCalls = (ifBlock.match(/encode_fused_qkv/g) || []).length;
      expect(fusedCalls).toBe(1);

      // The else block should have 3 separate matmul dispatches
      const elseBlock = source.slice(source.indexOf('} else {'));
      const matmulCalls = (elseBlock.match(/dispatch_matmul_tiled_f16_enc/g) || []).length;
      expect(matmulCalls).toBeGreaterThanOrEqual(3);
    });
  });
});

import { readFileSync } from 'fs';
import { describe, it, expect, beforeAll } from 'vitest';

describe('batched prefill', () => {
  let inferenceSource;
  let transformerSource;
  let ropeSource;
  let kvCacheSource;
  let flashAttentionSource;

  beforeAll(() => {
    inferenceSource = readFileSync('src/inference.h', 'utf-8');
    transformerSource = readFileSync('src/transformer_layer.h', 'utf-8');
    ropeSource = readFileSync('src/shaders/rope.wgsl', 'utf-8');
    kvCacheSource = readFileSync('src/shaders/kv_cache_update.wgsl', 'utf-8');
    flashAttentionSource = readFileSync('src/shaders/flash_attention.wgsl', 'utf-8');
  });

  describe('AC1: Prefill processes all tokens in one forward pass with M=seq_len', () => {
    it('calls embedding_lookup_batch for all input tokens at once', () => {
      expect(inferenceSource).toContain('embedding_lookup_batch(');
    });

    it('passes prefill_len (not 1) to transformer_layer_forward', () => {
      const match = inferenceSource.match(
        /transformer_layer_forward\([^)]*,\s*0\s*,\s*prefill_len\s*\)/s
      );
      expect(match).not.toBeNull();
    });

    it('does NOT have a per-token loop in the prefill phase', () => {
      const prefillStart = inferenceSource.indexOf('Prefill phase');
      const decodeStart = inferenceSource.indexOf('Decode phase');
      expect(prefillStart).toBeGreaterThan(-1);
      expect(decodeStart).toBeGreaterThan(-1);
      const prefillSection = inferenceSource.slice(prefillStart, decodeStart);
      const hasPerTokenLoop = /for\s*\(\s*uint32_t\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*prefill_len/.test(prefillSection);
      expect(hasPerTokenLoop).toBe(false);
    });

    it('prefill_len is derived from input_ids.size()', () => {
      expect(inferenceSource).toMatch(
        /prefill_len\s*=\s*static_cast<uint32_t>\(input_ids\.size\(\)\)/
      );
    });
  });

  describe('AC2: All matmuls, RoPE, attention, and KV cache handle M>1', () => {
    it('transformer_layer_forward accepts M parameter', () => {
      expect(transformerSource).toMatch(
        /transformer_layer_forward\([^)]*uint32_t\s+M\s*(=\s*1)?\s*\)/s
      );
    });

    it('RoPE shader reads M from params and computes per-token position', () => {
      expect(ropeSource).toMatch(/token_idx/);
      expect(ropeSource).toMatch(/seq_pos_offset/);
    });

    it('KV cache update shader handles M tokens', () => {
      expect(kvCacheSource).toMatch(/M/);
      expect(kvCacheSource).toMatch(/token_idx/);
    });

    it('flash attention shader supports multi-token query via workgroup_id.y', () => {
      expect(flashAttentionSource).toMatch(/wid\.y/);
    });

    it('attention dispatch uses M for workgroup count', () => {
      expect(transformerSource).toMatch(/q_seq_len/);
    });
  });

  describe('AC3: Output structure supports verification', () => {
    it('extracts last token hidden state from batched output', () => {
      expect(inferenceSource).toContain('prefill_len - 1');
      expect(inferenceSource).toContain('CopyBufferToBuffer');
    });

    it('returns generated tokens in result', () => {
      expect(inferenceSource).toMatch(/result\.tokens\.push_back\(/);
    });
  });

  describe('AC4: Prefill timing is measured for tok/s', () => {
    it('computes prefill_tok_per_sec', () => {
      expect(inferenceSource).toMatch(/prefill_tok_per_sec\s*=/);
    });

    it('divides token count by prefill time', () => {
      expect(inferenceSource).toMatch(
        /input_ids\.size\(\)\s*\)\s*\/\s*prefill_secs/
      );
    });
  });
});

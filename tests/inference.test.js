import { readFileSync, existsSync } from 'fs';
import { describe, it, expect, beforeAll } from 'vitest';

describe('inference.h', () => {
  const filePath = 'src/inference.h';

  it('file exists', () => {
    expect(existsSync(filePath)).toBe(true);
  });

  describe('source content', () => {
    let source;

    beforeAll(() => {
      source = readFileSync(filePath, 'utf-8');
    });

    it('defines GenerateResult struct with tokens field', () => {
      expect(source).toMatch(/struct\s+GenerateResult/);
      expect(source).toMatch(/std::vector<uint32_t>\s+tokens/);
    });

    it('GenerateResult has prefill_tok_per_sec field', () => {
      expect(source).toMatch(/double\s+prefill_tok_per_sec/);
    });

    it('GenerateResult has decode_tok_per_sec field', () => {
      expect(source).toMatch(/double\s+decode_tok_per_sec/);
    });

    it('defines generate() function returning GenerateResult', () => {
      expect(source).toMatch(/GenerateResult\s+generate\s*\(/);
    });

    it('includes weight_loader.h', () => {
      expect(source).toContain('#include "weight_loader.h"');
    });

    it('includes tokenizer.h', () => {
      expect(source).toContain('#include "tokenizer.h"');
    });

    it('includes transformer_layer.h', () => {
      expect(source).toContain('#include "transformer_layer.h"');
    });

    it('includes lm_head.h', () => {
      expect(source).toContain('#include "lm_head.h"');
    });

    it('includes sampling.h', () => {
      expect(source).toContain('#include "sampling.h"');
    });

    it('calls load_weights to load model weights', () => {
      expect(source).toContain('load_weights(');
    });

    it('calls BPETokenizer or tokenizer.encode', () => {
      expect(source).toContain('BPETokenizer');
      expect(source).toMatch(/tokenizer\.encode\(/);
    });

    it('calls transformer_layer_forward', () => {
      expect(source).toContain('transformer_layer_forward(');
    });

    it('calls lm_head_forward', () => {
      expect(source).toContain('lm_head_forward(');
    });

    it('calls sampling functions', () => {
      const hasGreedy = source.includes('sample_greedy(');
      const hasTopk = source.includes('sample_topk(');
      expect(hasGreedy || hasTopk).toBe(true);
    });

    it('has prefill phase with timing', () => {
      expect(source).toContain('prefill_start');
      expect(source).toContain('prefill_end');
      expect(source).toMatch(/prefill_tok_per_sec\s*=/);
    });

    it('has decode phase with timing', () => {
      expect(source).toContain('decode_start');
      expect(source).toContain('decode_end');
      expect(source).toMatch(/decode_tok_per_sec\s*=/);
    });

    it('prefill and decode are timed separately', () => {
      const prefillStartIdx = source.indexOf('prefill_start');
      const prefillEndIdx = source.indexOf('prefill_end');
      const decodeStartIdx = source.indexOf('decode_start');
      const decodeEndIdx = source.indexOf('decode_end');
      expect(prefillStartIdx).toBeLessThan(prefillEndIdx);
      expect(prefillEndIdx).toBeLessThan(decodeStartIdx);
      expect(decodeStartIdx).toBeLessThan(decodeEndIdx);
    });

    it('returns result with generated tokens', () => {
      expect(source).toMatch(/result\.tokens\.push_back\(/);
      expect(source).toMatch(/return\s+result/);
    });
  });
});

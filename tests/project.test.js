import { existsSync } from 'fs';
import { describe, it, expect } from 'vitest';

describe('project structure', () => {
  const requiredFiles = [
    'src/model_arch.h',
    'src/gguf_parser.h',
    'src/mmap_file.h',
    'src/model_config.h',
    'src/gpu_context.h',
    'src/gpu_context.cpp',
    'src/tensor.h',
    'src/main.cpp',
  ];

  it.each(requiredFiles)('source file %s exists', (file) => {
    expect(existsSync(file)).toBe(true);
  });
});

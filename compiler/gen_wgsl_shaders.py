#!/usr/bin/env python3
"""Generate runtimes/cpp/wgsl_shaders.h from compiler/kernels/*.wgsl.

Run this whenever kernel .wgsl files change:
    python compiler/gen_wgsl_shaders.py
"""
import os
import re

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "runtimes", "cpp", "wgsl_shaders.h")


def count_bindings(src: str) -> int:
    bindings = re.findall(r'@binding\((\d+)\)', src)
    return max(int(b) for b in bindings) + 1 if bindings else 0


def main():
    # Collect all .wgsl files
    kernels = []  # (name, category, source, n_bindings)
    for category in sorted(os.listdir(KERNEL_DIR)):
        cat_dir = os.path.join(KERNEL_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith('.wgsl'):
                continue
            name = fname[:-5]  # strip .wgsl
            with open(os.path.join(cat_dir, fname)) as f:
                src = f.read()
            n_bind = count_bindings(src)
            kernels.append((name, category, src, n_bind))

    # Generate header
    lines = []
    lines.append('#pragma once')
    lines.append('/**')
    lines.append(' * wgsl_shaders.h -- Auto-generated from compiler/kernels/*.wgsl')
    lines.append(' * Do not edit manually. Regenerate with: python compiler/gen_wgsl_shaders.py')
    lines.append(' */')
    lines.append('')
    lines.append('#include <string>')
    lines.append('#include <unordered_map>')
    lines.append('')
    lines.append('struct ShaderInfo {')
    lines.append('    const char* source;')
    lines.append('    uint32_t numBindings;')
    lines.append('};')
    lines.append('')

    # Emit each kernel as a raw string literal
    for name, category, src, n_bind in kernels:
        var_name = 'WGSL_' + name.upper()
        lines.append(f'// [{category}] {name} ({n_bind} bindings)')
        lines.append(f'static const char* {var_name} = R"WGSL(')
        lines.append(src.rstrip())
        lines.append(')WGSL";')
        lines.append('')

    # Registry
    lines.append('')
    lines.append('inline const std::unordered_map<std::string, ShaderInfo>& getEmbeddedKernels() {')
    lines.append('    static const std::unordered_map<std::string, ShaderInfo> kernels = {')
    for name, category, src, n_bind in kernels:
        var_name = 'WGSL_' + name.upper()
        lines.append(f'        {{"{name}", {{{var_name}, {n_bind}}}}},')
    lines.append('    };')
    lines.append('    return kernels;')
    lines.append('}')
    lines.append('')

    header = '\n'.join(lines)
    with open(OUTPUT, 'w', newline='\n') as f:
        f.write(header)

    print(f"Generated {OUTPUT}")
    print(f"  {len(kernels)} kernels from {KERNEL_DIR}")
    for name, category, src, n_bind in kernels:
        print(f"    [{category}] {name}: {n_bind} bindings, "
              f"{src.count(chr(10))+1} lines")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate wgsl_shaders.h from runtime/kernels/**/*.wgsl.

Reads every .wgsl file under runtime/kernels/, generates C++ string constants,
and emits a complete wgsl_shaders.h with:
  - ShaderInfo struct
  - All WGSL_* constants (sorted by category then name)
  - getEmbeddedKernels() registry function

Template kernels (containing ${T} markers) produce two constants:
  - WGSL_<NAME>   : f32-instantiated version (valid WGSL)
  - WGSL_<NAME>_T : raw template with ${T} markers (for instantiateTemplate())

Non-template .wgsl files produce just WGSL_<NAME>.

Metadata can be specified in .wgsl files via a comment on the first line:
  // @meta bindings=6 triton=true registry=custom_name noregistry=true

If 'bindings' is not specified, it is auto-counted from @binding(N) annotations.
If 'triton' is not specified, defaults to false.
If 'registry' is specified, the kernel uses that name as the registry key.
If 'noregistry=true', the kernel is excluded from getEmbeddedKernels().

Run this whenever kernel .wgsl files change:
    python runtime/gen_wgsl_shaders.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.join(_HERE, "kernels")
OUTPUT = os.path.join(_HERE, "wgsl_shaders.h")

# ── f32 template instantiation (mirrors wgsl_template.h) ─────────────────────

T_READ_F32 = """
fn t_read(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
    return (*buf)[idx];
}
"""

T_READ_RW_F32 = """
fn t_read_rw(buf: ptr<storage, array<f32>, read_write>, idx: u32) -> f32 {
    return (*buf)[idx];
}
"""

T_WRITE_F32 = """
fn t_write(buf: ptr<storage, array<f32>, read_write>, idx: u32, val: f32) {
    (*buf)[idx] = val;
}
"""

T_WRITE2_F32 = """
fn t_write2(buf: ptr<storage, array<f32>, read_write>, idx: u32, v0: f32, v1: f32) {
    (*buf)[idx] = v0;
    (*buf)[idx + 1u] = v1;
}
"""


def instantiate_f32(src: str) -> str:
    """Apply f32 template substitution (matches C++ instantiateTemplate)."""
    s = src
    s = s.replace("${T}", "f32")
    s = s.replace("${T_DTYPE}", "f32")
    s = s.replace("${T_BYTES}", "4")
    s = s.replace("${T_READ}", T_READ_F32)
    s = s.replace("${T_READ_RW}", T_READ_RW_F32)
    s = s.replace("${T_WRITE2}", T_WRITE2_F32)  # WRITE2 before WRITE
    s = s.replace("${T_WRITE}", T_WRITE_F32)
    return s


def is_template(src: str) -> bool:
    return "${T}" in src


def count_bindings(src: str) -> int:
    """Count bindings by finding max @binding(N) + 1."""
    bindings = re.findall(r'@binding\((\d+)\)', src)
    if not bindings:
        return 0
    return max(int(b) for b in bindings) + 1


def parse_meta(src: str):
    """Parse // @meta key=value from first few lines."""
    meta = {"bindings": None, "triton": False, "registry": None, "noregistry": False}
    for line in src.split("\n")[:5]:
        m = re.match(r'//\s*@meta\s+(.*)', line)
        if m:
            for token in m.group(1).split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    if k == "bindings":
                        meta["bindings"] = int(v)
                    elif k == "triton":
                        meta["triton"] = v.lower() == "true"
                    elif k == "registry":
                        meta["registry"] = v
                    elif k == "noregistry":
                        meta["noregistry"] = v.lower() == "true"
    return meta


# Patterns that are excluded from the registry by default
NOREGISTRY_SUFFIXES = ("_f16", "_f16w", "_f16wb")


def should_register(filename_base: str, meta: dict, is_tmpl: bool) -> bool:
    """Determine if a kernel should be in getEmbeddedKernels()."""
    if meta.get("noregistry"):
        return False
    # Template-only files that end with _t are explicitly named template variants
    # (like slice_t.wgsl) — they produce WGSL_SLICE_T but shouldn't be registered
    # since the non-template WGSL_SLICE is already registered
    if filename_base.endswith("_t") and is_tmpl:
        return False
    # Exclude f16/f16w variants (used via direct constant reference)
    for suffix in NOREGISTRY_SUFFIXES:
        if filename_base.endswith(suffix):
            return False
    return True


def main():
    if not os.path.isdir(KERNEL_DIR):
        print(f"Error: {KERNEL_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    # Collect all kernels grouped by category
    categories = {}
    for cat_name in sorted(os.listdir(KERNEL_DIR)):
        cat_dir = os.path.join(KERNEL_DIR, cat_name)
        if not os.path.isdir(cat_dir):
            continue
        kernels = []
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".wgsl"):
                continue
            fpath = os.path.join(cat_dir, fname)
            with open(fpath) as f:
                src = f.read()
            kernels.append((fname, src))
        if kernels:
            categories[cat_name] = kernels

    # Build output
    lines = []
    lines.append('#pragma once')
    lines.append('// wgsl_shaders.h -- Auto-generated from runtime/kernels/')
    lines.append('// Do not edit manually.  Regenerate with:')
    lines.append('//     python runtime/gen_wgsl_shaders.py')
    lines.append('')
    lines.append('#include <string>')
    lines.append('#include <unordered_map>')
    lines.append('')
    lines.append('struct ShaderInfo {')
    lines.append('    const char* source;')
    lines.append('    uint32_t numBindings;')
    lines.append('    bool isTritonGenerated;  // true = Triton-compiled, false = hand-written WGSL')
    lines.append('};')
    lines.append('')

    # Track registry entries
    registry_entries = []  # (key, var_name, bindings, triton)

    total_constants = 0

    for cat_name, kernels in categories.items():
        lines.append(f'// {"─" * 3} [{cat_name}] {"─" * (60 - len(cat_name))}')
        lines.append('')

        for fname, src in kernels:
            name = fname[:-5]  # strip .wgsl
            var_name = "WGSL_" + name.upper()
            meta = parse_meta(src)
            tmpl = is_template(src)

            # Determine binding count
            bindings = meta["bindings"]
            if bindings is None:
                bindings = count_bindings(src)

            if tmpl:
                # Template kernel: emit f32-instantiated WGSL_NAME + raw WGSL_NAME_T
                f32_src = instantiate_f32(src)

                # Remove @meta line from emitted source
                f32_clean = re.sub(r'//\s*@meta\s+.*\n', '', f32_src)
                src_clean = re.sub(r'//\s*@meta\s+.*\n', '', src)

                lines.append(f'// [{cat_name}] {name} — f32 instantiated')
                lines.append(f'static const char* {var_name} = R"WGSL(')
                lines.append(f32_clean.strip())
                lines.append(')WGSL";')
                lines.append('')
                total_constants += 1

                t_var = var_name + "_T"
                lines.append(f'// [{cat_name}] {name} — dtype template')
                lines.append(f'static const char* {t_var} = R"WGSL(')
                lines.append(src_clean.strip())
                lines.append(')WGSL";')
                lines.append('')
                total_constants += 1
            else:
                # Non-template: emit WGSL_NAME
                src_clean = re.sub(r'//\s*@meta\s+.*\n', '', src)
                lines.append(f'// [{cat_name}] {name}')
                lines.append(f'static const char* {var_name} = R"WGSL(')
                lines.append(src_clean.strip())
                lines.append(')WGSL";')
                lines.append('')
                total_constants += 1

            # Registry entry
            if should_register(name, meta, tmpl):
                reg_key = meta.get("registry") or name
                reg_var = var_name  # always point to the base constant (not _T)
                registry_entries.append((reg_key, reg_var, bindings, meta["triton"]))

    # Sort registry entries by key for stable output
    registry_entries.sort(key=lambda e: e[0])

    # Emit getEmbeddedKernels()
    lines.append('// ─── Registry ────────────────────────────────────────────────────────────')
    lines.append('')
    lines.append('inline const std::unordered_map<std::string, ShaderInfo>& getEmbeddedKernels() {')
    lines.append('    static const std::unordered_map<std::string, ShaderInfo> kernels = {')
    for reg_key, reg_var, bindings, triton in registry_entries:
        triton_str = "true" if triton else "false"
        lines.append(f'        {{"{reg_key}", {{{reg_var}, {bindings}, {triton_str}}}}},')
    lines.append('    };')
    lines.append('    return kernels;')
    lines.append('}')
    lines.append('')

    with open(OUTPUT, "w", newline="\n") as f:
        f.write("\n".join(lines))

    print(f"Generated {OUTPUT}")
    print(f"  {len(categories)} categories, {total_constants} constants, {len(registry_entries)} registry entries")
    print(f"  File size: {os.path.getsize(OUTPUT)} bytes")


if __name__ == "__main__":
    main()

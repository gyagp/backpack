#!/usr/bin/env python3
"""Update wgsl_shaders.h in-place from runtime/kernels/*.wgsl.

Reads the existing wgsl_shaders.h and patches only the constants that
correspond to .wgsl files.  Manual entries (Q8, Q4, _F16, etc.) are
preserved untouched.

Template kernels (containing ${T} markers) produce two constants:
  - WGSL_<NAME>   : f32-instantiated version (valid WGSL, backward compat)
  - WGSL_<NAME>_T : raw template with ${T} markers (for instantiateTemplate())

Non-template .wgsl files produce just WGSL_<NAME> as before.

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


def replace_constant(content: str, var_name: str, new_src: str) -> str:
    """Replace a WGSL constant's content in wgsl_shaders.h."""
    pattern = (
        r'(//[^\n]*\n)?'                       # optional comment line
        r'static const char\* ' + re.escape(var_name) +
        r' = R"WGSL\(' +
        r'.*?' +                                # old content (non-greedy)
        r'\)WGSL";'
    )
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None  # not found
    replacement = f'static const char* {var_name} = R"WGSL(\n{new_src.rstrip()}\n)WGSL";'
    return content[:match.start()] + replacement + content[match.end():]


def insert_after_constant(content: str, after_var: str, var_name: str,
                          src: str, comment: str = "") -> str:
    """Insert a new constant right after an existing one."""
    # Find end of the 'after_var' constant
    pattern = (
        r'static const char\* ' + re.escape(after_var) +
        r' = R"WGSL\(.*?\)WGSL";'
    )
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    insert_pos = match.end()

    block = "\n\n"
    if comment:
        block += f"// {comment}\n"
    block += f'static const char* {var_name} = R"WGSL(\n{src.rstrip()}\n)WGSL";'
    return content[:insert_pos] + block + content[insert_pos:]


def insert_at_end(content: str, var_name: str, src: str,
                  comment: str = "") -> str:
    """Insert a new constant before getEmbeddedKernels or #endif."""
    # Try to insert before getEmbeddedKernels function
    marker = "inline const std::unordered_map"
    pos = content.find(marker)
    if pos < 0:
        # fallback: before last #endif
        pos = content.rfind("#endif")
    if pos < 0:
        return None
    block = ""
    if comment:
        block += f"// {comment}\n"
    block += f'static const char* {var_name} = R"WGSL(\n{src.rstrip()}\n)WGSL";\n\n'
    return content[:pos] + block + content[pos:]


def main():
    if not os.path.exists(OUTPUT):
        print(f"Error: {OUTPUT} not found. Cannot update in-place.", file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT, "r") as f:
        content = f.read()

    original_len = len(content)

    # Collect template .wgsl files
    updated = 0
    added = 0

    for category in sorted(os.listdir(KERNEL_DIR)):
        cat_dir = os.path.join(KERNEL_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".wgsl"):
                continue
            name = fname[:-5]
            var_name = "WGSL_" + name.upper()
            with open(os.path.join(cat_dir, fname)) as f:
                src = f.read()

            if not is_template(src):
                # Non-template: update WGSL_<NAME> with latest .wgsl content
                result = replace_constant(content, var_name, src)
                if result:
                    content = result
                    updated += 1
                else:
                    # New non-template — insert at end
                    comment = f"[{category}] {name} (auto-generated from {fname})"
                    result = insert_at_end(content, var_name, src, comment)
                    if result:
                        content = result
                        added += 1
                        print(f"  Added {var_name}")
                    else:
                        print(f"  WARNING: could not insert {var_name}")
                continue

            # Template kernel: update WGSL_<NAME> with f32-instantiated version
            f32_src = instantiate_f32(src)
            result = replace_constant(content, var_name, f32_src)
            if result:
                content = result
                updated += 1
            else:
                # New template base — insert at end
                comment = f"[{category}] {name} — f32 instantiated (auto-generated from {fname})"
                result = insert_at_end(content, var_name, f32_src, comment)
                if result:
                    content = result
                    added += 1
                    print(f"  Added {var_name}")
                else:
                    print(f"  WARNING: could not insert {var_name}")

            # Add or update WGSL_<NAME>_T with raw template
            t_var = var_name + "_T"
            result = replace_constant(content, t_var, src)
            if result:
                content = result
                updated += 1
                print(f"  Updated {t_var}")
            else:
                # _T doesn't exist yet — insert after WGSL_<NAME>
                comment = f"[{category}] {name} — dtype template (use instantiateTemplate())"
                result = insert_after_constant(content, var_name, t_var, src, comment)
                if result:
                    content = result
                    added += 1
                    print(f"  Added {t_var}")
                else:
                    # Fallback: insert at end
                    result = insert_at_end(content, t_var, src, comment)
                    if result:
                        content = result
                        added += 1
                        print(f"  Added {t_var} (at end)")
                    else:
                        print(f"  WARNING: could not insert {t_var}")

    with open(OUTPUT, "w", newline="\n") as f:
        f.write(content)

    print(f"\nUpdated {OUTPUT}")
    print(f"  {updated} constants updated, {added} new _T constants added")
    print(f"  File size: {original_len} -> {len(content)} bytes")


if __name__ == "__main__":
    main()

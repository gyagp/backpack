**Purpose**
This is a C++ outline for the same subgroup-matrix Vulkan failure already captured in the Python repros, but written in a style that is easier to transplant into a local Dawn sample, scratch executable, or test harness.

Source outline: [docs/dawn_subgroup_matrix_cpp_repro_outline.cc](docs/dawn_subgroup_matrix_cpp_repro_outline.cc)

**What It Covers**
The outline keeps only the steps needed to reproduce the bug:
1. Create a Dawn `WGPUInstance`.
2. Request a Vulkan adapter with `allow_unsafe_apis` enabled.
3. Create a device requesting `Subgroups` and `ChromiumExperimentalSubgroupMatrix` when present.
4. Create a one-binding storage-buffer bind group layout.
5. Try to compile three compute pipelines:
   1. feature-only shader
   2. subgroup-matrix type declaration shader
   3. load/MMA/store shader

**Expected Behavior From Current Local Runs**
On the current vendored Dawn + NVIDIA Vulkan setup:
1. The feature-only shader compiles.
2. The type declaration shader fails with `Unknown configuration is M(8), N(0), K(8), f32`.
3. The load/MMA/store shader fails with the same reflected configuration error.

**Why This Exists Alongside The Raw Python Repro**
The raw Python repro in [docs/dawn_subgroup_matrix_raw_repro.py](docs/dawn_subgroup_matrix_raw_repro.py) is already enough for filing. The C++ outline is useful when someone on the Dawn side wants something that maps more directly onto their normal debugging environment.

**Practical Use**
The `.cc` file is intentionally written as an outline rather than a checked-in build target. The simplest way to use it is:
1. Drop it into a local Dawn scratch target.
2. Replace the proc-table note with the exact proc initialization expected by that build.
3. Keep the three WGSL shader strings unchanged.
4. Confirm the same pass/fail sequence.
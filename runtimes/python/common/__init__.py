"""Python runtime common library — WebGPU model base, kernels, profiler."""
import sys, os

# Ensure 'common.*' imports resolve correctly from within this package.
# model_base.py does 'from common.wgsl_kernels import ...' — this works
# because we add our parent dir (runtimes/python/) to sys.path, making
# 'common' resolve to this package.
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(os.path.dirname(_parent))
for p in [_parent, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

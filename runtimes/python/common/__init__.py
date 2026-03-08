"""Python runtime common library — WebGPU model base, kernels, profiler."""
import sys, os

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # runtimes/python
_root = os.path.dirname(os.path.dirname(_parent))                       # project root
for p in [_parent, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

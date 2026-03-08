"""Python runtime common library — WebGPU model base, kernels, profiler."""
import sys, os

# Ensure 'common.*' imports resolve correctly from within this package.
# model_base.py does 'from common.kernels import ...' which goes through
# the proxy stubs in models/common/ -> compiler/kernels.py
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # runtimes/python
_root = os.path.dirname(os.path.dirname(_parent))                       # project root
_models = os.path.join(_root, 'models')                                 # models/
for p in [_parent, _root, _models]:
    if p not in sys.path:
        sys.path.insert(0, p)

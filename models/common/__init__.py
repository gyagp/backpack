"""Compatibility shim: models/common/ -> runtimes/python/common/ + compiler/

Legacy per-model scripts import 'from common.model_base import ...'.
This package proxies those imports to the new canonical locations.
"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [os.path.join(_root, 'runtimes', 'python'), _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

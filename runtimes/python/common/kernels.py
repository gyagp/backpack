"""Proxy: runtimes/python/common/kernels.py -> compiler/kernels.py (legacy)"""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _root not in sys.path:
    sys.path.insert(0, _root)
from compiler.kernels import *  # noqa: F401,F403

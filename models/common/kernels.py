"""Proxy: models/common/kernels.py -> compiler/kernels.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from compiler.kernels import *  # noqa: F401,F403

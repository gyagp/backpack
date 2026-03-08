"""Proxy: models/common/gdn_kernel.py -> compiler/gdn_kernel.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from compiler.gdn_kernel import *  # noqa: F401,F403

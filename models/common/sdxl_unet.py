"""Proxy: models/common/sdxl_unet.py -> compiler/sdxl_unet.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from compiler.sdxl_unet import *  # noqa: F401,F403

"""Kernel registry — format-specific kernel collections.

Each format module exports the kernels needed for that model format.
Shared kernels (norm, attention, activation) are common to all formats.
Format-specific kernels (Q8 matmul, Q4 matmul, etc.) are isolated.

Usage:
    from runtimes.python.kernels.gguf_q8 import Q8_KERNELS
    from runtimes.python.kernels.shared import SHARED_KERNELS
"""

"""
Comprehensive CPU+GPU profiler for WebGPU model inference.

Provides:
  - GPUTimestampProfiler: GPU-side timing via WebGPU timestamp queries
  - CPUProfiler: CPU-side timing with hierarchical scopes
  - InferenceProfiler: unified profiler that correlates CPU and GPU timelines

Usage:
    from common.profiler import InferenceProfiler

    profiler = InferenceProfiler(model.cache.runner)

    # Option 1: Manual scoping
    profiler.begin_step("decode")
    profiler.begin_scope("layer_0")
    profiler.begin_gpu("qkv_linear")
    result = model._linear(...)  # GPU kernel
    profiler.end_gpu("qkv_linear")
    profiler.end_scope("layer_0")
    profiler.end_step()

    # Option 2: Context manager
    with profiler.step("decode"):
        with profiler.scope("layer_0"):
            with profiler.gpu("qkv_linear"):
                result = model._linear(...)

    profiler.report()
"""
import time
import ctypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import numpy as np

# D3D12 GPU timestamp frequency — used to convert GPU ticks to nanoseconds.
# For NVIDIA GPUs on D3D12, the timestamp period is typically 1ns.
# We'll query the actual period from the resolve buffer results.
_GPU_NS_PER_TICK = 1.0  # Adjusted at runtime if needed


# ---------------------------------------------------------------------------
# GPU Timestamp Profiler
# ---------------------------------------------------------------------------

@dataclass
class GPUTimestamp:
    """A pair of GPU timestamps for a compute dispatch."""
    name: str
    begin_ns: int = 0
    end_ns: int = 0
    cpu_submit_ns: int = 0  # CPU time when the dispatch was submitted

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.begin_ns) / 1000.0

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.begin_ns) / 1_000_000.0


class GPUTimestampProfiler:
    """Manages WebGPU timestamp queries for GPU-side profiling.

    Creates a timestamp query set and resolve buffer. Each profiled
    compute pass writes two timestamps (begin/end) which are resolved
    to a GPU buffer and read back after the command buffer completes.
    """

    # Maximum number of timestamp pairs (begin+end) per profiling session
    # 32 layers * 12 ops * 11 steps + overhead ≈ 4500 dispatches = 9000 timestamps
    MAX_TIMESTAMPS = 16384  # 8192 profiled dispatches

    def __init__(self, runner):
        self._runner = runner
        self._lib = runner._lib
        self._device = runner._device
        self._queue = runner._queue
        self._enabled = runner.has_timestamp_query
        self._query_set = None
        self._resolve_buf = None
        self._next_index = 0
        self._timestamps: List[GPUTimestamp] = []
        self._pending_names: Dict[int, str] = {}

        if self._enabled:
            self._create_query_set()

    def _create_query_set(self):
        """Create the timestamp query set and resolve buffer."""
        from triton.backends.webgpu.dawn_runner import (
            WGPUQuerySetDescriptor, WGPUStringView,
            WGPUBufferDescriptor,
            BUFFER_USAGE_COPY_SRC, BUFFER_USAGE_COPY_DST,
            BUFFER_USAGE_MAP_READ, BUFFER_USAGE_STORAGE,
        )

        desc = WGPUQuerySetDescriptor()
        desc.nextInChain = None
        desc.label = WGPUStringView.from_str("profiler_timestamps")
        desc.type = 0x00000002  # WGPUQueryType_Timestamp
        desc.count = self.MAX_TIMESTAMPS

        self._query_set = self._lib.wgpuDeviceCreateQuerySet(
            self._device, ctypes.byref(desc))

        # Resolve buffer: each timestamp is a uint64 (8 bytes)
        resolve_size = self.MAX_TIMESTAMPS * 8
        buf_desc = WGPUBufferDescriptor()
        buf_desc.nextInChain = None
        buf_desc.label = WGPUStringView.from_str("profiler_resolve")
        buf_desc.usage = (BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST
                          | BUFFER_USAGE_STORAGE)
        buf_desc.size = resolve_size
        buf_desc.mappedAtCreation = 0
        self._resolve_buf = self._lib.wgpuDeviceCreateBuffer(
            self._device, ctypes.byref(buf_desc))

        # Readback buffer for mapping
        rb_desc = WGPUBufferDescriptor()
        rb_desc.nextInChain = None
        rb_desc.label = WGPUStringView.from_str("profiler_readback")
        rb_desc.usage = BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST
        rb_desc.size = resolve_size
        rb_desc.mappedAtCreation = 0
        self._readback_buf = self._lib.wgpuDeviceCreateBuffer(
            self._device, ctypes.byref(rb_desc))

    @property
    def enabled(self) -> bool:
        return self._enabled and self._query_set is not None

    def reset(self):
        """Reset for a new profiling session."""
        self._next_index = 0
        self._timestamps.clear()
        self._pending_names.clear()

    def allocate_pair(self, name: str) -> Tuple[int, int]:
        """Allocate a pair of timestamp indices for begin/end.

        Returns (begin_index, end_index).
        Records the CPU submit time for later correlation.
        """
        if not self.enabled or self._next_index + 2 > self.MAX_TIMESTAMPS:
            return (-1, -1)
        begin = self._next_index
        end = self._next_index + 1
        self._next_index += 2
        cpu_ns = time.perf_counter_ns()
        self._pending_names[begin] = (name, cpu_ns)
        return (begin, end)

    def get_timestamp_writes_ptr(self, begin_idx: int, end_idx: int):
        """Create a WGPUPassTimestampWrites struct for a compute pass.

        Returns a ctypes pointer suitable for pass_desc.timestampWrites.
        """
        if begin_idx < 0:
            return None
        from triton.backends.webgpu.dawn_runner import WGPUPassTimestampWrites
        ts_writes = WGPUPassTimestampWrites()
        ts_writes.nextInChain = None
        ts_writes.querySet = self._query_set
        ts_writes.beginningOfPassWriteIndex = begin_idx
        ts_writes.endOfPassWriteIndex = end_idx
        # Keep alive
        self._current_ts_writes = ts_writes
        return ctypes.byref(ts_writes)

    def resolve_and_read(self):
        """Resolve all pending timestamps and read results.

        Call this after all profiled dispatches have been submitted.
        Encodes a resolve + copy + readback in a new command buffer.
        """
        if not self.enabled or self._next_index == 0:
            return

        from triton.backends.webgpu.dawn_runner import (
            WGPUCommandEncoderDescriptor, WGPUCommandBufferDescriptor,
            WGPUStringView, BufferMapCallback, WGPUBufferMapCallbackInfo,
            WGPUCallbackMode, WGPUFutureWaitInfo, WGPUMapAsyncStatus,
            MAP_MODE_READ, WGPUCommandBuffer,
        )

        lib = self._lib
        count = self._next_index

        # Ensure all prior GPU work is complete before resolving timestamps
        lib.wgpuDeviceTick(self._device)

        # Encode resolve + copy
        enc_desc = WGPUCommandEncoderDescriptor()
        enc_desc.nextInChain = None
        enc_desc.label = WGPUStringView.from_str("profiler_resolve")
        encoder = lib.wgpuDeviceCreateCommandEncoder(
            self._device, ctypes.byref(enc_desc))

        lib.wgpuCommandEncoderResolveQuerySet(
            encoder, self._query_set, 0, count,
            self._resolve_buf, 0)

        # Copy resolve → readback
        lib.wgpuCommandEncoderCopyBufferToBuffer(
            encoder, self._resolve_buf, 0,
            self._readback_buf, 0, count * 8)

        cb_desc = WGPUCommandBufferDescriptor()
        cb_desc.nextInChain = None
        cb_desc.label = WGPUStringView.from_str("")
        cmd_buf = lib.wgpuCommandEncoderFinish(encoder, ctypes.byref(cb_desc))
        cmd_bufs = (ctypes.c_void_p * 1)(cmd_buf)
        lib.wgpuQueueSubmit(self._queue, 1, cmd_bufs)

        # Map readback buffer
        map_done = [False]
        map_status = [0]

        @BufferMapCallback
        def on_map(status, message, ud1, ud2,
                   _md=map_done, _ms=map_status):
            _md[0] = True
            _ms[0] = status

        self._map_cb = on_map
        cb_info = WGPUBufferMapCallbackInfo()
        cb_info.nextInChain = None
        cb_info.mode = WGPUCallbackMode.WaitAnyOnly
        cb_info.callback = on_map
        cb_info.userdata1 = None
        cb_info.userdata2 = None

        future = lib.wgpuBufferMapAsync(
            self._readback_buf, MAP_MODE_READ, 0, count * 8, cb_info)
        wait_info = WGPUFutureWaitInfo()
        wait_info.future = future
        wait_info.completed = 0
        lib.wgpuInstanceWaitAny(
            self._runner._instance, 1,
            ctypes.byref(wait_info), ctypes.c_uint64(-1))

        if map_status[0] != WGPUMapAsyncStatus.Success:
            print(f"[Profiler] Failed to map readback buffer: {map_status[0]}")
            lib.wgpuCommandEncoderRelease(encoder)
            lib.wgpuCommandBufferRelease(cmd_buf)
            return

        data_ptr = lib.wgpuBufferGetConstMappedRange(
            self._readback_buf, 0, count * 8)
        raw = (ctypes.c_uint8 * (count * 8)).from_address(data_ptr)
        values = np.frombuffer(bytes(raw), dtype=np.uint64)

        lib.wgpuBufferUnmap(self._readback_buf)
        lib.wgpuCommandEncoderRelease(encoder)
        lib.wgpuCommandBufferRelease(cmd_buf)

        # Parse timestamp pairs
        for begin_idx, (name, cpu_ns) in sorted(self._pending_names.items()):
            end_idx = begin_idx + 1
            if end_idx < len(values):
                ts = GPUTimestamp(
                    name=name,
                    begin_ns=int(values[begin_idx]),
                    end_ns=int(values[end_idx]),
                    cpu_submit_ns=cpu_ns,
                )
                self._timestamps.append(ts)

    @property
    def timestamps(self) -> List[GPUTimestamp]:
        return self._timestamps

    def destroy(self):
        """Release GPU resources."""
        if self._query_set:
            self._lib.wgpuQuerySetDestroy(self._query_set)
            self._lib.wgpuQuerySetRelease(self._query_set)
            self._query_set = None
        if self._resolve_buf:
            self._lib.wgpuBufferDestroy(self._resolve_buf)
            self._lib.wgpuBufferRelease(self._resolve_buf)
            self._resolve_buf = None
        if self._readback_buf:
            self._lib.wgpuBufferDestroy(self._readback_buf)
            self._lib.wgpuBufferRelease(self._readback_buf)
            self._readback_buf = None


# ---------------------------------------------------------------------------
# CPU Profiler
# ---------------------------------------------------------------------------

@dataclass
class CPUEvent:
    """A CPU timing event."""
    name: str
    scope: str  # hierarchical scope path (e.g., "decode/layer_0/qkv")
    begin_ns: int
    end_ns: int = 0
    gpu_dispatch: Optional[str] = None  # associated GPU dispatch name

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.begin_ns) / 1000.0

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.begin_ns) / 1_000_000.0


@dataclass
class GPUDispatchEvent:
    """A GPU dispatch timed via CPU clock (perf_counter_ns).

    Since each run_kernel() call is synchronous (submit → poll → readback),
    the CPU begin/end timestamps tightly bracket the GPU execution time.
    This gives us a unified timeline even without hardware GPU timestamps.
    """
    name: str
    begin_ns: int  # CPU perf_counter_ns before dispatch
    end_ns: int    # CPU perf_counter_ns after readback complete

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.begin_ns) / 1000.0

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.begin_ns) / 1_000_000.0


class CPUProfiler:
    """CPU-side hierarchical profiler using time.perf_counter_ns."""

    def __init__(self):
        self._events: List[CPUEvent] = []
        self._scope_stack: List[str] = []
        self._active: Dict[str, CPUEvent] = {}

    def reset(self):
        self._events.clear()
        self._scope_stack.clear()
        self._active.clear()

    @property
    def current_scope(self) -> str:
        return "/".join(self._scope_stack) if self._scope_stack else ""

    def push_scope(self, name: str):
        self._scope_stack.append(name)

    def pop_scope(self):
        if self._scope_stack:
            self._scope_stack.pop()

    def begin(self, name: str, gpu_dispatch: str = None):
        scope = self.current_scope
        full_name = f"{scope}/{name}" if scope else name
        event = CPUEvent(
            name=name,
            scope=scope,
            begin_ns=time.perf_counter_ns(),
            gpu_dispatch=gpu_dispatch,
        )
        self._active[full_name] = event

    def end(self, name: str):
        scope = self.current_scope
        full_name = f"{scope}/{name}" if scope else name
        if full_name in self._active:
            event = self._active.pop(full_name)
            event.end_ns = time.perf_counter_ns()
            self._events.append(event)

    @property
    def events(self) -> List[CPUEvent]:
        return self._events


# ---------------------------------------------------------------------------
# Unified Inference Profiler
# ---------------------------------------------------------------------------

class InferenceProfiler:
    """Unified CPU+GPU profiler for WebGPU model inference.

    Correlates CPU and GPU timelines to provide a complete picture of
    where time is spent during inference. Supports hierarchical scoping
    for layer-level and op-level breakdowns.

    Usage:
        profiler = InferenceProfiler(runner)
        profiler.enable()

        with profiler.step("decode_token_1"):
            with profiler.scope("layer_0"):
                with profiler.gpu("qkv_linear"):
                    # ... GPU dispatch ...
                with profiler.cpu("rope"):
                    # ... CPU compute ...

        profiler.finish()  # resolve GPU timestamps
        profiler.report()
    """

    def __init__(self, runner=None):
        self._runner = runner
        self._cpu = CPUProfiler()
        self._gpu = GPUTimestampProfiler(runner) if runner else None
        self._dispatch_events: List[GPUDispatchEvent] = []
        self._enabled = False
        self._step_name = None

    def enable(self):
        """Enable profiling."""
        self._enabled = True

    def disable(self):
        """Disable profiling."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def gpu_enabled(self) -> bool:
        return self._gpu is not None and self._gpu.enabled

    def reset(self):
        """Reset all profiling data for a new session."""
        self._cpu.reset()
        if self._gpu:
            self._gpu.reset()
        self._dispatch_events.clear()

    def record_dispatch(self, name: str, begin_ns: int, end_ns: int):
        """Record a GPU dispatch event timed via CPU clock.

        Called by KernelCache.run() to record the CPU-timed duration of
        each GPU dispatch (submit → poll → readback). This gives a unified
        timeline on a single clock (perf_counter_ns).
        """
        if self._enabled:
            self._dispatch_events.append(
                GPUDispatchEvent(name=name, begin_ns=begin_ns, end_ns=end_ns))

    @contextmanager
    def step(self, name: str):
        """Context manager for a top-level inference step (e.g., one token)."""
        if not self._enabled:
            yield
            return
        self._step_name = name
        self._cpu.push_scope(name)
        self._cpu.begin("total")
        try:
            yield
        finally:
            self._cpu.end("total")
            self._cpu.pop_scope()

    @contextmanager
    def scope(self, name: str):
        """Context manager for a named scope (e.g., a layer)."""
        if not self._enabled:
            yield
            return
        self._cpu.push_scope(name)
        self._cpu.begin("scope_total")
        try:
            yield
        finally:
            self._cpu.end("scope_total")
            self._cpu.pop_scope()

    @contextmanager
    def cpu(self, name: str):
        """Context manager for a CPU-side operation."""
        if not self._enabled:
            yield
            return
        self._cpu.begin(name)
        try:
            yield
        finally:
            self._cpu.end(name)

    @contextmanager
    def gpu(self, name: str):
        """Context manager for a GPU dispatch.

        Records both CPU-side timing and GPU timestamps (if available).
        The GPU timestamps capture actual shader execution time on the GPU,
        while the CPU timing captures the full dispatch overhead (buffer
        setup, submission, readback).
        """
        if not self._enabled:
            yield
            return
        # CPU timing
        gpu_name = f"{self._cpu.current_scope}/{name}" if self._cpu.current_scope else name
        self._cpu.begin(name, gpu_dispatch=gpu_name)
        try:
            yield
        finally:
            self._cpu.end(name)

    def allocate_gpu_timestamps(self, name: str) -> Tuple[int, int]:
        """Allocate GPU timestamp indices for a dispatch.

        Returns (begin_index, end_index) for use with timestamp writes.
        Returns (-1, -1) if GPU profiling is not available.
        """
        if not self._enabled or not self.gpu_enabled:
            return (-1, -1)
        scope = self._cpu.current_scope
        full_name = f"{scope}/{name}" if scope else name
        return self._gpu.allocate_pair(full_name)

    def get_timestamp_writes_ptr(self, begin_idx: int, end_idx: int):
        """Get the WGPUPassTimestampWrites pointer for a compute pass."""
        if not self._enabled or not self.gpu_enabled or begin_idx < 0:
            return None
        return self._gpu.get_timestamp_writes_ptr(begin_idx, end_idx)

    def finish(self):
        """Finish profiling: resolve GPU timestamps and correlate timelines."""
        if self._gpu and self._gpu.enabled:
            self._gpu.resolve_and_read()

    def report(self, top_n: int = 20):
        """Print a comprehensive profiling report.

        Shows:
          1. CPU timeline summary (hierarchical)
          2. GPU dispatch timeline (CPU-timed)
          3. GPU kernel timeline (HW timestamps)
          4. Combined hotspot analysis
        """
        cpu_events = self._cpu.events
        gpu_timestamps = self._gpu.timestamps if self._gpu else []

        print("\n" + "=" * 70)
        print("  INFERENCE PROFILING REPORT")
        print("=" * 70)

        # --- CPU Summary ---
        print("\n--- CPU Timeline ---")
        if cpu_events:
            # Group by scope
            scope_totals: Dict[str, float] = {}
            op_times: Dict[str, List[float]] = {}
            total_cpu_ms = 0

            for e in cpu_events:
                key = f"{e.scope}/{e.name}" if e.scope else e.name
                if key not in op_times:
                    op_times[key] = []
                op_times[key].append(e.duration_ms)

                if e.name == "total" and "/" not in e.scope:
                    total_cpu_ms += e.duration_ms

            # Aggregate by operation type across layers
            agg: Dict[str, Tuple[float, int]] = {}  # name -> (total_ms, count)
            for key, times in op_times.items():
                parts = key.split("/")
                # Use the last part as the operation name
                op_name = parts[-1] if parts else key
                if op_name in ("total", "scope_total"):
                    continue
                if op_name not in agg:
                    agg[op_name] = (0.0, 0)
                agg[op_name] = (agg[op_name][0] + sum(times),
                                agg[op_name][1] + len(times))

            if total_cpu_ms > 0:
                print(f"  Total CPU time: {total_cpu_ms:.2f}ms")
            print(f"  {'Operation':<30s} {'Total':>8s} {'Count':>6s} "
                  f"{'Avg':>8s} {'%':>6s}")
            print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")

            for name, (total, count) in sorted(agg.items(),
                                                key=lambda x: -x[1][0]):
                avg = total / count if count else 0
                pct = (total / total_cpu_ms * 100) if total_cpu_ms > 0 else 0
                print(f"  {name:<30s} {total:>7.2f}ms {count:>5d}x "
                      f"{avg:>7.3f}ms {pct:>5.1f}%")
        else:
            print("  No CPU events recorded.")

        # --- GPU Dispatch Timeline (CPU-timed) ---
        print("\n--- GPU Dispatches (CPU-timed) ---")
        dispatch_events = self._dispatch_events
        if dispatch_events:
            disp_agg: Dict[str, Tuple[float, int]] = {}
            total_disp_ms = 0

            for de in dispatch_events:
                parts = de.name.split("/")
                op_name = parts[-1] if parts else de.name
                if op_name not in disp_agg:
                    disp_agg[op_name] = (0.0, 0)
                dur = de.duration_ms
                disp_agg[op_name] = (disp_agg[op_name][0] + dur,
                                     disp_agg[op_name][1] + 1)
                total_disp_ms += dur

            print(f"  Total GPU dispatch time: {total_disp_ms:.2f}ms "
                  f"({len(dispatch_events)} dispatches)")
            print(f"  {'Operation':<30s} {'Total':>8s} {'Count':>6s} "
                  f"{'Avg':>8s} {'%':>6s}")
            print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")

            for name, (total, count) in sorted(disp_agg.items(),
                                                key=lambda x: -x[1][0]):
                avg = total / count if count else 0
                pct = (total / total_disp_ms * 100) if total_disp_ms > 0 else 0
                print(f"  {name:<30s} {total:>7.2f}ms {count:>5d}x "
                      f"{avg:>7.3f}ms {pct:>5.1f}%")
        else:
            print("  No GPU dispatches recorded.")

        # --- GPU HW Timestamps ---
        print("\n--- GPU Kernels (HW Timestamps) ---")
        if gpu_timestamps:
            hw_agg: Dict[str, Tuple[float, int]] = {}
            total_hw_ms = 0

            for ts in gpu_timestamps:
                parts = ts.name.split("/")
                op_name = parts[-1] if parts else ts.name
                if op_name not in hw_agg:
                    hw_agg[op_name] = (0.0, 0)
                dur = ts.duration_ms
                hw_agg[op_name] = (hw_agg[op_name][0] + dur,
                                   hw_agg[op_name][1] + 1)
                total_hw_ms += dur

            print(f"  Total GPU HW time: {total_hw_ms:.2f}ms "
                  f"({len(gpu_timestamps)} kernels)")
            print(f"  {'Kernel':<30s} {'Total':>8s} {'Count':>6s} "
                  f"{'Avg':>8s} {'%':>6s}")
            print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")

            for name, (total, count) in sorted(hw_agg.items(),
                                                key=lambda x: -x[1][0]):
                avg = total / count if count else 0
                pct = (total / total_hw_ms * 100) if total_hw_ms > 0 else 0
                print(f"  {name:<30s} {total:>7.2f}ms {count:>5d}x "
                      f"{avg:>7.3f}ms {pct:>5.1f}%")
        else:
            print("  No GPU HW timestamps recorded.")

        # --- Hotspot Analysis ---
        print("\n--- Hotspot Analysis ---")
        if cpu_events:
            # Find the top CPU consumers
            all_ops = []
            for e in cpu_events:
                if e.name not in ("total", "scope_total"):
                    all_ops.append((e.duration_ms, e.name,
                                   e.scope, e.gpu_dispatch))

            all_ops.sort(key=lambda x: -x[0])
            print(f"  Top {min(top_n, len(all_ops))} slowest individual operations:")
            for i, (dur, name, scope, gpu) in enumerate(all_ops[:top_n]):
                label = f"{scope}/{name}" if scope else name
                gpu_tag = " [GPU]" if gpu else ""
                print(f"  {i+1:3d}. {dur:>7.3f}ms  {label}{gpu_tag}")

        # --- Op Counts per Phase ---
        print("\n--- Op Counts ---")
        step_ranges = []
        for e in cpu_events:
            if e.name == "total" and "/" not in e.scope:
                step_ranges.append((e.scope, e.begin_ns, e.end_ns))

        if step_ranges:
            def classify(ns):
                for sname, sb, se in step_ranges:
                    if sb <= ns < se:
                        return "prefill" if sname == "prefill" else "decode"
                return "other"

            prefill_cpu = sum(1 for e in cpu_events
                              if e.name not in ("total", "scope_total")
                              and classify(e.begin_ns) == "prefill")
            decode_cpu = sum(1 for e in cpu_events
                             if e.name not in ("total", "scope_total")
                             and classify(e.begin_ns) == "decode")
            prefill_gpu = sum(1 for de in dispatch_events
                              if classify(de.begin_ns) == "prefill")
            decode_gpu = sum(1 for de in dispatch_events
                             if classify(de.begin_ns) == "decode")
            n_decode = max(sum(1 for s in step_ranges if s[0] != "prefill"), 1)

            print(f"  Prefill:   {prefill_cpu} CPU ops, {prefill_gpu} GPU dispatches")
            print(f"  Decode:    {decode_cpu} CPU ops, {decode_gpu} GPU dispatches "
                  f"({n_decode} tokens)")
            print(f"  Per token: {decode_cpu//n_decode} CPU ops, "
                  f"{decode_gpu//n_decode} GPU dispatches")

        print("=" * 70)

    def destroy(self):
        """Release profiler resources."""
        if self._gpu:
            self._gpu.destroy()

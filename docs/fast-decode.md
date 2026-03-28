# Fast Decode

## Overview

Fast decode accelerates token generation by recording GPU dispatches during a
**capture stage** and replaying them on subsequent **replay stages**, skipping
the full ONNX Execute loop (~221 CPU-only ops + op dispatch overhead).

Two stages:
1. **Capture** — Run the model normally for one decode step, recording all GPU
   dispatches (pipelines, bind groups, workgroup sizes) into `CapturedFlush`
   vectors. Bind groups are AddRef'd for reuse. The step produces valid output.
2. **Replay** — Skip the ONNX Execute loop entirely. Update position-dependent
   params and the token ID buffer, then re-encode and submit the captured
   dispatches.

Auto-enabled for compatible models. Use `--no-fast-decode` to opt out.

## ORT WebGPU Architecture (Reference)

ORT's implementation intercepts GPU dispatch calls during capture:

1. **CaptureBegin** — flushes pending work, sets state to `Capturing`
2. **During capture** — saves `{pipeline, bind_group, dispatch_size}` without dispatching
3. **CaptureEnd** — resets state
4. **Replay** — encodes saved commands into compute passes and submits
5. **Buffer management** — `BufferCacheMode::Graph` ensures buffers are never freed

Key design: ORT's kernels are **pure GPU** — no CPU readbacks, no `writeBuffer`
uploads mid-graph. All intermediate data flows GPU-to-GPU through buffer handles.

## Requirements

1. **No dynamic control flow** — No `If`/`Loop`/`Scan` nodes
2. **No CPU readbacks mid-graph** — No `FlushPendingWork()` + `readBuffer()` during execution
3. **Fixed tensor shapes** — All tensor shapes identical between capture and replay
4. **Dedicated buffer management** — Buffers never freed during capture
5. **Warmup required** — Prefill tokens run through normal Execute before capture
6. **Conv or attention layers** — Model must have layers that benefit from fast decode

## LFM2-8B Compatibility

| Requirement | Status | Detail |
|---|---|---|
| No dynamic control flow | OK | Zero If/Loop/Scan nodes |
| Fixed tensor shapes | OK | GPU buffers pre-allocated to max; shapes fixed at decode time |
| No data-dependent shapes | OK | TopK x22 stays fully on GPU |
| No CPU readbacks | OK | 0 syncs per decode step confirmed via profiling |
| Fixed dispatch grid sizes | OK | All workgroup counts from constant tensor dimensions |
| Fixed buffer allocation | OK | Same buffer count (872) and pool hits every step |
| Constant param values | Partial | 24 GQA params vary — tracked via `ReplayParamUpdate` |

The 707 ONNX nodes break down as:
- **486 GPU-dispatch ops** (MatMulNBits x89, Add x70, RMSNorm x60, Mul x60, etc.)
- **221 CPU-only ops** (Reshape x46, Neg x45, Slice x38, Transpose x36, Shape x24, etc.)

CPU-only ops produce metadata but have zero impact on GPU dispatch correctness
for single-token decode.

## Implementation

### Infrastructure (graph_executor.h / graph_executor.cpp)
- `FastDecodeState` enum: `Off`, `Capturing`, `Replaying`
- `CapturedFlush` / `CapturedCommand` structs with pipeline, bind group, dispatch size, bind entries
- `CaptureBegin()` / `CaptureEnd()` — bracket the capture stage
- `ReplayWrites()` — update GQA position params + token ID buffer
- `ReplayDispatches()` — encode+submit captured dispatches
- `ReleaseCaptured()` — release bind groups and clear state
- `RegisterReplayParam()` for GQA position tracking (24 params)
- `CapturedTokenIdBuf` for GatherBlockQuantized embedding token ID update
- Self-clearing `moe_accum` kernel — eliminates MoE output zero-init `writeBuffer`
- `writeBuffer` recording callback in `GPUContext` for capture-time write tracking
- Param pool expansion during capture (prevents wrapping at 512-buffer pool size)

### Application (main.cpp)
- `CheckFastDecodeSupport()` — validates no dynamic control flow, has conv/attention layers
- `EnableFastDecode()` — sets `fastDecodeEnabled`, pre-allocates conv cast f16 buffers
- `RunStep()` implements three paths: replay (fast), capture (first decode after prefill), normal (prefill)
- Conv state pre-allocation: `convCastF16Bufs` for stable buffer handles across stages
- Conv cast flush removal: last captured flush removed, handled manually in replay with pre-built bind groups

### Embedding Token ID Update

The model uses `GatherBlockQuantized` for embedding lookup — a GPU dispatch that
dequantizes Q4 weights using `input_ids`. The op converts int64 to int32 in a
temp `gbq_idx32` buffer, then dispatches dequantization.

During capture, the bind group references this buffer. During replay:

1. **Capture** (`matmul.cpp`): When `fastDecodeState_ == Capturing`, register
   `gbq_idx32` in `capturedTokenIdBufs_`
2. **Replay** (`ReplayWrites()`): Write the new token ID (int32) to the buffer
3. **Caller** (`main.cpp`): Set `executor.replayTokenId_` before `ReplayWrites()`

### Per-Step Replay Updates

During each replay step, only 3 things change:
1. **24 GQA param buffers** — posOffset, pastSeq, totalSeq for 6 attention layers x 4 params
2. **1 GBQ token ID buffer** — new token ID (int32) for embedding lookup
3. **Conv cast dispatches** — f32->f16 conv state casts run after replay (not captured)

All other buffer contents persist from the capture stage.

## Performance

| Mode | tok/s | ms/tok | Improvement |
|---|---|---|---|
| Normal Execute | 54 | 18.6 | — |
| Fast Decode | 59 | 16.9 | **+10%** |

Tested with 200-token decode on LFM2-8B (RTX 5080, D3D12).

The modest speedup is due to Dawn/D3D12 per-submission overhead: encoding 975
dispatches into 19 command buffers takes ~5ms CPU, and the GPU fence wait adds
~10ms, despite only 0.6ms of actual GPU compute. The bottleneck is WebGPU API
call overhead and D3D12 command queue synchronization.

## How It Works

1. **Prefill**: All prompt tokens run through normal `Execute()` (707 ONNX nodes)
2. **Capture**: First decode token triggers `CaptureBegin()`, runs full `Execute()`
   (valid output), records 975 dispatches across 19 flushes. Bind groups AddRef'd.
3. **Replay**: Skip ONNX Execute loop. Update position params + token ID, then
   replay captured dispatches via WebGPU command buffers.

The speedup comes from eliminating ~221 CPU-only ops + op dispatch overhead while
keeping GPU work identical.

# Graph Capture for Fast Decode

## ORT WebGPU Graph Capture Architecture

ORT's graph capture works by intercepting GPU dispatch calls during a "capture" run:

1. **CaptureBegin** — flushes pending work, sets state to `Capturing`
2. **During capture** — `LaunchComputePipeline` creates bind groups but does NOT dispatch them. Instead it saves `{pipeline, bind_group, bind_group_layout, dispatch_size}` into a `CapturedCommandInfo` vector. The bind group is never submitted to any command buffer.
3. **CaptureEnd** — resets state
4. **Replay** — iterates saved commands, encodes each into compute passes, submits. Bind groups are used for the **first time** here. Batches up to 16 dispatches per compute pass.
5. **Buffer management** — `BufferCacheMode::Graph` ensures buffers are never freed. Same-size allocations return the same handle, guaranteeing bind group stability.

Key design: ORT's kernels are **pure GPU** — no CPU readbacks, no `writeBuffer` uploads of CPU-computed values mid-graph. All intermediate data flows GPU-to-GPU through buffer handles.

## ORT Graph Capture Requirements

1. **No dynamic shapes** — All tensor shapes identical between capture and replay
2. **No CPU readbacks mid-graph** — No `FlushPendingWork()` + `readBuffer()` during execution
3. **No dynamic control flow** — No `If`/`Loop`/`Scan` nodes
4. **Dedicated graph buffer manager** — Buffers never freed, deterministic handle allocation
5. **Warmup required** — N normal runs before capture to stabilize allocation patterns
6. **Contrib kernel restrictions** — Some kernels registered differently for graph mode

## LFM2-8B Model Compatibility

| Requirement | Status | Detail |
|---|---|---|
| No dynamic control flow | ✅ | Zero If/Loop/Scan nodes |
| Fixed tensor shapes | ⚠️ | ONNX declares dynamic (batch_size, sequence_length, past_sequence_length), but GPU buffers are fixed size at decode time — pre-allocated to max |
| No data-dependent shapes | ⚠️ | TopK ×22 has data-dep output, but stays fully on GPU (no CPU readback) |
| No CPU readbacks | ✅ | **0 syncs, 0 int-readback** per decode step confirmed via profiling |
| Fixed dispatch grid sizes | ✅ | All workgroup counts determined by constant tensor dimensions |
| Fixed buffer allocation | ✅ | Same buffer count (872) and pool hits every step |
| Constant param values | ⚠️ | Only 24 GQA params vary (posOffset/pastSeq/totalSeq) — tracked via `ReplayParamUpdate` |

The model satisfies all hard requirements. The 707 ONNX nodes break down as:
- **486 GPU-dispatch ops** (MatMulNBits ×89, Add ×70, RMSNorm ×60, Mul ×60, etc.)
- **221 CPU-only ops** (Reshape ×46, Neg ×45, Slice ×38, Transpose ×36, Shape ×24, etc.)

CPU-only ops produce metadata (shapes, positions) but have **zero** impact on GPU dispatch correctness for single-token decode — all position-dependent values are either constant (`-1`, `1` for seq_len=1) or handled by `ReplayParamUpdate` (GQA positions).

## Implementation Status: WORKING

Graph capture is fully functional for LFM2-8B decode. Enabled via `--fast-decode` flag.

### Infrastructure
- `CapturedFlush` / `CapturedCommand` structs with pipeline, bind group, dispatch size, bind entries
- `CaptureBegin()` / `CaptureEnd()` / `Replay()` / `ReplayWrites()` / `ReplayDispatches()`
- `RegisterReplayParam()` for GQA position tracking (24 params: posOffset, pastSeq, totalSeq)
- `CapturedTokenIdBuf` for GatherBlockQuantized embedding token ID update during replay
- `QueueDispatch()` wrapper that attaches bind entries for capture
- `SubmitAsync()` modified to attach bind entries during capture
- Self-clearing `moe_accum` kernel (slot 0: `dst = weight * src`, slot > 0: `dst += weight * src`) — eliminates MoE output zero-init `writeBuffer`
- `writeBuffer` recording callback in `GPUContext` for capture-time write tracking
- Param pool wrapping fix: unique buffer allocation during capture (prevents 993-dispatch wrapping at 512-buffer pool size)
- Conv state pre-allocation: `convCastF16Bufs` used in `ResetCaches` for stable buffer handles
- Conv cast flush removal: last captured flush (conv casts) removed, handled manually in replay path

### Key Fix: Embedding Token ID Update

The model uses `GatherBlockQuantized` for embedding lookup — a GPU dispatch that dequantizes Q4 weights on the fly. The op receives `input_ids` as int64, converts them to int32 in a temporary `gbq_idx32` buffer, then dispatches the dequantization kernel.

During graph capture, the bind group for this dispatch references the `gbq_idx32` buffer. During replay, the buffer still held the capture-step's token ID. The fix:

1. **Capture** (`opGatherBlockQuantized` in `matmul.cpp`): When `graphCaptureState_ == Capturing`, register the `gbq_idx32` buffer in `capturedTokenIdBufs_`
2. **Replay** (`ReplayWrites()` in `graph_executor.cpp`): Before replaying dispatches, write the new token ID (int32) to the registered buffer
3. **Caller** (`main.cpp`): Set `executor.replayTokenId_` before calling `ReplayWrites()`

### Per-Step Replay Updates

During each replay step, only 3 things are updated:
1. **24 GQA param buffers** — posOffset, pastSeq, totalSeq for 6 attention layers × 4 params
2. **1 GBQ token ID buffer** — new token ID (int32) for embedding lookup
3. **Conv cast dispatches** — f32→f16 conv state casts run after replay (not captured)

All other buffer contents persist from the capture step (constant initializer data, conv weights, etc.).

### Performance

| Mode | tok/s | ms/tok | Improvement |
|---|---|---|---|
| Normal Execute | 54 | 18.6 | — |
| Graph Capture Replay | 59 | 16.9 | **+10%** |

Tested with 200-token decode on LFM2-8B (RTX 5080, D3D12).

The modest speedup is due to Dawn/D3D12 per-submission overhead: encoding 975 dispatches into 19 command buffers takes ~5ms CPU, and the GPU fence wait adds ~10ms, despite only 0.6ms of actual GPU compute. The bottleneck is the WebGPU API call overhead and D3D12 command queue synchronization.

### How It Works

1. **Warmup phase**: All prompt tokens run through normal `Execute()` path (707 ONNX nodes evaluated)
2. **Capture step**: First decode token triggers `CaptureBegin()`, runs full `Execute()` (producing valid output), and records all GPU dispatches (975 dispatches across 19 flushes). Bind groups are AddRef'd for reuse.
3. **Replay steps**: Skip entire ONNX Execute loop. Update position params + token ID buffer, then replay captured dispatches directly via WebGPU command buffers.

The speedup comes from eliminating the CPU overhead of the Execute loop (~221 CPU-only ops + op dispatch overhead) while keeping GPU work identical.

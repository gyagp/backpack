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

## Implementation Status

### Infrastructure (complete)
- `CapturedFlush` / `CapturedCommand` structs with pipeline, bind group, dispatch size, bind entries
- `CaptureBegin()` / `CaptureEnd()` / `Replay()` / `ReplayWrites()` / `ReplayDispatches()`
- `RegisterReplayParam()` for GQA position tracking (24 params: posOffset, pastSeq, totalSeq)
- `QueueDispatch()` wrapper that attaches bind entries for capture
- `SubmitAsync()` modified to attach bind entries during capture
- Self-clearing `moe_accum` kernel (slot 0: `dst = weight * src`, slot > 0: `dst += weight * src`) — eliminates MoE output zero-init `writeBuffer`
- `writeBuffer` recording callback in `GPUContext` for capture-time write tracking
- Param pool wrapping fix: unique buffer allocation during capture (prevents 993-dispatch wrapping at 512-buffer pool size)
- Conv state pre-allocation: `convCastF16Bufs` used in `ResetCaches` for stable buffer handles
- Conv cast flush removal: last captured flush (conv casts) removed, handled manually in replay path

### Approaches Tried

| Approach | Result | Issue |
|---|---|---|
| **Capture+submit, AddRef bind groups** | Replay produces different logits from same inputs | Bind groups used in capture CB then reused in replay CB — Dawn produces different results on second use |
| **Capture-only (ORT pattern), pristine bind groups** | Same — different logits | Even pristine bind groups (never submitted before replay) produce wrong output |
| **Rebuild bind groups from saved entries** | Same — different logits | Fresh bind groups with identical buffer handles/offsets/sizes produce wrong output |
| **Replay all captured writes** | Overwrites idsBuf with capture-step token | Fixed via split ReplayWrites/ReplayDispatches and input writes after captured writes |
| **Replay zero-init writes only** | MoE accumulation not cleared | Fixed via self-clearing moe_accum kernel |
| **No write replay (GQA params only)** | Consistency test passes (replay is deterministic) | But replay output differs from normal Execute |

### Key Findings

1. **Replay is deterministic** — Two replays with same inputs produce identical output (verified)
2. **Replay differs from Execute** — Even with AddRef'd bind groups (identical WGPUBindGroup objects) or freshly rebuilt bind groups, replay produces different logit values from normal Execute
3. **No position-dependent data in dispatches** — Cross-reference of captured writes vs dispatch bind groups shows 0 position-dependent small writes consumed by any dispatch, 0 copies reading written buffers
4. **58 non-param writes bound in dispatches** — All are constants (conv bias zeros ×36, MoE expert_bias ×22). No stale state.
5. **Micro-test passes** — 1-2 dispatch capture+replay produces correct output. Full model (975 dispatches, 19 flushes) does not.

### Performance (when enabled, output degraded)

| Mode | tok/s | ms/tok |
|---|---|---|
| Normal Execute | 54 | 18.5 |
| Graph Capture Replay | 75 | 13.3 |
| Improvement | **+39%** | |

### Remaining Issue

The replay mechanism is **correct at the WebGPU API level** (bind groups, dispatch sizes, buffer contents all verified) but produces **numerically different GPU computation results** from normal Execute. This manifests as:
- Benchmark: tokens repeat (e.g., 37009 ×10) instead of varied generation
- Chat: first 2-3 tokens correct ("2 +") then degrades to repetition

The discrepancy persists across all tested configurations:
- Pristine bind groups (ORT pattern) vs AddRef'd (capture+submit) vs rebuilt from entries
- With/without write replay, with/without zero-init replay
- Single CB vs multi-CB replay, with/without waitForQueue between flushes

### Next Steps

1. **Test with Dawn Vulkan backend** — determine if the issue is D3D12-specific
2. **Create minimal 10-dispatch repro** — isolate the threshold where replay diverges
3. **Inspect Dawn's D3D12 bind group translation** — check if bind groups carry per-CB state
4. **Alternative: partial replay** — run the CPU metadata ops normally (~12 nodes, microseconds), then replay only the GPU dispatches. This avoids full graph capture but still skips the ~10ms CPU op loop overhead.

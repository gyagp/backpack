# Backpack Goal

Build Backpack into a continuously improving, portable WebGPU inference runtime for:

- Gemma 4 E2B IT QAT
- Qwen 3.5 2B
- Qwen 3.5 4B

The cared devices are `webgfx-104` (server/NVIDIA), `webgfx-103` (AMD), and
`webgfx-31` (Intel). All three devices must remain productively assigned to
conformance, measurement, or optimization work unless explicitly stopped or
blocked with a recorded reason.

## Required combinations

For every cared model and device, track and validate these independent pairs:

1. Backpack/WebGPU/GGUF compared with llama.cpp/Vulkan/GGUF.
2. Backpack/WebGPU/ONNX compared with ORT/WebGPU/ONNX.

Models are synchronized from `D:\workspace\project\agents\ai-models`.
Backpack and ORT are built once on webgfx-104, backed up with source revision
and date, and copied to compatible x64 devices. The latest llama.cpp release is
downloaded to `D:\backup\x64\llamacpp` and distributed in the same way.

## Acceptance criteria

### Correctness first

- Conformance is a hard gate for performance. A performance result is valid
  only when the exact tested artifact, model, command, options, and device have
  passed deterministic correctness checks.
- A candidate that fails conformance on any cared device must not be merged.
- llama.cpp and ORT are independent reference runtimes; their results are not
  gated by Backpack conformance.
- Graph capture is enabled for ORT by default. Qwen 3.5 may temporarily run
  without graph capture only when the result is marked clearly.
- Windows WebGPU implementations must not use subgroup-matrix operations.

### No unacceptable regressions

- **No confirmed large regression on any cared device may be merged.** A large
  regression means at least 5% loss in any protected conformance or performance
  metric under a like-for-like, repeated comparison.
- The default decision policy remains stricter: a repeatable regression beyond
  2% in a protected metric rejects the candidate unless further measurements
  prove it is noise or an explicitly approved tradeoff.
- A strong improvement specific to one device is acceptable when the other
  devices remain conformant and within the neutral/noise band.
- Base and candidate measurements must use the same prompt length, generated
  token count, model artifact, runtime options, warmup, graph-capture mode, and
  sample method. Incomparable measurements must never be drawn as a regression
  or used by the merge gate.
- Any apparent regression of 5% or more must be rerun before it is confirmed.
  A confirmed regression creates a task and blocks integration; if it is found
  after integration, the milestone must be reverted or repaired before more
  performance work is accepted.

### Performance direction

- Record prefill TPS and decode TPS separately, with bounded execution time.
- First make Backpack/ONNX faster than ORT/WebGPU for each cared model/device;
  then close and exceed the corresponding llama.cpp/Vulkan targets for GGUF.
- Preserve revision-linked history for every valid tested combination, including
  device, model, runtime/backend/format, command, conformance result, dates, and
  all measured TPS values.

## Continuous evolution

- Study upstream ONNX Runtime, ONNX Runtime GenAI, llama.cpp, Modular, and the
  accumulated experience in `docs/` regularly.
- Record each study date and its concrete potential tasks. Split ideas into
  atomic experiments that can be run independently and in parallel.
- Give every task a stable ID and source. Use a separate experiment branch per
  device/task; delete it after rejection or successful integration.
- Automatically queue learned conformance and performance tasks, feed measured
  gains back to their source study, and summarize meaningful accepted findings
  in the daily Digest.

## Milestone gate

A milestone may be pushed and synchronized as the next base only after:

1. deterministic conformance passes on all required cared devices;
2. comparable repeated performance evidence exists for all protected metrics;
3. no required device has a confirmed regression beyond policy;
4. every result is attached to the exact revision and dated artifact; and
5. the dashboard Tasks, Status, Digest, and performance history are updated.

Accepted milestones are pushed automatically, backed up, and synchronized to
all cared devices as the base for subsequent evolution.

# Backpack Goal

Build Backpack into a continuously improving, portable WebGPU inference runtime
that can expand to additional model families over time.

The current priority models are:

- Gemma 4 E2B IT QAT
- Qwen 3.5 2B
- Qwen 3.5 4B

This list is the present execution focus, not a permanent limit. New models may
be added to the cared-model set through Goal review as priorities evolve. Every
newly cared model inherits the same correctness, measurement, regression, and
milestone requirements below; adding one must not weaken coverage for models
that remain cared.

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
- In general, work on the largest validated performance gap first. Rank gaps by
  Backpack's relative deficit against the matching independent reference on the
  same model, device, format, prompt length, generation length, and options:
  ORT/WebGPU for ONNX and llama.cpp/Vulkan for GGUF. Missing or non-conformant
  measurements take precedence because a gap cannot be trusted until it is
  measured correctly.
- Prefer an experiment that can materially close the largest gap over one that
  only refines an already competitive metric. Recompute priorities whenever new
  valid Status evidence lands; do not let task creation order or an early popular
  research direction override measured gap size.
- After conformance is established, prioritize prefill throughput. The immediate
  GGUF objective is to close the large Backpack/WebGPU prefill gap against
  llama.cpp/Vulkan, while preserving exact output and decode performance on
  every cared device.
- First make Backpack/ONNX faster than ORT/WebGPU for each cared model/device;
  then close and exceed the corresponding llama.cpp/Vulkan targets for GGUF.
- Preserve revision-linked history for every valid tested combination, including
  device, model, runtime/backend/format, command, conformance result, dates, and
  all measured TPS values.

### Profile before optimizing

- Every performance task must start from a profile of the exact artifact,
  model, device, and workload it intends to change, and must name the specific
  kernel, dispatch, or host-side cost it is targeting together with that item's
  measured share. An optimization proposed without a profile behind it is a
  guess and must not be started ahead of a measured one.
- Attribute the gap between GPU and wall time before blaming a kernel. Record
  GPU hardware-timestamp time per token, wall time per token, dispatch count,
  submit/flush count, and fence wait separately. A change that speeds up a
  kernel will not show up if the workload is bound by host-side command
  encoding or submission instead, so identify which side dominates first.
- Eliminate host-side bottlenecks before optimizing GPU kernels. While the CPU
  side dominates, a faster kernel cannot show up in throughput and will be
  misread as a failed experiment. Reduce per-token host work first — dispatch
  count, buffer writes, and blocking queue waits — and only then rank and tune
  kernels. Reducing host work must not remove the CPU/GPU overlap that lets the
  GPU start on early command buffers while later ones are still being encoded;
  see `docs/opt-guide.md` §0.
- Prefer the profile's own units of blame: per-kernel totals with call counts
  and average duration, and effective memory bandwidth for weight-bound
  kernels. A kernel already running near the device's achievable bandwidth is
  not a target no matter how large its share; a small kernel invoked hundreds
  of times per token may be, because dispatch cost is per call.
- Verify the premise before implementing. Confirm from the artifact and the
  code path actually taken that the assumed inefficiency exists, and reject the
  task with the evidence when it does not.
- Profiling instrumentation perturbs what it measures. Note which numbers were
  taken under profiling and which under a clean run, and never mix the two in
  one comparison or report a profiled throughput as a result.
- Record the profile with the task and re-profile after the change, so the
  measured gain can be attributed to the item that was targeted. When a change
  regresses, keep the finding and the numbers in the code or the task so the
  same idea is not retried blindly.

## Continuous evolution

- Every day, measure the latest llama.cpp/Vulkan release and the latest built
  ORT/ORT GenAI WebGPU revision on all cared models and applicable cared
  devices. Run deterministic conformance first, then record standardized
  512-input/128-output prefill and decode TPS with runtime revisions, dates,
  graph-capture mode, commands, and artifacts.
- Study upstream ONNX Runtime, ONNX Runtime GenAI, llama.cpp, vLLM
  (`https://github.com/vllm-project/vllm`), Modular, and the accumulated
  experience in `docs/` regularly.
- Record each study date and its concrete potential tasks. Split ideas into
  atomic experiments that can be run independently and in parallel.
- Give every task a stable ID and source. Use a separate experiment branch per
  device/task; delete it after rejection or successful integration.
- Automatically queue learned conformance and performance tasks, feed measured
  gains back to their source study, and summarize meaningful accepted findings
  in the daily Digest.

### Multi-agent collaboration

Adapted from the published Gemma collaboration retrospective
(`https://huggingface.co/spaces/agent-collaborations/gemma-collab-lessons`).

- Preserve failure experience so later work does not repeat it. A settled
  direction — rejected, reverted, or contradicted by a recorded failure — must
  be surfaced against a new proposal that overlaps it, before that proposal is
  run. A rejection is only useful if the next agent sees it without having to
  re-run the experiment.
- Counter agent collapse. Agents converge quickly onto a few familiar
  directions and under-explore harder ones such as custom quantization, large
  fused kernels, and inference-engine restructuring. Reserve an explicit share
  of capacity for directions not already represented in recent work, and treat
  a queue that has narrowed to one theme as a defect to correct.
- Keep guidance balanced. Rules should push toward exploration and direct
  collaboration without prescribing the answer; an instruction specific enough
  to determine the result has removed the search that was the point.
- Bound message volume. High-frequency long messages are unreadable for a human
  reviewer and bias every later reader toward whatever was said first. Separate
  durable state from narration, keep narration short, and route it by topic
  rather than appending to one shared stream.
- Keep human review at the point of taste. Route decisions that turn on
  judgement rather than measurement — which direction is worth pursuing, when a
  neutral result should still be kept — to a human, and record the resolution
  where the next agent will read it.
- Preserve whole traces, not just outcomes. A summary and an artifact do not
  explain how a result was reached or which human prompt changed its direction.
  Retain the trace for accepted and rejected work alike, and attribute
  contributions across messages, artifacts, and traces.
- Keep metrics multi-dimensional. A single headline number invites optimizing
  the measurement instead of the system. Every performance claim stays paired
  with deterministic conformance and with the other protected metrics on the
  same artifact, and no result is accepted on one number alone.

## Operability and visibility

- Keep Dashboard Status, Tasks, Evolution, Digest, Devices, performance
  analysis, and related views synchronized with the latest valid execution
  state. Active work, failures, reasons, revisions, commands, and accepted
  results must be visible promptly; stale or invalid data must be corrected or
  removed.
- Keep the common LLM application under `apps/` working with every currently
  cared model and supported format, so a person can run an end-to-end prompt
  and review correctness manually before accepting performance evidence.

## Milestone gate

A milestone may be pushed and synchronized as the next base only after:

1. deterministic conformance passes on all required cared devices;
2. comparable repeated performance evidence exists for all protected metrics;
3. no required device has a confirmed regression beyond policy;
4. every result is attached to the exact revision and dated artifact; and
5. the dashboard Tasks, Status, Evolution, Digest, and performance history are
   updated; and
6. the common application remains conformant for the affected cared models.

Accepted milestones are pushed automatically, backed up, and synchronized to
all cared devices as the base for subsequent evolution.

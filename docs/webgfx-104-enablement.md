# webgfx-104 model enablement

Status as of 2026-10-01. The active scope in `goal.md` is this device only;
neither model enablement nor acceptance requires another machine.

## Device and execution scope

- NVIDIA GeForce RTX 5080, 16,303 MiB VRAM, driver 616.64.
- Backpack uses native Dawn/D3D12 WebGPU, with subgroup matrices disabled.
- The dashboard server scopes scheduling and its model matrix to webgfx-104.
  Out-of-scope pending runs are cancelled; historical evidence is retained.
  Remote registration, work claims, and provisioning are blocked.
- The worker checks the actual hostname before any sync or execution. Reference
  refresh scripts also enforce the hostname and contain no remote distribution.
- Downloads, builds, fixtures, reference binaries, and logs are under `gitignore/`.

## Qwen3.8-27B

The official model is `Qwen/Qwen3.8-27B`, architecture `qwen3_5`. Its text model
has 64 decoder layers, hidden size 5120, FFN size 17408, 24 attention heads,
4 KV heads, head size 256, and vocabulary size 248320.

ONNX discovery found these published artifacts:

- `amd/Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-onnx`, revision
  `3d00750aaa18e3e09e0b271a82e7762a4586718f`. Text and embedding data alone
  total 15,937,110,016 bytes (14.84 GiB), before working memory. The separate
  FP16 embedding tensor is 2,542,796,800 bytes, exceeding this Dawn device's
  2 GiB storage binding/buffer limit. The isolated enablement checkpoint below
  now uses CPU-mapped embedding rows and a native WebGPU text decoder. Larger
  contexts, memory residency, and performance remain under validation.
- AMD's FP32 embedding/vision variant requires still more memory.
- `tlwu/Qwen3.8-27B-NVFP4-ONNX` uses custom CUDA-oriented quantization and has
  21.7 GB of text data. It is not a verified native WebGPU artifact.

The text artifacts are now downloaded under `gitignore/models/Qwen3.8-27B-ONNX/`
and match the publisher's LFS hashes:

- `text.onnx.data`: 13,394,313,216 bytes,
  `9944f22d543669858768ab365cd8b09af68117396cf7e030c10635468db8a0e3`.
- `embedding.onnx.data`: 2,542,796,800 bytes,
  `41a8b66f4d3d229db720eb18f0e4567d721e784c3ef8f891274cfe1a362473bd`.

Task #744, branch `enable/webgfx104-qwen38-onnx`, checkpoint
`66204c6b2455e1d256ec62c626d7d11f6d702983`, adds block-size-128 Q4 matmul,
CPU-mapped FP16 embeddings, manifest-based decoder selection, and FP16 logits
readback. The CPU embedding path validates the text-only Gather/empty-media
ScatterND structure and rejects media placeholders. The original table and
weights are retained unchanged. It also repairs FP16 and capture failures:
explicit casts must retain their declared dtype, residual RMSNorm must write
FP16 outputs, and direct GPU submissions must be recorded during capture.
The latter fixes stale `Where` outputs in the decomposed rotary graph.

On webgfx-104, all 63 operator/session tests pass. The full model returns
`[19, 248044]` (`4` followed by EOS), exactly matching native ORT CPU and WebGPU.
ORT profiles place all heavy decoder operations on WebGPU; its original embedding
graph runs on CPU to avoid the oversized GPU buffer. Capture and ordinary Backpack
execution match the first 32 sky-response tokens and all 36 checked logits/state/
cache snapshots bit for bit. The complete 36-token sky response matches an ORT CPU
control with `MatMulNBits.accuracy_level=0`; that control changes the optional
accuracy hint only. The published accuracy-level-4 CPU and GPU references agree
on a shorter 29-token wording. These numerical differences remain documented.

The initial checkpoint used context capacity 128. Subsequent memory-ownership
fixes and bounded prefill have validated **512 input / 128 output tokens** at
context capacity **640**, with 32-token prefill batches. All 128 output tokens
match the original native ORT WebGPU export under the same batch schedule.
Three complete sky responses have stable live allocations and unchanged output.
Peak logical GPU allocation was 15.15 GiB, with 13.40 GiB live after decode;
free pools and driver staging are separate, so full physical residency is unproven.

The combined local integration preserves the accepted GGUF tile implementation
and adds these ONNX changes plus exact-workload benchmarking. The main workspace
executable uses the following validated configuration:

```powershell
gitignore/runtime/build/backpack_llm.exe --model gitignore/models/Qwen3.8-27B-ONNX --backend d3d12 --max-seq-len 640 --prefill-chunk 32 --max-tokens 128 --chat "Explain why the sky is blue in one sentence."
```

Current evidence: `gitignore/logs/qwen38-onnx/long-context-checkpoint.json`,
`gitignore/logs/benchmark-protocol-audit/checkpoint.json`, and
`gitignore/logs/workspace-integration-20260930/`. Parameter arenas and the rejected
OS-residency toggle are not selected defaults or part of this integration.
Comparable 27B native warm throughput and broader performance targets remain open.

Evidence is in `gitignore/logs/qwen38-onnx/checkpoint.json`, `full-reference-*`,
`full-backpack-reference/`, `capture-replay-checks.json`, and `sky/`. The raw ORT
runs use capture disabled for diagnostics; they are not benchmark results.
The first-layer extraction has been checked against the completed, hashed model.
The validated GGUF path remains available in the root build.

Initial GGUF enablement uses the exact model from
`unsloth/Qwen3.8-27B-GGUF`, revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`:

```text
Qwen3.8-27B-UD-IQ3_S.gguf
size:   12040883104 bytes
sha256: d847e2c1e4aa276e4b7b8e9ad7628050e61e165d49ab995407bc36677a6f3864
```

The complete download matches the publisher's LFS SHA-256. This dynamic quant
contains 14 quantized types, including IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS,
IQ2_S, IQ3_S, IQ4_XS, IQ4_NL, Q2_K through Q6_K, and Q8_0. The filename does
not describe every tensor's type. Alternative Unsloth UD-Q3_K_XL and UD-Q2_K_XL
headers also contain mixed IQ tensors.

Implemented format fixes:

- Decode IQ1_S, IQ2_XXS, and IQ2_XS, including their codebooks.
- Correct Q2_K and Q3_K field offsets; the downloaded model uses Q3_K embeddings.
- Interpret the GGUF block count of 65 as 64 target layers plus one MTP layer.
- Raise an error for an unknown dequantization type instead of leaving output
  unwritten.
- Keep all mixed-IQ projections in native packed storage, including separately
  quantized gate/up matrices and codebooks. No Q8 expansion is needed for them.
- Detect the independent Q5_K output head from the tensor index and retain
  the Q3_K input embedding table for native GPU gathers.
- Use continuous RoPE frequencies for keys in both serial and batched paths,
  matching queries and llama.cpp. Previously keys reset at section boundaries.
- Apply the Qwen35 Unicode pre-tokenizer, honor the requested GGUF context
  length, and preserve UTF-8 Windows prompts in the shared LLM application.

The CPU format checks match the independent `gguf==0.19.0` decoder exactly on
513 randomized blocks for each new IQ type and 128 sampled blocks for every
quantized type in the downloaded model (Q8_0 and IQ4_NL use 32-value blocks).
Native GPU matmul and exact embedding gathers pass 98 independent-reference
cases spanning the 14 quantized types, padded parameter buffers, partial tiles,
and split-output strides. Forty multilingual tokenizer cases match llama.cpp
token IDs. A separate GPU regression checks continuous key RoPE frequencies at
nonzero positions and section boundaries.

The shared LLM app matches llama.cpp exactly on six prompts in both serial and
batched modes (12 comparisons): arithmetic, a capital city, a longer sequence
continuation, a sky explanation, Chinese, and digit output. Evidence is in
`gitignore/logs/qwen38-native/conformance-rope-fixed/`. See `apps/llm/README`
for runnable commands. This establishes initial GGUF text enablement; ONNX,
image generation, and performance targets remain incomplete.

Batched 27B prefill uses bounded command submissions on D3D12 to avoid GPU
timeouts. `submitOnly` consumes bind-group references; keeping those references
for a second cleanup caused heap corruption and was corrected. A 512-token
profile then exposed the old fixed-size parameter arena; allocation now scales
with layer count. Both failed runs are retained under
`gitignore/logs/qwen38-native/` so these directions are not repeated blindly.

The completed 512-token hardware profile records 44.307 seconds of GPU work,
1,085 dispatches, and about 44.701 seconds for build/submission/wait/cleanup.
Those wall phases overlap GPU execution and are not independent CPU costs.
Feed-forward projections account for 71.2% of GPU time; unpacking the same
weights separately for each prompt row is the next measured optimization
target. Profiled throughput is not a benchmark result. Raw timing and the
interactive trace are under `gitignore/logs/qwen38-native/profile512-fixed/`.

Five clean 512/128 runs are stored under
`gitignore/logs/qwen38-native/benchmark512/`, with a runtime allocation peak of
12,686,502,752 bytes (11.82 GiB) at context length 1024. These establish a slow
initial baseline: mean prefill 11.543 tokens/s (0.088% CV), mean decode
9.929 tokens/s (0.123% CV). No performance improvement is claimed. The exact tested binaries and
source hashes are archived in
`gitignore/runtime/artifacts/qwen38-rope-fixed-20260930/`. Final revalidation
after the arena-sizing change is recorded separately in
`gitignore/logs/qwen38-native/final-validation/`.
The final binary passes the exact-answer check with a 512-token prompt, and
the three earlier cared GGUF models pass their prompt checks again. Dashboard
observation `obs-goal104-qwen38-native-20260930` binds those checks, the five
timing samples, and binary/source hashes.

### Native prefill tile (task #743)

The accepted experiment replaces per-prompt-row weight decoding with a 16x16
shared-memory tile. It keeps 32 partial sums in the original reduction order
and pads the transposed weight tile to avoid shared-memory bank conflicts.
The ordinary decode shader is unchanged.

Five alternating base/candidate pairs on webgfx-104 measured:

| Metric | Base median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| Prefill, 512 tokens | 11.547 tokens/s | 53.897 tokens/s | +366.75% (4.67x) |
| Decode, 128 tokens | 9.924 tokens/s | 9.926 tokens/s | +0.021% (neutral) |
| Runtime allocation peak | 11.82 GiB | 11.82 GiB | unchanged |

Both binaries passed the 512-token exact-answer check. The final candidate also
passed 204 packed-matmul/gather GPU cases and all 12 serial/batched output
comparisons. The repeated comparison had a maximum CV of 0.37%. A final
hardware profile records 9.722 seconds of GPU prefill work versus 44.307 seconds
for the initial implementation. Profiled times are separate from the clean
timing samples above.

The experiment branch is `experiment/webgfx104-qwen38-native-prefill-tile`;
base snapshot `0af75ca`, candidate `a3d1cc897510683b44716059c2d698681a67fdbe`.
Commands, binary hashes, correctness, samples, and profiles are retained under
`gitignore/logs/task743/`. The local experiment gate returned `accept`.
The validated delta is applied to the main workspace, and a main-workspace
smoke check confirms the optimized default and exact `4` output. Publication
of the combined milestone remains pending the broader goal audit.

The default applies only to Qwen3.8-27B IQ3_S on RTX 5080. Setting
`BP_NATIVE_QUANT_PREFILL_TILE=0` retains the scalar implementation for diagnosis.
A wider K=128 tile was explored and retained in commit `fd7c821`; it was not
selected. Its first run showed an unexplained decode collapse, which did not
recur. The repeat measured 48.9 prefill tokens/s and normal decode, below the
padded K=32 tile. The repeat's external memory sampling is marked in its logs;
do not treat the transient decode result as a confirmed code regression.

llama.cpp Vulkan b11256 loads the artifact and returns exactly `4` for the
arithmetic check when fed the official non-thinking prefix, which matches
`app::applyQwenUserTemplate`. The ordinary conversation flags produced a
reasoning trace instead of a final answer within 32 tokens; that run failed
the exact-answer check. No throughput from either short prompt is a protected
512/128 result.

The corrected five-repeat llama.cpp reference adapter passes the exact answer
check, then measures prefill separately and decode with 512 tokens already in
context (`llama-bench -d 512`). On webgfx-104 it reports 2004.09 prefill tokens/s
and 49.34 decode tokens/s (standard deviations 2.24 and 0.47). The older
`-p 512 -n 128` invocation measures decode at depth zero and must not be used
as a like-for-like 512-context decode comparison. Commands and individual
samples are recorded in `gitignore/logs/qwen38-reference/validated-adapter.log`
and dashboard observation `obs-goal104-qwen38-llamacpp-20260929`.

Evidence is in `gitignore/models/discovery/`, the model's `artifact.json`,
`gitignore/logs/qwen38-reference/`, and
`gitignore/logs/qwen38-gguf-reference-tests.log`.

To repeat CPU quantization validation after building `backpack_gguf_test`:

```powershell
gitignore/venvs/gguf-validation/Scripts/python.exe -B runtime/tests/test_gguf_reference.py --model gitignore/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf
```

## Image model

Exact-name and official-Qwen Hugging Face API searches returned no
Qwen-Image-3.0 artifact. The official catalog includes Qwen-Image-2.1.
The 2026-10-01 recheck also covers Unsloth's catalog. A similarly named
community repository, `UnifiedHorusRA/qwen-image-3`, contains metadata but no
weights. The exact official repository endpoint returns HTTP 401, which does
not establish whether a private or unpublished artifact exists. Evidence:
`gitignore/logs/image-identity-20261001/`.
A repository link or confirmation of a different release has been requested;
the original image-model requirement remains pending (task #741).
The current `apps/image` implementation is for Z-Image-Turbo and does not
establish support for either Qwen image model.

## Existing cared models

Gemma 4 E2B QAT, Qwen3.5-2B, and Qwen3.5-4B pass the catalog's GGUF prompt
checks with the rebuilt application. The ONNX fixes are now applied to the root
working tree and the shared app passes the current packages. They repair these
previous failures:

- Gemma: `Inconsistent packed Gemma PLE layer 15`.
- Qwen 2B/4B: missing `LinearAttentionGate` handling, followed by a
  `CausalConvWithState` optional-input error and access violation.

The archived `cf43e74-20260804` binary reproduces all three failures with the
same commands/artifacts. These are existing artifact compatibility gaps,
tracked as task #742, and must be repaired before full goal acceptance.
Commands, binary hashes, source diff hash, and logs are in
`gitignore/logs/webgfx104-conformance-20260929/`.

An isolated ONNX candidate is now on branch
`enable/webgfx104-qwen-onnx-current`, revision
`ecc49ac9ea7e98c3b1f6396c386a910eef3ba63c`, built under
`gitignore/runtime/build/task742/`. It adds `LinearAttentionGate`,
`GatedRMSNorm`, and three-axis `MRotaryEmbedding`, binds the exports' compact
cache names, reads the explicit attention head dimension, and supplies all
three text position axes during prefill and replay. Unknown operators now
raise a clear error instead of producing invalid text.

The ONNX tokenizer now reads nested text configuration and all EOS IDs,
honors the tokenizer's explicit chat-end token, and decodes added tokens.
Both Qwen3.5 ONNX models return exactly `4` and stop correctly, and both
produce coherent bounded English responses. All 53 operator tests pass on
D3D12, including new gate, grouped gated-RMS, multi-axis rotary, and failure
tests; added-token/EOS checks pass on both real tokenizer packages.

The isolated candidate now advances to `b78de1b` with FP16 recurrent output
boundaries and replay fixes. FP32 cache storage retains FP16-rounded values
when the graph declares FP16 state, while accumulation within a multi-token
call remains FP32. FP16 activation outputs preserve their input dtype. Tests
cover scalar prefill, vector decode, promoted caches, and final-boundary versus
per-token rounding. Native f16 conversion is required on this D3D12 path:
`pack2x16float` truncated the tested values. Odd-sized FP16 graph outputs now
copy the padded final word instead of silently dropping the final element.

Both floating-point `Cast` kernels previously submitted work outside the
capture queue. The current exports put these casts after rotary embedding,
so replay reused the capture-time keys even though the values and recurrent
states advanced. Routing the casts through `QueueDispatch` fixes this. The
new rotary operator also casts FP16 inputs on the GPU, allowing changed
inputs to participate in replay. All **58 operator tests pass on D3D12**,
including changed-input capture tests for both cast directions and three-axis
FP16 rotary embedding.

Independent native ORT 1.31.0 reference uses revision
`96f73115c95968a3f31f2a110b33c164d847dda4`, from the verified
`D:/backup/x64/ort/20260930-011912` build. It selects D3D12 adapter 0, the
RTX 5080 (LUID 0x121A9). Its Dawn build has Agility SDK disabled; no
subgroup-matrix programs were observed. Profiles place the model's heavy
operators on WebGPU. Reference graph capture is **disabled** for these
state-dump diagnostics; their wall times are not performance measurements.

Five identical-token workloads were compared for each Qwen3.5 model:
arithmetic, capital, one-sentence sky explanation, 479-token code recall,
and up to 128 generated tokens explaining photosynthesis. Ordinary Backpack
matches the complete native ORT token sequence in 8 of 10 cases. Both models
pass arithmetic, capital, and code recall. The other exact matches are the
2B sky response and all 128 tokens of the 4B explanation. The 2B explanation
and 4B sky wording differ; this does not establish full numerical parity.
Native ORT GPU also differs from ORT CPU, so these differences cannot all be
attributed to a Backpack defect.

With capture-only Q4 prequantization disabled, capture reproduces ordinary
Backpack tokens exactly in all four longer sky/explanation cases. Default
capture uses a different activation quantization path: both 2B responses
match ordinary execution, while the 4B wording changes. The default captured
4B sky response matches ORT exactly; its longer explanation does not.
This numerical-path difference remains explicit and is not the frozen-cast
replay failure. No performance acceptance is claimed.

Arithmetic state dumps confirm finite, FP16-representable layer-0 convolution
and recurrent states after both prefill calls and decode. Against ORT CPU,
recurrent-state relative L2 error is 0.00086 to 0.00110, while logits differ more
(0.047 to 0.122). Whole-model precision tolerances, further reference validation,
and comparable performance remain pending. The candidate was subsequently applied to the root working tree after the
Gemma fixes below. Qwen3.8 ONNX memory/operator enablement remains open.

Evidence is under `gitignore/logs/task742/`: `precision-reference-summary.json`,
`precision-first-comparison.json`, `replay-cast-summary.json`,
`replay-cast-op-tests.log`, and `capture-isolation/`. Native reference runners
are in `benchmarks/ort_state_reference.cpp` and
`benchmarks/backpack_state_reference.cpp`; requests contain exact token batches
so tokenizer or chat formatting differences cannot mask runtime behavior.

## Gemma ONNX and further reference validation

The candidate is now `111d7be1a951d399cd2395b4a11a1ffa278c5de1` on the same
isolated branch. The Gemma package is
`webai-community/ai-models`, revision `6d5cd933ae1008d4cb78f88730248c464d3a50d1`,
source model `gemma-4-e2b-it`. Its decoder contains 35 transformer layers but
only 15 cache inputs. The loader now derives the full transformer depth from
the graph, retaining all 35 PLE tables and 20 shared-KV layers. It also preserves
and concatenates the 3840-row producer and 5120-row consumer PLE projection
matrices. Previously their names mapped to one entry, so the consumer matrix
overwrote the producer and left the combined projection incomplete.

All **59 operator/loader tests pass on D3D12**. A small loader fixture verifies
35 transformer layers with 15 caches, all PLE tables, distinct producer and
consumer rows, embedding-only graphs, and rejection of invalid layer numbering.

Native ORT's original embedding and decoder graphs now provide an independent
Gemma text reference. The reference passes zero image/audio tokens through the
original embedding graph. On webgfx-104, Backpack matches every generated token
from both ORT CPU and ORT WebGPU for four workloads: Paris, arithmetic, a sky
explanation, and a **520-token context recall**. Both serial and batched
Backpack execution match the Paris reference. The shared LLM CLI returns Paris
for Gemma and exactly `4` for both Qwen ONNX packages, with correct stopping.
Heavy embedding and decoder operations are assigned to ORT WebGPU; CPU work is
limited to token/mask control operations. These tensor-dump reference runs have
capture disabled and are diagnostics, not accepted throughput evidence.

Further Qwen checks feed each runtime the same 24 reference continuation tokens
for the two previously differing responses. Backpack matches ORT CPU's top
choice at all **52 checked positions** (26 per case); ORT WebGPU matches 50.
Mean probability total variation versus CPU is 0.0190/0.0147 for Backpack and
0.0193/0.0171 for ORT WebGPU. This explains the observed wording differences
without requiring exact greedy-text identity between the two GPU paths.
These are ordinary-execution checks: single-token `Prefill` does not exercise
capture. Extra runs initially named `capture` are marked as excluded capture
evidence in their command metadata.

The GenAI model-creation failure (`0xC0000409`) was traced to the reference
runner requesting `adapterIndex=0`: GenAI's initialization session does not
forward that option and had already initialized ORT's shared context using
automatic selection. The runner now uses consistent automatic high-performance
D3D12 selection. NVIDIA's process listing confirms execution on the RTX 5080.
Gemma reproduces all four reference responses with capture enabled, and native
ORT logs confirm graph replay. Qwen's existing capture-disabled setting remains
in use for accepted reference measurements; both models reproduce the uncaptured
native reference responses. Capture-enabled Qwen wording differs and is not used
for this performance baseline.

Evidence: `gitignore/logs/task742/gemma-conformance-summary.json`,
`gemma-reference/`, `gemma-first-token/`, `gemma-all-op-tests.log`,
`teacher-force-comparison.json`, and `final-app-smoke.json`. The unsuccessful
GenAI reference attempt is retained under `gemma-reference/genai-capture/`;
its diagnostic trace and pinned headers are retained under `task742/`.
The working reference runner is `benchmarks/genai_state_reference.cpp`, source
revision `e1a13c8b62c469030508542c56edf286e7d5e4ea`.

## Working-tree integration and current reference baselines

The runtime compatibility delta `111d7be` is applied locally without replacing
the validated Qwen3.8 native-quantization optimization. Root validation passes:
59 operator/loader tests, 204 native quantization tests, both Qwen tokenizer
checks, six ONNX/GGUF catalog app checks, and all 12 Qwen3.8 llama.cpp comparisons.
The control plane passes 102 tests and still exposes only webgfx-104. The source
patch, untracked-file hashes, binary hashes, model hashes, commands, and results
are recorded in `gitignore/logs/task742/workspace-integration/provenance.json` and
`root-validation/`. No milestone has been published; the comparable Backpack
performance audit remains pending.

The fresh ORT/GenAI backup is staged locally at
`gitignore/evolution/backups/ort/20260930-011912/`. The benchmark adapter accepts
the build's `build-metadata.json` and native conformance runner, preserves model
files through an ignored package view, and explicitly selects D3D12 for every
WebGPU session. Original packages are unchanged. It uses fixed prompt text,
checks the native tokenizer's reported counts, and retains raw benchmark logs.
Artifact fingerprints and workload metadata now separate incompatible regression
series; older exports or prompt protocols must not be treated as current regressions.

Five repeats after one warmup, reusing the generator, measured the following
independent ORT reference baselines on this device:

| Model | Prefill tokens/s | Decode tokens/s | Capture |
|---|---:|---:|---|
| Gemma 4 E2B ONNX | Invalidated: rewind failure | Invalidated | Enabled |
| Qwen3.5-2B ONNX | 5693.43 | 110.046 | Disabled temporarily |
| Qwen3.5-4B ONNX | 3056.80 | 94.7605 | Disabled temporarily |

The workload has 512 input tokens, 128 generated tokens, and context capacity
640. ORT reports first-token sampling separately, leaving 127 timed decode
steps per repeat (635 samples total). The fixed text is 512 space-separated
`A` tokens, SHA-256
`2167630e46c013754a4b4af6957cb45088c6c1a439709b78adc1074ab724acc3`.
These results establish reference baselines; they are not an accepted comparison
against a differently configured Backpack benchmark. Raw statistics and commands
are in `gitignore/logs/task742/reference-benchmarks-fixed/`; capture and adapter
proof is in `genai-capture-validation/` and `final-genai-gemma.log`.


Control-plane validation: 102 tests passed, including device-scope/admission tests
and two tests for Qwen prompt formatting and decode context validation.
The live local API lists only webgfx-104 and rejects remote provisioning before
any provisioning action.
Unresolved artifact identities remain in the full goal audit and model matrix;
validated available models can proceed with experiments while identity
clarification is pending. An available artifact always retains its conformance
gate, even if its identity metadata has not been updated.

## Exact application benchmark protocol (2026-09-30)

The combined candidate uses `BenchmarkTokens` and the same prefill entry point
as the shared application. Warmup is a complete separate workload. Each measured
512-input/128-output run performs exactly 127 decode calls, ending at processed
position 639 within capacity 640. Raw token IDs and per-run counts are retained.

Protocol `llm-fixed-text-reuse-first-token-v2` includes the first prediction in
prefill timing. The native figures below are derived from the original raw ORT
prompt-processing plus separate first-token-sampling durations. Artifact and
prompt hashes, capacity, output counts, warmup and reuse settings match. These
are application-path baselines, not measurements of the older benchmark-only
shortcuts. Qwen3.5 ORT capture remains disabled temporarily and explicitly marked.

| Model | Backpack prefill tok/s | ORT prefill tok/s | Backpack decode tok/s | ORT decode tok/s |
|---|---:|---:|---:|---:|
| gemma-4-e2b-it-qat ONNX | 192.47 | Invalidated: rewind failure | 159.75 | Invalidated |
| qwen3.5-2b ONNX | 773.93 | 5670.87 | 196.29 | 110.05 |
| qwen3.5-4b ONNX | 679.45 | 3048.05 | 126.41 | 94.76 |

The largest measured prefill gap is Gemma ONNX (about 17x). Its older GGUF
batched benchmark shortcut produced different outputs across identical runs;
those timings are excluded. The current application path is deterministic.
Evidence and exact commands: `gitignore/logs/benchmark-protocol-audit/`.

The parameter-arena experiment remains opt-in and outside this integration.
It reduces native placement requirements, but 27B decode variability exceeds
the acceptance threshold. The OS-managed-residency-only experiment was rejected
after a repeated fivefold prefill slowdown. No performance milestone is claimed.

## Gemma ONNX application prefill on RTX 5080

Runtime checkpoint `5a4b31e114fb115c0a48f233cb7e1a7da56112ed` enables
batched application prefill for the validated Gemma 4 E2B ONNX configuration
(35 layers, hidden size 1536, vocabulary 262144) on RTX 5080/D3D12.
Prompts longer than 16 tokens use the existing 128-token chunks. Set
`BP_GEMMA_SERIAL_PREFILL=1` to restore serial prefill. GGUF routing and
chunk-completion waits are unchanged.

Five alternating pairs, each with a complete warmup and two measured runs,
used the same binary with the serial fallback or the default route. The exact
workload was 512 raw prompt tokens, 128 output tokens, and context capacity 640:

| Metric | Serial fallback | Batched default |
| --- | ---: | ---: |
| Mean prefill tokens/s | 192.87 | 1564.07 |
| Mean decode tokens/s | 166.44 | 166.67 |

Prefill improved 8.11 times. The largest sample coefficient of variation was
1.51%; every paired decode difference was within 0.6%. Every generated token
matched native ORT. Separate D3D12 validation covered arithmetic, a sky
explanation, 520-token recall, and the 512/128 workload, with three streaming
repetitions per Backpack route. All 64 operator/session tests passed.

The profile explains the gain: serial prefill issued 343,045 dispatches and
512 submissions; batched prefill issued 2,117 dispatches and four submissions.
Timestamped GPU regions fell from 2644.52 ms to 297.64 ms. These instrumented
times are diagnostic and are separate from the clean throughput table.
The largest remaining batched region is gate/up Q4 multiplication, about 41%
of GPU time. The earlier native ORT warm result of about 3280 prefill tokens/s
was subsequently invalidated: GenAI's multimodal rewind changes the sky answer
after reset. A valid native warm comparison remains pending; the Backpack
serial-versus-batched improvement and fresh-generator conformance remain valid.

Repeated streaming also exposed a session-reset bug: queued decode readbacks
remained pending when the next prompt reused their staging buffers. Reset now
completes and unmaps the outstanding ring slots before clearing cache state.
The bounded benchmark had already drained its own submissions and did not
exercise this streaming path.

Evidence: `gitignore/logs/gemma-onnx-app-prefill/`, including
`conformance-final/report.json`, `paired-benchmark-final/summary.json`,
`evaluation.json`, and the separate timestamp/counter profiles.
The device-local policy evaluation accepts both protected metrics.
Milestone publication and the broader goal remain pending.

## Qwen3.8 native ORT reference and rewind audit

The original published Qwen3.8 text and embedding graphs now have an independent
native ORT runner in `benchmarks/ort_resident_reference.cpp`. It binds recurrent
and KV state outputs to WebGPU, retains those GPU tensors for the next call,
and discards them at reset. It executes the original embedding graph on CPU and
reads only logits for CPU greedy sampling. GPU tensor allocation is verified
for every state output; physical VRAM residency is not claimed.

The 512/128 workload uses sixteen explicit 32-token prefill batches, exactly
127 decode calls, capacity 640, a complete warmup, and five measured repetitions.
First-token-only, arithmetic, and sky reset checks validate more than the
repeated `A` benchmark. The sky result matches the original accuracy-level-4
ORT CPU/WebGPU wording (29 tokens), as distinct from Backpack's previously
documented accuracy-level-0 CPU match.

The capture-disabled diagnostic measures about 209 prefill and 27 decode
tokens/s. It matches every benchmark output token, and profiling confirms that
MatMulNBits, CausalConvWithState, LinearAttention, and GroupQueryAttention run
on WebGPU. CPU shape/control nodes remain. Native graph capture rejects this
original graph during session initialization, so a capture-enabled performance
baseline remains open. These results do not establish a new performance
milestone or a paired Backpack improvement.

GenAI revision `66a5cacc86171d866e2125846521a32e17c7efce` has a separate rewind
problem: `MultiModalPipelineState` does not override the default no-op
`State::RewindTo`. Recurrent state and `is_prompt_` survive rewind, and subsequent
prefill skips configured chunking. Repeated `A` tokens concealed the defect.
Qwen arithmetic changes from token 19 to 2523 after reset; Gemma's 28-token sky
answer changes to five tokens. The preliminary GenAI Qwen measurements and the
two current GenAI Gemma warm observations have been invalidated. Fresh-generator
conformance is retained. The extended GenAI reference helper now detects changed
continuations and saves both sequences in a failure artifact.
The automatic ORT benchmark adapter rejects these two model types on the known
bad GenAI revision so the invalid warm measurements cannot be recreated by the
daily reference cycle. Qwen3.5-2B/4B use a separate decoder-only rewind path.

Evidence and reproducible commands are under
`gitignore/logs/qwen38-ort-benchmark/`. ORT is pinned to
`96f73115c95968a3f31f2a110b33c164d847dda4` (DLL 1.31.0); original graph and
weight hashes are retained. All execution is on webgfx-104 / RTX 5080 / D3D12.

## Qwen3.8 submission and residency experiments

Task #752 profiled the complete warm 512/128 workload at capacity 640 with
32-token prefill batches. At decode step 8, the counter-only run took 1340.65 ms,
including 1204.30 ms inside queue submission. The timestamp control measured
119.78 ms of GPU regions over a 1456.28 ms span. All output tokens matched the
independent native ORT reference in both repetitions.

Optional native-memory diagnostics in `backpack_state_reference.cpp` distinguish
logical buffer bytes from Dawn's backend allocator. Build that helper with
C++20 and `BP_NATIVE_MEMORY_DIAGNOSTICS`, then request `native_memory: true`.
At warm decode, Backpack reported 14.39 GB of logical live buffers, while Dawn
reported 16.72 GB used and 18.66 GB allocated. The dump includes 16,384 parameter
buffers requiring about 1 GiB of 64 KiB native placements. These counters cover
different allocation layers and do not prove physical residency.

Packing parameter slots and trimming the free buffer pool together reduced
native used bytes to 14.61 GB, but sampled decode still took 1083.59 ms. Splitting
replay into groups of 16 dispatches took 1484.99 ms. Neither control was selected.

A read-only priming pass copies four bytes from each captured resource to a
private scratch buffer before the existing inference submissions. This lets
Dawn see the entire replay working set together without removing CPU/GPU
overlap or completion waits. The profiled decode step fell to 204.84 ms,
including 2.33 ms for priming and 16.03 ms in submission. The full 128-token
continuation, three repeated arithmetic/sky answers, and 65 D3D12 tests passed.
DXGI local usage increased to about 15.37 GB against a reported 13.86 GB budget.

The candidate was **rejected** after two clean alternating pairs:

| Pair | Baseline prefill | Candidate prefill | Baseline decode | Candidate decode |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.63 | 15.75 | 0.906 | 3.591 |
| 2 | 20.70 | 8.94 | 0.849 | 3.809 |

Values are tokens/s. Each run used a complete warmup and exact 512/128 counts.
The 23.7% and 56.8% prefill losses block integration despite the decode gain.
Remaining runs were stopped; the incomplete third pair is not evidence.
The main application and its defaults remain unchanged. Follow-up task #753
investigates memory handling when returning from primed decoding to prefill.

Evidence, source patches, failed binaries, and the policy rejection are retained
under `gitignore/logs/qwen38-memory-pressure/`. Failed runtime candidate:
`43d83806c5893e18f367332f2eb39d9951b5ef9e`.

## Native multimodal reset correction on webgfx-104

Local GenAI reference patch `90d1e2e75a7c2dad49541c1505c37e766afc8d0a`
is based on upstream `66a5cacc86171d866e2125846521a32e17c7efce`. A full
multimodal rewind now recreates request state while retaining the loaded model
sessions. This restores positions, recurrent/KV state, and the prompt phase,
and releases captured graphs whose buffer addresses belong to the old request.
Unsupported partial multimodal rewinds fail before changing the search state.
This local reference patch has not been published upstream. It is installed as
the local daily reference after validation through the actual automatic adapter
for Gemma and Qwen3.5-2B/4B. The guard against the original faulty DLL remains.

Fresh-versus-reused generator tests passed across six changes of prompt length
for Gemma with capture enabled and disabled, and Qwen3.8 without capture.
Separate checks reproduced Gemma's original 28-token sky answer and Qwen's
first arithmetic token, 29-token sky answer, and exact 512/128 continuation
after three resets each. A diagnostic Gemma run logged 78 native WebGPU graph
replays across three distinct request graph IDs. Its process was observed on
the local RTX 5080. Qwen's original-graph capture rejection remains unresolved.

The corrected Gemma warm reference uses capacity 640, 512 input tokens,
128 generated tokens, exactly 127 decode calls, a complete warmup, and five
measured repetitions. All output tokens match the fresh native reference:

| Metric | Mean tokens/s | Coefficient of variation |
| --- | ---: | ---: |
| Prefill | 3421.61 | 1.46% |
| Decode | 124.17 | 0.91% |

These clean measurements exclude profiling, verbose logging, and external GPU
sampling. Reset costs are recorded separately (1.15–1.33 ms for Gemma).
The model fingerprint is
`e7119225b6db5eef2bd47448c8ec162409de3ce02b0788fd94ecd86fd77a95de`;
ORT remains `96f73115c95968a3f31f2a110b33c164d847dda4`.

The isolated binaries and exact commands are under
`gitignore/runtime/ort-reference/task755/` and
`gitignore/logs/genai-multimodal-reset/`. The reset patch is validated for the
text reference workloads; media inputs, adapters, and runtime-option overrides
are outside this validation. Backpack runtime defaults are unchanged.

The complete daily package is
`gitignore/evolution/backups/ort/20260930-genai90d1e2e-local-reset/`.
Its manifest records the patched library, unchanged upstream benchmark driver,
binary hashes, and archived GenAI source. The automatic adapter measured Gemma
at 3369.67 / 124.84, Qwen3.5-2B at 5796.49 / 110.52, and Qwen3.5-4B at
3047.15 / 94.89 prefill/decode tokens/s. Each used five measured 512/128 runs.
Gemma used capture; Qwen3.5 retained its explicit temporary capture exception.
Compared with the prior valid Qwen3.5 references, the largest observed loss was
0.03%. Qwen3.8's unconfigured original package and capture rejection remain open.

Task #754's combined Qwen3.8 candidate completed its first five-repeat
comparison with exact output: mean prefill 20.64 → 22.39 tokens/s and decode
0.96 → 3.89 tokens/s. Candidate prefill variation (5.67%) and baseline decode
variation (5.56%) exceed policy, so this is still inconclusive and unaccepted.
The reverse-order follow-up also reproduced exact output. With all ten samples
per variant retained, candidate prefill variation remains 5.98% and baseline
decode variation is 7.22%. Mean gains are 8.48% prefill and 4.31 times decode,
but policy remains inconclusive. This needs a diagnosis of the variance before
further acceptance measurements. Evidence is under
`gitignore/logs/qwen38-q4-prefill-rows/`, including `pooled-evaluation.json`.

## Qwen3.8 native captured reference

Task #756 establishes native WebGPU capture for the unchanged Qwen3.8 text
graph and weights on webgfx-104. Local ORT revision
`a27f28316f0adba7c43bf769055ab85ba760a244` adds exact int64 support to the
metadata/view operations `Shape` and `Squeeze`. All 41 relevant provider tests
pass, including 18 cases with full-range int64 values, CPU fallback disabled,
and observed RTX 5080 execution. Windows subgroup matrices remain disabled.

Local GenAI revision `8412ddd8550fa52676643f3d3fc0db76fb476ccb` provides the
opt-in `ORTGENAI_QWEN_STATIC_CAPTURE=1` configuration for this 64-layer,
5120-hidden, 248320-vocabulary model at batch 1 and capacity 640. It specializes
the session's batch and total-length dimensions, preserves a zero-padded
640-element mask, and reuses position/mask GPU buffers. The embedding table
stays on CPU; only requested embedding rows are copied to the decoder.

Long-prompt validation exposed another native bug: prefill chunking added byte
offsets to opaque WebGPU buffer handles. The corrected path uses the device copy
API when the device does not support offset tensor views. The failed probe and
its source/binaries are retained; no timing from that failed version is valid.

The final build passes changed-prompt and reset checks, first-token generation,
arithmetic, the original 29-token sky continuation, and full 512/128 generation.
A varied 512-token prompt also matches 32 original eager-reference output tokens
exactly in two repetitions per runtime. A separate diagnostic run records 78
WebGPU graph replays across six graph IDs. Profiling places MatMulNBits,
CausalConvWithState, LinearAttention, and GroupQueryAttention entirely on
WebGPU. Only shape-related Gather and Slice nodes remain on CPU; the decoder
graph has no internal Memcpy nodes.

Five clean repetitions after a complete warmup, with capacity 640 and 32-token
prefill chunks, produce:

| Metric | Mean tokens/s | Coefficient of variation |
| --- | ---: | ---: |
| Prefill | 211.00 | 0.22% |
| Decode | 39.55 | 0.31% |

Every 128-token continuation is exact, with 127 timed decode calls. Reset costs
are reported separately (57.13–62.21 ms). Profiling, verbose logs, and external
GPU sampling are excluded from these measurements. The text graph SHA-256 is
`44b712df872250c1c1614c3aa717788a103f3a814e0cfd24b22196d3b99509cd`, unchanged
from the publisher; the original artifact fingerprint remains
`ec7b289f16fc47d6e2cda28ce055bee8a24fc216ae650a0fcda4fe0a8041991d`.

Exact requests, commands, tokens, profiles, and binary hashes are under
`gitignore/logs/qwen38-ort-capture-audit/`. The independently archived reference
package is under
`gitignore/evolution/backups/ort-models/20260930-qwen38-capture-a27f2831-8412ddd/`,
including both source archives and patches. The model registry now routes native
Qwen3.8 reference jobs through the validated capture view and the package at
`gitignore/evolution/backups/ort-models/20260930-qwen38-capture-routed/`.
The existing task #755 daily package serves the other models, and Backpack
runtime defaults remain unchanged. Media, beam search, other context
capacities, and appending another turn after decoding are outside this capture
validation.

The automatic adapter preserves the publisher artifact fingerprint while
recording the native configuration view separately. It verifies the original
graphs, weights, and tokenizer; the capture capability and binary hashes; the
640-token capacity; and exact continuation across warm resets before timing.
The real adapter passed five 512/128 repetitions at 210.67 prefill and 38.99
decode tokens/s. All 111 framework tests pass, including routing, artifact
identity, reset checks, and protection of completed task commands.

To run that same native reference path on webgfx-104:

```powershell
python -B evolution/benchmark_ort.py --model gitignore/models/Qwen3.8-27B-ONNX --reference-model gitignore/models/Qwen3.8-27B-ORT-WebGPU --bin-dir gitignore/evolution/backups/ort-models/20260930-qwen38-capture-routed --qwen-static-capture --prompt "Answer with only the number. What is 2 + 2?" --required-fact 4 --expected-output 4 --prompt-tokens 512 --generation-tokens 128 --repetitions 5 --evidence-root gitignore/logs/qwen38-native-reference
```

Routing evidence is under `gitignore/logs/qwen38-native-routing/`.
The daily upstream refresh helper's missing external script is a separate
remaining maintenance task; model routing does not bypass that refresh gate.

## Qwen3.8 parameter packing with resource priming

Task #760 combines the existing opt-in parameter arenas with the unaccepted
rows4, decode priming, and reset cleanup candidate. Source
`6c7e6fb4cd57ea49963af7d5cb81be70854a6ae5` preserves captured parameter field
offsets, makes arena owners copy sources for priming, and retires each owner
once after captured references are released. Both arena modes pass 67 D3D12
operator/session tests, two exact full 512/128 repetitions, and the 36-token
Backpack sky continuation across three resets.

The allocator/host-counter diagnostic reports 17.43 → 16.36 GB native used
allocation after prefill, and 16.72 → 15.65 GB at warm decode. The sampled
first warm 32-token prefill chunk falls from 1793.81 to 1123.00 ms, including
submission time of 924.43 → 633.67 ms. Dispatch, submit, and write counts are
unchanged. These instrumented times are diagnostic, not acceptance throughput.

The first clean five-repeat series improves mean prefill from 21.09 to 26.18
tokens/s and decode from 3.75 to 4.43 tokens/s, with exact outputs. A
reverse-order follow-up retains all samples: candidate decode variability is
still 5.24% over ten observations, exceeding the 5% policy limit. Pooled mean
gains are 18.68% prefill and 12.70% decode. The candidate
remains unaccepted and is absent from root runtime defaults. Evidence is under
`gitignore/logs/qwen38-primed-param-arenas/`.

Task #761 tested read-only prefill resource priming at revision
`dfa073f97ec806d2e72f6e086d1cfa040411ae1f`, preserving the original dispatch
submissions and chunk fences. All 67 tests and the full 128-token continuations
passed. The hypothesis was rejected: sampled warm prefill submission time did
not fall (617.71 → 637.56 ms), and the pass added a submission without reducing
native allocation. Full instrumented warm prefill was 19.39 → 21.89 seconds;
this is diagnostic evidence, not a confirmed clean-throughput regression.
No clean campaign or integration was pursued for this failed premise. Preserve
`gitignore/logs/qwen38-prefill-resource-priming/` before considering related ideas.

Task #762 tested releasing the 1.03 GB unused pool after decode capture. It
preserved exact output and reduced native used allocation from 15.65 to 14.61 GB,
but Dawn's reserved heaps remained at 17.47 GB. The later decode GPU sample did
not improve. Live buffer bytes alone therefore did not explain the remaining
decode cost.

The pinned Dawn D3D12 backend did not implement recycled-heap cleanup in its
public `ReduceMemoryUsage` API. An isolated patch at Dawn revision
`e98c18fbe81f3b67408b19aad92f061dae3525cc` adds fenced cleanup and updates the
allocation counters at the completed serial. Three pooled-allocation tests
preserve pending copies and live contents while reducing reservation from
150.99 to 12.58 MB; all 67 Backpack operator/session tests pass with that DLL.
The build recompiles two implementation objects and relinks the remaining
cached objects from the matching Dawn base `181bf8634c1d6e7773e1852d1ca95a79d0b0abe5`.
The installed DLL and source checkout remain unchanged.

On Qwen3.8, that API lowers reserved allocation to 15.68 GB and preserves all
128 output tokens across both repetitions. The sampled GPU step is essentially
unchanged (186.93 → 183.83 ms), and reduction costs 104–168 ms. Task #762 is
rejected as a speed experiment; the validated memory API fix and its failed
development probes remain isolated. These are diagnostics, not clean speed or
regression claims. `MatMulNBitsQ4Blocked` still accounts for about 86% of decode
GPU time. Evidence: `gitignore/logs/qwen38-primed-pool-trim/`.

## Explicit GQA cache capacity

The standalone correctness checkpoint
`2eca773` fixes GQA inferring static cache layout from a rounded allocation.
A packed 511-token cache can occupy a buffer rounded to 512 tokens by the pool;
that does not add padding between heads. Treating it as a static 512-token
layout shifts every head after the first and can read old pooled contents.
Larger Q4 test fixtures exposed this bug even with the Q4 optimization disabled.

`GpuTensor::kvCacheCapacity` now declares the sequence stride of reserved KV
storage. LLM sessions set it when allocating their caches. Ordinary packed
inputs use the dynamic append path regardless of excess allocation capacity.
The regression test poisons recycled buffers, checks both packed and explicitly
reserved caches, and compares the full returned key/value tensors with CPU data.

All 65 standalone operator/session tests pass on RTX 5080/D3D12. Qwen3.5-2B/4B
sky outputs match fresh root controls across three resets. Five measured
512/128 runs per configuration show no protected regression beyond policy;
the highest coefficient of variation is 4.84%. All eight cared LLM model/format
paths retain exact 128-token application output. Only the four correctness
files and matching application binaries were integrated. The user's root index
is preserved, and no broader performance milestone is claimed.

Evidence and the previous binaries are under
`gitignore/logs/gqa-explicit-kv-capacity/`. The initial 35-token Qwen2B sky
expectation came from an obsolete diagnostic artifact; fresh accepted root and
candidate controls both produce the validated 29-token output, and that audit
is retained.

Task #763's separate constant-layout Q4 decode prototype passes bitwise GPU
checks on all nine real model shapes and independent CPU samples. Its corrected
shader preserves the uniform binding after layout constants are substituted.
Two full-model repetitions per mode match all 128 expected tokens, and profiling
confirms 497 specialized calls. Sampled Q4 GPU time is 155.07 → 149.98 ms.
The clean comparison retained ten samples per mode, including reverse order.
Mean prefill/decode gains were 2.53%/3.15%, but baseline decode variability was
5.53%, above policy. The result is inconclusive and is not integrated. Task
#766 separately validates the complete scoped configuration against the accepted
runtime. Evidence: `gitignore/logs/qwen38-q4-decode-layout/`.

## Local native reference refresh

Task #765 replaces the removed external build helper with
`evolution/build_native_reference.py`. Builds and backup publication enforce
webgfx-104 and keep generated files under `gitignore/`. The driver preserves
existing build generators, builds the shared-provider library, normalizes the
ORT SDK, and builds GenAI and the structured conformance helper. The refresh
script uses the emitted build paths and publishes complete, hashed backups.

All 117 framework tests pass. A real build of pinned ORT `a27f28316f` and GenAI
`8412ddd855` passes reset conformance and five 512/128 benchmark repetitions
on Gemma E2B, Qwen3.5-2B/4B, and Qwen3.8-27B. The exact post-build script sections
also build `model_chat`, publish an isolated backup, and pass a Gemma chat check.
Packaged runtime hashes match the four-model validation.

This validates the local orchestration repair. Fetching and building the latest
upstream revisions was a separate follow-up; this task preserved selected backups.
Evidence: `gitignore/logs/native-refresh-local-helper/integration.json`.

Task #769 completed that follow-up on webgfx-104. It fetched ORT `967a8e064d`
and GenAI `fa55959bc6`, then retained the validated local fixes at ORT `9a60fe69d9`
and GenAI `19a003a751`. The GenAI merge uses upstream session-owned embedding
staging and retains safe chunk copies for opaque WebGPU handles.

All four native LLM adapters pass reset conformance and five 512/128 measurements.
The rebuilt binaries preserve all 128 tokens against the previous validated
native pair on every model, including internal Qwen prefill chunking. The original
Qwen sky prompt stops at the same 29 tokens. Eighteen int64 metadata/view cases
pass with CPU fallback disabled. Profiles confirm heavy decoder operations on
WebGPU for all four models; Qwen capture/replay markers and the native process's
RTX 5080 GPU UUID are recorded. Qwen3.5-2B/4B still use the explicitly marked
temporary capture-disabled configuration.

The complete validated package is selected at
`gitignore/evolution/backups/ort/20260930-183323-ort-9a60fe69d9-genai-19a003a751/`.
Older packages are preserved. The adapter accepts both existing manifest hash
formats while checking every required binary; all 117 framework tests pass.
Evidence: `gitignore/logs/native-latest-refresh-20261001/`.

Task #766's separate Backpack default bundle remains isolated. Ten samples per
version show mean prefill 20.63 → 26.05 tok/s and decode 0.926 → 4.334 tok/s,
with exact output. All 69 operator tests and eight LLM model/format checks pass.
Other-model regression checks also pass; Gemma's independent complete-session
measurements retain the slow repetition seen in both versions. The default gate
still marks Qwen baseline decode variability (5.79%) inconclusive. A tested,
opt-in separated-sample exception is awaiting user approval; production policy
and Backpack runtime binaries have not changed.

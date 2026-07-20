# Backpack self-evolution framework

Windows WebGPU policy: subgroup-matrix operations are unsupported for both
Backpack and native ONNX Runtime. Correctness and optimization work must use
portable WebGPU paths such as DP4A, scalar, or f16 fallbacks; the scheduler
must not create subgroup-matrix enablement tasks on Windows.

Status: initial architecture proposal (2026-07-17)

## 1. Objective

Turn Backpack's existing optimization, correctness, profiling, and comparison
workflows into a continuous evidence-driven loop:

```text
observe upstreams -> propose idea -> implement candidate -> validate on devices
        ^                                                   |
        |                                                   v
 learn from outcomes <- integrate/reject <- decide <- evaluate evidence
```

The system should accelerate engineering work without allowing an agent to
silently weaken correctness, cherry-pick performance results, or commit an
unreviewed change to the main branch.

## 2. Design principles

1. **Evidence is immutable.** Raw measurements, environment fingerprints,
   logs, and the exact candidate commit are retained for every conclusion.
2. **A candidate is tested, not a mutable checkout.** Every device evaluates
   the same commit and benchmark manifest in an isolated worktree.
3. **Correctness is a hard gate.** Performance cannot compensate for a failed
   correctness or stability requirement.
4. **Performance is statistical.** Decisions use repeated samples and an
   explicit noise budget, not one before/after measurement.
5. **Device policy is task-specific.** A task declares required ("cared") and
   informational devices before execution. Missing required evidence cannot be
   interpreted as success.
6. **Automation is policy bounded.** Agents may propose and test freely;
   integration occurs only through a dedicated integrator after all configured
   gates pass.
7. **Humans see exceptions and consequential choices.** The dashboard should
   foreground blocked work, mixed results, policy overrides, and decisions,
   rather than require watching routine execution.

## 3. Reuse from webgfx-agents

`D:\workspace\project\agents\webgfx-agents` provides a useful control-plane
starting point:

| Existing mechanism | Reuse | Required change for Backpack |
|---|---|---|
| TypeScript server, REST API, SSE dashboard | Reuse pattern | Add evolution/task/evidence/decision views |
| WebSocket agent registration and heartbeat | Reuse | Fingerprint GPU, driver, Dawn, backend, compiler, power mode, and calibration status |
| Job -> per-agent Run materialization | Reuse | Materialize from a frozen experiment manifest and required device selector |
| One active task per machine reservation | Reuse directly | Also acquire a device lease and reject concurrent GPU workloads |
| FIFO scheduler and reconnect handling | Reuse | Add capability matching, task priority, and experiment cancellation |
| Script executor with streamed output | Reuse concept | Replace arbitrary dashboard shell commands with versioned task adapters and typed results |
| JSON files for jobs/results | Useful for prototype | Move authoritative relational state to SQLite initially; store large artifacts in `gitignore/evolution/` |
| Repository/model synchronization | Reuse pattern | Fetch exact candidate/base commits; verify model and binary hashes |
| Baseline comparison | Reuse concept | Baselines are keyed by device fingerprint and benchmark manifest version |

The webgfx scheduler should be extracted or ported, not coupled to its AI-test
result format. In particular, its `Job` and `Run` remain execution primitives;
an `EvolutionTask` owns one or more jobs across implementation, validation,
debate, and integration phases.

## 4. Logical architecture

```text
                       Dashboard / REST / SSE
                                |
                     Evolution control plane
        +-----------------------+------------------------+
        | task state machine    | policy/evidence engine |
        | scheduler             | decision inbox         |
        | source learner        | integration manager    |
        +-----------------------+------------------------+
                    | WebSocket             | git
                    v                       v
          Machine agents / device pool   isolated worktrees
             |        |        |             candidate refs
          NVIDIA     AMD     Intel/other          |
             +--------+--------+------------------+
                              |
                artifact store + SQLite metadata
                 (`gitignore/evolution/` locally)
```

### Components

- **Control plane:** authoritative state machine, scheduling, authentication,
  audit log, APIs, and live events.
- **Machine agent:** advertises capabilities, leases one GPU, prepares an exact
  revision, runs typed adapters, and uploads signed/hash-addressed evidence.
- **Implementer:** converts an accepted proposal into a candidate commit in an
  isolated branch/worktree. It never writes directly to the integration branch.
- **Evidence engine:** normalizes results, checks completeness and correctness,
  calculates confidence/regression classifications, and produces a verdict.
- **Debate coordinator:** invokes independent advocate and skeptic reviews for
  mixed evidence, then a judge applies the declared policy. It stores arguments
  and evidence references, not just a prose conclusion.
- **Human decision service:** records approve/reject/request-more-data actions,
  rationale, actor, timestamp, and the policy version being overridden.
- **Integrator:** verifies candidate ancestry, reruns the required pre-merge
  gate, creates the commit/merge, and records its resulting SHA. This is the
  only component permitted to update the integration branch.
- **Source learner:** watches llama.cpp, ONNX Runtime, and Modular MAX, extracts
  relevant changes with provenance, deduplicates them, and creates proposals.

## 5. Core domain model

### EvolutionTask

The durable unit shown in the dashboard:

```yaml
id: evo-2026-0042
title: Use a backend-specific Q8 decode tile
kind: optimization                 # optimization | correctness | feature | maintenance
state: validating
origin:
  type: upstream                   # human | upstream | experiment | regression
  source: llama.cpp
  source_revision: abc123
  source_urls: []
hypothesis: Reduce Q8 decode memory traffic without changing logits.
base_sha: 012345
candidate_sha: 6789ab
owner_agent: implementer-1
manifest_id: decode-q8-v3
device_policy_id: windows-webgpu-core-v1
decision_policy_id: conservative-perf-v1
created_at: 2026-07-17T12:00:00Z
```

### Experiment manifest

A versioned, reviewable declaration frozen when validation begins:

```yaml
id: decode-q8-v3
models:
  - id: qwen3.5-2b-q4_k_m
    sha256: "..."
  - id: gemma-4-e2b-q4_k_xl
    sha256: "..."
backends: [d3d12, vulkan]
workloads:
  - {mode: correctness, prompt_set: smoke-v2, temperature: 0}
  - {mode: decode, prompt_tokens: 128, generated_tokens: 128}
warmups: 3
samples: 10
metrics: [decode_tok_s, prefill_tok_s, peak_gpu_memory_mb, gpu_time_ms]
correctness:
  op_tests: required
  kernel_tests: required
  max_logit_error: 0.001
artifacts: [stdout, stderr, result_json, profile_on_regression]
```

### Device policy

Selectors describe cared devices; they do not bind a task to transient machine
names. Resolution is frozen into a device matrix before the first run.

```yaml
id: windows-webgpu-core-v1
required:
  - selector: {os: windows, gpu_vendor: nvidia, backend: d3d12}
    count: 1
  - selector: {os: windows, gpu_vendor: amd, backend: d3d12}
    count: 1
informational:
  - selector: {os: windows, backend: vulkan}
timeout_hours: 24
missing_required: block
```

Each resolved target records machine ID, GPU PCI ID, driver version, WebGPU
adapter properties/features/limits, Dawn revision, compiler/build flags, CPU,
RAM, OS build, power configuration, model hashes, and calibration result.

### Evidence and verdicts

One `RunEvidence` represents one base or candidate execution on one resolved
target. It references raw artifacts and contains normalized samples. An
`Evaluation` compares paired base/candidate evidence on the same device.

Possible per-device verdicts are:

- `positive`: correctness passes and the primary metric improves beyond noise;
- `neutral`: correctness passes and the confidence interval is within the
  declared equivalence band;
- `negative`: correctness/stability fails or a protected metric regresses;
- `inconclusive`: missing, noisy, interrupted, or environment drifted.

The aggregate verdict is `accept`, `reject`, `debate`, or `blocked`.

### Decision

A decision is a first-class record with status `pending`, `approved`,
`rejected`, `more_evidence_requested`, or `expired`. It contains the concrete
question, available options, recommendation, evidence links, risk, deadline,
and final rationale. Dashboard comments alone are not decisions.

## 6. Task state machine

```text
proposed -> triaged -> implementing -> candidate_ready -> validating
   |          |             |                 |              |
   v          v             v                 v              v
 rejected   blocked       failed            failed    evaluating
                                                           |
                         +---------------------------------+------------------+
                         |                                 |                  |
                    all positive                       mixed/neutral       hard fail
                         |                                 |                  |
                    ready_to_merge <--- debate ---> awaiting_human          rejected
                         |
                    integrating -> integrated -> observing
                                         |
                                      reverted
```

Transitions are commands checked by the control plane, not arbitrary status
edits. Every transition appends an audit event. Retrying creates a new attempt
under the same task so earlier evidence remains visible.

### Default decision policy

1. Any correctness failure on a required device: reject; no debate can waive it
   without an explicit human policy override.
2. Any missing/inconclusive required device: block and retry within budget.
3. All required devices positive or neutral, with at least one positive and no
   protected-metric regression: ready to merge.
4. Required devices have mixed positive/negative performance: structured
   debate.
5. A backend-specific optimization may be accepted when its applicability is
   reliably gated and non-target devices are correctness-neutral. The debate
   must verify the dispatch predicate and maintenance cost.
6. The judge may recommend accept/reject/more evidence. High-risk changes,
   policy exceptions, and uncertain judgments enter `awaiting_human`.

Policy is versioned and attached to the task before measurements begin.

## 7. Reliable performance evaluation

For each required device, the agent should:

1. Confirm an idle machine, fixed power mode, sufficient free memory, and no
   conflicting GPU process.
2. Build base and candidate from clean worktrees using the same toolchain.
3. Calibrate; reject the run if thermal or clock variability exceeds policy.
4. Alternate base and candidate samples (ABBA or randomized paired order) to
   reduce temperature and time drift.
5. Report all samples. The server calculates median, dispersion, paired delta,
   confidence interval, and outliers according to the manifest.
6. Automatically repeat an inconclusive comparison up to a fixed budget.

Initial conservative thresholds should be configurable per metric. A practical
starting point for decode throughput is: regression below -2% is negative,
improvement above +2% is positive, and the interval between is neutral, with a
95% paired bootstrap confidence interval. These numbers require calibration on
the actual device pool before enabling automatic integration.

Correctness should include the current op/kernel tests plus deterministic model
checks. Exact generated text is useful as a smoke test but should not replace
logit/tensor tolerances and task-specific invariants.

## 8. Structured debate for mixed results

The debate is a bounded evidence review, not an open-ended conversation:

1. **Analyst** summarizes the device matrix and identifies whether the conflict
   correlates with backend, vendor, driver, model, or workload.
2. **Advocate** argues for integration, citing evidence IDs and possible safe
   capability gates.
3. **Skeptic** argues against integration, focusing on correctness, hidden
   regressions, complexity, and evidence quality.
4. Both may request a small predefined set of additional experiments.
5. **Judge** applies the frozen decision policy and emits a machine-readable
   recommendation plus rationale and dissent.

No role may invent measurements. Every factual claim in the stored debate must
reference evidence or source provenance. Debate budgets and maximum extra runs
prevent loops.

## 9. Source-learning pipeline

Create one source adapter for each of llama.cpp, ONNX Runtime, and Modular MAX:

```text
poll pinned upstream -> identify new commits/releases/docs -> relevance filter
-> retrieve patch/context -> extract technique -> compare with Backpack
-> deduplicate -> proposal with provenance -> human/automatic triage
```

A learned idea contains source URL/revision, license, affected subsystem,
mechanism, expected benefit, hardware applicability, implementation sketch,
validation plan, and confidence. The learner proposes techniques; it must not
copy incompatible code or create a task from a headline without inspecting the
source diff/context.

Useful initial relevance categories are quantized matmul, attention/KV cache,
operator fusion, command submission, memory planning, shader generation,
backend capability dispatch, and benchmark methodology. Deduplication keys on
source revision and semantic similarity to open/rejected tasks. Rejected ideas
remain searchable to avoid repeated churn.

## 10. Dashboard information architecture

### Overview

- evolution velocity: proposed, validating, integrated, rejected;
- current benchmark headline by model/device/backend versus baseline and
  llama.cpp/ORT;
- active experiments and device utilization;
- regressions, blocked tasks, unhealthy/offline machines;
- a prominent human decision inbox.

### Task detail

- hypothesis, origin/provenance, candidate diff and commits;
- lifecycle timeline and live run progress;
- device-by-model matrix with base, candidate, delta, confidence, correctness,
  and artifact links;
- debate transcript structured by claims/evidence;
- pending decision with recommendation and explicit actions;
- integration/revert history.

### Machines

Reuse the webgfx agent setup model: configured and connected machines, hardware
inventory, labels, capabilities, health, current lease, last calibration, and
recent failure/noise rate. Secrets are write-only and never returned by the API.

### Ideas

An inbox grouped by upstream source, with provenance, relevance, novelty,
estimated cost/benefit, duplicate links, and actions to accept, defer, or reject.

## 11. Persistence and repository layout

Proposed source layout:

```text
evolution/
  shared/               typed protocol and schemas
  server/               API, scheduler, state machine, policy engine
  agent/                machine service and typed task adapters
  dashboard/            web UI
  adapters/             build/test/benchmark/source adapters
  policies/             versioned decision and device policies
  manifests/            versioned experiment manifests
docs/
  self-evolution-framework.md
```

All runtime state and generated material follows the repository rule:

```text
gitignore/evolution/
  state.db
  worktrees/<task>/<base-or-candidate>/
  artifacts/<sha256>/...
  logs/...
  models/...
```

Version-control policies/manifests and source code. Do not version-control the
database, checkouts, builds, downloaded models, logs, profiles, or raw results.
SQLite in WAL mode is sufficient for the first single-server deployment;
encapsulate persistence behind repositories so PostgreSQL can replace it if
multiple control-plane replicas are later needed.

Minimum tables are `evolution_tasks`, `task_attempts`, `experiment_manifests`,
`device_policies`, `machines`, `device_snapshots`, `jobs`, `runs`, `evidence`,
`evaluations`, `debates`, `decisions`, `ideas`, `artifacts`, and append-only
`audit_events`.

## 12. Security and integration safety

- Authenticate dashboard mutations; use role-based permissions for operator,
  reviewer, and integrator actions.
- Give machine agents scoped enrollment tokens, rotate them, and use TLS.
- Allow only signed/versioned task adapters. Do not expose a general remote
  shell through the dashboard.
- Redact secrets from streamed logs and cap artifact size/retention.
- Verify commit and manifest hashes on the device and again on upload.
- Require clean worktrees and prohibit candidate code from modifying the
  control plane, policies, benchmark parser, or acceptance thresholds in the
  same task unless a human explicitly approves that scope.
- Protect the integration branch. The integrator uses a service identity and
  only accepts a short-lived merge authorization tied to task, candidate SHA,
  evidence set, and policy verdict.
- Run a post-integration canary; automatically open a revert decision when
  protected metrics regress relative to the accepted evidence.

## 13. Implementation roadmap

### Phase 0: freeze contracts and calibrate (about 1 week)

- Define JSON Schemas for task, manifest, machine fingerprint, evidence, and
  decision.
- Wrap `runtime` tests and `benchmarks/run_benchmark.py` in one typed local
  adapter that emits normalized JSON.
- Measure benchmark noise on two representative devices and set provisional
  thresholds.
- Move new generated outputs to `gitignore/evolution/`.

Exit: the same base/candidate comparison can be replayed locally from a frozen
manifest and produces a deterministic evaluation record.

### Phase 1: observable read-only dashboard (about 1-2 weeks)

- Stand up server, SQLite migrations, audit events, REST/SSE, and dashboard.
- Import the current `goal.md`, `autopo.yaml`, benchmark results, and machine
  inventory as read-only views; do not automate commits.
- Implement decision inbox and task timeline.

Exit: current work, device status, perf evidence, and pending decisions are
visible in one place.

### Phase 2: distributed experiment runner (about 2 weeks)

- Port agent registration, heartbeat, system info, reservation, scheduler, and
  result upload patterns from webgfx-agents.
- Add capability matching, exact-revision worktrees, device leases, artifact
  hashing, model caching, retry/idempotency, and base/candidate paired runs.

Exit: a manually created candidate is evaluated on every required device, and
missing evidence blocks the verdict.

### Phase 3: policy evaluation and debate (about 1-2 weeks)

- Implement correctness/performance gates and aggregate verdicts.
- Add advocate, skeptic, analyst, and judge roles with evidence citations and
  bounded follow-up experiments.
- Route policy exceptions to authenticated human decisions.

Exit: positive, negative, mixed, and inconclusive matrices take distinct,
auditable paths.

### Phase 4: guarded integration (about 1 week)

- Add protected integration branch, merge authorization, pre-merge recheck,
  integration record, and post-merge canary/revert workflow.
- Begin in mandatory-human-approval mode. Enable automatic integration only
  after a sustained shadow period demonstrates zero unsafe recommendations.

Exit: accepted candidates can be integrated reproducibly and traced back to
their complete evidence set.

### Phase 5: upstream learning loop (about 2 weeks)

- Implement pinned adapters for llama.cpp, ONNX Runtime, and Modular MAX.
- Add provenance/licensing checks, relevance classification, deduplication,
  proposal ranking, and outcome feedback.

Exit: upstream changes produce well-specified, deduplicated proposals that enter
the same lifecycle as human ideas.

## 14. First vertical slice

The first implementation should deliberately be narrow:

1. One optimization task created manually.
2. One frozen manifest: Qwen3.5-2B, deterministic correctness plus decode 128.
3. Two configured Windows machines, one required D3D12 target each.
4. Base and candidate paired measurements with ten samples.
5. Dashboard device matrix and one human approve/reject decision.
6. Integration produces a merge commit only after approval.

This slice exercises the whole trust chain before adding autonomous
implementation, debate, or source learning.

## 15. Decisions to make before implementation

These should appear as the first dashboard decision records:

1. **Control-plane codebase:** implement under Backpack or generalize/fork
   webgfx-agents. Recommendation: build Backpack-specific domain modules while
   porting its proven transport/scheduler patterns; avoid coupling the two
   repositories' release cycles initially.
2. **Required device fleet:** enumerate the minimum GPU/vendor/backend matrix
   and which machine can satisfy each selector.
3. **Integration mode:** mandatory human approval during the shadow period
   versus immediate policy-driven merges. Recommendation: mandatory approval.
4. **Performance policy:** protected models/metrics, equivalence bands, sample
   counts, and acceptable benchmark duration, calibrated from measured noise.
5. **Model distribution:** shared network storage versus content-addressed
   agent caches. Recommendation: agent caches with hashes and an internal
   server/object-store source.
6. **Agent implementation authority:** whether candidates are generated by an
   existing coding-agent service or initially supplied as human-created refs.
   Recommendation: start with supplied refs, then add autonomous implementers
   after validation and integration are trustworthy.

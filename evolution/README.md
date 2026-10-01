# Backpack Evolution MVP

This directory contains the first end-to-end implementation of the
self-evolution framework described in
[`docs/self-evolution-framework.md`](../docs/self-evolution-framework.md).
It uses only the Python standard library. Runtime state, experiment output, and
the SQLite database are written beneath `gitignore/evolution/`.

The active goal targets **webgfx-104 / NVIDIA GeForce RTX 5080 only**. The server limits scheduling and
the current device matrix to that machine, cancels queued work for excluded
devices, and rejects remote enrollment/provisioning. The agent also checks the
actual hostname before synchronizing or executing work. Existing fleet history
is retained; it does not add devices to the active scope.

## Start the dashboard

From the repository root:

```powershell
python -m evolution.server
```

Open <http://127.0.0.1:8787>. The server binds only to localhost by default.
Use `--host` and `--port` to change the listener.

## Register the local device

On webgfx-104:

```powershell
$env:BP_EVOLUTION_BACKEND = "webgpu"
python -m evolution.agent --server http://127.0.0.1:8787 register
```

`BP_EVOLUTION_GPU` and `BP_EVOLUTION_DRIVER` can override automatic Windows
GPU discovery. Labels can describe selectors that are not detected directly:

```powershell
python -m evolution.agent --server http://127.0.0.1:8787 `
  --label gpu_vendor=nvidia --label pool=required register
```

The returned machine `id` is used when creating experiment evidence.

Populate the local model directory with available cared-model formats from the
local catalog:

```powershell
python -m evolution.agent --server http://127.0.0.1:8787 sync-models
```

The catalog cares about Gemma 4 E2B IT QAT, Qwen 3.5 4B, Qwen 3.5 2B,
Qwen 3.8 27B, and the requested Qwen-Image-3.0 (artifact identity pending).
Missing artifacts remain pending; catalog membership does not establish support.

After an accepted task becomes an integration milestone, webgfx-104 can fetch
the exact new base into an isolated local worktree without modifying the
developer checkout:

```powershell
python -m evolution.agent --server http://127.0.0.1:8787 sync-base
```

For continuous heartbeat and automatic base synchronization, run the watcher as
a startup service on webgfx-104:

```powershell
python -m evolution.agent --server http://127.0.0.1:8787 --label role=required watch
```

Remote worker provisioning is disabled for this goal. Keep generated agent
state under `gitignore/` on webgfx-104.

## Daily reference cycle

Performance records include `measured_processes` and
`measured_repetitions_per_process`, along with per-process `warmup_runs` and
`reuse_generator`. Status compares regression evidence only within the same
recorded schedule: reset/allocator state can differ between two conversations
per process and five. Missing historical counts remain unknown; total samples
and prose descriptions do not establish process boundaries. Both observation
quarantine and confirmed-regression grouping use these fields. The numerical
regression thresholds and optimization CV policy are unchanged.

The control plane creates one dated refresh task for ORT/ORT GenAI and one for
llama.cpp. `webgfx-104` builds or downloads each x64 artifact once, records its
source revision and date below `gitignore/evolution/backups/`, and keeps all
artifacts on this device. Refresh scripts enforce the local hostname. ORT builds use isolated latest-`origin/main`
worktrees under the deliberately short `gitignore/o/` path (required by
MSBuild/DXC path limits), leaving developer checkouts
untouched.

A successful refresh unlocks dated reference jobs for all cared models on
webgfx-104. Each job runs deterministic conformance first and then the standard
512-input/128-output, five-repetition benchmark. Qwen exact-answer checks are
enforced by both reference adapters. The local worker copies the adapters with the
revisioned runtimes, so measurements do not depend on an older accepted
Backpack worktree containing the newest orchestration code.

Idle claim polls are heartbeats. A device without a poll for 150 seconds is
shown as stale and is not counted as online; queued work remains visible until
the worker returns or an operator records a pause reason.

The integrator automatically pushes the accepted SHA to
`refs/heads/evolution/base` using `--force-with-lease`. Publication requires an
`accept` verdict and an `integrating` task; a failed push is recorded and the
previous active milestone remains authoritative.

## Run a paired experiment

The experiment runner alternates base and candidate executions, extracts a
numeric metric from the last suitable JSON output line, and writes normalized
evidence plus raw logs under `gitignore/evolution/experiments/`.

```powershell
python -m evolution.experiment `
  --task-id evo-example `
  --machine-id machine-example `
  --base-sha BASE_COMMIT `
  --candidate-sha CANDIDATE_COMMIT `
  --base-command '["gitignore/runtime/base/backpack_llm.exe","--benchmark"]' `
  --candidate-command '["gitignore/runtime/candidate/backpack_llm.exe","--benchmark"]'
```

Commands must be JSON arrays and are launched directly without a shell. Each
command must exit successfully and emit a JSON object containing
`decode_tok_s` (or the value supplied with `--metric`). The runner does not
prepare worktrees; operators must ensure both executables correspond to the
declared SHAs. The server rejects evidence whose SHA differs from the task's
frozen base or candidate SHA.

Upload the resulting files:

```powershell
python -m evolution.agent --server http://control-host:8787 upload `
  gitignore/evolution/experiments/evo-example/machine-example/base-evidence.json
python -m evolution.agent --server http://control-host:8787 upload `
  gitignore/evolution/experiments/evo-example/machine-example/candidate-evidence.json
```

Then invoke `POST /api/tasks/<id>/evaluate` or use an API client. The dashboard
will show the resulting device matrix.

## HTTP API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/status` | Overview counts |
| `GET`, `POST` | `/api/tasks` | List/create tasks |
| `GET` | `/api/tasks/:id` | Task, evidence, evaluations, decisions, audit |
| `GET` | `/api/tasks/:id/context` | Build a deterministic bounded agent context |
| `POST` | `/api/tasks/:id/delegate` | Delegate one task to a bounded leaf-agent role |
| `POST` | `/api/tasks/:id/transition` | Guarded lifecycle transition |
| `POST` | `/api/tasks/:id/candidate` | Freeze base and candidate SHAs |
| `POST` | `/api/tasks/:id/evaluate` | Run the policy engine |
| `GET`, `POST` | `/api/memory` | Query or reinforce structured memory records |
| `GET` | `/api/memory/status` | Memory, context-budget, and learning-cursor summary |
| `GET`, `POST` | `/api/agent-sessions` | Inspect or start delegated agent sessions |
| `POST` | `/api/machines/register` | Register or heartbeat a machine |
| `POST` | `/api/machines/configure` | Add an expected offline fleet member |
| `GET` | `/api/machines` | Device pool |
| `POST` | `/api/evidence` | Upload normalized evidence |
| `GET` | `/api/models/matrix` | Cared-model conformance/performance matrix |
| `GET` | `/api/models/manifest` | Available model files for synchronization |
| `POST` | `/api/observations` | Add conformance and performance observation |
| `GET`, `POST` | `/api/history` | Verified optimization gains and evidence |
| `GET` | `/api/milestones/current` | Exact accepted base for device synchronization |
| `GET` | `/api/decisions?status=pending` | Human decision inbox |
| `POST` | `/api/decisions/:id/resolve` | Record a human resolution |
| `GET` | `/api/events` | Server-sent live events |

Mutating requests accept `X-Evolution-Actor` for audit attribution. This MVP
has no network authentication and must remain on a trusted network. Add TLS,
enrollment tokens, and role-based authentication before exposing it remotely.

## Tests

```powershell
python -m unittest evolution.tests.test_framework -v
```

The current tests cover state transition guards, commit binding, missing
evidence, correctness rejection, and positive performance acceptance.

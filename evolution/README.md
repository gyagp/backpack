# Backpack Evolution MVP

This directory contains the first end-to-end implementation of the
self-evolution framework described in
[`docs/self-evolution-framework.md`](../docs/self-evolution-framework.md).
It uses only the Python standard library. Runtime state, experiment output, and
the SQLite database are written beneath `gitignore/evolution/`.

## Start the dashboard

From the repository root:

```powershell
python -m evolution.server
```

Open <http://127.0.0.1:8787>. The server binds only to localhost by default.
Use `--host` and `--port` to change the listener.

## Register a device

On each benchmark machine:

```powershell
$env:BP_EVOLUTION_BACKEND = "webgpu"
python -m evolution.agent --server http://control-host:8787 register
```

`BP_EVOLUTION_GPU` and `BP_EVOLUTION_DRIVER` can override automatic Windows
GPU discovery. Labels can describe selectors that are not detected directly:

```powershell
python -m evolution.agent --server http://control-host:8787 `
  --label gpu_vendor=nvidia --label pool=required register
```

The returned machine `id` is used when creating experiment evidence.

Sync all available cared-model formats from the control machine with:

```powershell
python -m evolution.agent --server http://control-host:8787 sync-models
```

The catalog currently cares about Gemma 4 E2B IT QAT, Qwen 3.5 4B, and Qwen
3.5 2B. Missing GGUF or ORT formats are reported as unavailable and are not
treated as failures.

After an accepted task becomes an integration milestone, each device can fetch
the exact new base into an isolated worktree without modifying its developer
checkout:

```powershell
python -m evolution.agent --server http://control-host:8787 sync-base
```

For continuous heartbeat and automatic base synchronization, run the watcher as
a startup service on every cared device:

```powershell
python -m evolution.agent --server http://control-host:8787 --label role=required watch
```

For first-time Windows worker setup, download and inspect the bootstrap before
running it in PowerShell on that worker:

```powershell
Invoke-WebRequest http://10.172.21.28:8787/api/bootstrap.ps1 -OutFile $env:TEMP\bootstrap-backpack.ps1
notepad $env:TEMP\bootstrap-backpack.ps1
powershell -ExecutionPolicy Bypass -File $env:TEMP\bootstrap-backpack.ps1
```

This registers the worker, syncs cared models, installs an `ONLOGON` scheduled
watcher, and keeps all generated agent state outside the source checkout.

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
| `POST` | `/api/tasks/:id/transition` | Guarded lifecycle transition |
| `POST` | `/api/tasks/:id/candidate` | Freeze base and candidate SHAs |
| `POST` | `/api/tasks/:id/evaluate` | Run the policy engine |
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

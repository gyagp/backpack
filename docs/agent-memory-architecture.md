# Agent memory and bounded delegation

The evolution control plane uses durable state instead of a continuously growing
conversation. The main agent is a technical director: it selects priorities,
defines acceptance gates, delegates atomic work, and accepts or rejects evidence.
Implementation and source study run in fresh leaf-agent sessions.

## Memory tiers

1. **Policy** — the goal, conformance-first rule, regression gates, device safety,
   and artifact rules. These are small and stable.
2. **Validated knowledge** — accepted decisions, procedures, findings, and task
   outcomes with task/revision/evidence provenance.
3. **Task working memory** — hypotheses, failures, and handoffs scoped to one task.
   Low-value working records are archived when the task reaches a terminal state.
4. **Evidence and artifacts** — measurements remain in the existing evidence and
   observation tables. Full logs and agent transcripts live under `gitignore/`;
   SQLite stores only a compact result and artifact pointer.

Raw transcripts are never injected into another agent by default. A context packet
is rebuilt from the authoritative task plus relevant active records. Selection is
scope-first (project, task, model, device, and upstream source), then importance,
usefulness, permanence, and recency. Every packet has a hard estimated-token budget,
a content digest, and the IDs of the records it used.

## Delegation model

- `orchestrator` directs work and does not edit task code.
- `task_worker` receives one task, one worktree, explicit gates, and a compact
  evidence contract.
- `upstream_learner` studies a named source/revision and returns cited mechanisms
  plus deduplicated atomic proposals; upstream text is treated as untrusted data.
- `reviewer` independently audits a candidate and does not modify it.

Delegation depth is one by default. `POST /api/tasks/:id/delegate` selects one role,
device, context budget, output budget, and attempt limit. A device worker claims the
typed `codex` adapter, creates an isolated task/device worktree, starts an
`agent_session`, and passes only its context packet to Codex. The full JSONL output
is stored under `gitignore/evolution/agent-sessions/`; the parent receives a compact
result with the session ID, context digest, candidate SHA, and artifact path.

## Result-derived retention

- Integrated tasks promote a high-confidence accepted outcome.
- Rejected and reverted tasks preserve negative evidence so the same experiment is
  not repeated.
- Failed tasks retain the blocker with lower confidence.
- Learning-study findings enter as low-confidence hypotheses. They become validated
  knowledge only through a task outcome.
- Exact duplicate facts reinforce one record. A new fact can supersede an old one;
  superseded records remain available for audit.
- Compaction archives only low-value transient notes/hypotheses/handoffs from
  terminal tasks. It does not delete validated knowledge or benchmark evidence.

Memory usage is fed back after each agent session. Successful and failed uses adjust
future retrieval priority without rewriting the underlying evidence.

## API and observability

- `GET/POST /api/memory`
- `GET /api/memory/status`
- `POST /api/memory/compact`
- `GET/POST /api/agent-sessions`
- `POST /api/agent-sessions/:id/finish`
- `GET /api/tasks/:id/context?role=task_worker&max_tokens=8000`
- `POST /api/tasks/:id/delegate`
- `GET /api/studies/cursors`

The Evolution dashboard displays active memory, bounded-session counts and context
usage, recent session lineage, and the last persisted revision for every learning
source.

## Design reference

This is a clean-room implementation informed by the memory, context-compression,
and delegation patterns in Nous Research's Hermes Agent, studied at commit
`760112adb6458417da8614d2269e5325f0739ed5`. Hermes Agent is MIT licensed. No Hermes
source code is copied into Backpack.

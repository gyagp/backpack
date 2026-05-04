# Autopo test Session

## Work Unit
Add fence-wait-time instrumentation to measure time spent in completeAsyncMapI32/waitForQueue

## Acceptance Criteria
- readArgmax records time spent blocking in completeAsyncMapI32
- Cumulative fence wait time is reported in benchmark output (ms total and % of decode time)
- Build succeeds

## Rules
# Rules


- **For build system tasks (CMake, Makefile), validate the approach with a minimal config before adding complexity** — wu-001 hit 50 turns trying to configure Dawn via FetchContent — likely hit repeated cmake errors and kept iterating fixes instead of stepping back
  Learned: iteration 1, wu-001

- **If cmake configure fails 3 times in a row, stop and reassess the approach entirely rather than making incremental fixes** — Build system errors compound — each fix can introduce new errors, leading to turn exhaustion
  Learned: iteration 1, wu-001

- **Dawn is a complex dependency with many transitive deps — prefer pre-built binaries or git submodule over FetchContent when possible** — FetchContent for large Chromium-adjacent projects pulls enormous dependency trees that are hard to configure correctly in limited turns
  Learned: iteration 1, wu-001

- **For WGSL shader work units, start by reading the GGUF spec or reference implementation (e.g., llama.cpp) to understand the exact binary layout before writing any code** — The current work unit (Q4_K_M dequantization) requires precise knowledge of block structure — guessing leads to incorrect bit manipulation that's hard to debug without a test harness
  Learned: iteration 2, wu-002 planning

- **When a work unit fails by hitting max turns twice, escalate: either decompose it into smaller units or flag it as blocked rather than retrying the same approach** — wu-001 failed twice at 50 turns each (total ~3000s) with the same fundamental issue. Retrying without structural change wastes budget
  Learned: iteration 1, wu-001 (both attempts)

- **Before executing a work unit, verify it isn't already complete by checking if acceptance criteria are already met** — wu-001 attempt 3 spent turns only to conclude 'No changes were needed — the work unit was already complete', wasting an entire execution cycle
  Learned: iteration 1, wu-001 (third attempt)

- **Work units that pass on first attempt with low turn counts (<5 s
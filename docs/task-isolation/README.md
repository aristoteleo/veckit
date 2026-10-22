# Isolated task imports for concurrent Python scoring

This standalone community contribution is based on upstream `aristoteleo/veckit` commit `cc3859ab2a2305efc4544815d3f44f660efa77d4`. It is available on branch `fix/task-module-isolation` in this public fork and has not been merged upstream. It is independent of the other fixes in this fork.

## Problem and change

The local scorer loaded task files under process-global names such as `metrics`, `metrics_v2` and `_reuse`, deleting/replacing these aliases on each call. Mixed T1/T2/T3 threads could import another task's module or observe a missing/partially initialized alias. Even serial calls replaced modules owned by the calling application.

The patch uses the existing installable T1/T2/T3 packages with task-qualified imports and Python's module import locks. Metric modules use explicit package imports, and no longer insert paths or manufacture generic aliases. The local scorer still never loads task `data.py` or the legacy `_reuse` loader. No metric formula changes are included.

## Recorded evidence

The reproducer starts a fresh process, installs sentinel modules under the five affected generic names, and runs three rounds of 12 real `veckit.score()` calls across six threads before comparing with serial T1/T2/T3 results.

| Check | Upstream run | Patched run |
|---|---:|---:|
| Failed calls | 15 / 36 | 0 / 36 |
| Foreign generic aliases preserved | 0 / 5 | 5 / 5 |
| `sys.path` unchanged | No | Yes |
| Input files unchanged | Yes | Yes |

All 36 patched outputs match serial scoring exactly (including equal-NaN comparison), and serial metrics are identical before/after. The baseline failure count is scheduling-dependent and is not a fixed failure probability. This establishes the tested module-loading behavior, not a universal guarantee of thread safety for every dependency, arbitrary shared mutable object or concurrent writes to the same output path.

See [before.json](before.json), [after.json](after.json), and [verification.json](verification.json) for the raw evidence and environment. All contribution fixtures are invented and generated in temporary directories; no participant predictions or private/held-out data are needed.

## Validation

The full suite passes 10 tests (6 contribution tests plus 4 existing tests). It covers stable task-qualified module identities, cold-start mixed-task concurrent scoring, preservation of foreign aliases and `sys.path`, invalid task rejection, and an import guard that rejects any attempt to load task data or `_reuse`.

A wheel was built from this branch, installed into a temporary directory outside the source checkout, and used for real T1/T2/T3 Python API and CLI calls. All three tasks completed and their API/CLI results matched. Installed-wheel checks are recorded in `verification.json`.

## Reproduce

```bash
git clone --branch fix/task-module-isolation --single-branch https://github.com/eason-hk-barcelona/veckit.git veckit-task-isolation
cd veckit-task-isolation
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest
python -m pytest tests -q
python examples/reproduce_concurrent_scoring.py --expect isolated --out after.json
```

To reproduce the original behavior with the same script, while staying in the contribution checkout:

```bash
git fetch https://github.com/aristoteleo/veckit.git cc3859ab2a2305efc4544815d3f44f660efa77d4
git worktree add --detach ../veckit-baseline cc3859ab2a2305efc4544815d3f44f660efa77d4
python examples/reproduce_concurrent_scoring.py --source ../veckit-baseline --expect collision --out before.json
```

The commands print and optionally save a report. Temporary synthetic H5AD files are removed automatically. Timings and race outcomes are machine/run-specific. This work concerns the published local scorer; it makes no claim about the hosted leaderboard implementation.

## 中文摘要

以 T1/T2/T3 独立模块路径取代共享 metrics 别名，解决混合任务线程间的导入冲突。本次 36 次并发调用从 15 次失败变为全部成功，结果与串行相同，并保留调用方已有的同名模块。

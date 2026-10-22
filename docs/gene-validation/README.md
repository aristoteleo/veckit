# Linear-time gene-panel mismatch validation

This standalone community contribution is based on upstream `aristoteleo/veckit` commit `cc3859ab2a2305efc4544815d3f44f660efa77d4`. It is available on branch `fix/linear-gene-validation` in this public fork and has not been merged upstream. It is independent of the other fixes in this fork.

## Problem and change

When a prediction's gene names differ from the reference, `_prediction()` rebuilt the complete expected/received gene set inside each list-comprehension iteration. A simple gene-order error therefore triggered quadratic work before the actionable error message appeared. This patch constructs each set once and reuses it for membership checks and explicit reordering.

## Recorded evidence

On temporary synthetic H5AD inputs with two cells and reversed gene order:

| Genes | Upstream, seconds (one run) | Patched, seconds (median of five) |
|---:|---:|---:|
| 500 | 0.00517 | 0.00183 |
| 2,000 | 0.06417 | 0.00200 |
| 8,000 | 1.36095 | 0.00488 |
| 32,285 | 34.19675 | 0.01484 |

The timings include reading H5AD and validating the mismatch, **not the full scoring pipeline**. Runs share one machine but have unequal repeat counts; hardware, filesystem caches and scheduling affect the numbers. Error messages are identical before/after and the input bytes are unchanged. The algorithmic change removes repeated set construction; no timing threshold is imposed on tests.

See [before.json](before.json), [after.json](after.json), and [verification.json](verification.json) for the raw evidence and environment. All contribution fixtures are invented and generated in temporary directories; no participant predictions or private/held-out data are needed.

## Validation

The full suite passes 19 tests (15 contribution tests plus 4 existing tests). Coverage includes dense/CSR/CSC matrices, exact panels, opt-in reordering with aligned expression/metadata, ordered missing/extra diagnostics capped at ten names, length mismatches and reference bootstrapping.

A wheel was built from this branch, installed into a temporary directory outside the source checkout, and used for real T1/T2/T3 Python API and CLI calls. All three tasks completed and their API/CLI results matched. Installed-wheel checks are recorded in `verification.json`.

## Reproduce

```bash
git clone --branch fix/linear-gene-validation --single-branch https://github.com/eason-hk-barcelona/veckit.git veckit-gene-validation
cd veckit-gene-validation
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest
python -m pytest tests -q
python examples/benchmark_gene_validation.py --repeats 5 --out after.json
```

To reproduce the original behavior with the same script, while staying in the contribution checkout:

```bash
git fetch https://github.com/aristoteleo/veckit.git cc3859ab2a2305efc4544815d3f44f660efa77d4
git worktree add --detach ../veckit-baseline cc3859ab2a2305efc4544815d3f44f660efa77d4
python examples/benchmark_gene_validation.py --source ../veckit-baseline --repeats 1 --out before.json
```

The commands print and optionally save a report. Temporary synthetic H5AD files are removed automatically. Timings and race outcomes are machine/run-specific. This work concerns the published local scorer; it makes no claim about the hosted leaderboard implementation.

## 中文摘要

将基因面板错误校验中的重复集合构建移出循环。32,285 个合成基因的本机错误校验从 34.20 秒降至约 0.0148 秒；这是校验路径的耗时，不是整个评分流程的速度。

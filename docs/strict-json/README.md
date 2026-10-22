# Strict JSON reports for undefined metric values

This standalone community contribution is based on upstream `aristoteleo/veckit` commit `cc3859ab2a2305efc4544815d3f44f660efa77d4`. It is available on branch `fix/strict-json-results` in this public fork and has not been merged upstream. It is independent of the other fixes in this fork.

## Problem and change

A valid finite expression matrix can produce undefined metrics (for example Pearson correlation for constant expression), as can collapsed spatial coordinates. Both CLI and Python API report files previously emitted bare `NaN`, and Python's default serializer also permits `Infinity`/`-Infinity`. These tokens are rejected by strict JSON consumers even though the CLI exits successfully.

A shared output encoder now writes non-finite floating-point values as JSON `null`, recursively handles report dictionaries/lists/tuples, and sets `allow_nan=False`. The Python return dictionary retains its raw numeric NaN/infinity values. Finite reports keep their previous serialization. `null` means unavailable/undefined, never a zero score; downstream aggregators must decide how to handle missing metrics.

## Recorded evidence

The real CLI and Python API were exercised on four synthetic fixtures (eight exported reports): normal T1, constant-expression T1, collapsed-coordinate T2, and collapsed-coordinate T3. Before the patch, the six reports from edge cases failed strict parsing; after it, all eight parse successfully. CLI stdout matches its output file, raw API NaNs remain available, and all input bytes are unchanged. Unit coverage additionally checks positive/negative infinity, NumPy floating types, nested structures, and preservation of literal text such as "NaN".

See [before.json](before.json), [after.json](after.json), and [verification.json](verification.json) for the raw evidence and environment. All contribution fixtures are invented and generated in temporary directories; no participant predictions or private/held-out data are needed.

## Validation

The full suite passes 19 tests (15 contribution tests plus 4 existing tests). Expected constant-correlation and scikit-learn warnings do not affect test success. Real finite T1/T2/T3 results are equal with and without JSON export. This is an export-boundary change; it does not change metric formulas or substitute a score for undefined values.

A wheel was built from this branch, installed into a temporary directory outside the source checkout, and used for real T1/T2/T3 Python API and CLI calls. All three tasks completed and their API/CLI results matched. Installed-wheel checks are recorded in `verification.json`.

## Reproduce

```bash
git clone --branch fix/strict-json-results --single-branch https://github.com/eason-hk-barcelona/veckit.git veckit-strict-json
cd veckit-strict-json
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest
python -m pytest tests -q
python examples/reproduce_strict_json.py --expect strict --out after.json
```

To reproduce the original behavior with the same script, while staying in the contribution checkout:

```bash
git fetch https://github.com/aristoteleo/veckit.git cc3859ab2a2305efc4544815d3f44f660efa77d4
git worktree add --detach ../veckit-baseline cc3859ab2a2305efc4544815d3f44f660efa77d4
python examples/reproduce_strict_json.py --source ../veckit-baseline --expect nonstandard --out before.json
```

The commands print and optionally save a report. Temporary synthetic H5AD files are removed automatically. Timings and race outcomes are machine/run-specific. This work concerns the published local scorer; it makes no claim about the hosted leaderboard implementation.

## 中文摘要

CLI/API 的 JSON 导出把无定义的非有限数值写为 null，避免下游严格解析器拒绝文件；Python 内存返回值保留 NaN/Infinity。8 个真实导出测试均通过，有限指标不变。

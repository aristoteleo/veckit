# Prevent JSON output from overwriting VEC input data

**Contribution by Yichen (Eason) Guo / Eason_hk.**

This is a small patch to the official `veckit` local scorer. It prevents an output-path
mistake from destroying a prediction, target, reference or wild-type H5AD file. It is
useful for participants running local evaluations manually or through experiment scripts.

The patch is based on upstream commit
[`cc3859ab2a2305efc4544815d3f44f660efa77d4`](https://github.com/aristoteleo/veckit/commit/cc3859ab2a2305efc4544815d3f44f660efa77d4).
It has not been merged upstream or released by the organisers. It changes file-output
validation only; it does not change a metric, model, challenge rule or official submission.

## Reproduced problem

Both the command-line entry point in `score_h5ad.py` and the Python `score()` API in
`veckit.py` call `Path.write_text()` on the result path after loading and scoring the data.
When that path is also a supplied H5AD input, the old implementation completes scoring
successfully and then replaces the HDF5 contents with JSON. The original file can no longer
be read as H5AD. A different-looking symlink or hard link can identify the same input too.

The [reproduction script](../../examples/reproduce_output_collision.py) creates its own
120-cell, 40-gene invented data in a temporary directory. It accepts no external data
paths and touches no participant files. These dimensions are for a local software test,
not a competition-valid prediction. The script checks both real CLI and API calls,
SHA-256 fingerprints and H5AD readability.

| Check | Unpatched upstream | Patched branch |
|---|---|---|
| CLI output points to prediction | Succeeds, replaces H5AD with JSON | Rejects before scoring |
| API output points to prediction | Succeeds, replaces H5AD with JSON | Rejects before scoring |
| Prediction bytes after the collision attempt | Changed | Unchanged |
| Prediction still readable as H5AD | No | Yes |

Recorded results: [before.json](before.json), [after.json](after.json),
[verification.json](verification.json). Exception paths in the reports are replaced with
`<temporary-directory>` by the reproduction script. File fingerprints are unmodified.

## What the patch does

- Checks the output against every supplied `input`, `target`, `reference` and `wt` path.
- Resolves relative paths and symlinks, and checks filesystem identity for hard links.
- Rejects collisions before any expensive metric calculation.
- Rechecks immediately before writing, to catch an alias introduced during scoring.
- Keeps normal output behavior: return/stdout without `out`, creation of a distinct report,
  and replacement of an existing report that is separate from the named data files.

The error identifies the conflicting argument and asks for a separate JSON result file.
For example, use `--out results/task1.json` or `out="results/task1.json"`.

This protects against accidental aliases at the checks. It is not a lock or an atomic
filesystem transaction: a concurrent process changing a path after the final check can
still race the write. Files that are not named as inputs are outside this guard. Normal
report replacement remains intentional behavior.

## Install and verify this branch

Use an isolated environment. From a clone of this fork, on `fix/prevent-output-overwriting-inputs`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . pytest
.venv/bin/python -m pytest tests -q
.venv/bin/python examples/reproduce_output_collision.py --expect protected
```

The exact tested dependency versions are recorded in `verification.json`. The new tests
need no competition account, downloads, GPUs or held-out data. The existing upstream
rotation test uses its already-bundled public sample.

To reproduce the original failure without changing this checkout, install the upstream
revision in a second environment and run the same script with it:

```bash
python3 -m venv .venv-before
.venv-before/bin/python -m pip install \
  'git+https://github.com/aristoteleo/veckit.git@cc3859ab2a2305efc4544815d3f44f660efa77d4'
.venv-before/bin/python examples/reproduce_output_collision.py --expect vulnerable
```

Run that command without a custom `PYTHONPATH` that points to a different checkout.
The script runs its CLI subprocess in the temporary directory so the current working
directory cannot silently select a different `score_h5ad.py`.

## Validation

Tested on macOS arm64 with Python 3.11.16 on 2026-10-23 (Asia/Shanghai):

- The 50 focused output tests produce **36 failures and 14 passes on the original code**.
  After the patch all 50 pass. They exercise actual argument parsing, API calls and file
  writes; only expensive metric functions are stubbed in these focused tests.
- Three additional integration tests use the real T1, T2 and T3 metric panels on invented
  expression/coordinate data. With a separate report, returned and saved results match a
  no-output call exactly, and all input bytes stay unchanged.
- Four existing upstream shape tests still pass.
- **Full suite: 57 passed**, with six upstream scikit-learn `n_jobs` deprecation warnings.

This is a file-integrity result, not evidence of predictive quality or official ranking.
The public scorer retains its upstream version string `0.1.1`; the branch/commit identifies
this patch. No new biological data or participant files are included.

## 中文说明

官方本地评分器原来允许把结果路径设成输入文件。例如将 `out` 指向预测的 `.h5ad`，
评分会完成，但随后写入的 JSON 会覆盖原 HDF5 数据。命令行和 Python API 都有此问题。

本补丁在评分前和写入前检查路径冲突，覆盖相对路径、符号链接和硬链接。冲突时会指出
对应的输入参数并拒绝写入。请把结果写到独立的文件，例如 `results/task1.json`。
已有独立 JSON 报告仍可更新，评分算法没有改动。

上面的复现脚本只创建和操作临时合成数据，不接收真实文件路径。`before.json` 与
`after.json` 保存了修复前后的真实运行结果。新测试覆盖输入保护和三项任务的正常评分，
连同原项目测试共 57 项通过。此成果是社区补丁，尚未由主办方合并或发布；不涉及主赛提交。

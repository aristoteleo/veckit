"""veckit -- Virtual Embryo Challenge local scorer: score a submitted .h5ad against the T1/T2/T3 metric
panels entirely offline, against reference files YOU supply. This package NEVER imports T1/T2/T3's
`data.py` (real Sherlock/Oak paths to the held-out validation/test data) -- see score_h5ad.py's
`_load_task_metrics` for how that's structurally guaranteed, not just a convention.

    pip install git+https://github.com/aristoteleo/vec_baselines.git

    from veckit import score
    result = score(task="T1", input="pred.h5ad", target="T1/9.5.h5ad", reference="T1/8.5.h5ad")
    print(result["metrics"])

Or from the shell, after the same install:

    veckit --task T1 --input pred.h5ad --target T1/9.5.h5ad --reference T1/8.5.h5ad

Every keyword here mirrors a `score_h5ad.py` CLI flag 1:1 -- this module is a thin wrapper so metric
updates land in ONE place (`common/core_metrics.py`, `T1/metrics.py`, ...) and both the CLI and this API
pick them up immediately on the next `pip install --upgrade`; nothing here duplicates scoring logic.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import score_h5ad as _s

_DEFAULTS = dict(reference=None, wt=None, out=None, seed=0, allow_reorder=False, setting="heart")
_PATH_FIELDS = ("input", "target", "reference", "wt", "out")


def score(task: str, input, target, **kw) -> dict:
    """Score `input` for `task` ('T1'/'T2'/'T3') against `target` -- a local file YOU supply, standing in
    for the pseudo target. `reference` (T1/T2) / `wt` (T3) is the DE/shift baseline's reference
    expression; omit it to default to `input` itself (correct only when testing a no-change baseline like
    copy_last/wt_identity -- pass a real one for a real model's `de_score`/`de_direction`/`severity_slope`
    to be meaningful). Returns {"meta": ..., "metrics": ...} with numeric metric values. Undefined
    values remain NaN/infinity in memory and are encoded as null in CLI/API JSON reports.
    Unknown kwargs raise TypeError, so a typo'd flag name fails loudly instead of being silently
    ignored."""
    if task not in ("T1", "T2", "T3"):
        raise ValueError(f"task must be 'T1'/'T2'/'T3', got {task!r}")
    unknown = set(kw) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"score() got unexpected keyword(s): {sorted(unknown)}; "
                        f"valid: {sorted(_DEFAULTS)}")

    fields = {"task": task, "input": input, "target": target, **_DEFAULTS, **kw}
    for name in _PATH_FIELDS:
        if fields[name] is not None:
            fields[name] = Path(fields[name])
    args = Namespace(**fields)

    meta, scores = {"T1": _s.score_t1, "T2": _s.score_t2, "T3": _s.score_t3}[task](args)
    result = {"meta": meta, "metrics": scores}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(_s._result_json(result) + "\n", encoding="utf-8")
    return result

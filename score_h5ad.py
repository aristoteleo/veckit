#!/usr/bin/env python
"""veckit -- score a submitted h5ad against the T1/T2/T3 metric panels, entirely offline, against
reference files YOU supply. This package NEVER reads T1/T2/T3's `data.py` -- those files hold real
Sherlock/Oak paths to the held-out validation/test data, so nothing here even imports them, let alone
opens them.

Examples
--------
Testing a copy_last-style submission (--input IS the preceding stage, so it doubles as --reference too):

    python score_h5ad.py --task T1 --input T1/8.5.h5ad --target T1/9.5.h5ad

Scoring a real prediction, with the genuine preceding stage as --reference (needed for `de_score`/
`de_direction` -- "did you predict the right *change*", not just "does this look plausible"; can't be
computed from a single snapshot):

    python score_h5ad.py --task T1 --input pred.h5ad --target T1/9.5.h5ad --reference T1/8.5.h5ad
    python score_h5ad.py --task T2 --setting heart --input pred.h5ad --target b.h5ad --reference a.h5ad
    python score_h5ad.py --task T3 --input pred.h5ad --target mab21l2_ko.h5ad --wt wt.h5ad

`--target` stands in for the target you're predicting (an extrapolation-shaped pseudo target, the same
direction as the real val/test split -- see T1/TASK1.md etc. for the real one). `--reference` (T1/T2) /
`--wt` (T3) is the reference expression `de_score`/`de_direction`/`severity_slope` are computed against;
if you don't pass one, it defaults to --input itself (only meaningful for a no-change baseline like
copy_last / wt_identity -- those metrics then correctly read as "no predicted change"; pass a real
reference/wt explicitly to get a meaningful number for an actual model).

This is NOT a preview of your real score: the pseudo target was training data a moment ago, so it is
typically an easier question than the real held-out one.

Also installable (`pip install git+https://github.com/aristoteleo/vec_baselines.git`) as `veckit` -- a
`veckit` console command and `from veckit import score` Python API wrap this same code; see veckit.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import signal
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from common.core_metrics import _log


ROOT = Path(__file__).resolve().parent


def _as_float32(X, name: str):
    """Convert to float32 while PRESERVING sparsity. The old version eagerly densified the whole
    cells x genes matrix, which alone was ~2 GiB per file at T1's full 17k x 32k scale (and several copies
    were held at once) -- the root of the OOM. Sparse stays sparse (CSR); only the stored nonzeros are
    validated, never a dense full-matrix op."""
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32, copy=False)
        values = X.data
    else:
        X = np.asarray(X, dtype=np.float32)
        values = X
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D cells x genes matrix, got shape {X.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return X


def _prediction(path: Path, genes, *, need_coords: bool, allow_reorder: bool):
    if not path.exists():
        raise FileNotFoundError(path)
    a = ad.read_h5ad(path)

    pred_genes = [str(g) for g in a.var_names]
    # genes=None means no panel to check against yet -- this call IS what defines the gene panel (the
    # bootstrap file, --input). Every later _prediction() call in the run passes the panel this one
    # returns, so validation still applies to every OTHER file.
    expected = pred_genes if genes is None else [str(g) for g in genes]
    if pred_genes != expected:
        if allow_reorder and set(pred_genes) == set(expected) and len(pred_genes) == len(expected):
            a = a[:, expected].copy()
        else:
            missing = [g for g in expected if g not in set(pred_genes)][:10]
            extra = [g for g in pred_genes if g not in set(expected)][:10]
            raise ValueError(
                "Prediction var_names do not match the task gene panel/order. "
                f"expected {len(expected)} genes, got {len(pred_genes)}. "
                f"missing head={missing}, extra head={extra}. "
                "Use --allow-reorder only if the same genes are present in a different order."
            )

    X = _as_float32(a.X, "prediction.X")
    # validate on stored nonzeros only; for sparse the implicit zeros are 0 (non-negative) so no full-matrix
    # comparison is needed.
    neg_vals = X.data if sparse.issparse(X) else X
    if (neg_vals < 0).any():
        raise ValueError("prediction.X contains negative values; metrics expect log-normalized nonnegative data")

    ct = np.asarray(a.obs["celltype"]).astype(str) if "celltype" in a.obs else np.array(["NA"] * a.n_obs)
    C = None
    if need_coords:
        if "spatial_3D" not in a.obsm:
            raise KeyError("Prediction needs obsm['spatial_3D'] for this task")
        C = np.asarray(a.obsm["spatial_3D"], dtype=np.float32)
        if C.ndim != 2 or C.shape[0] != a.n_obs or C.shape[1] < 3:
            raise ValueError(f"obsm['spatial_3D'] must have shape cells x >=3, got {C.shape}")
        C = C[:, :3]
        if not np.isfinite(C).all():
            raise ValueError("obsm['spatial_3D'] contains NaN or infinite values")

    return X, C, ct, a


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_task_metrics(task: str):
    """Load `{task}/metrics.py` and `{task}/metrics_v2.py` WITHOUT ever touching any task's `data.py`.

    T2's and T3's own `metrics.py` pull T1's (and, for T3, T2's) canonical functions via `from _reuse
    import t1_metrics[, t2_metrics]` -- but the real `_reuse.py` ALSO eagerly loads `data.py` as an
    (unused, by metrics.py) side effect, and this package must not import that file at all, not even for
    its inert path strings. So instead of the real `_reuse.py`, this registers a clean stand-in under
    that name holding only what `metrics.py` actually needs, sourced straight from `T1/metrics.py` (and
    `T2/metrics.py` for T3) -- both confirmed to contain zero references to Sherlock/Oak paths.
    """
    for name in ("_reuse", "metrics", "metrics_v2"):
        sys.modules.pop(name, None)

    t1_metrics = _load_module("t1_metrics", ROOT / "T1" / "metrics.py")
    reuse = types.ModuleType("_reuse")
    reuse.t1_metrics = t1_metrics
    sys.modules["_reuse"] = reuse   # register BEFORE loading t2_metrics, which itself needs _reuse.t1_metrics
    if task in ("T2", "T3"):
        reuse.t2_metrics = _load_module("t2_metrics", ROOT / "T2" / "metrics.py")

    metrics = t1_metrics if task == "T1" else _load_module("metrics", ROOT / task / "metrics.py")
    sys.modules["metrics"] = metrics
    metrics_v2 = _load_module("metrics_v2", ROOT / task / "metrics_v2.py")
    return metrics, metrics_v2


def _resolve_reference(args, ref_field: str, need_coords: bool):
    """--reference (T1/T2) / --wt (T3): defaults to --input itself when not given. Loaded FIRST (with
    genes=None) so it also defines the gene panel every other file is validated against."""
    ref_path = getattr(args, ref_field)
    defaulted = ref_path is None
    if defaulted:
        ref_path = args.input
        print(f"[veckit] --{ref_field.replace('_', '-')} not given -- defaulting to --input itself. "
             f"de_score/de_direction{'/severity_slope' if args.task == 'T3' else ''} will correctly read "
             f"as 'no predicted change', which is only meaningful if you're testing a no-change baseline "
             f"(copy_last/wt_identity). Pass --{ref_field.replace('_', '-')} explicitly for a real model.",
             file=sys.stderr)
    X, C, ct, A = _prediction(ref_path, None, need_coords=need_coords, allow_reorder=False)
    return X, C, ct, A, defaulted


def _subprocess_ctx():
    # `fork` lets the child inherit the already-loaded (sparse) matrices via copy-on-write, so no
    # re-read or pickling of large arrays happens. Falls back to the platform default elsewhere.
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def _terminate(p):
    if p.is_alive():
        p.terminate()
        p.join(timeout=10)
    if p.is_alive():
        p.kill()


def _run_compute(fn, *args):
    """Run `fn(*args)` in a child process and return its result.

    The parent stays responsive to SIGINT/SIGTERM while the child is inside a potentially
    uninterruptible native (sklearn/BLAS) call such as LogisticRegression.fit or PCA.fit: the parent
    merely polls a pipe on a 0.2s cycle, so a signal is delivered to the Python main thread promptly, the
    child is terminated/joined, and the KeyboardInterrupt / SystemExit is re-raised in the caller. This
    changes execution isolation only -- never the function's math.
    """
    ctx = _subprocess_ctx()
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    def _child():
        try:
            res = fn(*args)
            child_conn.send(("ok", res))
        except BaseException as exc:
            try:
                child_conn.send(("err", exc))
            except BaseException:
                pass
        finally:
            try:
                child_conn.close()
            except BaseException:
                pass

    p = ctx.Process(target=_child)
    p.start()
    status = payload = None
    try:
        while True:
            if parent_conn.poll(0.2):
                status, payload = parent_conn.recv()
                break
            if not p.is_alive():
                raise RuntimeError("compute worker exited unexpectedly")
    except BaseException:
        _terminate(p)
        raise
    else:
        _terminate(p)
        if status == "ok":
            return payload
        raise payload
    finally:
        try:
            parent_conn.close()
        except BaseException:
            pass

def score_t1(args):
    metrics, metrics_v2 = _load_task_metrics("T1")

    _log("loading reference...")
    ref_X, _, ref_ct, ref_A, ref_defaulted = _resolve_reference(args, "reference", need_coords=False)
    genes = [str(g) for g in ref_A.var_names]
    _log("loading target...")
    true_X, _, true_ct, _ = _prediction(args.target, genes, need_coords=False,
                                        allow_reorder=args.allow_reorder)
    _log("training frozen probe...")
    probe = _run_compute(metrics.train_frozen_probe, true_X, true_ct)
    _log("loading prediction...")
    pred_X, _, pred_ct, A = _prediction(args.input, genes, need_coords=False, allow_reorder=args.allow_reorder)
    _log("computing metrics...")
    scores = _run_compute(metrics_v2.score_task1_v2,
                          pred_X, pred_ct, true_X, true_ct, ref_X, probe, args.seed)
    _log("done")
    meta = {
        "task": "T1",
        "target_source": str(args.target),
        "reference_defaulted_to_input": ref_defaulted,
        "truth_cells": int(true_X.shape[0]),
        "prediction_cells": int(A.n_obs),
        "genes": len(genes),
    }
    return meta, scores


def score_t2(args):
    metrics, metrics_v2 = _load_task_metrics("T2")

    _log("loading reference...")
    ref_X, ref_C, ref_ct, ref_A, ref_defaulted = _resolve_reference(args, "reference", need_coords=True)
    genes = [str(g) for g in ref_A.var_names]
    _log("loading target...")
    true_X, true_C, true_ct, _ = _prediction(args.target, genes, need_coords=True,
                                             allow_reorder=args.allow_reorder)
    _log("training frozen probe...")
    probe = _run_compute(metrics.train_frozen_probe, true_X, true_ct)
    _log("loading prediction...")
    pred_X, pred_C, pred_ct, A = _prediction(args.input, genes, need_coords=True, allow_reorder=args.allow_reorder)
    _log("computing metrics...")
    scores = _run_compute(metrics_v2.score_task2_v2,
                          pred_X, pred_C, pred_ct, true_X, true_C, true_ct, ref_X, probe, args.seed)
    _log("done")
    meta = {
        "task": "T2",
        "setting": args.setting,
        "target_source": str(args.target),
        "reference_defaulted_to_input": ref_defaulted,
        "truth_cells": int(true_X.shape[0]),
        "prediction_cells": int(A.n_obs),
        "genes": len(genes),
    }
    return meta, scores


def score_t3(args):
    metrics, metrics_v2 = _load_task_metrics("T3")

    _log("loading wt...")
    wt_X, wt_C, wt_ct, wt_A, wt_defaulted = _resolve_reference(args, "wt", need_coords=True)
    genes = [str(g) for g in wt_A.var_names]
    _log("loading target...")
    true_X, true_C, true_ct, _ = _prediction(args.target, genes, need_coords=True,
                                             allow_reorder=args.allow_reorder)
    _log("training frozen probe...")
    probe = _run_compute(metrics.train_frozen_probe, true_X, true_ct)
    _log("loading prediction...")
    pred_X, pred_C, pred_ct, A = _prediction(args.input, genes, need_coords=True, allow_reorder=args.allow_reorder)
    _log("computing metrics...")
    scores = _run_compute(metrics_v2.score_task3_v2,
                          pred_X, pred_C, pred_ct, true_X, true_C, true_ct, wt_X, probe, args.seed)
    _log("done")
    meta = {
        "task": "T3",
        "target_source": str(args.target),
        "wt_defaulted_to_input": wt_defaulted,
        "truth_cells": int(true_X.shape[0]),
        "prediction_cells": int(A.n_obs),
        "genes": len(genes),
    }
    return meta, scores


def _on_sigterm(signum, frame):
    raise SystemExit(143)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    ap.add_argument("--input", required=True, type=Path, help="submitted .h5ad")
    ap.add_argument("--target", required=True, type=Path,
                    help="the pseudo target -- a local file YOU supply, standing in for the target you're "
                        "predicting. See the module docstring's examples.")
    ap.add_argument("--reference", type=Path,
                    help="T1/T2 only: your own local copy of the DE/shift reference stage (the preceding "
                        "stage the change is measured against). Defaults to --input if omitted.")
    ap.add_argument("--wt", type=Path,
                    help="T3 only: your own local copy of the matched wild type. Defaults to --input if "
                        "omitted.")
    ap.add_argument("--out", type=Path, help="write JSON result here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-reorder", action="store_true",
                    help="allow identical gene sets in a different order and reorder to the task panel")
    ap.add_argument("--setting", default="heart", choices=["heart", "embryo"],
                    help="T2 only: cosmetic, recorded in the output; doesn't change what's loaded")
    args = ap.parse_args()

    # Interruptibility: SIGINT (Ctrl+C) raises KeyboardInterrupt (default); SIGTERM is mapped to a
    # SystemExit(143) so the shell's conventional 128+15 code is preserved. Because the heavy native
    # (sklearn/BLAS) work runs in isolated worker subprocesses (see `_run_compute`), the Python main thread
    # stays responsive and these land promptly instead of waiting out a minutes-long native call.
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if args.task == "T1":
            meta, scores = score_t1(args)
        elif args.task == "T2":
            meta, scores = score_t2(args)
        else:
            meta, scores = score_t3(args)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code == 143:
            print("Terminated.", file=sys.stderr)
        return code
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    result = {"meta": meta, "metrics": scores}
    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    sys.exit(main())

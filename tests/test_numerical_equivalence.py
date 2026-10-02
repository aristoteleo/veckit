"""Regression test: the memory-safe refactor must NOT change any metric value.

Runs the ORIGINAL (pre-refactor) T1 scorer -- vendored in `legacy_reference.py` -- and the refactored
scorer on the SAME synthetic fixture, and asserts every reported metric is numerically equal. This is the
core guarantee of the PR: "lower memory, but keep the scorer numerically equivalent".

Tolerances: metrics with float-order changes (sparse-vs-dense pseudobulk accumulation, moment-form
variance, sparse PCA basis) allow a small relative tolerance; rounded outputs are compared at ~1 rounding
unit. Deterministic metrics (energy_distance, variogram, library_size_ratio) should match to ~1e-6.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_h5ad as _s                       # noqa: E402
from legacy_reference import (  # noqa: E402
    legacy_score_task1_v2,
    mmd_unbiased as legacy_mmd_unbiased,
    train_frozen_probe as legacy_train_probe,
)
from _fixture import make_t1_fixture          # noqa: E402
from common.core_metrics import _mean_gene_variance, mmd_unbiased  # noqa: E402

# rounding granularity of each reported metric (the dict values are already rounded)
_TOL = {
    "de_score": 2e-4, "de_direction": 2e-4,
    "energy_distance": 2e-5, "mmd_u": 2e-5, "variogram": 2e-6,
    "pb_rel_err": 2e-4, "library_size_ratio": 2e-3, "variance_ratio": 2e-3,
    "composition_JSD": 2e-4, "pseudobulk_pearson": 2e-4,
    "_de_raw": 2e-4, "_de_chance": 2e-4, "_de_chance_unif": 2e-4,
}


def _new_scorer(pred_X, pred_ct, true_X, true_ct, ref_X, seed=0):
    metrics, metrics_v2 = _s._load_task_metrics("T1")
    probe = metrics.train_frozen_probe(true_X, true_ct)
    return metrics_v2.score_task1_v2(pred_X, pred_ct, true_X, true_ct, ref_X, probe=probe, seed=seed)


def _legacy_scorer(pred_X, pred_ct, true_X, true_ct, ref_X, seed=0):
    probe = legacy_train_probe(true_X, true_ct)
    return legacy_score_task1_v2(pred_X, pred_ct, true_X, true_ct, ref_X, probe=probe, seed=seed)


def _assert_close(new, old, label):
    for k in sorted(set(new) | set(old)):
        a, b = new.get(k), old.get(k)
        if k in ("_n_up", "_n_dn"):
            assert a == b, f"[{label}] {k}: new={a} old={b}"
        elif a is None or b is None:
            assert a == b, f"[{label}] {k}: new={a!r} old={b!r}"
        elif isinstance(a, float) or isinstance(b, float):
            tol = _TOL.get(k, 1e-3)
            diff = abs(a - b)
            assert diff <= tol, f"[{label}] {k}: new={a} old={b} (diff={diff:.2e} tol={tol})"
        else:
            assert float(a) == float(b), f"[{label}] {k}: new={a} old={b}"


def _fixture_arrays():
    return make_t1_fixture()


def test_new_sparse_vs_legacy_dense():
    ref, true, pred, ct_r, ct_t, ct_p, _ = _fixture_arrays()
    new = _new_scorer(sparse.csr_matrix(pred), ct_p,
                      sparse.csr_matrix(true), ct_t,
                      sparse.csr_matrix(ref), seed=0)
    old = _legacy_scorer(pred, ct_p, true, ct_t, ref, seed=0)   # legacy densifies anyway
    _assert_close(new, old, "new-sparse vs legacy-dense")
    # sanity: de_score should be a real number, not None, on this fixture
    assert new["de_score"] is not None and new["_n_up"] + new["_n_dn"] > 0


def test_new_csc_vs_legacy():
    ref, true, pred, ct_r, ct_t, ct_p, _ = _fixture_arrays()
    new = _new_scorer(sparse.csc_matrix(pred), ct_p,
                      sparse.csc_matrix(true), ct_t,
                      sparse.csc_matrix(ref), seed=0)
    old = _legacy_scorer(pred, ct_p, true, ct_t, ref, seed=0)
    _assert_close(new, old, "new-csc vs legacy-dense")


def test_new_dense_vs_legacy_dense():
    # dense path must also be unchanged
    ref, true, pred, ct_r, ct_t, ct_p, _ = _fixture_arrays()
    new = _new_scorer(pred, ct_p, true, ct_t, ref, seed=0)
    old = _legacy_scorer(pred, ct_p, true, ct_t, ref, seed=0)
    _assert_close(new, old, "new-dense vs legacy-dense")


def test_sparse_vs_dense_self_consistent():
    # the refactored scorer should give the same result for sparse and dense inputs of identical values
    ref, true, pred, ct_r, ct_t, ct_p, _ = _fixture_arrays()
    s = _new_scorer(sparse.csr_matrix(pred), ct_p, sparse.csr_matrix(true), ct_t, sparse.csr_matrix(ref), seed=0)
    d = _new_scorer(pred, ct_p, true, ct_t, ref, seed=0)
    _assert_close(s, d, "new-sparse vs new-dense")


def test_multiple_seeds():
    ref, true, pred, ct_r, ct_t, ct_p, _ = _fixture_arrays()
    for seed in (0, 1, 2):
        new = _new_scorer(sparse.csr_matrix(pred), ct_p,
                          sparse.csr_matrix(true), ct_t,
                          sparse.csr_matrix(ref), seed=seed)
        old = _legacy_scorer(pred, ct_p, true, ct_t, ref, seed=seed)
        _assert_close(new, old, f"seed={seed}")


def test_mmd_matches_legacy_at_official_gene_scale():
    """Sparse input must retain the released dense-PCA metric at Task 1 width."""
    rng = np.random.default_rng(17)
    true = rng.normal(size=(75, 32_285)).astype(np.float32)
    pred = (true + rng.normal(scale=0.1, size=true.shape)).astype(np.float32)

    expected = legacy_mmd_unbiased(pred, true, n=75, seed=3)
    actual = mmd_unbiased(sparse.csr_matrix(pred), sparse.csr_matrix(true), n=75, seed=3)

    assert round(actual, 5) == round(expected, 5), (
        f"official-width MMD changed at reported precision: new={actual}, legacy={expected}"
    )


def test_variance_is_storage_format_independent():
    rng = np.random.default_rng(23)
    dense = rng.lognormal(size=(150, 500)).astype(np.float32)

    sparse_value = _mean_gene_variance(sparse.csr_matrix(dense))
    dense_value = _mean_gene_variance(dense)

    np.testing.assert_allclose(sparse_value, dense_value, rtol=1e-12, atol=1e-12)

"""Input validation tests: NaN/Inf detection, negative-value rejection, gene-order mismatch, and
sparse/dense/float32 handling through the real `_prediction` loader path."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
import anndata as ad

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_h5ad as _s                            # noqa: E402


def _write(tmp_path, name, X, ct=None, var_names=None, obsm=None):
    ct = ct if ct is not None else np.array(["t0"] * X.shape[0])
    vn = var_names if var_names is not None else [f"g{i}" for i in range(X.shape[1])]
    import pandas as pd
    a = ad.AnnData(X=X, obs={"celltype": ct}, var=pd.DataFrame(index=vn))
    if obsm is not None:
        a.obsm["spatial_3D"] = obsm
    path = tmp_path / name
    a.write(path)
    return path


def _genes(G=100):
    return [f"g{i}" for i in range(G)]


def test_as_float32_preserves_sparse():
    X = sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(3, 4))
    Y = _s._as_float32(X, "x")
    assert sparse.issparse(Y) and Y.format == "csr"
    assert Y.dtype == np.float32
    assert (Y.toarray() == X.toarray()).all()


def test_as_float32_raises_on_nan(tmp_path):
    X = sparse.csr_matrix(np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="NaN|finite"):
        _s._as_float32(X, "x")


def test_as_float32_raises_on_inf(tmp_path):
    X = sparse.csr_matrix(np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="NaN|finite"):
        _s._as_float32(X, "x")


def test_prediction_rejects_negative_sparse(tmp_path):
    G = _genes()
    X = sparse.csr_matrix(np.full((10, 100), 0.5))
    X[0, 0] = -0.1  # stored negative nonzero
    p = _write(tmp_path, "neg.h5ad", X, var_names=G)
    with pytest.raises(ValueError, match="negative"):
        _s._prediction(p, None, need_coords=False, allow_reorder=False)


def test_prediction_ok_nonnegative_sparse(tmp_path):
    G = _genes()
    X = sparse.csr_matrix(np.random.default_rng(0).lognormal(0, 0.5, size=(10, 100)).astype(np.float32))
    p = _write(tmp_path, "ok.h5ad", X, var_names=G)
    out, _, _, _ = _s._prediction(p, None, need_coords=False, allow_reorder=False)
    assert sparse.issparse(out) and out.format == "csr"


def test_gene_order_mismatch(tmp_path):
    G = _genes()
    X = sparse.csr_matrix(np.random.default_rng(1).lognormal(0, 0.5, size=(10, 100)).astype(np.float32))
    p = _write(tmp_path, "order.h5ad", X, var_names=list(reversed(G)))
    with pytest.raises(ValueError, match="var_names"):
        _s._prediction(p, G, need_coords=False, allow_reorder=False)


def test_gene_reorder_allowed(tmp_path):
    G = _genes()
    X = sparse.csr_matrix(np.random.default_rng(2).lognormal(0, 0.5, size=(10, 100)).astype(np.float32))
    p = _write(tmp_path, "reorder.h5ad", X, var_names=list(reversed(G)))
    out, _, _, _ = _s._prediction(p, G, need_coords=False, allow_reorder=True)
    assert out.shape == (10, 100)

"""Numerical compatibility with the original multi-bandwidth RBF MMD."""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from benchmarks.mmd_reference import mmd_reference
from common.core_metrics import mmd_unbiased


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fmt", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
@pytest.mark.parametrize("seed", [0, 29])
def test_matches_original_with_unequal_sizes(dtype, fmt, seed):
    rng = np.random.default_rng(912)
    a = rng.normal(size=(91, 37)).astype(dtype)
    b = rng.normal(size=(139, 37)).astype(dtype)
    a = fmt(a)
    b = fmt(b)
    args = dict(n=117, n_pc=16, seed=seed, scales=(0.25, 0.7, 1.0, 4.0))
    expected = mmd_reference(a, b, **args)
    actual = mmd_unbiased(a, b, **args)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("identical", [False, True])
def test_duplicate_cells_and_single_kernel(identical):
    rng = np.random.default_rng(23)
    b = np.repeat(rng.normal(size=(20, 5)), 7, axis=0)
    a = b.copy() if identical else b[:80] + 0.2
    args = dict(n=119, n_pc=5, scales=(1.0,))
    np.testing.assert_allclose(mmd_unbiased(a, b, **args), mmd_reference(a, b, **args), rtol=0, atol=1e-12)


@pytest.mark.parametrize("pair", [
    ("sample_8.5.h5ad", "sample_9.5.h5ad"),
    ("sample_heart_9.25.h5ad", "sample_heart_9.5.h5ad"),
    ("sample_wt.h5ad", "sample_mab21l2_ko.h5ad"),
])
def test_public_real_sample_pairs(pair):
    import anndata as ad
    directory = Path(__file__).resolve().parents[1] / "data"
    a, b = [ad.read_h5ad(directory / name) for name in pair]
    genes = a.var_names.intersection(b.var_names, sort=False)
    x, y = a[:, genes].X, b[:, genes].X
    np.testing.assert_allclose(mmd_unbiased(x, y), mmd_reference(x, y), rtol=0, atol=1e-7)


def test_public_t2_score_panel_matches_original(monkeypatch):
    import common.core_metrics as core
    from veckit import score
    directory = Path(__file__).resolve().parents[1] / "data"
    kwargs = dict(task="T2", input=directory / "sample_heart_9.25.h5ad",
                  target=directory / "sample_heart_9.5.h5ad",
                  reference=directory / "sample_heart_9.25.h5ad")
    monkeypatch.setattr(core, "mmd_unbiased", mmd_reference)
    expected = score(**kwargs)
    monkeypatch.setattr(core, "mmd_unbiased", mmd_unbiased)
    actual = score(**kwargs)
    assert actual["meta"] == expected["meta"]
    assert actual["metrics"].keys() == expected["metrics"].keys()
    for key, value in expected["metrics"].items():
        if value is None:
            assert actual["metrics"][key] is None
        else:
            np.testing.assert_allclose(actual["metrics"][key], value, rtol=0, atol=0, equal_nan=True)

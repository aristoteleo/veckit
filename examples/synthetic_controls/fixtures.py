"""Small deterministic synthetic fixtures for T1, T2, and T3 control demos."""
from __future__ import annotations

import anndata as ad
import numpy as np


FIXTURE_SEED = 1101


def _celltypes(n_cells: int) -> np.ndarray:
    return np.asarray([f"synthetic_type_{i % 3}" for i in range(n_cells)])


def _expression(
    rng: np.random.Generator,
    n_cells: int,
    n_genes: int,
    delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    celltypes = _celltypes(n_cells)
    X = rng.gamma(2.2, 0.42, size=(n_cells, n_genes))
    for type_index in range(3):
        start = 24 + type_index * 12
        X[celltypes == f"synthetic_type_{type_index}", start : start + 12] += 0.8
    if delta is not None:
        X += delta[None, :]
    return np.clip(X, 0.0, None).astype(np.float32), celltypes


def _coordinates(rng: np.random.Generator, celltypes: np.ndarray, scale: float) -> np.ndarray:
    centers = np.asarray(
        [[-1.0, 0.1, 0.2], [0.75, 0.8, -0.15], [0.15, -0.9, 0.5]],
        dtype=float,
    )
    type_index = np.asarray([int(value.rsplit("_", 1)[1]) for value in celltypes])
    noise = rng.normal(0.0, 0.17, size=(len(celltypes), 3))
    return ((centers[type_index] + noise) * scale).astype(np.float32)


def _anndata(X: np.ndarray, celltypes: np.ndarray, genes: list[str], coords=None) -> ad.AnnData:
    out = ad.AnnData(X=X)
    out.obs_names = [f"synthetic_cell_{i:04d}" for i in range(out.n_obs)]
    out.obs["celltype"] = celltypes
    out.obs["synthetic_batch"] = np.asarray([i % 2 for i in range(out.n_obs)])
    out.var_names = genes
    out.var["synthetic_feature"] = True
    out.varm["synthetic_loadings"] = np.arange(out.n_vars * 2, dtype=float).reshape(out.n_vars, 2)
    out.varp["synthetic_gene_graph"] = np.eye(out.n_vars, dtype=np.float32)
    out.uns["fixture"] = "generated; no Challenge data"
    out.layers["synthetic_unmodified"] = X.copy()
    if coords is not None:
        out.obsm["spatial_3D"] = np.asarray(coords, dtype=np.float32)
    return out


def make_synthetic_fixtures(seed: int = FIXTURE_SEED) -> dict[str, dict[str, ad.AnnData]]:
    """Return task fixtures containing only generated genes, cells, expression, and coordinates."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n_genes = 72
    genes = [f"synthetic_gene_{i:03d}" for i in range(n_genes)]
    delta = np.zeros(n_genes, dtype=float)
    delta[:12] = np.linspace(0.35, 0.75, 12)
    delta[12:24] = -np.linspace(0.25, 0.55, 12)

    t1_ref_X, t1_ref_ct = _expression(rng, 120, n_genes)
    t1_target_X, t1_target_ct = _expression(rng, 132, n_genes, delta)

    t2_ref_X, t2_ref_ct = _expression(rng, 120, n_genes)
    t2_target_X, t2_target_ct = _expression(rng, 135, n_genes, delta)
    t2_ref_C = _coordinates(rng, t2_ref_ct, 0.82)
    t2_target_C = _coordinates(rng, t2_target_ct, 1.0)

    t3_wt_X, t3_wt_ct = _expression(rng, 126, n_genes)
    t3_target_X, t3_target_ct = _expression(rng, 129, n_genes, delta * 0.75)
    t3_wt_C = _coordinates(rng, t3_wt_ct, 1.0)
    t3_target_C = _coordinates(rng, t3_target_ct, 1.05)

    return {
        "T1": {
            "reference": _anndata(t1_ref_X, t1_ref_ct, genes),
            "target": _anndata(t1_target_X, t1_target_ct, genes),
        },
        "T2": {
            "reference": _anndata(t2_ref_X, t2_ref_ct, genes, t2_ref_C),
            "target": _anndata(t2_target_X, t2_target_ct, genes, t2_target_C),
        },
        "T3": {
            "wt": _anndata(t3_wt_X, t3_wt_ct, genes, t3_wt_C),
            "target": _anndata(t3_target_X, t3_target_ct, genes, t3_target_C),
        },
    }

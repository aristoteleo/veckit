"""Executable implementations of the controls described on the VEC Baselines page.

These functions implement the public formulas. They are not copies of the private
``bake.py`` implementation and do not claim server equivalence.
"""
from __future__ import annotations

from numbers import Integral
from typing import Iterable

import anndata as ad
import numpy as np
from scipy import sparse


BASELINES_URL = "https://virtualembryo.ai/challenge/baselines"
AUDIT_KEY = "vec_synthetic_control"


def _as_dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _validate_expression(a: ad.AnnData, name: str) -> None:
    if not isinstance(a, ad.AnnData):
        raise TypeError(f"{name} must be an AnnData object")
    if a.X is None or a.X.ndim != 2:
        shape = None if a.X is None else a.X.shape
        raise ValueError(f"{name}.X must be a 2D cells x genes matrix, got {shape}")
    if a.n_obs == 0 or a.n_vars == 0:
        raise ValueError(f"{name}.X must contain at least one cell and one gene")
    values = a.X.data if sparse.issparse(a.X) else np.asarray(a.X)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}.X contains NaN or infinite values")
    if (values < 0).any():
        raise ValueError(f"{name}.X contains negative values")
    if not a.var_names.is_unique:
        raise ValueError(f"{name}.var_names must be unique")


def _spatial(a: ad.AnnData, name: str) -> np.ndarray:
    _validate_expression(a, name)
    if "spatial_3D" not in a.obsm:
        raise KeyError(f"{name} needs obsm['spatial_3D']")
    coords = np.asarray(a.obsm["spatial_3D"], dtype=float)
    if coords.shape != (a.n_obs, 3):
        raise ValueError(
            f"{name}.obsm['spatial_3D'] must have shape ({a.n_obs}, 3), got {coords.shape}"
        )
    if not np.isfinite(coords).all():
        raise ValueError(f"{name}.obsm['spatial_3D'] contains NaN or infinite values")
    return coords


def _rng(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return np.random.Generator(np.random.PCG64(int(seed)))


def _same_genes(left: ad.AnnData, right: ad.AnnData, left_name: str, right_name: str) -> None:
    _validate_expression(left, left_name)
    _validate_expression(right, right_name)
    if list(map(str, left.var_names)) != list(map(str, right.var_names)):
        raise ValueError(f"{left_name} and {right_name} must have identical var_names in the same order")


def _record(out: ad.AnnData, name: str, formula: str, **details) -> ad.AnnData:
    out.uns[AUDIT_KEY] = {
        "name": name,
        "formula": formula,
        "public_source": f"{BASELINES_URL}#ref-{name}",
        "implementation_scope": "public-description-only; not bake.py-equivalent",
        **details,
    }
    return out


def _scale(a: ad.AnnData, source_name: str, control_name: str, factor: float) -> ad.AnnData:
    _validate_expression(a, source_name)
    out = a.copy()
    out.X = out.X * factor
    _validate_expression(out, "output")
    return _record(out, control_name, f"{source_name}.X * {factor}", factor=float(factor))


def ctrl_scale_ref(reference: ad.AnnData) -> ad.AnnData:
    """Return the reference with expression multiplied by 2.0."""
    return _scale(reference, "reference", "ctrl_scale_ref", 2.0)


def ctrl_shrink_ref(reference: ad.AnnData) -> ad.AnnData:
    """Return the reference with expression multiplied by 0.99."""
    return _scale(reference, "reference", "ctrl_shrink_ref", 0.99)


def ctrl_one_cell(reference: ad.AnnData, n_cells: int) -> ad.AnnData:
    """Tile the reference pseudobulk mean to an explicitly requested cell count."""
    _validate_expression(reference, "reference")
    if isinstance(n_cells, bool) or not isinstance(n_cells, Integral) or n_cells <= 0:
        raise ValueError("n_cells must be a positive integer")

    mean_cell = np.asarray(_as_dense(reference.X).mean(axis=0), dtype=float).reshape(1, -1)
    X = np.repeat(mean_cell, int(n_cells), axis=0)
    obs_names = [f"ctrl_one_cell_{i:06d}" for i in range(int(n_cells))]
    out = ad.AnnData(
        X=X,
        var=reference.var.copy(deep=True),
    )
    out.obs_names = obs_names
    for key, value in reference.varm.items():
        out.varm[key] = value.copy()
    for key, value in reference.varp.items():
        out.varp[key] = value.copy()
    return _record(
        out,
        "ctrl_one_cell",
        "repeat(mean(reference.X, axis=0), n_cells, axis=0)",
        n_cells=int(n_cells),
    )


def _normalise_bounds(bounds: Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray(bounds, dtype=float)
    if values.shape != (3, 2):
        raise ValueError(f"bounds must have shape (3, 2), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("bounds contain NaN or infinite values")
    if (values[:, 0] > values[:, 1]).any():
        raise ValueError("each bounds row must satisfy min <= max")
    return values


def ctrl_random_cube(
    reference: ad.AnnData,
    *,
    seed: int,
    synthetic_target: ad.AnnData | None = None,
    bounds: Iterable[Iterable[float]] | None = None,
) -> ad.AnnData:
    """Keep reference expression and redraw coordinates uniformly inside target bounds."""
    _spatial(reference, "reference")
    if (synthetic_target is None) == (bounds is None):
        raise ValueError("provide exactly one of synthetic_target or bounds")
    if synthetic_target is not None:
        target_coords = _spatial(synthetic_target, "synthetic_target")
        resolved = np.column_stack([target_coords.min(axis=0), target_coords.max(axis=0)])
        bounds_source = "synthetic_target"
    else:
        resolved = _normalise_bounds(bounds)
        bounds_source = "explicit"

    generator = _rng(seed)
    coords = generator.uniform(resolved[:, 0], resolved[:, 1], size=(reference.n_obs, 3))
    out = reference.copy()
    out.obsm["spatial_3D"] = coords
    return _record(
        out,
        "ctrl_random_cube",
        "reference.X unchanged; coordinates ~ Uniform(target axis minima, maxima)",
        seed=int(seed),
        bounds=resolved.tolist(),
        bounds_source=bounds_source,
    )


def ctrl_squashed_ref(reference: ad.AnnData) -> ad.AnnData:
    """Stretch centered coordinates by (4.0, 1.0, 0.25), then recenter."""
    coords = _spatial(reference, "reference")
    center = coords.mean(axis=0)
    factors = np.array([4.0, 1.0, 0.25])
    out = reference.copy()
    out.obsm["spatial_3D"] = (coords - center) * factors + center
    _spatial(out, "output")
    return _record(
        out,
        "ctrl_squashed_ref",
        "(coordinates - center) * (4.0, 1.0, 0.25) + center",
        factors=factors.tolist(),
        center=center.tolist(),
    )


def ctrl_scale_wt(wt: ad.AnnData) -> ad.AnnData:
    """Return wild type with expression multiplied by 2.0."""
    _spatial(wt, "wt")
    return _scale(wt, "wt", "ctrl_scale_wt", 2.0)


def ctrl_shrink_wt(wt: ad.AnnData) -> ad.AnnData:
    """Return wild type with expression multiplied by 0.75."""
    _spatial(wt, "wt")
    return _scale(wt, "wt", "ctrl_shrink_wt", 0.75)


def ctrl_random_dir(wt: ad.AnnData, synthetic_target: ad.AnnData, *, seed: int) -> ad.AnnData:
    """Add a random gene-level shift with the target pseudobulk shift's pre-clip L2 norm."""
    _spatial(wt, "wt")
    _spatial(synthetic_target, "synthetic_target")
    _same_genes(wt, synthetic_target, "wt", "synthetic_target")
    generator = _rng(seed)

    wt_X = np.asarray(_as_dense(wt.X), dtype=float)
    target_X = np.asarray(_as_dense(synthetic_target.X), dtype=float)
    target_shift = target_X.mean(axis=0) - wt_X.mean(axis=0)
    target_norm = float(np.linalg.norm(target_shift))
    if target_norm <= 0.0:
        raise ValueError("synthetic WT-to-target pseudobulk shift must have non-zero L2 norm")

    random_shift = generator.normal(size=wt.n_vars)
    random_norm = float(np.linalg.norm(random_shift))
    if random_norm <= 0.0:
        raise RuntimeError("sampled a zero Gaussian vector")
    random_shift *= target_norm / random_norm

    pre_clip = wt_X + random_shift[None, :]
    output_X = np.clip(pre_clip, 0.0, None)
    realised_shift = output_X.mean(axis=0) - wt_X.mean(axis=0)

    out = wt.copy()
    out.X = output_X
    _spatial(out, "output")
    return _record(
        out,
        "ctrl_random_dir",
        "clip(wt.X + scaled Gaussian pseudobulk shift, 0, infinity)",
        seed=int(seed),
        target_shift_l2=target_norm,
        requested_random_shift_l2=float(np.linalg.norm(random_shift)),
        realised_post_clip_shift_l2=float(np.linalg.norm(realised_shift)),
        clipped_fraction=float(np.mean(pre_clip < 0.0)),
    )

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import anndata as ad
import numpy as np

from veckit import score as veckit_score


EXPECTED_PATH = Path(__file__).resolve().parent / "expected" / "t2_synthetic_v011.json"
FIXTURE_SEED = 1101
UPSTREAM_COMMIT = "46d41e63f42a9aab815db20b742feeccd249cb17"
BOARD_ANCHOR_METRICS = (
    "de_score",
    "de_direction",
    "mmd_u",
    "variogram",
    "d2_shape",
    "occupancy_dice",
    "scale_log_ratio",
    "neighborhood_mmd",
)


def _make_anndata(X, celltype, coords, genes):
    a = ad.AnnData(X=np.asarray(X, dtype=np.float32))
    a.obs_names = [f"cell_{i}" for i in range(a.n_obs)]
    a.obs["celltype"] = np.asarray(celltype, dtype=str)
    a.var_names = np.asarray(genes, dtype=str)
    a.obsm["spatial_3D"] = np.asarray(coords, dtype=np.float32)
    return a


def make_t2_fixture(seed: int = FIXTURE_SEED):
    """Synthetic-only T2 fixture with fake genes, fake cell types, and deterministic coordinates."""
    rng = np.random.default_rng(seed)
    n_cells = 180
    n_genes = 96
    n_types = 3

    genes = [f"g{i:03d}" for i in range(n_genes)]
    celltype = np.array([f"fake_type_{i % n_types}" for i in range(n_cells)])

    base = rng.gamma(shape=2.0, scale=0.55, size=(n_cells, n_genes)).astype(np.float32)
    cell_offsets = rng.normal(0.0, 0.08, size=(n_cells, 1)).astype(np.float32)
    reference_X = np.maximum(base + cell_offsets, 0.0).astype(np.float32)

    delta = np.zeros(n_genes, dtype=np.float32)
    delta[:10] = np.linspace(0.35, 0.7, 10, dtype=np.float32)
    delta[10:20] = -np.linspace(0.3, 0.6, 10, dtype=np.float32)
    type_effect = np.zeros((n_cells, n_genes), dtype=np.float32)
    type_effect[celltype == "fake_type_0", 20:36] = 0.55
    type_effect[celltype == "fake_type_1", 36:52] = 0.55
    type_effect[celltype == "fake_type_2", 52:68] = 0.55

    target_X = np.maximum(reference_X + delta + type_effect, 0.0).astype(np.float32)
    pred_noise = rng.normal(0.0, 0.035, size=(n_cells, n_genes)).astype(np.float32)
    prediction_X = np.maximum(target_X * 0.97 + 0.03 * reference_X + pred_noise, 0.0).astype(np.float32)

    centers = np.array(
        [[-1.0, 0.0, 0.2], [0.8, 0.7, -0.1], [0.2, -0.9, 0.5]],
        dtype=np.float32,
    )
    target_C = centers[np.arange(n_cells) % n_types] + rng.normal(0.0, 0.16, size=(n_cells, 3)).astype(np.float32)

    theta = 0.37
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    prediction_C = (target_C @ rot.T) * np.array([1.08, 0.94, 1.03], dtype=np.float32)
    prediction_C = prediction_C + rng.normal(0.0, 0.035, size=(n_cells, 3)).astype(np.float32)

    return {
        "reference": _make_anndata(reference_X, celltype, target_C, genes),
        "target": _make_anndata(target_X, celltype, target_C, genes),
        "prediction": _make_anndata(prediction_X, celltype, prediction_C, genes),
    }


def _write_fixture_files(fixture, tmpdir: Path):
    ref_path = tmpdir / "reference.h5ad"
    target_path = tmpdir / "target.h5ad"
    pred_path = tmpdir / "prediction.h5ad"
    fixture["reference"].write_h5ad(ref_path)
    fixture["target"].write_h5ad(target_path)
    fixture["prediction"].write_h5ad(pred_path)
    return ref_path, target_path, pred_path


def score_t2_public_result(fixture, *, score_seed: int = 0, setting: str = "heart"):
    with tempfile.TemporaryDirectory() as tmp:
        ref_path, target_path, pred_path = _write_fixture_files(fixture, Path(tmp))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="'n_jobs' has no effect.*", category=FutureWarning)
            result = veckit_score(
                task="T2",
                input=pred_path,
                target=target_path,
                reference=ref_path,
                seed=score_seed,
                setting=setting,
            )
    return result


def score_t2_public_api(fixture, *, score_seed: int = 0, setting: str = "heart"):
    return score_t2_public_result(fixture, score_seed=score_seed, setting=setting)["metrics"]


def _load_expected():
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


def _assert_metric_vector_close(actual, expected, tolerances):
    for key in BOARD_ANCHOR_METRICS:
        assert key in actual, key
        assert key in expected, key
        assert np.isfinite(actual[key]), key
        difference = abs(float(actual[key]) - float(expected[key]))
        tolerance = tolerances.get(key, 1e-5)
        assert difference <= tolerance, (
            f"{key}: actual={actual[key]!r}, expected={expected[key]!r}, "
            f"difference={difference}, tolerance={tolerance}"
        )


def test_t2_synthetic_conformance_v011():
    expected = _load_expected()
    assert expected["version"] == "veckit==0.1.1"
    assert expected["upstream_commit"] == UPSTREAM_COMMIT
    fixture = make_t2_fixture(seed=expected["fixture_seed"])
    actual = score_t2_public_api(fixture, score_seed=expected["score_seed"])
    _assert_metric_vector_close(actual, expected["metrics"], expected["tolerances"])


def test_t2_public_api_repeatable():
    fixture = make_t2_fixture(seed=FIXTURE_SEED)
    first = score_t2_public_api(fixture, score_seed=0)
    second = score_t2_public_api(fixture, score_seed=0)
    _assert_metric_vector_close(first, second, {k: 1e-9 for k in BOARD_ANCHOR_METRICS})


def test_t2_setting_is_metadata_only():
    fixture = make_t2_fixture(seed=FIXTURE_SEED)
    heart = score_t2_public_result(fixture, score_seed=0, setting="heart")
    embryo = score_t2_public_result(fixture, score_seed=0, setting="embryo")
    assert heart["meta"]["setting"] == "heart"
    assert embryo["meta"]["setting"] == "embryo"
    assert heart["metrics"] == embryo["metrics"]


def test_t2_board_anchor_partition():
    fixture = make_t2_fixture(seed=FIXTURE_SEED)
    metrics = score_t2_public_api(fixture, score_seed=0)
    assert set(BOARD_ANCHOR_METRICS) <= set(metrics)
    assert "count_log_ratio" not in BOARD_ANCHOR_METRICS
    assert "count_log_ratio" in metrics

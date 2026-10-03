from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from examples.synthetic_controls.controls import (
    AUDIT_KEY,
    ctrl_one_cell,
    ctrl_random_cube,
    ctrl_random_dir,
    ctrl_scale_ref,
    ctrl_scale_wt,
    ctrl_shrink_ref,
    ctrl_shrink_wt,
    ctrl_squashed_ref,
)
from examples.synthetic_controls.fixtures import make_synthetic_fixtures
from examples.synthetic_controls.run_controls import build_manifest


def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


@pytest.fixture(scope="module")
def fixtures():
    return make_synthetic_fixtures()


@pytest.mark.parametrize(
    ("task", "source", "function", "factor"),
    [
        ("T1", "reference", ctrl_scale_ref, 2.0),
        ("T1", "reference", ctrl_shrink_ref, 0.99),
        ("T3", "wt", ctrl_scale_wt, 2.0),
        ("T3", "wt", ctrl_shrink_wt, 0.75),
    ],
)
def test_scale_controls_change_only_x_and_audit(fixtures, task, source, function, factor):
    original = fixtures[task][source]
    before = _dense(original.X).copy()
    output = function(original)

    np.testing.assert_allclose(output.X, before * factor)
    np.testing.assert_array_equal(original.X, before)
    assert output.obs.equals(original.obs)
    assert output.var.equals(original.var)
    assert list(output.var_names) == list(original.var_names)
    assert output.uns["fixture"] == original.uns["fixture"]
    np.testing.assert_array_equal(output.layers["synthetic_unmodified"], original.X)
    assert output.uns[AUDIT_KEY]["factor"] == factor
    if task == "T3":
        np.testing.assert_array_equal(output.obsm["spatial_3D"], original.obsm["spatial_3D"])


def test_one_cell_is_the_pseudobulk_mean_with_explicit_count(fixtures):
    reference = fixtures["T1"]["reference"]
    n_cells = fixtures["T1"]["target"].n_obs
    output = ctrl_one_cell(reference, n_cells)
    expected = _dense(reference.X).mean(axis=0)

    assert output.shape == (n_cells, reference.n_vars)
    np.testing.assert_allclose(output.X, np.repeat(expected[None, :], n_cells, axis=0))
    np.testing.assert_allclose(_dense(output.X).mean(axis=0), expected)
    assert list(output.var_names) == list(reference.var_names)
    np.testing.assert_array_equal(output.varm["synthetic_loadings"], reference.varm["synthetic_loadings"])
    np.testing.assert_array_equal(output.varp["synthetic_gene_graph"], reference.varp["synthetic_gene_graph"])
    assert len(output.obs.columns) == 0
    assert len(output.obsm) == 0
    assert [key for key in output.layers.keys() if key is not None] == []
    assert output.raw is None
    assert output.obs_names[0] == "ctrl_one_cell_000000"


def test_random_cube_is_seeded_and_bounded(fixtures):
    reference = fixtures["T2"]["reference"]
    target = fixtures["T2"]["target"]
    original_coords = reference.obsm["spatial_3D"].copy()
    bounds = np.column_stack(
        [target.obsm["spatial_3D"].min(axis=0), target.obsm["spatial_3D"].max(axis=0)]
    )
    first = ctrl_random_cube(reference, synthetic_target=target, seed=17)
    second = ctrl_random_cube(reference, bounds=bounds, seed=17)
    third = ctrl_random_cube(reference, bounds=bounds, seed=18)

    np.testing.assert_array_equal(first.X, reference.X)
    np.testing.assert_array_equal(first.obsm["spatial_3D"], second.obsm["spatial_3D"])
    assert not np.array_equal(first.obsm["spatial_3D"], third.obsm["spatial_3D"])
    assert np.all(first.obsm["spatial_3D"] >= bounds[:, 0])
    assert np.all(first.obsm["spatial_3D"] <= bounds[:, 1])
    np.testing.assert_array_equal(reference.obsm["spatial_3D"], original_coords)


def test_squashed_ref_uses_public_centered_formula(fixtures):
    reference = fixtures["T2"]["reference"]
    coords = np.asarray(reference.obsm["spatial_3D"], dtype=float)
    center = coords.mean(axis=0)
    expected = (coords - center) * np.array([4.0, 1.0, 0.25]) + center
    output = ctrl_squashed_ref(reference)

    np.testing.assert_allclose(output.obsm["spatial_3D"], expected)
    np.testing.assert_allclose(output.obsm["spatial_3D"].mean(axis=0), center)
    np.testing.assert_array_equal(output.X, reference.X)
    np.testing.assert_array_equal(reference.obsm["spatial_3D"], coords)


def test_random_dir_matches_preclip_norm_then_clips(fixtures):
    wt = fixtures["T3"]["wt"]
    target = fixtures["T3"]["target"]
    output = ctrl_random_dir(wt, target, seed=23)

    wt_X = _dense(wt.X).astype(float)
    original_wt_X = wt_X.copy()
    target_shift = _dense(target.X).astype(float).mean(axis=0) - wt_X.mean(axis=0)
    generator = np.random.Generator(np.random.PCG64(23))
    random_shift = generator.normal(size=wt.n_vars)
    random_shift *= np.linalg.norm(target_shift) / np.linalg.norm(random_shift)
    expected = np.clip(wt_X + random_shift[None, :], 0.0, None)

    np.testing.assert_allclose(output.X, expected)
    np.testing.assert_allclose(np.linalg.norm(random_shift), np.linalg.norm(target_shift))
    assert np.all(output.X >= 0)
    np.testing.assert_array_equal(output.obsm["spatial_3D"], wt.obsm["spatial_3D"])
    audit = output.uns[AUDIT_KEY]
    assert audit["requested_random_shift_l2"] == pytest.approx(audit["target_shift_l2"])
    assert audit["realised_post_clip_shift_l2"] <= audit["requested_random_shift_l2"]
    np.testing.assert_array_equal(wt.X, original_wt_X)


def test_validation_errors_are_specific(fixtures):
    t1 = fixtures["T1"]
    t2 = fixtures["T2"]
    t3 = fixtures["T3"]

    with pytest.raises(ValueError, match="n_cells"):
        ctrl_one_cell(t1["reference"], 0)
    with pytest.raises(ValueError, match="exactly one"):
        ctrl_random_cube(t2["reference"], seed=0)
    with pytest.raises(ValueError, match="bounds"):
        ctrl_random_cube(t2["reference"], bounds=[[0, 1], [0, 1]], seed=0)
    with pytest.raises(ValueError, match="seed"):
        ctrl_random_cube(t2["reference"], synthetic_target=t2["target"], seed=-1)

    zero_target = t3["wt"].copy()
    with pytest.raises(ValueError, match="non-zero"):
        ctrl_random_dir(t3["wt"], zero_target, seed=0)

    empty = t1["reference"][:0].copy()
    with pytest.raises(ValueError, match="at least one cell"):
        ctrl_scale_ref(empty)


def test_scalar_and_mean_controls_accept_sparse_expression(fixtures):
    sparse_reference = fixtures["T1"]["reference"].copy()
    sparse_reference.X = sparse.csr_matrix(sparse_reference.X)
    scaled = ctrl_scale_ref(sparse_reference)
    one_cell = ctrl_one_cell(sparse_reference, 4)

    assert sparse.issparse(scaled.X)
    np.testing.assert_allclose(scaled.X.toarray(), sparse_reference.X.toarray() * 2.0)
    sparse_mean = sparse_reference.X.toarray().mean(axis=0, keepdims=True)
    np.testing.assert_allclose(one_cell.X, np.repeat(sparse_mean, 4, axis=0))


def test_all_controls_run_through_public_scorer():
    manifest = build_manifest(task="all", fixture_seed=1101, control_seed=1101, score_seed=0)
    assert manifest["synthetic_only"] is True
    assert len(manifest["controls"]) == 9
    assert set(CONTROL["name"] for CONTROL in manifest["controls"]) == {
        "ctrl_scale_ref",
        "ctrl_shrink_ref",
        "ctrl_one_cell",
        "ctrl_random_cube",
        "ctrl_squashed_ref",
        "ctrl_scale_wt",
        "ctrl_shrink_wt",
        "ctrl_random_dir",
    }
    for control in manifest["controls"]:
        assert control["formula_check"] == "PASS"
        assert control["metrics"]

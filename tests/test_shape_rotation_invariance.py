"""Shape metrics must not depend on the frame an embryo was imaged in (issue #7).

A proper rigid rotation of the same point cloud must score as identical on `sliced_wasserstein` and
`occupancy_dice`, while a mirror image must still not be credited as identical (the metrics minimise over
proper sign flips only, by design). Uses the bundled `data/sample_heart_9.5.h5ad`.

    python -m pytest tests/test_shape_rotation_invariance.py -q
"""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.shape_metrics import _canonicalise, occupancy_dice, sliced_wasserstein  # noqa: E402


@pytest.fixture(scope="module")
def cloud():
    a = ad.read_h5ad(ROOT / "data" / "sample_heart_9.5.h5ad")
    return np.asarray(a.obsm["spatial_3D"], float)


def _rot_z(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        out.append(q)
    return out


ROTATIONS = [_rot_z(d) for d in range(0, 360, 5)] + _random_rotations(50)


def test_canonical_frame_is_right_handed(cloud):
    for R in ROTATIONS:
        C = cloud @ R.T
        X = C - C.mean(0)
        Z, rms = _canonicalise(C)
        basis = np.linalg.lstsq(X, Z * rms, rcond=None)[0]   # X @ basis = Z * rms
        assert np.linalg.det(basis) > 0


def test_fewer_than_three_points_does_not_raise_in_canonicalise():
    for n in (1, 2):
        _canonicalise(np.random.default_rng(0).normal(size=(n, 3)))


def test_proper_rotations_score_as_identical(cloud):
    bad = []
    for i, R in enumerate(ROTATIONS):
        rotated = cloud @ R.T
        sw, _ = sliced_wasserstein(rotated, cloud)
        dice, _ = occupancy_dice(rotated, cloud)
        if sw > 1e-8 or dice < 1.0 - 1e-12:
            bad.append((i, sw, dice))
    assert not bad, f"{len(bad)} of {len(ROTATIONS)} proper rotations were penalised, first: {bad[:3]}"


def test_mirror_image_is_still_not_credited(cloud):
    mirrored = cloud * np.array([1.0, 1.0, -1.0])
    sw, _ = sliced_wasserstein(mirrored, cloud)
    dice, _ = occupancy_dice(mirrored, cloud)
    assert sw > 1e-3 and dice < 0.99

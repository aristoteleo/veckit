"""Virtual Embryo Challenge — 3D shape and growth metrics.

Nothing in the suite scored tissue SHAPE. `fgw_spatial` and `geometry_discrimination` were both removed for
good reasons (an oracle of 0.26 against a floor of 0.30 in one case, four candidates and an L1 bias in the
other), and what remained -- `volume_log_ratio` and `density_log_ratio` -- turned out to measure less than it
looked: both were computed on each cloud's **own axis-aligned bounding box**, so neither is rotation-invariant,
a cube of uniform random points scored 0.055 / 0.422 against the real model's 0.053 / 0.420, and a
volume-preserving anisotropic squash of the *true* cloud by (4x, 1x, 0.25x) scored volume_log_ratio 0.00000.
`neighborhood_mmd` cannot cover the gap either: a 3x anisotropic stretch costs it 0.00016, 0.1% of its range.

The three shape terms here were selected against VEC's constraints -- no cell correspondence, each stage
imaged in its own arbitrary rigid frame, genuine growth between stages, differing cell counts -- and
critically against the requirement that **the oracle must actually reach the ideal value**, which is what
disqualified FGW.

  * `d2_distance`      alignment-free, so zero registration risk. Measured monotone in developmental distance
                       (target 0.0022 < previous stage 0.0089 < two stages back 0.0334), robust to a 16x cell
                       count difference (0.0036), and one of only two candidates tested that ranks a real
                       prediction ABOVE a featureless blob (blob 0.0179, ellipsoid 0.0264).
  * `sliced_wasserstein` best floor-to-signal ratio of anything tested (target 0.0061 vs previous stage
                       0.0559, ~9x), and the ratio improves with cell count rather than degrading.
  * `occupancy_dice`   most stable separation measured (0.939 vs 0.724 at the 10% release, 0.974 vs 0.753 at
                       full density), and it is the rotation-invariant replacement for the envelope role
                       `volume_log_ratio` was playing.

LATERALITY IS NOT SCORED, AND THAT MUST BE STATED RATHER THAN DISCOVERED
-----------------------------------------------------------------------
A **mirrored** embryo scores `d2_distance` 0.002197 -- bit-identical to the split-half ceiling. The same holds
by construction for Gromov-Wasserstein, 3D Zernike moments and the Laplace-Beltrami spectrum, all of which are
reflection-invariant; Chamfer distance and F-score at a distance threshold actually rank the mirror ABOVE every
genuine baseline. `sliced_wasserstein` and `occupancy_dice` minimise over proper rotations only (det = +1), so
they are not reflection-invariant by construction. `_canonicalise` forces a right-handed frame, so a proper
rotation of a cloud is scored as identical and a mirror image is not (tests/test_shape_rotation_invariance.py).
That is a check on the frame, not a calibrated laterality test, so do not read them as one.

Since dextral looping is the flagship phenotype of these conditional knockouts, that is a real gap. The
honest place for it is a calibrated topological diagnostic (persistent homology with a density-aware
filtration for H1 handles and H2 cavities, plus medial-axis torsion, which is the direct readout of looping in
the developmental literature). It is not implemented here: on 2,500 cells an alpha complex yields ~6,000 H1
classes, i.e. mostly sampling noise, and it needs a scale-normalised filtration and a summary statistic chosen
in advance before it can rank anything.
"""
from __future__ import annotations
import numpy as np


def _canonicalise(C):
    """Centre, rotate onto principal axes, scale by RMS radius. Returns (Z, rms_radius).

    This is what makes the metrics below comparable across embryos imaged in different frames: translation
    goes with the centroid, rotation with the PCA basis, and scale with the RMS radius (a rotation-invariant
    size, unlike a bounding-box extent). PCA leaves each axis' SIGN undetermined, which is why the callers
    that use the rotated coordinates minimise over the sign combinations with determinant +1. That search only
    works if every frame has the same handedness, so the basis is made right-handed (det +1) here.
    """
    C = np.asarray(C, float)
    mu = C.mean(0); X = C - mu
    rms = float(np.sqrt((X ** 2).sum(1).mean()))
    if rms < 1e-12: return X, rms
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    if Vt.shape[0] == 3 and np.linalg.det(Vt) < 0:   # fewer than 3 points gives a non-square Vt
        Vt[-1] *= -1   # keep the frame right-handed, see issue #7
    return (X @ Vt.T) / rms, rms


_PROPER_FLIPS = [f for f in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]   # det = +1


def _match_n(A, B, seed=0):
    """Subsample both clouds to a common cell count.

    Shape must be scored independently of how many cells a submission contains, otherwise count becomes a
    lever on a shape metric: a geometrically *perfect* prediction at half the truth's count used to score
    worse on `density_log_ratio` than a model that got the geometry wrong. Count is scored on its own by
    `count_log_ratio`.
    """
    rng = np.random.default_rng(seed)
    n = min(A.shape[0], B.shape[0])
    return (A[rng.choice(A.shape[0], n, replace=False)],
            B[rng.choice(B.shape[0], n, replace=False)], n)


# ---------------------------------------------------------------- shape
def d2_distance(pred_C, true_C, n_pairs=200000, seed=0):
    """**D2 shape distribution distance** — alignment-free, lower is better, 0 = same shape.

    Sample random point pairs from each cloud, take their distances normalised by the cloud's RMS radius, and
    compare the two distance distributions with the 1-D Wasserstein distance. Because it only ever uses
    *pairwise distances within* a cloud, it needs no rotation, no translation, no correspondence and no shared
    frame -- the registration problem simply does not arise. Normalising by RMS radius makes it scale-free, so
    it measures form and leaves size to `scale_log_ratio`.

    Not reflection-invariant only in the trivial sense that it is *blind* to reflection: a mirrored embryo
    scores identically to a perfect one. See the module docstring.
    """
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(seed)

    def dists(C):
        C = np.asarray(C, float); X = C - C.mean(0)
        rms = float(np.sqrt((X ** 2).sum(1).mean()))
        if rms < 1e-12: return np.zeros(1)
        i = rng.integers(0, X.shape[0], n_pairs); j = rng.integers(0, X.shape[0], n_pairs)
        keep = i != j
        return np.linalg.norm(X[i[keep]] - X[j[keep]], axis=1) / rms

    return float(wasserstein_distance(dists(pred_C), dists(true_C)))


def sliced_wasserstein(pred_C, true_C, n_proj=200, seed=0):
    """**Sliced Wasserstein distance on canonicalised coordinates** — lower is better, 0 = same shape.

    Both clouds are centred, PCA-aligned and RMS-scaled, then compared by averaging the 1-D Wasserstein
    distance over random directions. PCA fixes the axes but not their signs, so the score is **minimised over
    the four sign combinations with determinant +1** (proper rotations only, so a mirror image is not silently
    credited). The spread across those four is returned too: a large spread means the canonicalisation was
    ambiguous (a near-spherical or near-degenerate cloud) and the value should be distrusted.

    O(n_proj * n log n), so it needs no subsampling even at full release.
    """
    A, B, n = _match_n(np.asarray(pred_C, float), np.asarray(true_C, float), seed)
    Za, _ = _canonicalise(A); Zb, _ = _canonicalise(B)
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n_proj, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    Pb = np.sort(Zb @ D.T, axis=0)
    vals = []
    for f in _PROPER_FLIPS:
        Pa = np.sort((Za * np.asarray(f, float)) @ D.T, axis=0)
        vals.append(float(np.abs(Pa - Pb).mean()))
    return float(min(vals)), float(max(vals) - min(vals))


def occupancy_dice(pred_C, true_C, n_grid=16, extent=3.0, seed=0):
    """**Occupancy Dice on a shared canonical grid** — higher is better, 1 = same occupied region.

    Canonicalise both clouds (so they share a frame up to axis signs), voxelise each on the *same* fixed grid
    spanning [-extent, extent] RMS radii, and take the Dice coefficient of the occupied-voxel sets, maximised
    over proper sign flips. Unlike the `volume_log_ratio` it replaces, the grid is shared and defined in a
    rotation-invariant frame rather than per-cloud axis-aligned, so an anisotropic squash of the truth no
    longer scores perfectly.

    Also returns the voxel edge in units of the truth's median nearest-neighbour spacing. Read the Dice value
    only when that ratio is comfortably above ~10: below it the grid is resolving individual cells rather than
    tissue, and the metric becomes a sampling-density statistic.
    """
    from sklearn.neighbors import NearestNeighbors
    A, B, n = _match_n(np.asarray(pred_C, float), np.asarray(true_C, float), seed)
    Za, _ = _canonicalise(A); Zb, rms_b = _canonicalise(B)

    def occ(Z):
        v = np.floor((np.clip(Z, -extent, extent - 1e-9) + extent) / (2 * extent) * n_grid).astype(int)
        return set(map(tuple, v))

    ob = occ(Zb)
    best = 0.0
    for f in _PROPER_FLIPS:
        oa = occ(Za * np.asarray(f, float))
        d = 2 * len(oa & ob) / max(len(oa) + len(ob), 1)
        best = max(best, d)
    # resolution check, in units of the truth's median NN spacing (canonical units)
    nn = NearestNeighbors(n_neighbors=2).fit(Zb).kneighbors(Zb)[0][:, 1]
    edge = (2 * extent) / n_grid
    return float(best), float(edge / max(float(np.median(nn)), 1e-12))


# ---------------------------------------------------------------- growth, as two separate questions
def scale_log_ratio(pred_C, true_C):
    """Signed log ratio of RMS radius — **rotation-invariant** tissue size. 0 = the right size.

    Replaces `volume_log_ratio`'s question with an estimator that does not depend on the coordinate frame.
    The RMS radius is invariant to rotation and translation by construction, whereas the occupied volume of an
    axis-aligned bounding box is not. Signed, so overshoot and undershoot are distinguishable.
    """
    def rms(C):
        C = np.asarray(C, float); X = C - C.mean(0)
        return float(np.sqrt((X ** 2).sum(1).mean()))
    a, b = rms(pred_C), rms(true_C)
    return float(np.log(a / b)) if a > 0 and b > 0 else float("nan")


def count_log_ratio(n_pred, n_true):
    """Signed log ratio of cell number — proliferation as its own axis. 0 = the right number of cells.

    Count used to enter only entangled inside `density_log_ratio = |log(n_p/n_t) - log(V_p/V_t)|`, which meant
    a model that got volume right and count wrong was penalised twice while one that got both wrong in
    compensating directions scored 0, and in-place duplication of cells moved the metric by exactly log(k).
    Proliferation and tissue expansion are distinct biological predictions and are now scored separately.
    """
    return float(np.log(max(n_pred, 1) / max(n_true, 1)))

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
they are not reflection-invariant by construction -- but their sensitivity to handedness has not been
calibrated here, so do not read them as laterality tests either.

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
    that use the rotated coordinates minimise over the sign combinations with determinant +1.
    """
    C = np.asarray(C, float)
    mu = C.mean(0); X = C - mu
    rms = float(np.sqrt((X ** 2).sum(1).mean()))
    if rms < 1e-12: return X, rms
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
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


def scale_ratio(pred_C, true_C):
    """**Unsigned, bounded** form of `scale_log_ratio`: min(r, 1/r) in (0, 1]. 1 = the right size.

    Same question and same estimator (RMS radius, rotation- and translation-invariant); only the reporting
    scale differs, and that scale is what decides whether the metric can be searched rather than modelled.

    `scale_log_ratio` is SIGNED, which is exactly right for a diagnostic — a reader wants to know whether a
    prediction is too big or too small — and exactly wrong for a ranked score on a public leaderboard. Size
    is a single scalar degree of freedom, and rescaling a point cloud about its centroid changes nothing
    else that is scored: `d2_shape` and `occupancy_dice` both canonicalise by RMS radius first, so they are
    invariant to it. Measured on the embryo interpolation board, rescaling the FLOOR's coordinates takes its
    size score from 50 to 100 in five submissions of bisection search, with `occupancy_dice` fixed at 0.7047
    and `d2_shape` at 0.0531 to four decimals throughout. The sign is what makes five submissions enough:
    each score tells the searcher which way to move next.

    Folding the ratio removes that gradient. Over- and under-shoot by the same factor now score the same, so
    a submission's score no longer says which side of the target it is on, and a searcher has to explore
    rather than bisect. It does not make size unlearnable — a model that predicts growth correctly still
    scores 1.0 — and it does not make search impossible, only uninformed.

        r = 1     -> 1.00      (exactly the right size)
        r = 1.25  -> 0.80      |  r = 0.8  -> 0.80
        r = 2     -> 0.50      |  r = 0.5  -> 0.50

    `scale_log_ratio` remains available and is reported alongside as the signed diagnostic.
    """
    def rms(C):
        C = np.asarray(C, float); X = C - C.mean(0)
        return float(np.sqrt((X ** 2).sum(1).mean()))
    a, b = rms(pred_C), rms(true_C)
    if not (a > 0 and b > 0):
        return float("nan")
    r = a / b
    return float(min(r, 1.0 / r))


def median_nn_distance(C, seed=0, n=4000):
    """Median nearest-neighbour distance — the CELL-level length scale of a point cloud."""
    from sklearn.neighbors import NearestNeighbors
    C = np.asarray(C, float)
    if len(C) < 2:
        return float("nan")
    if len(C) > n:
        C = C[np.random.default_rng(seed).choice(len(C), n, replace=False)]
    d, _ = NearestNeighbors(n_neighbors=2).fit(C).kneighbors(C)
    return float(np.median(d[:, 1]))


def size_fidelity(pred_C, true_C, seed=0):
    """Did the tissue reach the right SIZE — at both the tissue scale and the cell scale?

    Mean of two folded ratios, each in (0, 1], 1 = exactly right:

        tissue scale : min(r, 1/r)   with r = RMS radius ratio            (how big the organ is)
        cell scale   : min(q, 1/q)   with q = median NN distance ratio    (how far apart cells sit)

    WHY TWO SCALES AND NOT ONE. Size measured only as an RMS radius is a single scalar that a submission
    can set freely, and setting it costs nothing elsewhere: `d2_shape` and `occupancy_dice` canonicalise
    by RMS radius before comparing, so they are invariant to a rescale. Rescaling the FLOOR's coordinates
    about their centroid -- a prediction with no modelling in it whatsoever -- took the old size score
    from 50 to 100 while leaving occupancy_dice at 0.6748 and d2_shape at 0.0308, unchanged to four
    decimals. Signed reporting made it worse: each leaderboard score revealed which way to move, so five
    submissions of bisection search were enough to land it.

    Cells, however, do not change size when an embryo grows; the tissue gets bigger by acquiring more of
    them. That is visible in the released data, and it is what makes the second scale an anchor rather
    than a redundant copy of the first:

        heart   E8.25   E8.5    E8.75   E9.5    E10.5   E12.5
        RMS     355.6   254.6   216.3   335.2   439.9   511.5     <- swings 2.4x, and not monotonically
        NN dist  17.10   17.76   16.27   17.17   16.83   17.06     <- constant to +/-4%

    A real prediction that grows the tissue correctly leaves cell spacing alone and scores well on both
    terms. A uniform rescale moves them TOGETHER, so buying the first term costs the second: on the heart
    interpolation board the same rescale attack now scores 53.6 instead of 100.0, and on embryo 84.1
    instead of 100.0, while the attainable ceiling still scores 100.0 and the floor still 50.0.

    (The heart NN row also explains the RMS row: a length scale that is constant across stages while the
    organ radius swings non-monotonically points at the imaged field of view differing between stages,
    not at the heart shrinking. See `common/scale_comparability.py`.)

    ASSUMPTION. This reads a nearest-neighbour distance, so it assumes a submission returns coordinates
    for INDIVIDUAL CELLS. A method that emits a density field or an aggregated representation has no cell
    spacing to measure, and this metric is not meaningful for it. `scale_log_ratio` (signed) and
    `scale_ratio` (unsigned) remain available and are reported alongside, so a size result can always be
    decomposed into which of the two scales went wrong.

    Weighting is 50/50 rather than tuned: the two terms answer equally necessary halves of "is this the
    right size", and no board was used to choose a split between them.
    """
    def rms(C):
        C = np.asarray(C, float); X = C - C.mean(0)
        return float(np.sqrt((X ** 2).sum(1).mean()))

    def fold(x):
        return float(min(x, 1.0 / x)) if x > 0 and np.isfinite(x) else 0.0

    a, b = rms(pred_C), rms(true_C)
    tissue = fold(a / b) if (a > 0 and b > 0) else 0.0
    p, q = median_nn_distance(pred_C, seed=seed), median_nn_distance(true_C, seed=seed)
    cell = fold(p / q) if (np.isfinite(p) and np.isfinite(q) and q > 0) else 0.0
    return float(0.5 * (tissue + cell))


def count_log_ratio(n_pred, n_true):
    """Signed log ratio of cell number — proliferation as its own axis. 0 = the right number of cells.

    Count used to enter only entangled inside `density_log_ratio = |log(n_p/n_t) - log(V_p/V_t)|`, which meant
    a model that got volume right and count wrong was penalised twice while one that got both wrong in
    compensating directions scored 0, and in-place duplication of cells moved the metric by exactly log(k).
    Proliferation and tissue expansion are distinct biological predictions and are now scored separately.
    """
    return float(np.log(max(n_pred, 1) / max(n_true, 1)))

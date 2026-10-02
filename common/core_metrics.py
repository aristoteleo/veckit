"""Virtual Embryo Challenge — shared core metrics.

Read `../METRICS_V2.md` for the audit history. The short version: the first metric suite could not tell the
baselines apart, its replacement could be gamed, and this is the third pass. Everything here is stated as a
measurement, and the attacks that motivated each design choice are named so they can be re-run.

WHAT AN ATTACK ON THIS SUITE LOOKED LIKE, AND WHY THE CODE BELOW IS SHAPED THIS WAY
----------------------------------------------------------------------------------
Submitting the *input* stage's expression matrix multiplied by a constant -- zero target information, no
parameters, identical for every stage and every knockout -- scored `de_score` 0.61 on Task 1 against 0.17 for
the best real baseline and 1.0 for the oracle. It worked because:

  * the effect-size floor made the DE set a pure |mean shift| threshold (the Wilcoxon + BH step was verified
    to remove exactly zero genes: 107/107 on Task 1, 78/78 on Task 3), so the score was a global magnitude
    knob -- multiplying a real prediction's delta by 8 moved it 0.062 -> 0.467 with the gene ranking and the
    direction unchanged;
  * scaling the reference makes |lfc| proportional to the reference's own mean expression, and differentially
    expressed genes *are* the highly expressed ones, so the attack selects the right genes for the wrong
    reason;
  * the chance correction assumed a uniformly random gene subset (chance = n_pred/G), which no real
    submission is. The operative null was ~0.70, not 0.

The three changes that close it, which must stay together:
  (a) **rank-based selection** -- the prediction contributes exactly the K genes the truth declares, ranked by
      its own logFC, so any positive rescaling of a prediction is a no-op;
  (b) **sign-split matching** -- up-genes are matched against up-genes and down against down, so negating a
      prediction no longer scores (it previously scored *higher*: 0.63 flipped vs 0.42 correct);
  (c) **an empirical expression-magnitude null** -- chance is measured as the score achieved by ranking genes
      by the reference's mean expression, i.e. by the attack itself, rather than assumed.

`de_direction` had its own version of the same hole (0.99 * reference scored 0.31), closed by partialling out
the reference expression level.

REMOVED, and why -- so nobody reintroduces them
-----------------------------------------------
* **The discrimination family** (`stage_discrimination`, `perturbation_discrimination`,
  `geometry_discrimination`). Ranked candidates by L1 distance and reported the true target's position. With
  only 4 candidate stages (3 knockouts) the score can take just {0, 1/3, 2/3, 1} and a random guess averages
  0.5, so every baseline tied at 0.667. Ranking by L1 on *unnormalized* change vectors is also biased: a zero
  prediction sits at distance ||delta_t||_1 from every candidate, so a do-nothing model always identifies the
  smallest-effect candidate -- which handed Task 3's floor a perfect 1.000 as an artefact. The *question* is
  legitimate and currently unasked; a repair needs a large candidate pool (all released stages x region x
  probe-defined type, not four) and a scale-free distance (cosine, never L1 on raw magnitudes).
* **`progress` / `effect_progress`** -- L1 path length from reference to target. Removed at the task owner's
  call.
* **`fgw_spatial`** -- subsampled 600 cells per side independently, so two disjoint subsamples of the *same*
  embryo already scored 0.26 against a floor of 0.30. Its whole dynamic range was sampling noise.
* **`pseudobulk_pearson_celltype`** -- rewarded the floor, and consumed submitted labels.
* **`deg_jaccard`** -- its top-K set was a union over both tails, so `top(-d) == top(d)`: a perfect
  anti-phenotype scored identically to the correct one.
* **`deg_rank_spearman`** -- a duplicate of `de_direction` on the same vectors.
* **`delta_pearson`, `delta_mae`, `delta_magnitude_ratio`** -- `delta_mae`'s wild-type reference cancels
  algebraically (mean|dp-dt| == mean|pb(pred) - pb(true_ko)|); `delta_magnitude_ratio` depends only on the two
  norms, so every rotation of a correct-norm delta scores identically; `delta_pearson` is the Pearson twin of
  `de_direction`. Severity is now scored by `severity_slope` below, which cannot be won by a random direction.
* **`density_log_ratio`** -- exactly, not approximately, redundant: log(rho_p/rho_t) = log(n_p/n_t) -
  log(V_p/V_t). Count is now its own term.
* **`volume_log_ratio` as implemented** -- the question survives, the estimator does not: it voxelised each
  cloud on its *own* axis-aligned bounding box, so it was not rotation-invariant, and a cube of uniform random
  points scored the same as the real model. Replaced by the rotation-invariant scale/shape group in
  `shape_metrics.py`.
* **`spatial_celltype_coloc` on Task 2** -- consumed *submitted* cell-type labels, which the protocol says
  participants never submit.
"""
from __future__ import annotations
import sys
import time as _time
import numpy as np
from scipy import sparse


_LOG_T0 = _time.time()
def _log(msg):
    """Stage-progress log to stderr (never stdout, so the JSON result stays clean)."""
    print(f"[veckit] {msg} ({_time.time() - _LOG_T0:.1f}s)", file=sys.stderr, flush=True)



def to_dense(X): return X.toarray() if sparse.issparse(X) else np.asarray(X)
def _dense(X):
    """Densify a (small, already-sampled/chunked) block, PRESERVING the source dtype."""
    if sparse.issparse(X):
        arr = X.toarray()                      # scipy returns float64
        if arr.dtype != X.dtype:
            arr = arr.astype(X.dtype, copy=False)   # keep float32 arithmetic identical to the old path
        return np.ascontiguousarray(arr)       # force C order so BLAS matmul matches the legacy layout
    return np.asarray(X)

def pseudobulk(X):
    """Per-gene mean expression. Sparse-aware: mean(axis=0) on a CSR/CSC matrix never densifies the full
    matrix -- only an O(nnz) reduction over stored nonzeros. Same quantity as `to_dense(X).mean(0)` up to
    float32 accumulation order."""
    return np.asarray(X.mean(axis=0)).ravel()


def _pairwise_euclidean(A, B):
    """Pairwise Euclidean distance via the BLAS matmul identity ||a-b||^2 = ||a||^2+||b||^2-2a.b, instead
    of `scipy.spatial.distance.cdist`. Mathematically identical (up to float precision -- measured max
    relative error ~1e-6 against cdist on this data's scale); the only difference is that a matmul routes
    through BLAS's optimized SGEMM instead of cdist's serial C loop, which matters a lot at T1's full
    32,285-gene width: ~95x faster on a 1500x1500x32285 benchmark (30s -> 0.3s per call), because
    `energy_distance` below makes three such calls."""
    aa = np.einsum("ij,ij->i", A, A)
    bb = np.einsum("ij,ij->i", B, B)
    d2 = aa[:, None] + bb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0, out=d2)   # fp cancellation can push near-zero distances slightly negative
    return np.sqrt(d2, out=d2)


def _ranks(v):
    from scipy.stats import rankdata
    return rankdata(np.asarray(v, float))


# ---------------------------------------------------------------- differential expression
MIN_LFC = 0.25    # minimum |mean log-expression shift| for a gene to count as differentially expressed
SEVERITY_LOG_WORST = float(np.log(1e-3))   # severity_slope of a no-/wrong-response prediction: complete
#                                            undershoot (beta -> 0), reported as a finite floor (~ -6.9) so
#                                            it can anchor skill instead of collapsing the group to NaN.

def de_genes(X_cond, X_ref, alpha=0.05, min_lfc=MIN_LFC, min_cells=10, chunk=256):
    """Significant DE genes, split by direction, by Wilcoxon rank-sum + Benjamini-Hochberg FDR at `alpha`
    and an effect of at least `min_lfc` in mean log-expression.

    Returns (up_indices, down_indices, lfc). Note the FDR step is, on this data, inert once the effect-size
    floor is applied -- the returned set is identical to {|lfc| >= min_lfc}, verified 107/107 on Task 1 and
    78/78 on Task 3. It is kept because it will start to bind on smaller cell counts or a wider panel, but
    the honest description of this selection is "genes whose mean moved by at least `min_lfc`", and the
    scoring below is built so that this does not matter: only the *ranking* of the truth's genes is used.

    Column-chunked so the full dense `A`, `B` (n_cells x n_genes) are never materialised at once: each gene
    (column) is independent under Mann-Whitney, so per-chunk columns are bit-identical to the full call.
    """
    from scipy.stats import mannwhitneyu
    lfc = pseudobulk(X_cond) - pseudobulk(X_ref)
    if X_cond.shape[0] < min_cells or X_ref.shape[0] < min_cells:
        return np.array([], int), np.array([], int), lfc
    c_cond = X_cond.tocsc() if sparse.issparse(X_cond) else X_cond
    c_ref = X_ref.tocsc() if sparse.issparse(X_ref) else X_ref
    G = X_cond.shape[1]
    p = np.empty(G, float); stat = np.empty(G, float)
    with np.errstate(all="ignore"):
        for g0 in range(0, G, chunk):
            g1 = min(g0 + chunk, G)
            Ac = _dense(c_cond[:, g0:g1])
            Bc = _dense(c_ref[:, g0:g1])
            s, pv = mannwhitneyu(Ac, Bc, axis=0, alternative="two-sided")   # tie-corrected by default
            stat[g0:g1] = s; p[g0:g1] = pv
    p = np.nan_to_num(p, nan=1.0)
    order = np.argsort(p); m = len(p)
    below = p[order] <= alpha * (np.arange(1, m + 1) / m)                 # Benjamini-Hochberg
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    sig = order[:k]
    if min_lfc > 0: sig = sig[np.abs(lfc[sig]) >= min_lfc]
    return sig[lfc[sig] > 0], sig[lfc[sig] < 0], lfc


def _signed_overlap(lfc_pred, up_true, dn_true):
    """Take the prediction's top-|up_true| genes by logFC and bottom-|dn_true|, and intersect like with like.

    Rank-based, so multiplying the prediction's delta by any positive constant leaves this unchanged; and
    sign-split, so negating the prediction moves its up-genes into the down slot where they no longer match.
    """
    n_up, n_dn = len(up_true), len(dn_true)
    if n_up + n_dn == 0: return float("nan"), 0
    order = np.argsort(-np.asarray(lfc_pred, float))                      # most positive first
    pred_up = order[:n_up]
    pred_dn = order[len(order) - n_dn:] if n_dn else np.array([], int)
    hit = len(np.intersect1d(pred_up, up_true)) + len(np.intersect1d(pred_dn, dn_true))
    return hit / (n_up + n_dn), n_up + n_dn


def de_score(pred_X, true_X, ref_X, alpha=0.05, min_lfc=MIN_LFC):
    """**Differential-expression score** — did the prediction name the genes that actually change, and move
    them the right way, beyond what a biology-free submission achieves?

    The truth defines how many genes go up and how many go down (`n_up`, `n_dn`) against the reference
    condition. The prediction is then asked for exactly that many, ranked by its own logFC, and up is matched
    against up and down against down. The score is reported relative to a **measured** null: the same
    construction applied to a ranking by the reference's own mean expression, which is precisely what the
    "resubmit the input stage, scaled" attack produces.

        de_score = (overlap - chance) / (1 - chance)

    0 = no better than ranking genes by how highly expressed they already are; negative = worse; 1 = the
    truth's own DE sets. A prediction with no change scores 0 (guarded relatively, because `pseudobulk`
    averages float32 and two bit-identical arrays in different buffers differ by ~1e-5).

    Returns a dict: score, raw, chance (expression-magnitude null), chance_uniform (the hypergeometric null,
    reported for continuity with earlier versions), n_up, n_dn, n_true.
    """
    up_t, dn_t, _ = de_genes(true_X, ref_X, alpha, min_lfc)
    G = int(true_X.shape[1]); nan = float("nan")
    n_true = len(up_t) + len(dn_t)
    out = {"score": nan, "raw": nan, "chance": nan, "chance_uniform": nan,
           "n_up": len(up_t), "n_dn": len(dn_t), "n_true": n_true}
    if n_true == 0: return out

    dp = pseudobulk(pred_X) - pseudobulk(ref_X)
    dt = pseudobulk(true_X) - pseudobulk(ref_X)
    # The expression-magnitude null, in BOTH orientations. Scaling the reference UP gives a delta proportional
    # to +pb(ref); scaling it DOWN gives -pb(ref), which puts the most highly expressed genes in the
    # down-regulated slot. Both are biology-free, and only covering the first left the "0.99 x reference"
    # submission scoring 0.29 — above a real model — because many genuinely down-regulated genes are highly
    # expressed. The null is the BETTER of the two, i.e. the best a submission can do knowing only how highly
    # expressed each gene already is.
    pb_ref = pseudobulk(ref_X)
    c_up, _ = _signed_overlap(pb_ref, up_t, dn_t)
    c_dn, _ = _signed_overlap(-pb_ref, up_t, dn_t)
    chance = max(c_up, c_dn)
    out["chance"] = float(chance)
    out["chance_uniform"] = float(n_true / G)
    if np.std(dt) < 1e-12: return out
    if np.std(dp) < 1e-2 * np.std(dt):                                    # predicted no change
        out.update(score=0.0, raw=0.0); return out

    raw, _ = _signed_overlap(dp, up_t, dn_t)
    out["raw"] = float(raw)
    out["score"] = float((raw - chance) / (1 - chance)) if chance < 1 else nan
    return out


def de_direction(pred_X, true_X, ref_X):
    """Signed agreement of the change, **with the reference expression level partialled out**.

    Rank-partial correlation of the predicted and observed logFC vectors, controlling for the reference's own
    pseudobulk. Without the control, submitting 0.99 x the reference scored 0.31 (Task 1) and 0.54 (Task 3),
    beating every real model: its delta is exactly proportional to -pb(ref), and the true developmental change
    is itself correlated with expression level, so the two agree for a reason that has nothing to do with the
    prediction. Partialling that shared component out leaves only agreement the reference cannot explain.
    Returns 0.0 for a prediction with no change.
    """
    dp = pseudobulk(pred_X) - pseudobulk(ref_X)
    dt = pseudobulk(true_X) - pseudobulk(ref_X)
    if np.std(dt) < 1e-12 or np.std(dp) < 1e-2 * np.std(dt): return 0.0
    rp, rt, rr = _ranks(dp), _ranks(dt), _ranks(pseudobulk(ref_X))
    C = np.corrcoef(np.vstack([rp, rt, rr]))
    if not np.all(np.isfinite(C)): return 0.0
    # If the prediction's change is (anti)collinear with the reference's expression level, it carries no
    # information the reference does not already have, and the partial correlation is 0/0. Return 0 rather
    # than the amplified float32 noise that 0/0 produces: a submission of 0.99 x the reference has
    # |corr(rank dp, rank pb_ref)| = 1 up to rounding, and the unguarded ratio handed it 0.047.
    if 1.0 - abs(C[0, 2]) < 1e-6: return 0.0
    num = C[0, 1] - C[0, 2] * C[1, 2]
    den = np.sqrt(max((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2), 1e-12))
    r = num / den
    return float(np.clip(r, -1.0, 1.0)) if np.isfinite(r) else 0.0


def severity_slope(pred_X, true_X, ref_X, alpha=0.05, min_lfc=MIN_LFC):
    """**Did the prediction get the SIZE of the response right, given that it got the direction right?**

    Signed log of the OLS slope of predicted logFC on observed logFC, over the truth's DE genes only:

        beta = argmin_b || dp[DE] - b * dt[DE] ||^2 ,     reported as log(beta)

    0 = exactly the right severity, positive = overshoot, negative = undershoot. This replaces
    `delta_magnitude_ratio`, which depended only on the two vectors' norms and therefore gave every rotation
    of a correct-norm delta an identical (perfect) score -- a random direction with the right magnitude won
    it outright. A slope cannot be won that way: a random direction regresses to beta ~ 0, i.e. log beta ->
    -inf.

    **A prediction with no response of its own** (`wt_identity`, whose prediction IS the reference) has the
    WORST severity, not an undefined one: it undershoots the true response completely. It is reported as a
    finite maximal-undershoot floor `SEVERITY_LOG_WORST = log(1e-3)` rather than NaN, so that (a) the metric
    can be skill-normalised against it as a genuine floor (returning NaN made the floor's value NaN, which
    `skill()` then propagated to zero *every* model, collapsing T3's scale to 37.5/75 and leaving this whole
    group inert), and (b) a real submission that gets the response SIZE right scores above it. A truly
    undefined board (the truth itself has no response / no DE genes to regress on) is still NaN -- that is a
    property of the board, not a prediction failure.
    """
    up_t, dn_t, _ = de_genes(true_X, ref_X, alpha, min_lfc)
    idx = np.concatenate([up_t, dn_t]).astype(int)
    nan = float("nan")
    if len(idx) < 5: return nan, nan                       # board undefined: too few truth DE genes
    dp_all = pseudobulk(pred_X) - pseudobulk(ref_X)
    dt_all = pseudobulk(true_X) - pseudobulk(ref_X)
    if np.std(dt_all) < 1e-12: return nan, nan             # board undefined: the truth has no response
    # No predicted response of its own (wt_identity etc.): complete undershoot = worst severity, finite so it
    # can anchor the floor. Checks the SAME no-effect condition de_score/de_direction use, on the full delta
    # before restricting to DE genes -- the only version immune to the float32 rounding noise (~1e-5 between
    # two bit-identical arrays) that let the r2 gate below hand `wt_identity` a spurious slope of -14.9.
    if np.std(dp_all) < 1e-2 * np.std(dt_all): return SEVERITY_LOG_WORST, 0.0
    dp, dt = dp_all[idx], dt_all[idx]
    ss = float(dt @ dt)
    if ss < 1e-12: return nan, nan                         # board undefined: truth DE delta degenerate
    beta = float(dt @ dp) / ss
    resid = dp - beta * dt
    r2 = 1.0 - float(resid @ resid) / max(float(((dp - dp.mean()) ** 2).sum()), 1e-12)
    # Inverted (beta<=0) or unrelated (r2<=0.01): the prediction has a response but it does not track the
    # truth's -- worst severity (a finite floor), not credited and not dropped.
    if beta <= 0 or r2 <= 0.01: return SEVERITY_LOG_WORST, round(r2, 4)
    return float(np.log(beta)), round(r2, 4)


# ---------------------------------------------------------------- distribution
def mmd_unbiased(pred_X, true_X, n=2000, n_pc=30, seed=0, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """**Unbiased** multi-kernel RBF-MMD^2 in a PCA space fitted on the observed cells only.

    Two fixes over the previous version. The kernel diagonals are dropped (n(n-1) and m(m-1) denominators),
    because the biased estimator's diagonal term put the "oracle" at 0.0004 rather than 0 -- ~2/n, a pure
    finite-sample artefact that the skill rescaling then treated as the ceiling. And the bandwidth is a
    mixture over `scales` around the truth's median heuristic rather than one value, so a submission cannot
    sit in the blind spot of a single kernel width. The PCA basis and the bandwidth are both properties of
    the target alone, identical for every submission.

    The PCA is fit on a dense copy of the FULL `true_X`, preserving the released metric's solver selection
    and numerical semantics. This is one unavoidable transient allocation: sklearn's `auto` selects ARPACK
    for sparse input but may select the full or randomized solver for dense input, and those bases diverge at
    the official 32,285-gene scale. Only the sampled rows are transformed afterward, directly from sparse.

    Still note the residual limitation, which no bandwidth mixture fixes: PCA to `n_pc` components is a
    linear projection, so any structure in the discarded subspace is invisible. Use `energy_distance` below
    for a full-space check.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import rbf_kernel
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=min(n_pc, true_X.shape[1]), random_state=0).fit(to_dense(true_X))
    pa = rng.choice(pred_X.shape[0], min(n, pred_X.shape[0]), replace=False)
    pb = rng.choice(true_X.shape[0], min(n, true_X.shape[0]), replace=False)
    A = pca.transform(pred_X[pa])
    B = pca.transform(true_X[pb])
    d2 = np.sum((B[:, None] - B[None, :]) ** 2, -1)
    gamma0 = 1.0 / (np.median(d2[d2 > 0]) + 1e-9)
    na, nb = A.shape[0], B.shape[0]
    tot = 0.0
    for s in scales:
        g = gamma0 * s
        Kaa, Kbb, Kab = rbf_kernel(A, A, g), rbf_kernel(B, B, g), rbf_kernel(A, B, g)
        np.fill_diagonal(Kaa, 0.0); np.fill_diagonal(Kbb, 0.0)
        tot += (Kaa.sum() / (na * (na - 1)) + Kbb.sum() / (nb * (nb - 1)) - 2 * Kab.mean())
    return float(tot / len(scales))


def energy_distance(pred_X, true_X, n=1500, seed=0):
    """Energy distance (E-statistic) in the FULL gene space — no projection, so no blind subspace.

        E = 2 E||a-b|| - E||a-a'|| - E||b-b'||    (>= 0, zero iff the distributions coincide)

    This exists because every other distributional term in the suite lives in a 30-PC space, which is a
    linear map: perturbations inside its 470-dimensional null space leave the score bit-identical. Uses
    n(n-1) denominators for the within-sample terms, so the split-half ceiling is not inflated by a diagonal.

    Sampling semantics preserved: the same rng(seed) row choice as before, but rows are selected from the
    sparse matrix FIRST and only the ~n sampled rows are densified (instead of densifying the whole matrix).
    """
    rng = np.random.default_rng(seed)
    pa = rng.choice(pred_X.shape[0], min(n, pred_X.shape[0]), replace=False)
    pb = rng.choice(true_X.shape[0], min(n, true_X.shape[0]), replace=False)
    A, B = _dense(pred_X[pa]), _dense(true_X[pb])
    na, nb = A.shape[0], B.shape[0]
    Daa, Dbb = _pairwise_euclidean(A, A), _pairwise_euclidean(B, B)
    return float(2 * _pairwise_euclidean(A, B).mean()
                 - Daa.sum() / (na * (na - 1)) - Dbb.sum() / (nb * (nb - 1)))


def variogram_score(pred_X, true_X, n_pairs=20000, n_cells=1500, p=0.5, seed=0, chunk=2000):
    """**Is the gene-gene covariance structure right?** — the only term in the suite that constrains the joint.

    For randomly sampled gene pairs (i, j), compare E|x_i - x_j|^p between prediction and truth:

        VS = mean over pairs of ( E_pred|x_i - x_j|^p - E_true|x_i - x_j|^p )^2

    Every other expression metric here is a per-gene marginal, a 500-dim mean, or a kernel distance in a
    30-PC space, so a submission with perfect marginals and completely scrambled co-expression passes all of
    them. The variogram score is the component of the proper-scoring-rule literature designed for exactly
    that failure (the energy score is known to be weak on correlation structure). Lower is better; 0 for a
    perfect match.

    Semantics preserved: the same rng(seed) cell and gene-pair draws, but rows are sampled from the sparse
    matrix first and only the ~n_cells sampled rows are densified. The ~n_pairs x n_cells difference array
    is built in `chunk`-pair blocks so no large temporary is held at once; per-pair means are accumulated
    verbatim and the (va-vb)^2 reduction is unchanged.
    """
    rng = np.random.default_rng(seed)
    pa = rng.choice(pred_X.shape[0], min(n_cells, pred_X.shape[0]), replace=False)
    pb = rng.choice(true_X.shape[0], min(n_cells, true_X.shape[0]), replace=False)
    A, B = _dense(pred_X[pa]), _dense(true_X[pb])
    G = A.shape[1]
    i = rng.integers(0, G, n_pairs); j = rng.integers(0, G, n_pairs)
    keep = i != j; i, j = i[keep], j[keep]
    va = np.empty(len(i)); vb = np.empty(len(i))
    for c0 in range(0, len(i), chunk):
        c1 = min(c0 + chunk, len(i))
        ic, jc = i[c0:c1], j[c0:c1]
        va[c0:c1] = (np.abs(A[:, ic] - A[:, jc]) ** p).mean(0)
        vb[c0:c1] = (np.abs(B[:, ic] - B[:, jc]) ** p).mean(0)
    return float(((va - vb) ** 2).mean())


def pb_rel_err(pred_X, true_X):
    """Relative L2 error of the pseudobulk profile — the absolute-scale term the suite lacked.

    `pseudobulk_pearson` is affine-invariant: 5*X+3 and 0.01*X both score 0.8815629 against a floor of
    0.8815629, identical to seven significant figures, so nothing in the panel noticed a submission on the
    wrong scale. Lower is better, oracle exactly 0, no affine invariance. Intended as a **gate** (flag a
    submission outside a stated band) as much as a score.
    """
    a, b = pseudobulk(pred_X), pseudobulk(true_X)
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def library_size_ratio(pred_X, true_X, batch=256):
    """Median implied library size of the submission over the target's — the normalisation-convention gate.

    The released data is log(1 + 1e4 * c / sum c), so `expm1(X).sum(1)` should be ~1e4. Nothing checked this:
    `data.py::_needs_norm` only renormalises *integer count* matrices, so a submission log-normalised to
    target_sum = 1e6 instead of 1e4 -- a uniform +log(100) offset on every entry -- was accepted as-is and
    silently scored on a different scale. An unchanged do-nothing submission could then score anywhere from
    0.000 to 0.358 on the old `de_score` purely by normalisation choice.

    Computed in `batch`-cell blocks so `clip`/`expm1` never materialise a full n_cells x G temporary; only
    one scalar (library size) per cell is retained. Formula unchanged.
    """
    la = _median_library_size(pred_X, batch)
    lb = _median_library_size(true_X, batch)
    return float(la / lb) if lb > 0 else float("nan")


def _median_library_size(X, batch):
    n = X.shape[0]; sizes = np.empty(n)
    for b0 in range(0, n, batch):
        b1 = min(b0 + batch, n)
        chunk = _dense(X[b0:b1])
        sizes[b0:b1] = np.expm1(np.clip(chunk, 0, 50)).sum(1)
    return float(np.median(sizes))


def variance_ratio(pred_X, true_X):
    """Mean per-gene within-population variance of the prediction over the truth's.

    A guard, not a score. A one-row submission broadcast to many cells scored `pseudobulk_pearson` exactly
    equal to the copy floor and `delta_pearson` exactly 1.0; anything far below ~0.1 here means the
    submission is not a population and its mean-based metrics should not be read as if it were.

    Sparse uses Var = E[X^2] - E[X]^2 via two O(nnz) reductions (no densification). Both paths accumulate
    in float64 so identical values score consistently regardless of storage format.
    """
    va = _mean_gene_variance(pred_X); vb = _mean_gene_variance(true_X)
    return float(va / vb) if vb > 0 else float("nan")


def _mean_gene_variance(X):
    if sparse.issparse(X):
        Xf = X.tocsr().astype(np.float64, copy=False)
        mean = np.asarray(Xf.mean(axis=0)).ravel()
        mean_sq = np.asarray(Xf.multiply(Xf).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
        return float(var.mean())
    return float(np.asarray(X, dtype=np.float64).var(axis=0).mean())


# ---------------------------------------------------------------- aggregation
def skill(value, floor, ceiling, lower_is_better=False):
    """Rescale a metric to (0, 1] — bounded **by construction**, not by clipping: 0.5 = the do-nothing floor,
    1.0 = the attainable ceiling, and arbitrarily bad values compress smoothly toward 0 without ever
    reaching or crossing it.

    An earlier version of this used a linear ratio, (value-floor)/(ceiling-floor) -- unbounded below by
    design, so that a model far worse than the floor could read as skill -500% or worse. That's not a bug in
    the linear formula (dividing by a small floor-ceiling gap is exactly what makes a near-saturated metric
    like `mmd_u` swing so hard), but a weighted average of unbounded numbers is itself unbounded, which
    fights a leaderboard's basic need for a readable top-line range. This version fixes that at the root by
    using a **hyperbolic** map instead of a linear one:

        d(value)   = |value - ceiling|, signed so it's 0 at the ceiling and grows as value gets worse
        d(floor)   = |ceiling - floor|          (a fixed, metric-specific "one floor-unit" of distance)
        skill      = d(floor) / (d(floor) + d(value))     -- clipped above at 1.0 only

    At value=ceiling, d(value)=0, skill=1.0. At value=floor, d(value)=d(floor), skill=0.5. As value gets
    arbitrarily worse, d(value)->infinity and skill->0 asymptotically -- it never hits 0, so two very
    different degrees of failure (skill 0.08 vs 0.001) are still distinguishable, but neither one can pull a
    weighted average negative or unbounded. A model that beats the ceiling itself (a real, if rare,
    occurrence since the ceiling is a half-vs-half estimate, not a true optimum) is clipped to 1.0 exactly
    like the old version clipped above.

    `ceiling` must be the **attainable** ceiling (see `split_half_ceiling`), not the truth resubmitted.
    """
    if not all(np.isfinite([value, floor, ceiling])): return float("nan")
    d_floor = abs(ceiling - floor)
    if d_floor < 1e-12: return float("nan")
    d = (value - ceiling) if lower_is_better else (ceiling - value)
    denom = d_floor + d
    if denom <= 1e-9: return 1.0
    return float(min(d_floor / denom, 1.0))


def summarise_seeds(rows):
    """Collapse a list of per-seed metric dicts into "mean +/- sd", suppressing digits finer than the interval.

    Every number in this kit used to be a single run quoted to four decimals, on metrics where resampling
    moved the value by more than the gap between two models: one measured case had a model's score span
    0.038-0.075 across three seeds on a metric whose entire occupied range was 0-0.17. A value without an
    interval is not a measurement, and a ranking inside an interval is not a result.
    """
    keys = [k for k in rows[0] if not k.startswith("_")]
    out = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        if not vals: out[k] = None; continue
        m, sd = float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        # report to one digit finer than the interval, never more
        dig = 4 if sd == 0 else int(np.clip(-np.floor(np.log10(max(sd, 1e-9))) + 1, 1, 6))
        out[k] = round(m, dig); out["_sd_" + k] = round(sd, dig + 1)
    return out


def split_half(n, seed=0):
    """Two disjoint halves of an index range — the basis of the attainable ceiling.

    Every `oracle` row in this kit used to be the target resubmitted verbatim, which no honest generative
    model can reach: it makes sampling-based metrics read 0 when a perfect sampler at the same cell count
    scores an order of magnitude worse (a correct sampler gave MMD 0.004 at 200 cells against an "oracle" of
    0.0004), and it makes the skill denominator fiction. Scoring one half of the target against the other
    gives a ceiling a real model could actually attain, at the sample size the evaluation actually uses.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[: n // 2], idx[n // 2:]

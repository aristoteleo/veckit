"""Virtual Embryo Challenge — Task 2 evaluation metrics (spatial-temporal).

Task 2 = Task-1 gene-expression + cell-type-composition metrics  PLUS  spatial-organization accuracy:
  * local spatial-expression fidelity, cell-type co-occurrence and per-gene spatial autocorrelation.
The expression / composition metrics are reused from Task 1.
(An FGW term was removed — see the note above `neighborhood_mmd`.)
"""
from __future__ import annotations
import numpy as np
from _reuse import t1_metrics
pseudobulk_pearson, pseudobulk_pearson_celltype = t1_metrics.pseudobulk_pearson, t1_metrics.pseudobulk_pearson_celltype
composition_jsd, rbf_mmd, deg_recovery = t1_metrics.composition_jsd, t1_metrics.rbf_mmd, t1_metrics.deg_recovery
train_frozen_probe, to_dense = t1_metrics.train_frozen_probe, t1_metrics.to_dense


# fgw_distance (fused Gromov-Wasserstein) was REMOVED. It subsampled 600 cells per side independently, so
# two disjoint subsamples of the SAME embryo already scored 0.26 against a copy_last floor of 0.30 — a usable
# range of 0.04, narrower than the single-seed sampling noise, with all four baselines inside it. Reinstating
# it needs enough cells (or averaging over enough subsamples) that the oracle approaches zero, plus a
# reported interval. See ../METRICS_V2.md.


def _knn_neighborhood_pb(X, coords, k=15):
    """Per-cell spatial-neighborhood pseudobulk: mean expression of each cell's k nearest spatial neighbors."""
    from sklearn.neighbors import NearestNeighbors
    _, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
    return to_dense(X)[idx].mean(1)

def neighborhood_mmd(pred_X, pred_coords, true_X, true_coords, k=15, n=2000):
    """Coupling-INDEPENDENT local spatial-expression fidelity: MMD between the distributions of
    spatial-neighborhood pseudobulks (lower = better).

    Uses `common.core_metrics.mmd_unbiased`, NOT the v1 `rbf_mmd` this function used to call. The v1 version
    fitted its PCA basis on predicted and observed cells POOLED and took the kernel bandwidth from the
    PREDICTION, so a submission could bend the space it was scored in and inflating its variance drove the
    score toward 0. The v2 estimator fits both on the observed cells only and drops the kernel diagonals.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.core_metrics import mmd_unbiased
    return mmd_unbiased(_knn_neighborhood_pb(pred_X, pred_coords, k),
                        _knn_neighborhood_pb(true_X, true_coords, k), n=n)

def spatial_celltype_coloc(pred_coords, pred_ct, true_coords, true_ct, k=15):
    """Spatial cell-type organization: Pearson correlation between the predicted and observed cell-type
    neighborhood CO-OCCURRENCE matrices (higher = better; 1 = identical spatial neighborhoods)."""
    from sklearn.neighbors import NearestNeighbors
    types = sorted(set(pred_ct) & set(true_ct)); ti = {t: i for i, t in enumerate(types)}; K = len(types)
    def coloc(coords, ct):
        _, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
        codes = np.array([ti.get(c, -1) for c in ct]); C = np.zeros((K, K)); cnt = np.zeros(K)
        for i, a in enumerate(codes):
            if a < 0: continue
            neigh = codes[idx[i]]; neigh = neigh[neigh >= 0]
            for b in neigh: C[a, b] += 1
            cnt[a] += max(len(neigh), 1)
        return C / (cnt[:, None] + 1e-9)
    Cp, Ct = coloc(pred_coords, pred_ct), coloc(true_coords, true_ct)
    return float(np.corrcoef(Cp.ravel(), Ct.ravel())[0, 1])

def morans_I_profile(X, coords, k=15, n_genes=200, seed=0, genes=None):
    """Per-gene spatial autocorrelation (Moran's I) using a kNN spatial weight matrix.
    I_g = (N/W) * sum_ij w_ij (x_ig - mean_g)(x_jg - mean_g) / sum_i (x_ig - mean_g)^2 ,
    with w_ij = 1 if j is among the k nearest spatial neighbours of i (row-normalized here)."""
    from sklearn.neighbors import NearestNeighbors
    X = to_dense(X); N = X.shape[0]
    # Omitting X makes sklearn exclude each query row by index, even when
    # coincident coordinates mean the query is not the first distance tie.
    _, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors()
    if genes is None:                                                 # caller must pass a FIXED gene set
        genes = np.argsort(-X.var(0))[:n_genes]
    Z = X[:, genes] - X[:, genes].mean(0)                             # centred
    lag = Z[idx].mean(1)                                              # spatially lagged value (row-normalized weights)
    num = (Z * lag).sum(0); den = (Z ** 2).sum(0) + 1e-12
    return num / den                                                  # [n_genes] Moran's I per gene

def morans_I_agreement(pred_X, pred_coords, true_X, true_coords, k=15, n_genes=200):
    """Do the predicted genes show the same SPATIAL PATTERNING strength as the real ones?
    Pearson correlation between predicted and observed per-gene Moran's I (higher = better).

    The gene set is chosen ONCE, on the truth, and the SAME indices are used on both sides. The previous
    version selected the top-variance genes on each side independently and then correlated them paired by
    rank position, i.e. it compared two different gene sets — gene 1 of the prediction against gene 1 of the
    truth regardless of which genes those were. That made the value uninterpretable and let 500 random plane
    waves score 0.47.
    """
    genes = np.argsort(-to_dense(true_X).var(0))[:n_genes]             # fixed on the truth, used for both
    ip = morans_I_profile(pred_X, pred_coords, k, n_genes, genes=genes)
    it = morans_I_profile(true_X, true_coords, k, n_genes, genes=genes)
    return float(np.corrcoef(ip, it)[0, 1])

def tissue_size_metrics(pred_coords, true_coords, n_grid=24):
    """Growth / packing: does the predicted embryo have the right SIZE and CELL DENSITY?
    (The other spatial metrics use normalized intra-embryo distances and are scale-invariant by
    design, so none of them can see growth at all.)
      volume_log_ratio = |log(V_pred / V_true)|      via the occupied-voxel volume of a common grid
      density_log_ratio = |log(rho_pred / rho_true)|, rho = n_cells / V
    Both 0 = perfect; higher = worse."""
    def occupied_volume(C, n_grid):
        lo, hi = C.min(0), C.max(0); span = np.maximum(hi - lo, 1e-9)
        vox = np.floor((C - lo) / span * (n_grid - 1e-9)).astype(int)
        n_occ = len(np.unique(vox, axis=0))
        cell_vol = np.prod(span / n_grid)
        return n_occ * cell_vol, n_occ
    Vp, _ = occupied_volume(pred_coords, n_grid); Vt, _ = occupied_volume(true_coords, n_grid)
    rho_p = pred_coords.shape[0] / (Vp + 1e-12); rho_t = true_coords.shape[0] / (Vt + 1e-12)
    return (float(abs(np.log((Vp + 1e-12) / (Vt + 1e-12)))),
            float(abs(np.log((rho_p + 1e-12) / (rho_t + 1e-12)))))

def score_task2(pred_X, pred_coords, pred_ct, true_X, true_coords, true_ct, ref_X, probe=None):
    """All Task-2 metrics: Task-1 expression/composition + spatial-organization (FGW + neighborhood MMD +
    spatial cell-type co-occurrence). ref_X = preceding stage (for DEG)."""
    if probe is None: probe = train_frozen_probe(true_X, true_ct)
    dj, dr = deg_recovery(pred_X, true_X, ref_X)
    return {
        # --- gene-expression accuracy (Task 1) ---
        "pseudobulk_pearson":          round(pseudobulk_pearson(pred_X, true_X), 4),
        "pseudobulk_pearson_celltype": round(pseudobulk_pearson_celltype(pred_X, pred_ct, true_X, true_ct), 4),
        "rbf_mmd":                     round(rbf_mmd(pred_X, true_X), 5),
        "deg_jaccard":                 round(dj, 4),
        "deg_rank_spearman":           round(dr, 4),
        # --- cell-type composition (Task 1) ---
        "composition_JSD":             round(composition_jsd(pred_X, probe, true_ct), 4),
        # --- spatial organization (Task 2) ---
        "neighborhood_mmd":            round(neighborhood_mmd(pred_X, pred_coords, true_X, true_coords), 5),      # local micro-environment (coupling-free)
        "spatial_celltype_coloc":      round(spatial_celltype_coloc(pred_coords, pred_ct, true_coords, true_ct), 4),  # cell-type architecture (higher better)
        "morans_I_agreement":          round(morans_I_agreement(pred_X, pred_coords, true_X, true_coords), 4),    # spatial patterning strength (higher better)
        "volume_log_ratio":            round(tissue_size_metrics(pred_coords, true_coords)[0], 4),                # growth: embryo size (0 = perfect)
        "density_log_ratio":           round(tissue_size_metrics(pred_coords, true_coords)[1], 4),                # growth: cell packing (0 = perfect)
    }

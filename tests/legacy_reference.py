"""Vendored ORIGINAL (pre-memory-fix) implementations of the veckit T1 metric panel.

This is a frozen regression oracle: every function here is a verbatim copy of the metric code as it existed
at commit 46d41e6 ("Execute the tutorial notebook for real") -- BEFORE the memory-safe / interruptible
refactor. The `test_numerical_equivalence.py` test runs BOTH this legacy scorer and the refactored scorer on
the same synthetic fixture and asserts the metric values match, proving the refactor did not change the
metric semantics.

The legacy functions eagerly densify (to_dense) and do full-matrix work -- exactly the behaviour the
refactor removed. They are kept here ONLY as the reference oracle for the regression test, never imported by
the package itself.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ---- helpers (legacy: eager densify) ----
def to_dense(X): return X.toarray() if sparse.issparse(X) else np.asarray(X)
def pseudobulk(X): return np.asarray(to_dense(X).mean(0)).ravel()


def _pairwise_euclidean(A, B):
    aa = np.einsum("ij,ij->i", A, A)
    bb = np.einsum("ij,ij->i", B, B)
    d2 = aa[:, None] + bb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0, out=d2)   # fp cancellation can push near-zero distances slightly negative
    return np.sqrt(d2, out=d2)


def _ranks(v):
    from scipy.stats import rankdata
    return rankdata(np.asarray(v, float))


# ---- differential expression (legacy) ----
MIN_LFC = 0.25
SEVERITY_LOG_WORST = float(np.log(1e-3))


def de_genes(X_cond, X_ref, alpha=0.05, min_lfc=MIN_LFC, min_cells=10):
    from scipy.stats import mannwhitneyu
    A, B = to_dense(X_cond), to_dense(X_ref)
    lfc = pseudobulk(A) - pseudobulk(B)
    if A.shape[0] < min_cells or B.shape[0] < min_cells:
        return np.array([], int), np.array([], int), lfc
    with np.errstate(all="ignore"):
        stat, p = mannwhitneyu(A, B, axis=0, alternative="two-sided")     # tie-corrected by default
    p = np.nan_to_num(p, nan=1.0)
    order = np.argsort(p); m = len(p)
    below = p[order] <= alpha * (np.arange(1, m + 1) / m)                 # Benjamini-Hochberg
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    sig = order[:k]
    if min_lfc > 0: sig = sig[np.abs(lfc[sig]) >= min_lfc]
    return sig[lfc[sig] > 0], sig[lfc[sig] < 0], lfc


def _signed_overlap(lfc_pred, up_true, dn_true):
    n_up, n_dn = len(up_true), len(dn_true)
    if n_up + n_dn == 0: return float("nan"), 0
    order = np.argsort(-np.asarray(lfc_pred, float))                      # most positive first
    pred_up = order[:n_up]
    pred_dn = order[len(order) - n_dn:] if n_dn else np.array([], int)
    hit = len(np.intersect1d(pred_up, up_true)) + len(np.intersect1d(pred_dn, dn_true))
    return hit / (n_up + n_dn), n_up + n_dn


def de_score(pred_X, true_X, ref_X, alpha=0.05, min_lfc=MIN_LFC):
    up_t, dn_t, _ = de_genes(true_X, ref_X, alpha, min_lfc)
    G = int(true_X.shape[1]); nan = float("nan")
    n_true = len(up_t) + len(dn_t)
    out = {"score": nan, "raw": nan, "chance": nan, "chance_uniform": nan,
           "n_up": len(up_t), "n_dn": len(dn_t), "n_true": n_true}
    if n_true == 0: return out
    dp = pseudobulk(pred_X) - pseudobulk(ref_X)
    dt = pseudobulk(true_X) - pseudobulk(ref_X)
    pb_ref = pseudobulk(ref_X)
    c_up, _ = _signed_overlap(pb_ref, up_t, dn_t)
    c_dn, _ = _signed_overlap(-pb_ref, up_t, dn_t)
    chance = max(c_up, c_dn)
    out["chance"] = float(chance)
    out["chance_uniform"] = float(n_true / G)
    if np.std(dt) < 1e-12: return out
    if np.std(dp) < 1e-2 * np.std(dt):
        out.update(score=0.0, raw=0.0); return out
    raw, _ = _signed_overlap(dp, up_t, dn_t)
    out["raw"] = float(raw)
    out["score"] = float((raw - chance) / (1 - chance)) if chance < 1 else nan
    return out


def de_direction(pred_X, true_X, ref_X):
    dp = pseudobulk(pred_X) - pseudobulk(ref_X)
    dt = pseudobulk(true_X) - pseudobulk(ref_X)
    if np.std(dt) < 1e-12 or np.std(dp) < 1e-2 * np.std(dt): return 0.0
    rp, rt, rr = _ranks(dp), _ranks(dt), _ranks(pseudobulk(ref_X))
    C = np.corrcoef(np.vstack([rp, rt, rr]))
    if not np.all(np.isfinite(C)): return 0.0
    if 1.0 - abs(C[0, 2]) < 1e-6: return 0.0
    num = C[0, 1] - C[0, 2] * C[1, 2]
    den = np.sqrt(max((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2), 1e-12))
    r = num / den
    return float(np.clip(r, -1.0, 1.0)) if np.isfinite(r) else 0.0


# ---- distribution (legacy: densify then sample) ----
def mmd_unbiased(pred_X, true_X, n=2000, n_pc=30, seed=0, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import rbf_kernel
    pred_X, true_X = to_dense(pred_X), to_dense(true_X)
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=min(n_pc, true_X.shape[1]), random_state=0).fit(true_X)
    A = pca.transform(pred_X[rng.choice(pred_X.shape[0], min(n, pred_X.shape[0]), replace=False)])
    B = pca.transform(true_X[rng.choice(true_X.shape[0], min(n, true_X.shape[0]), replace=False)])
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
    A, B = to_dense(pred_X), to_dense(true_X)
    rng = np.random.default_rng(seed)
    A = A[rng.choice(A.shape[0], min(n, A.shape[0]), replace=False)]
    B = B[rng.choice(B.shape[0], min(n, B.shape[0]), replace=False)]
    na, nb = A.shape[0], B.shape[0]
    Daa, Dbb = _pairwise_euclidean(A, A), _pairwise_euclidean(B, B)
    return float(2 * _pairwise_euclidean(A, B).mean()
                 - Daa.sum() / (na * (na - 1)) - Dbb.sum() / (nb * (nb - 1)))


def variogram_score(pred_X, true_X, n_pairs=20000, n_cells=1500, p=0.5, seed=0):
    A, B = to_dense(pred_X), to_dense(true_X)
    rng = np.random.default_rng(seed)
    A = A[rng.choice(A.shape[0], min(n_cells, A.shape[0]), replace=False)]
    B = B[rng.choice(B.shape[0], min(n_cells, B.shape[0]), replace=False)]
    G = A.shape[1]
    i = rng.integers(0, G, n_pairs); j = rng.integers(0, G, n_pairs)
    keep = i != j; i, j = i[keep], j[keep]
    va = (np.abs(A[:, i] - A[:, j]) ** p).mean(0)
    vb = (np.abs(B[:, i] - B[:, j]) ** p).mean(0)
    return float(((va - vb) ** 2).mean())


def pb_rel_err(pred_X, true_X):
    a, b = pseudobulk(pred_X), pseudobulk(true_X)
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def library_size_ratio(pred_X, true_X):
    a, b = to_dense(pred_X), to_dense(true_X)
    la = float(np.median(np.expm1(np.clip(a, 0, 50)).sum(1)))
    lb = float(np.median(np.expm1(np.clip(b, 0, 50)).sum(1)))
    return float(la / lb) if lb > 0 else float("nan")


def variance_ratio(pred_X, true_X):
    va = float(to_dense(pred_X).var(0).mean()); vb = float(to_dense(true_X).var(0).mean())
    return float(va / vb) if vb > 0 else float("nan")


# ---- T1/metrics.py: frozen probe + composition + pearson (legacy) ----
def _jsd2(p, q, eps=1e-6):
    p = p + eps; q = q + eps; p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(x * np.log2(x / y))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def train_frozen_probe(target_X, target_ct, cap=60000, seed=0):
    target_X = to_dense(target_X); rng = np.random.default_rng(seed); n = target_X.shape[0]
    idx = rng.choice(n, min(cap, n), replace=False) if n > cap else np.arange(n)
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, n_jobs=-1))
    clf.fit(target_X[idx], target_ct[idx]); return clf


def composition_jsd(pred_X, probe, true_ct):
    classes = probe.classes_
    pi_hat = probe.predict_proba(to_dense(pred_X)).mean(0)
    u, cnt = np.unique(true_ct, return_counts=True); dd = dict(zip(u, cnt))
    pi_obs = np.array([dd.get(c, 0) for c in classes], float)
    return _jsd2(pi_obs, pi_hat)


def pseudobulk_pearson(pred_X, true_X):
    return float(np.corrcoef(pseudobulk(pred_X), pseudobulk(true_X))[0, 1])


# ---- legacy T1 score_task1_v2 (mirrors T1/metrics_v2.py original) ----
def legacy_score_task1_v2(pred_X, pred_ct, true_X, true_ct, ref_X, probe=None, seed=0):
    if probe is None: probe = train_frozen_probe(true_X, true_ct)
    de = de_score(pred_X, true_X, ref_X)
    return {
        "de_score":            round(de["score"], 4) if np.isfinite(de["score"]) else None,
        "de_direction":        round(de_direction(pred_X, true_X, ref_X), 4),
        "energy_distance":     round(energy_distance(pred_X, true_X, seed=seed), 5),
        "mmd_u":               round(mmd_unbiased(pred_X, true_X, seed=seed), 5),
        "variogram":           round(variogram_score(pred_X, true_X, seed=seed), 6),
        "pb_rel_err":          round(pb_rel_err(pred_X, true_X), 4),
        "library_size_ratio":  round(library_size_ratio(pred_X, true_X), 3),
        "variance_ratio":      round(variance_ratio(pred_X, true_X), 3),
        "composition_JSD":     round(composition_jsd(pred_X, probe, true_ct), 4),
        "pseudobulk_pearson":  round(pseudobulk_pearson(pred_X, true_X), 4),
        "_de_raw":             round(de["raw"], 4) if np.isfinite(de["raw"]) else None,
        "_de_chance":          round(de["chance"], 4) if np.isfinite(de["chance"]) else None,
        "_de_chance_unif":     round(de["chance_uniform"], 4) if np.isfinite(de["chance_uniform"]) else None,
        "_n_up": int(de["n_up"]), "_n_dn": int(de["n_dn"]),
    }

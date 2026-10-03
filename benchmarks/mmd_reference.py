"""Frozen 0.1.1 comparison, copied from the MIT-licensed project baseline.

Source: common/core_metrics.py at 46d41e63f42a9aab815db20b742feeccd249cb17.
Copyright (c) 2026 Virtual Embryo Challenge organizers. See ../LICENSE.
Kept independent of the optimized function to detect numerical drift.
"""
import numpy as np
from scipy import sparse


def mmd_reference(pred_X, true_X, n=2000, n_pc=30, seed=0,
                  scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import rbf_kernel
    pred_X = pred_X.toarray() if sparse.issparse(pred_X) else np.asarray(pred_X)
    true_X = true_X.toarray() if sparse.issparse(true_X) else np.asarray(true_X)
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
        np.fill_diagonal(Kaa, 0.0)
        np.fill_diagonal(Kbb, 0.0)
        tot += (Kaa.sum() / (na * (na - 1)) + Kbb.sum() / (nb * (nb - 1)) - 2 * Kab.mean())
    return float(tot / len(scales))

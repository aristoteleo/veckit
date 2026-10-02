"""Shared synthetic T1 fixture builder for the regression tests.

Builds log-normalised non-negative expression matrices (sparse-able) with a genuine differential-expression
signal between reference and target, so that de_score/de_direction are well-defined (non-null) and every
metric in the panel is exercised away from its degenerate guard thresholds.
"""
from __future__ import annotations
import numpy as np


def make_t1_fixture(seed=7, cells_ref=200, cells_true=220, cells_pred=220, genes=500,
                    de_frac=0.12, de_lfc=1.0):
    rng = np.random.default_rng(seed)
    G = genes
    base_mean = rng.uniform(0.3, 2.5, G)
    n_de = max(2, int(G * de_frac))
    n_up = n_de // 2
    n_dn = n_de - n_up
    up = rng.choice(G, n_up, replace=False)
    dn = rng.choice(np.setdiff1d(np.arange(G), up), n_dn, replace=False)
    delta = np.zeros(G); delta[up] = de_lfc; delta[dn] = -de_lfc

    ref = rng.lognormal(mean=base_mean, sigma=0.8, size=(cells_ref, G)).astype(np.float32)
    true = rng.lognormal(mean=base_mean + delta, sigma=0.8, size=(cells_true, G)).astype(np.float32)
    pred = true * rng.uniform(0.9, 1.1, size=(cells_pred, 1)) + rng.normal(0, 0.05, size=(cells_pred, G))
    pred = np.maximum(pred, 0.0).astype(np.float32)

    ct_ref = rng.integers(0, 4, size=cells_ref).astype(str)
    ct_true = rng.integers(0, 4, size=cells_true).astype(str)
    ct_pred = rng.integers(0, 4, size=cells_pred).astype(str)
    genes_names = [f"g{i}" for i in range(G)]
    return ref, true, pred, ct_ref, ct_true, ct_pred, genes_names

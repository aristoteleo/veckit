"""Virtual Embryo Challenge — Task 1 metric panel.

  PRIMARY (these rank models)
    de_score        did you name the genes that change, and move them the right way, beyond what ranking
                    genes by how highly expressed they already are achieves? Rank-based and sign-split, so
                    rescaling or negating a prediction is no longer rewarded.
    de_direction    signed agreement of the change, with the reference expression level partialled out
    energy_distance full-500-gene-space distributional distance (no PCA blind subspace)
    mmd_u           unbiased multi-kernel MMD in a truth-fitted PCA space
    variogram       gene-gene covariance structure — the only term that constrains the joint
  CONSTRAINT (reported, not ranked)
    pb_rel_err      absolute pseudobulk scale — a GATE, since every other term is affine- or scale-free
    variance_ratio  is the submission a population at all, or one cell broadcast?
    composition_JSD cell-type proportions via the frozen probe
    pseudobulk_pearson  kept as a diagnostic only: 0.88 for a do-nothing model, and affine-invariant

Removed: `rbf_mmd` (biased estimator, replaced by `mmd_u`). See ../METRICS_V2.md and common/core_metrics.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.core_metrics import (de_score, de_direction, mmd_unbiased, energy_distance,
                                 variogram_score, pb_rel_err, variance_ratio,
                                 library_size_ratio, _log)
from metrics import composition_jsd, train_frozen_probe, pseudobulk_pearson


def score_task1_v2(pred_X, pred_ct, true_X, true_ct, ref_X, probe=None, seed=0):
    """Full Task-1 panel. `ref_X` = the preceding stage (the DE reference)."""
    if probe is None: probe = train_frozen_probe(true_X, true_ct)
    _log("computing de_score..."); de = de_score(pred_X, true_X, ref_X)
    _log("computing de_direction..."); de_dir = de_direction(pred_X, true_X, ref_X)
    _log("computing energy_distance..."); ed = energy_distance(pred_X, true_X, seed=seed)
    _log("computing mmd_u..."); mmd = mmd_unbiased(pred_X, true_X, seed=seed)
    _log("computing variogram..."); vg = variogram_score(pred_X, true_X, seed=seed)
    _log("computing constraints...")
    return {
        # ---- primary ----
        "de_score":            round(de["score"], 4) if np.isfinite(de["score"]) else None,
        "de_direction":        round(de_dir, 4),
        "energy_distance":     round(ed, 5),
        "mmd_u":               round(mmd, 5),
        "variogram":           round(vg, 6),
        # ---- constraint ----
        "pb_rel_err":          round(pb_rel_err(pred_X, true_X), 4),
        "library_size_ratio":  round(library_size_ratio(pred_X, true_X), 3),
        "variance_ratio":      round(variance_ratio(pred_X, true_X), 3),
        "composition_JSD":     round(composition_jsd(pred_X, probe, true_ct), 4),
        "pseudobulk_pearson":  round(pseudobulk_pearson(pred_X, true_X), 4),
        # ---- audit trail for de_score ----
        "_de_raw":             round(de["raw"], 4) if np.isfinite(de["raw"]) else None,
        "_de_chance":          round(de["chance"], 4) if np.isfinite(de["chance"]) else None,
        "_de_chance_unif":     round(de["chance_uniform"], 4) if np.isfinite(de["chance_uniform"]) else None,
        "_n_up": int(de["n_up"]), "_n_dn": int(de["n_dn"]),
    }

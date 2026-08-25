"""Virtual Embryo Challenge — Task 3 metric panel (mutant perturbation).

Scored on the response relative to the matched wild type, because a do-nothing prediction already reaches
pseudobulk_pearson 0.956.

  PRIMARY — response:     de_score, de_direction, severity_slope
  PRIMARY — distribution: mmd_u, variogram (same "is the cell-state distribution right" question T1/T2
                          score, computed the same way -- directly on pred vs. true, no WT-relative
                          construction; `wt_identity`'s own score on these IS the floor, by the same
                          convention `copy_last` is T1's floor, so there is nothing to fix here)
  PRIMARY — shape:        d2_shape, sliced_wasserstein, occupancy_dice, scale_log_ratio, count_log_ratio,
                          neighborhood_mmd   — half of what a T3 entrant submits is 3D coordinates, and the
                          previous panel scored NONE of it, so morphological knockout phenotypes could be
                          neither right nor wrong. The same spatial group as Task 2 now applies here, under
                          one rule rather than "include it where the floor does not win".
  CONSTRAINT:             pb_rel_err, variance_ratio, composition_JSD, pseudobulk_pearson
  DIAGNOSTIC:             energy_distance (full-gene-space audit of the same question mmd_u already scores)

REMOVED: `delta_pearson` (the Pearson twin of de_direction on the same vectors), `delta_mae` (its wild-type
reference cancels algebraically), `delta_magnitude_ratio` (depends only on the two norms, so every rotation of
a correct-norm delta scored identically and a random direction with the right magnitude won it),
`deg_jaccard` / `deg_rank_spearman` (the top-K set was a union over both tails, so a perfect anti-phenotype
scored the same as the correct one), `perturbation_discrimination`, `effect_progress`, `fgw_spatial`.
Severity is now `severity_slope`, which a random direction cannot win. See ../METRICS_V2.md.

Also tried and REMOVED: `response_mmd`, an MMD between (pred - matched WT cell-type centroid) and
(true_ko - the same centroid), meant to score the KO-vs-WT *shift* distribution instead of the raw KO
distribution (motivated by `mmd_u`/`energy_distance` on the raw distributions structurally favoring
`wt_identity`, which reproduces the ~95% of cells a real knockout doesn't touch). Measured instead: EVERY
real model (`shift_transfer`, `gene_ko`, `perturb_ode`) scored far *below* `wt_identity`'s floor on it
(skill -71 to -97), the opposite of the intended fix. Per-cell residual variance -- each real model's own
prediction noise stacked on top of the KO's true (and biologically subtle) shift -- dominated the MMD kernel
comparison, while `wt_identity`'s residual is just wild type's own natural single-cell noise centered at
exactly zero; a real model's *extra* variance made its residual cloud look less like the true residual
cloud than "no response at all" did, for reasons having nothing to do with getting the shift's direction or
size right. `de_score`/`de_direction`/`severity_slope` avoid this because they operate on pseudobulk/DE-gene
aggregates, where averaging over many cells is exactly what suppresses single-cell noise relative to the
population-level signal -- a single-cell-level distributional metric doesn't get that averaging for free.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from common.core_metrics import (de_score, de_direction, severity_slope, severity_ratio, mmd_unbiased,
                                 energy_distance, variogram_score, pb_rel_err, variance_ratio,
                                 library_size_ratio)
from common.shape_metrics import (d2_distance, sliced_wasserstein, occupancy_dice,
                                  scale_log_ratio, count_log_ratio)
from metrics import composition_jsd, train_frozen_probe, pseudobulk_pearson, neighborhood_mmd


def score_task3_v2(pred_X, pred_C, pred_ct, true_ko_X, true_C, true_ct, wt_X, probe=None, seed=0):
    """Full Task-3 panel. `wt_X` = the matched wild type: both the DE reference and the do-nothing baseline."""
    if probe is None: probe = train_frozen_probe(true_ko_X, true_ct)
    de = de_score(pred_X, true_ko_X, wt_X)
    slope, slope_r2 = severity_slope(pred_X, true_ko_X, wt_X)
    sev_ratio, _ = severity_ratio(pred_X, true_ko_X, wt_X)
    out = {
        # ---- primary: the response ----
        "de_score":           round(de["score"], 4) if np.isfinite(de["score"]) else None,
        "de_direction":       round(de_direction(pred_X, true_ko_X, wt_X), 4),
        # `severity_ratio` is what ranks (bounded, no hand-chosen floor constant); `severity_slope` is
        # kept alongside it as the SIGNED diagnostic, so a reader can still tell overshoot from undershoot.
        "severity_ratio":     round(sev_ratio, 4) if np.isfinite(sev_ratio) else None,
        "severity_slope":     round(slope, 4) if np.isfinite(slope) else None,
        # ---- primary: the mutant state as a distribution ----
        "energy_distance":    round(energy_distance(pred_X, true_ko_X, seed=seed), 5),
        "mmd_u":              round(mmd_unbiased(pred_X, true_ko_X, seed=seed), 5),
        "variogram":          round(variogram_score(pred_X, true_ko_X, seed=seed), 6),
        # ---- constraint ----
        "pb_rel_err":         round(pb_rel_err(pred_X, true_ko_X), 4),
        "library_size_ratio": round(library_size_ratio(pred_X, true_ko_X), 3),
        "variance_ratio":     round(variance_ratio(pred_X, true_ko_X), 3),
        "composition_JSD":    round(composition_jsd(pred_X, probe, true_ct), 4),
        "pseudobulk_pearson": round(pseudobulk_pearson(pred_X, true_ko_X), 4),
        # ---- audit ----
        "_de_raw":            round(de["raw"], 4) if np.isfinite(de["raw"]) else None,
        "_de_chance":         round(de["chance"], 4) if np.isfinite(de["chance"]) else None,
        "_n_up": int(de["n_up"]), "_n_dn": int(de["n_dn"]),
        "_slope_r2":          slope_r2,
    }
    if pred_C is not None and true_C is not None:
        sw, sw_spread = sliced_wasserstein(pred_C, true_C, seed=seed)
        dice, vox_ratio = occupancy_dice(pred_C, true_C, seed=seed)
        out.update({
            "d2_shape":           round(d2_distance(pred_C, true_C, seed=seed), 5),
            "sliced_wasserstein": round(sw, 5),
            "occupancy_dice":     round(dice, 4),
            "scale_log_ratio":    round(scale_log_ratio(pred_C, true_C), 4),
            "count_log_ratio":    round(count_log_ratio(pred_X.shape[0], true_ko_X.shape[0]), 4),
            "neighborhood_mmd":   round(neighborhood_mmd(pred_X, pred_C, true_ko_X, true_C), 5),
            "_sw_flip_spread":    round(sw_spread, 5),
            "_dice_voxel_over_nn": round(vox_ratio, 1),
        })
    return out

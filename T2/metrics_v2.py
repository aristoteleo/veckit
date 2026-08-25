"""Virtual Embryo Challenge — Task 2 metric panel (expression + 3D shape).

  PRIMARY — expression: de_score, de_direction, energy_distance, mmd_u, variogram (see ../T1/metrics_v2.py)
  PRIMARY — shape:      d2_shape, sliced_wasserstein, occupancy_dice   (rotation-invariant, count-matched)
  PRIMARY — growth:     scale_log_ratio (rotation-invariant size), count_log_ratio (proliferation)
  PRIMARY — local:      neighborhood_mmd (now on the unbiased truth-fitted estimator)
  CONSTRAINT:           pb_rel_err, variance_ratio, composition_JSD, pseudobulk_pearson, morans_I_agreement

REMOVED
  * `volume_log_ratio` / `density_log_ratio` — computed on each cloud's OWN axis-aligned bounding box, so not
    rotation-invariant; a cube of uniform random points scored the same as the real model, and a
    volume-preserving anisotropic squash of the TRUTH scored 0.00000. Their two questions (size, cell number)
    are now `scale_log_ratio` and `count_log_ratio`, which are separate and frame-independent.
  * `spatial_celltype_coloc` — consumed SUBMITTED cell-type labels, which the protocol says participants never
    submit, and a k-means relabelling scored 1.0000, tying the ceiling. Local organisation is covered by
    `neighborhood_mmd`; a label-free replacement needs a permutation null and is not implemented yet.
  * `fgw_spatial`, `stage_discrimination`, `geometry_discrimination`, `progress` — see ../METRICS_V2.md.

`morans_I_agreement` is retained as a CONSTRAINT now that its gene pairing is fixed (the set is chosen once on
the truth and both sides are indexed with it). Laterality is still unscored — see common/shape_metrics.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from common.core_metrics import (de_score, de_direction, mmd_unbiased, energy_distance,
                                 variogram_score, pb_rel_err, variance_ratio,
                                 library_size_ratio)
from common.shape_metrics import (d2_distance, sliced_wasserstein, occupancy_dice, size_fidelity,
                                  scale_log_ratio, scale_ratio, count_log_ratio)
from metrics import (composition_jsd, train_frozen_probe, pseudobulk_pearson,
                     neighborhood_mmd, morans_I_agreement)


def score_task2_v2(pred_X, pred_coords, pred_ct, true_X, true_coords, true_ct, ref_X, probe=None, seed=0):
    """Full Task-2 panel: expression + shape + growth + local organisation."""
    if probe is None: probe = train_frozen_probe(true_X, true_ct)
    de = de_score(pred_X, true_X, ref_X)
    sw, sw_spread = sliced_wasserstein(pred_coords, true_coords, seed=seed)
    dice, vox_ratio = occupancy_dice(pred_coords, true_coords, seed=seed)
    return {
        # ---- primary: expression ----
        "de_score":            round(de["score"], 4) if np.isfinite(de["score"]) else None,
        "de_direction":        round(de_direction(pred_X, true_X, ref_X), 4),
        "energy_distance":     round(energy_distance(pred_X, true_X, seed=seed), 5),
        "mmd_u":               round(mmd_unbiased(pred_X, true_X, seed=seed), 5),
        "variogram":           round(variogram_score(pred_X, true_X, seed=seed), 6),
        # ---- primary: shape (rotation-invariant, cell-count matched) ----
        "d2_shape":            round(d2_distance(pred_coords, true_coords, seed=seed), 5),
        "sliced_wasserstein":  round(sw, 5),
        "occupancy_dice":      round(dice, 4),
        # ---- primary: growth, as two separate questions ----
        # `scale_ratio` is what ranks (unsigned, so a leaderboard score gives no search
        # direction); `scale_log_ratio` is kept as the SIGNED diagnostic -- a reader still needs
        # to know whether a prediction is too big or too small.
        # `size_fidelity` is what ranks: tissue scale AND cell scale, so a uniform rescale that
        # buys the first pays for it in the second. `scale_ratio` (unsigned) and
        # `scale_log_ratio` (signed) stay as diagnostics that decompose it.
        "size_fidelity":       round(size_fidelity(pred_coords, true_coords), 4),
        "scale_ratio":         round(scale_ratio(pred_coords, true_coords), 4),
        "scale_log_ratio":     round(scale_log_ratio(pred_coords, true_coords), 4),
        "count_log_ratio":     round(count_log_ratio(pred_X.shape[0], true_X.shape[0]), 4),
        # ---- primary: local organisation ----
        "neighborhood_mmd":    round(neighborhood_mmd(pred_X, pred_coords, true_X, true_coords), 5),
        # ---- constraint ----
        "pb_rel_err":          round(pb_rel_err(pred_X, true_X), 4),
        "library_size_ratio":  round(library_size_ratio(pred_X, true_X), 3),
        "variance_ratio":      round(variance_ratio(pred_X, true_X), 3),
        "composition_JSD":     round(composition_jsd(pred_X, probe, true_ct), 4),
        "pseudobulk_pearson":  round(pseudobulk_pearson(pred_X, true_X), 4),
        "morans_I_agreement":  round(morans_I_agreement(pred_X, pred_coords, true_X, true_coords), 4),
        # ---- audit ----
        "_de_raw":             round(de["raw"], 4) if np.isfinite(de["raw"]) else None,
        "_de_chance":          round(de["chance"], 4) if np.isfinite(de["chance"]) else None,
        "_n_up": int(de["n_up"]), "_n_dn": int(de["n_dn"]),
        "_sw_flip_spread":     round(sw_spread, 5),
        "_dice_voxel_over_nn": round(vox_ratio, 1),
    }

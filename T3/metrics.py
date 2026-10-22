"""Virtual Embryo Challenge — Task 3 evaluation metrics (mutant perturbation prediction).

Task 3 reuses the Task-1 expression/composition metrics and the Task-2 spatial metrics, but adds the
metrics that matter specifically for a *perturbation*: a model must predict the KNOCKOUT EFFECT, i.e. the
shift away from wild type, not merely produce an embryo-like cell set. A prediction that simply returns the
wild type scores well on absolute-expression metrics while capturing no biology at all, so the perturbation
metrics below are computed on the WT->KO delta.
"""
from __future__ import annotations
import numpy as np
from T1 import metrics as t1_metrics
from T2 import metrics as t2_metrics

CAP = 5.0     # cap on |log| magnitude ratio (e^5 ~ 150x off)

# reuse: expression + composition (Task 1)
pseudobulk = t1_metrics.pseudobulk
to_dense = t1_metrics.to_dense
pseudobulk_pearson = t1_metrics.pseudobulk_pearson
composition_jsd = t1_metrics.composition_jsd
rbf_mmd = t1_metrics.rbf_mmd
train_frozen_probe = t1_metrics.train_frozen_probe
# reuse: spatial (Task 2)
neighborhood_mmd = t2_metrics.neighborhood_mmd
spatial_celltype_coloc = t2_metrics.spatial_celltype_coloc


# ---------- perturbation-effect metrics (the core of Task 3) ----------
def perturbation_delta(ko_X, wt_X):
    """The knockout effect: pseudobulk(KO) - pseudobulk(WT)."""
    return pseudobulk(ko_X) - pseudobulk(wt_X)

def delta_pearson(pred_X, true_ko_X, wt_X):
    """Do you predict the KO EFFECT (not just an embryo-like profile)?
    Pearson between predicted and observed WT->KO pseudobulk shift. Higher is better.
    A 'predict the wild type' baseline has zero delta and scores 0 here by construction.

    The no-effect guard is RELATIVE to the true effect, not an absolute 1e-12. It has to be: `pseudobulk`
    averages a float32 matrix, and two bit-identical arrays in different buffers sum to slightly different
    values, so `wt_identity` — whose prediction *is* the reference, element for element — produced a delta
    of norm ~1e-5 rather than 0. That sailed past an absolute 1e-12 threshold and the metric happily
    correlated pure rounding noise against the truth, handing the do-nothing floor +0.126 while every model
    that predicts an actual effect scored negative. Since this metric is the Task-3 primary and the floor of
    the skill normalization, a floor set by float32 noise contaminated everything downstream.
    `deg_recovery_perturbation` below already used the relative form; this now matches it."""
    dp, dt = perturbation_delta(pred_X, wt_X), perturbation_delta(true_ko_X, wt_X)
    if np.std(dt) < 1e-12: return 0.0
    if np.std(dp) < 1e-2 * np.std(dt): return 0.0          # no predicted effect (see docstring)
    return float(np.corrcoef(dp, dt)[0, 1])

def delta_mae(pred_X, true_ko_X, wt_X):
    """Absolute error of the predicted effect size (lower is better)."""
    dp, dt = perturbation_delta(pred_X, wt_X), perturbation_delta(true_ko_X, wt_X)
    return float(np.mean(np.abs(dp - dt)))

def delta_magnitude_ratio(pred_X, true_ko_X, wt_X):
    """|log| ratio of predicted to observed effect MAGNITUDE (0 = perfect; lower is better).
    Separates 'right direction, wrong strength' from 'wrong direction' — a model can get delta_pearson
    right while systematically under- or over-shooting the severity of the phenotype."""
    dp, dt = perturbation_delta(pred_X, wt_X), perturbation_delta(true_ko_X, wt_X)
    np_, nt_ = float(np.linalg.norm(dp)), float(np.linalg.norm(dt))
    if nt_ < 1e-12: return 0.0
    if np_ < 1e-6 * nt_: return CAP                  # predicts no effect at all -> worst possible
                                                     # (relative, for the float32 reason in delta_pearson)
    return float(min(abs(np.log(np_ / nt_)), CAP))   # capped so a near-zero prediction can't dominate

def deg_recovery_perturbation(pred_X, true_ko_X, wt_X, topk=50):
    """DE-gene recovery for the perturbation: which genes respond to the knockout, and in which
    direction. Top-K up/down Jaccard + signed rank agreement, both computed on the WT->KO delta."""
    from scipy.stats import spearmanr
    dp, dt = perturbation_delta(pred_X, wt_X), perturbation_delta(true_ko_X, wt_X)
    if float(np.std(dp)) < 1e-2 * float(np.std(dt)): return 0.0, 0.0     # predicts no effect -> recovers none
    top = lambda d: set(np.argsort(d)[:topk]) | set(np.argsort(d)[-topk:])
    sp, st = top(dp), top(dt)
    return float(len(sp & st) / len(sp | st)), float(spearmanr(dp, dt).correlation)


def score_task3(pred_X, pred_coords, pred_ct, true_ko_X, true_coords, true_ct, wt_X, probe=None):
    """All Task-3 metrics. `wt_X` = the matched wild-type expression (the reference the effect is measured
    against). Spatial metrics are skipped when coordinates are unavailable."""
    if probe is None: probe = train_frozen_probe(true_ko_X, true_ct)
    dj, dr = deg_recovery_perturbation(pred_X, true_ko_X, wt_X)
    out = {
        # --- perturbation effect (Task-3 core) ---
        "delta_pearson":          round(delta_pearson(pred_X, true_ko_X, wt_X), 4),
        "delta_mae":              round(delta_mae(pred_X, true_ko_X, wt_X), 5),
        "delta_magnitude_ratio":  round(delta_magnitude_ratio(pred_X, true_ko_X, wt_X), 4),
        "deg_jaccard":            round(dj, 4),
        "deg_rank_spearman":      round(dr, 4),
        # --- absolute mutant state (Task-1 metrics) ---
        "pseudobulk_pearson":     round(pseudobulk_pearson(pred_X, true_ko_X), 4),
        "rbf_mmd":                round(rbf_mmd(pred_X, true_ko_X), 5),
        "composition_JSD":        round(composition_jsd(pred_X, probe, true_ct), 4),
    }
    if pred_coords is not None and true_coords is not None:
        out["neighborhood_mmd"] = round(neighborhood_mmd(pred_X, pred_coords, true_ko_X, true_coords), 5)
        # KO and WT files use different cell-type vocabularies, so raw labels do not intersect.
        # Use the FROZEN PROBE labels for both sides — the official protocol, and it makes the
        # vocabularies identical by construction.
        pl = probe.classes_[probe.predict_proba(to_dense(pred_X)).argmax(1)]
        tl = probe.classes_[probe.predict_proba(to_dense(true_ko_X)).argmax(1)]
        v = spatial_celltype_coloc(pred_coords, pl, true_coords, tl)
        out["spatial_celltype_coloc"] = round(v, 4) if np.isfinite(v) else None
    return out

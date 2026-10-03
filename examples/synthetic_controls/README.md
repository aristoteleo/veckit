# Executable Synthetic Adversarial Controls

This example implements the eight adversarial controls described on the VEC
[Baselines page](https://virtualembryo.ai/challenge/baselines#ref-ctrl_scale_ref), using generated fixtures
with fake genes, cells, expression, cell types, and coordinates.

It is a public-description-only implementation. It is not a copy of the private `bake.py`, does not claim
server equivalence, and cannot predict a hidden leaderboard score.

## Run all controls

From a clean checkout:

```bash
python -m pip install -e ".[test]"
python examples/synthetic_controls/run_controls.py --task all --seed 1101 \
  --out artifacts/synthetic_controls_manifest.json
```

The command generates every control, checks its formula, scores it through the public `veckit.score(...)`
API, prints a compact metric summary, and writes a manifest containing versions, seeds, fixture hashes,
formulas, shapes, and all returned diagnostics. Synthetic `.h5ad` files are temporary by default. Preserve
them only when needed:

```bash
python examples/synthetic_controls/run_controls.py --task T2 \
  --write-artifacts artifacts/synthetic_h5ad
```

## Controls

| Name | Task | Public formula implemented here |
|---|---|---|
| `ctrl_scale_ref` | T1/T2 | `reference.X * 2.0` |
| `ctrl_shrink_ref` | T1 | `reference.X * 0.99` |
| `ctrl_one_cell` | T1 | Reference pseudobulk mean repeated to an explicit cell count |
| `ctrl_random_cube` | T2 | Reference expression with coordinates sampled uniformly inside target bounds |
| `ctrl_squashed_ref` | T2 | Centered coordinates multiplied by `(4.0, 1.0, 0.25)`, then recentered |
| `ctrl_scale_wt` | T3 | `wt.X * 2.0` |
| `ctrl_shrink_wt` | T3 | `wt.X * 0.75` |
| `ctrl_random_dir` | T3 | Gaussian gene shift scaled to the WT-to-target pseudobulk L2 norm, added to WT, then clipped at zero |

`ctrl_scale_ref` is evaluated once for each of its published tasks, so an all-task run contains nine rows
covering eight distinct controls.

## Explicit interpretations

The public page says `ctrl_one_cell` has the right cell count but does not name which input supplies it. The
function therefore requires `n_cells`; this walkthrough uses `synthetic_target.n_obs`.

For `ctrl_random_dir`, the shift is the gene-level pseudobulk vector used by `veckit`'s T3 perturbation
metrics: `mean(target.X, axis=0) - mean(wt.X, axis=0)`. The equal-norm assertion applies before clipping,
matching the operation order on the public page. The manifest separately records the post-clip realised norm.
The random vector is not orthogonalised: the published control says Gaussian random direction, not a forced
90-degree direction.

For `ctrl_random_cube`, callers must provide either a target-shaped AnnData object or explicit `3 x 2`
axis bounds. This walkthrough uses only a synthetic target.

## Tests

```bash
python -m pytest tests/test_synthetic_controls.py -q
```

Tests assert formulas, input immutability, metadata preservation, bounds, deterministic RNG behavior,
pre/post-clipping semantics, validation failures, and successful public scorer execution for all controls.
Metric values are diagnostics, not pass/fail thresholds.

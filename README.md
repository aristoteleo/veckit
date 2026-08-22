# veckit

Local scorer for the [Virtual Embryo Challenge](https://virtualembryo.ai): score a submitted `.h5ad`
against the T1/T2/T3 metric panels, entirely offline, against reference files **you** supply. This package
never reads the official (held-out) validation/test data, and never imports anything that names where that
data lives -- it only knows how to compute a metric panel from arrays you hand it.

```bash
pip install veckit
```

## Getting the real data

Register and download the released T1/T2/T3 stages from the official challenge site:
[virtualembryo.ai/challenge/data](https://virtualembryo.ai/challenge/data) (see also
[aristoteleo/virtualembryo](https://github.com/aristoteleo/virtualembryo)). `veckit_tutorial.ipynb` in this
repo also ships a few tiny (150-cell) samples if you just want to try the tool first.

## CLI

Full, meaningful panel (a real prediction, scored against the real preceding stage):

```bash
veckit --task T1 --input pred.h5ad --target T1/9.5.h5ad --reference T1/8.5.h5ad
veckit --task T2 --setting heart --input pred.h5ad --target b.h5ad --reference a.h5ad
veckit --task T3 --input pred.h5ad --target mab21l2_ko.h5ad --wt wt.h5ad
```

Quick copy_last / wt_identity check (`--input` doubles as `--reference`/`--wt` when omitted):

```bash
veckit --task T1 --input T1/8.5.h5ad --target T1/9.5.h5ad
```

## Python

```python
from veckit import score
result = score(task="T1", input="pred.h5ad", target="T1/9.5.h5ad", reference="T1/8.5.h5ad")
print(result["metrics"])
```

`--target` (all tasks) is the pseudo target your `--input` prediction is scored against.
`--reference` (T1/T2) / `--wt` (T3) is the reference expression `de_score`/`de_direction`/`severity_slope`
are computed relative to -- these are PRIMARY metrics ("did you predict the right *change*", not just
"does this look plausible"), so pass a real one whenever you're scoring a real model; omitting it defaults
to `--input` itself, which is only correct when you're deliberately testing a no-change baseline
(`copy_last`/`wt_identity`) -- those metrics then correctly read as exactly 0, matching the official
baseline tables.

**This is not a preview of your real competition score.** Whatever you pass as `--target` is, by
definition, data you already had — so this only tells you the scoring pipeline runs and your submission
format is valid, not how well you'll do on the real held-out target.

## Synthetic adversarial controls

The [`examples/synthetic_controls`](examples/synthetic_controls) walkthrough provides synthetic-only,
executable implementations of the eight adversarial controls described on the Challenge Baselines page.
It includes deterministic fixtures, formula/property tests, and a one-command run through the public scorer:

```bash
python examples/synthetic_controls/run_controls.py --task all --seed 1101
```

These examples implement the public descriptions; they are not server-equivalent controls or leaderboard
predictors. See the example README for the explicit `ctrl_one_cell` and `ctrl_random_dir` interpretations.

## More

Full task definitions and metric rationale: [virtualembryo.ai](https://virtualembryo.ai) and
[aristoteleo/virtualembryo](https://github.com/aristoteleo/virtualembryo).

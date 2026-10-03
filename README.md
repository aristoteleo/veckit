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

## MMD performance check

The multi-kernel MMD computes its median bandwidth in row blocks and reuses
three squared-distance matrices across bandwidths. The PCA fit, sampling,
bandwidth values and unbiased estimator are unchanged.

From a source checkout:

```bash
python -m pip install -e '.[test,benchmark]'
python -m pytest -q
python -m benchmarks.benchmark_mmd --output mmd-local-results.json
```

The benchmark runs each method three times in fresh processes, alternating
order and limiting BLAS to eight threads. It compares the complete MMD
function, including PCA. Peak memory includes interpreter and import overhead.
The frozen baseline is from commit `46d41e63f42a9aab815db20b742feeccd249cb17`.

Recorded Windows / Python 3.13.5 medians in
[`benchmarks/mmd-results.json`](benchmarks/mmd-results.json):

| Input | Original time | Updated time | Original peak | Updated peak |
|---|---:|---:|---:|---:|
| Public T1 sample pair | 80.4 ms | 77.1 ms | 278.6 MiB | 278.6 MiB |
| Public T2 heart sample pair | 20.7 ms | 16.4 ms | 178.4 MiB | 175.6 MiB |
| T2 samples resampled to 2,000 cells each | 486.3 ms | 268.4 ms | 1,099.9 MiB | 250.8 MiB |

All recorded scores are identical between implementations. Tests also compare
float32 / float64, CSR / CSC / dense inputs, unequal cell counts, repeated
cells, custom bandwidths, and the three real public sample pairs.

The 2,000-cell case resamples the supplied 150-cell fixtures for a memory
stress test; it is **not** 2,000 independent biological observations or a new
prediction result. These are MMD measurements, not full scoring-pipeline
speedups. Small inputs are dominated by PCA/import costs and save little memory.

## More

Full task definitions and metric rationale: [virtualembryo.ai](https://virtualembryo.ai) and
[aristoteleo/virtualembryo](https://github.com/aristoteleo/virtualembryo).

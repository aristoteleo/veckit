# Synthetic T2 Conformance Checks

This directory holds a small synthetic-only regression check for the VEC `veckit` scorer.

## What it is for

- Verify the T2 scorer still returns the expected board-anchor metric vector on a fixed synthetic fixture.
- Give contributors a runnable example of a reproducible, Challenge-independent conformance check through the public `veckit.score(...)` API.
- Catch unintended scorer or dependency changes before they reach users.

## What it is not for

- It is not a leaderboard predictor.
- It does not use Challenge validation/test truth.
- It does not change scorer semantics.
- It is intentionally small and synthetic-only.

## How to run

From a clean checkout, install the package and test dependency:

```bash
python -m pip install -e ".[test]"
```

Then run the standalone check:

```bash
python tests/run_t2_synthetic_conformance.py
```

To refresh the frozen JSON after an upstream baseline update:

```bash
python tests/generate_t2_synthetic_expected.py --write
```

If `pytest` is available, you can also run:

```bash
python -m pytest tests/test_t2_synthetic_conformance.py
```

## Notes

- The fixture is synthetic-only and uses fake genes, fake cell types, and deterministic coordinates.
- The test writes temporary `.h5ad` files and runs the public `veckit.score(...)` API.
- The frozen JSON covers the eight T2 board-anchor metrics and records both `veckit==0.1.1` and the upstream commit used to generate it.
- The checks also verify repeatability, `heart`/`embryo` metadata-only behavior, and metric-key partitioning.
- Per-metric tolerances allow one unit at the precision returned by the public scorer.
- These checks do not predict hidden leaderboard scores or establish equivalence with a future server scorer.
- Refresh the frozen JSON only after the upstream baseline settles and the values have been reviewed.

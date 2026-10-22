"""Run the real metric panels on invented data with and without JSON output."""
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import veckit  # noqa: E402


@pytest.mark.parametrize("task", ["T1", "T2", "T3"])
def test_separate_output_preserves_actual_scores(task, tmp_path):
    rng = np.random.default_rng(42)
    labels = np.repeat(["invented_a", "invented_b", "invented_c"], 40)
    X = rng.gamma(2, 1, (120, 40)).astype(np.float32)
    X[labels == "invented_b", :10] += 1
    X[labels == "invented_c", 10:20] += 1
    coords = rng.normal(size=(120, 3)).astype(np.float32)
    paths = {}
    for role, scale in (("reference", 1.0), ("target", 1.1), ("input", 1.05)):
        expression = X.copy()
        if role != "reference":
            expression[:, :12] *= 2.0 if role == "target" else 1.7
            expression[:, 12:24] *= 0.4 if role == "target" else 0.6
        a = ad.AnnData(expression, obs=pd.DataFrame(
            {"celltype": labels}, index=[f"cell_{i}" for i in range(120)]))
        a.var_names = [f"invented_gene_{i}" for i in range(40)]
        a.obsm["spatial_3D"] = coords * scale
        paths[role] = tmp_path / f"{role}.h5ad"
        a.write_h5ad(paths[role])
    before = {role: path.read_bytes() for role, path in paths.items()}
    reference = {"wt" if task == "T3" else "reference": paths["reference"]}
    expected = veckit.score(task, paths["input"], paths["target"], **reference, seed=0)
    out = tmp_path / "results" / "metrics.json"
    actual = veckit.score(task, paths["input"], paths["target"], **reference, seed=0, out=out)
    assert actual == expected
    assert json.loads(out.read_text(encoding="utf-8")) == expected
    assert all(path.read_bytes() == before[role] for role, path in paths.items())

"""Reproduce output/input collisions using disposable, invented data only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import veckit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("vulnerable", "protected"), required=True)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="vec-output-repro-") as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(42)
        labels = np.repeat(["synthetic_a", "synthetic_b", "synthetic_c"], 40)
        X = rng.gamma(2, 1, size=(120, 40)).astype(np.float32)
        X[labels == "synthetic_b", :10] += 1
        X[labels == "synthetic_c", 10:20] += 1
        for interface in ("api", "cli"):
            folder = root / interface
            folder.mkdir()
            for name, values in (("reference", X), ("target", X * 1.1), ("prediction", X * 1.05)):
                a = ad.AnnData(values.copy(), obs=pd.DataFrame(
                    {"celltype": labels}, index=[f"synthetic_cell_{i}" for i in range(len(labels))]))
                a.var_names = [f"synthetic_gene_{i}" for i in range(40)]
                a.write_h5ad(folder / f"{name}.h5ad")
            pred = folder / "prediction.h5ad"
            before = digest(pred)
            error = None
            if interface == "api":
                try:
                    veckit.score("T1", pred, folder / "target.h5ad", reference=folder / "reference.h5ad", out=pred)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            else:
                proc = subprocess.run([sys.executable, "-m", "score_h5ad", "--task", "T1", "--input", str(pred),
                                       "--target", str(folder / "target.h5ad"), "--reference", str(folder / "reference.h5ad"),
                                       "--out", str(pred)], capture_output=True, text=True, cwd=root)
                if proc.returncode:
                    error = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}"
            try:
                ad.read_h5ad(pred)
                readable = True
            except OSError:
                readable = False
            if error:
                error = error.replace(str(root), "<temporary-directory>")
            rows.append({"interface": interface, "sha256_before": before, "sha256_after": digest(pred),
                         "unchanged": before == digest(pred), "h5ad_readable": readable, "error": error})
    print(json.dumps({"python": sys.version.split()[0],
                      "packages": {m: importlib.metadata.version(m) for m in ("veckit", "anndata", "numpy", "scipy", "scikit-learn")},
                      "results": rows}, indent=2))
    if args.expect == "protected":
        ok = all(row["unchanged"] and row["h5ad_readable"] and
                 row["error"] and row["error"].startswith("ValueError: Output path") for row in rows)
    else:
        ok = all(not row["unchanged"] and not row["h5ad_readable"] and row["error"] is None for row in rows)
    if not ok:
        raise SystemExit(f"Observed behavior did not match --expect {args.expect}")


if __name__ == "__main__":
    main()

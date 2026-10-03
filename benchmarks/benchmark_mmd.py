"""Compare full MMD wall time and process peak memory in fresh subprocesses.

Run from the repository root:
    python -m benchmarks.benchmark_mmd --output benchmarks/mmd-results.json
The 2,000-cell case is a resampling stress test, not new biological evidence.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "T1_public": ("sample_8.5.h5ad", "sample_9.5.h5ad"),
    "T2_public": ("sample_heart_9.25.h5ad", "sample_heart_9.5.h5ad"),
    "T2_resampled_2000": ("sample_heart_9.25.h5ad", "sample_heart_9.5.h5ad"),
}


def worker(case, arm):
    import anndata as ad
    import numpy as np
    import psutil
    from scipy import sparse
    from sklearn.decomposition import PCA  # Exclude one-time lazy imports from timing.
    from sklearn.metrics.pairwise import rbf_kernel
    from threadpoolctl import threadpool_limits
    from benchmarks.mmd_reference import mmd_reference
    from common.core_metrics import mmd_unbiased

    a, b = [ad.read_h5ad(ROOT / "data" / name) for name in PAIRS[case]]
    genes = a.var_names.intersection(b.var_names, sort=False)
    arrays = [v[:, genes].X for v in (a, b)]
    arrays = [v.toarray() if sparse.issparse(v) else np.asarray(v) for v in arrays]
    original_shapes = [v.shape for v in arrays]
    if case == "T2_resampled_2000":
        rng = np.random.default_rng(61027)
        arrays = [v[rng.choice(len(v), 2000, replace=True)] for v in arrays]
    fn = mmd_reference if arm == "baseline" else mmd_unbiased
    with threadpool_limits(limits=8):
        start = time.perf_counter()
        score = fn(*arrays)
        elapsed = time.perf_counter() - start
    mem = psutil.Process().memory_info()
    if hasattr(mem, "peak_wset"):
        peak = mem.peak_wset
        memory_kind = "Windows peak process working set"
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
        memory_kind = "ru_maxrss process peak"
    return {"case": case, "arm": arm, "score": score, "seconds": elapsed,
            "peak_memory_bytes": int(peak), "memory_measurement": memory_kind,
            "original_shapes": original_shapes, "evaluated_shapes": [v.shape for v in arrays],
            "dtype": str(arrays[0].dtype), "threads": 8,
            "data_sha256": {name: hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
                            for name in PAIRS[case]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", choices=PAIRS)
    p.add_argument("--arm", choices=["baseline", "candidate"])
    p.add_argument("--output", default="benchmarks/mmd-results.json")
    a = p.parse_args()
    if a.worker:
        print(json.dumps(worker(a.worker, a.arm)))
        return
    output = Path(a.output)
    if output.exists():
        raise FileExistsError("Use a different output path; keep previous measurements.")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    result = {"started_at": datetime.datetime.now().astimezone().isoformat(),
              "platform": platform.platform(), "python": platform.python_version(),
              "baseline_commit": "46d41e63f42a9aab815db20b742feeccd249cb17",
              "source_sha256": hashlib.sha256((ROOT / "common/core_metrics.py").read_bytes()).hexdigest(),
              "source_lf_sha256": hashlib.sha256((ROOT / "common/core_metrics.py").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
              "versions": {p: importlib.metadata.version(p) for p in
                           ["numpy", "scipy", "scikit-learn", "anndata", "psutil", "threadpoolctl"]},
              "note": "Fresh process per run; full MMD time, process-wide peak. Stress resamples real public T2 cells, not independent observations.",
              "runs": []}
    wall_start = time.monotonic()
    for repeat in range(3):
        for case in PAIRS:
            order = ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
            for arm in order:
                if time.monotonic() - wall_start > 600:
                    raise TimeoutError("Ten-minute benchmark cap reached")
                proc = subprocess.run([sys.executable, "-X", "utf8", "-m", "benchmarks.benchmark_mmd",
                                       "--worker", case, "--arm", arm], cwd=ROOT, env=env,
                                      check=True, capture_output=True, text=True, encoding="utf-8", timeout=90)
                row = json.loads(proc.stdout)
                row["repeat"] = repeat
                result["runs"].append(row)
                output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({k: row[k] for k in ["case", "arm", "repeat", "score", "seconds", "peak_memory_bytes"]}), flush=True)
    summary = {}
    for case in PAIRS:
        rows = {arm: [r for r in result["runs"] if r["case"] == case and r["arm"] == arm]
                for arm in ["baseline", "candidate"]}
        med = {arm: {k: statistics.median(r[k] for r in data)
                     for k in ["seconds", "peak_memory_bytes", "score"]} for arm, data in rows.items()}
        summary[case] = {"medians": med,
                         "speedup": med["baseline"]["seconds"] / med["candidate"]["seconds"],
                         "memory_reduction": 1 - med["candidate"]["peak_memory_bytes"] / med["baseline"]["peak_memory_bytes"],
                         "absolute_score_difference": abs(med["candidate"]["score"] - med["baseline"]["score"])}
    result["summary"] = summary
    result["gate"] = "All score differences <=1e-7; stress peak reduction>=60%; no case >10% slower."
    result["passed"] = all(s["absolute_score_difference"] <= 1e-7 and s["speedup"] >= 1/1.1 for s in summary.values()) and summary["T2_resampled_2000"]["memory_reduction"] >= .6
    result["wall_seconds"] = time.monotonic() - wall_start
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate and score the eight public VEC adversarial controls on synthetic fixtures."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.synthetic_controls.controls import (  # noqa: E402
    AUDIT_KEY,
    BASELINES_URL,
    ctrl_one_cell,
    ctrl_random_cube,
    ctrl_random_dir,
    ctrl_scale_ref,
    ctrl_scale_wt,
    ctrl_shrink_ref,
    ctrl_shrink_wt,
    ctrl_squashed_ref,
)
from examples.synthetic_controls.fixtures import FIXTURE_SEED, make_synthetic_fixtures  # noqa: E402
from veckit import score as veckit_score  # noqa: E402


def _dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _fixture_hash(a) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(_dense(a.X)).tobytes())
    digest.update("\0".join(map(str, a.var_names)).encode("utf-8"))
    if "celltype" in a.obs:
        digest.update("\0".join(map(str, a.obs["celltype"])).encode("utf-8"))
    if "spatial_3D" in a.obsm:
        digest.update(np.ascontiguousarray(a.obsm["spatial_3D"]).tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_control_outputs(fixtures, seed: int) -> list[tuple[str, str, object]]:
    t1, t2, t3 = fixtures["T1"], fixtures["T2"], fixtures["T3"]
    return [
        ("ctrl_scale_ref", "T1", ctrl_scale_ref(t1["reference"])),
        ("ctrl_scale_ref", "T2", ctrl_scale_ref(t2["reference"])),
        ("ctrl_shrink_ref", "T1", ctrl_shrink_ref(t1["reference"])),
        ("ctrl_one_cell", "T1", ctrl_one_cell(t1["reference"], t1["target"].n_obs)),
        (
            "ctrl_random_cube",
            "T2",
            ctrl_random_cube(t2["reference"], synthetic_target=t2["target"], seed=seed),
        ),
        ("ctrl_squashed_ref", "T2", ctrl_squashed_ref(t2["reference"])),
        ("ctrl_scale_wt", "T3", ctrl_scale_wt(t3["wt"])),
        ("ctrl_shrink_wt", "T3", ctrl_shrink_wt(t3["wt"])),
        ("ctrl_random_dir", "T3", ctrl_random_dir(t3["wt"], t3["target"], seed=seed)),
    ]


def _selected(task: str, selection: str) -> bool:
    return selection == "all" or selection == task


def _score_control(name, task, output, fixtures, directory: Path, score_seed: int) -> dict:
    task_fixture = fixtures[task]
    input_path = directory / f"{task}_{name}.h5ad"
    target_path = directory / f"{task}_target.h5ad"
    output.write_h5ad(input_path)
    if not target_path.exists():
        task_fixture["target"].write_h5ad(target_path)

    kwargs = {"task": task, "input": input_path, "target": target_path, "seed": score_seed}
    if task in ("T1", "T2"):
        reference_path = directory / f"{task}_reference.h5ad"
        if not reference_path.exists():
            task_fixture["reference"].write_h5ad(reference_path)
        kwargs["reference"] = reference_path
        if task == "T2":
            kwargs["setting"] = "heart"
    else:
        wt_path = directory / "T3_wt.h5ad"
        if not wt_path.exists():
            task_fixture["wt"].write_h5ad(wt_path)
        kwargs["wt"] = wt_path

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'n_jobs' has no effect.*", category=FutureWarning)
        return veckit_score(**kwargs)


def _formula_check(name: str, task: str, output, fixtures, control_seed: int) -> None:
    source = fixtures[task]["wt" if task == "T3" else "reference"]
    actual_X = _dense(output.X)
    source_X = _dense(source.X)

    if name in ("ctrl_scale_ref", "ctrl_scale_wt"):
        np.testing.assert_allclose(actual_X, source_X * 2.0)
    elif name == "ctrl_shrink_ref":
        np.testing.assert_allclose(actual_X, source_X * 0.99)
    elif name == "ctrl_shrink_wt":
        np.testing.assert_allclose(actual_X, source_X * 0.75)
    elif name == "ctrl_one_cell":
        mean_cell = source_X.mean(axis=0)
        np.testing.assert_allclose(actual_X, np.repeat(mean_cell[None, :], output.n_obs, axis=0))
    elif name == "ctrl_random_cube":
        target_coords = np.asarray(fixtures[task]["target"].obsm["spatial_3D"], dtype=float)
        bounds = np.column_stack([target_coords.min(axis=0), target_coords.max(axis=0)])
        expected = np.random.Generator(np.random.PCG64(control_seed)).uniform(
            bounds[:, 0], bounds[:, 1], size=(source.n_obs, 3)
        )
        np.testing.assert_array_equal(output.obsm["spatial_3D"], expected)
        np.testing.assert_array_equal(actual_X, source_X)
    elif name == "ctrl_squashed_ref":
        coords = np.asarray(source.obsm["spatial_3D"], dtype=float)
        center = coords.mean(axis=0)
        expected = (coords - center) * np.array([4.0, 1.0, 0.25]) + center
        np.testing.assert_allclose(output.obsm["spatial_3D"], expected)
        np.testing.assert_array_equal(actual_X, source_X)
    elif name == "ctrl_random_dir":
        target_X = _dense(fixtures[task]["target"].X).astype(float)
        wt_X = source_X.astype(float)
        target_shift = target_X.mean(axis=0) - wt_X.mean(axis=0)
        random_shift = np.random.Generator(np.random.PCG64(control_seed)).normal(size=source.n_vars)
        random_shift *= np.linalg.norm(target_shift) / np.linalg.norm(random_shift)
        np.testing.assert_allclose(np.linalg.norm(random_shift), np.linalg.norm(target_shift))
        np.testing.assert_allclose(actual_X, np.clip(wt_X + random_shift[None, :], 0.0, None))
    else:
        raise AssertionError(f"no formula check registered for {task}/{name}")

    if not np.isfinite(actual_X).all() or (actual_X < 0).any():
        raise AssertionError(f"{task}/{name} produced invalid expression")
    if list(output.var_names) != list(source.var_names):
        raise AssertionError(f"{task}/{name} changed the gene panel or order")


def build_manifest(
    *,
    task: str = "all",
    fixture_seed: int = FIXTURE_SEED,
    control_seed: int = 1101,
    score_seed: int = 0,
    artifact_dir: Path | None = None,
) -> dict:
    fixtures = make_synthetic_fixtures(fixture_seed)
    outputs = [row for row in make_control_outputs(fixtures, control_seed) if _selected(row[1], task)]

    manifest = {
        "schema_version": 1,
        "title": "Executable synthetic adversarial controls for veckit",
        "synthetic_only": True,
        "scope": "public-description-only; not bake.py-equivalent; not leaderboard-predictive",
        "public_source": BASELINES_URL,
        "versions": {
            "veckit": importlib.metadata.version("veckit"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "anndata": importlib.metadata.version("anndata"),
            "git_commit": _git_commit(),
        },
        "seeds": {
            "fixture": int(fixture_seed),
            "control": int(control_seed),
            "score": int(score_seed),
        },
        "fixture_hashes": {
            task_name: {name: _fixture_hash(value) for name, value in values.items()}
            for task_name, values in fixtures.items()
        },
        "controls": [],
    }

    with tempfile.TemporaryDirectory() as temporary:
        score_dir = Path(temporary)
        for name, control_task, output in outputs:
            try:
                _formula_check(name, control_task, output, fixtures, control_seed)
            except AssertionError as error:
                raise AssertionError(f"{control_task}/{name} formula check failed: {error}") from error
            try:
                result = _score_control(name, control_task, output, fixtures, score_dir, score_seed)
            except Exception as error:
                raise RuntimeError(f"{control_task}/{name} public scoring failed: {error}") from error
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                output.write_h5ad(artifact_dir / f"{control_task}_{name}.synthetic.h5ad")
            manifest["controls"].append(
                {
                    "name": name,
                    "task": control_task,
                    "shape": [int(output.n_obs), int(output.n_vars)],
                    "formula_check": "PASS",
                    "audit": output.uns[AUDIT_KEY],
                    "metrics": result["metrics"],
                }
            )
    return manifest


def _summary_metrics(task: str, metrics: dict) -> str:
    keys = {
        "T1": ("de_score", "de_direction", "variance_ratio"),
        "T2": ("d2_shape", "occupancy_dice", "scale_log_ratio", "neighborhood_mmd"),
        "T3": ("de_direction", "severity_slope", "variance_ratio"),
    }[task]
    return ", ".join(f"{key}={metrics.get(key)}" for key in keys)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "T1", "T2", "T3"), default="all")
    parser.add_argument("--fixture-seed", type=int, default=FIXTURE_SEED)
    parser.add_argument("--seed", type=int, default=1101, help="control RNG seed")
    parser.add_argument("--score-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write the machine-readable manifest JSON")
    parser.add_argument("--write-artifacts", type=Path, help="persist generated synthetic h5ad files")
    args = parser.parse_args(argv)

    manifest = build_manifest(
        task=args.task,
        fixture_seed=args.fixture_seed,
        control_seed=args.seed,
        score_seed=args.score_seed,
        artifact_dir=args.write_artifacts,
    )
    for control in manifest["controls"]:
        print(
            f"[PASS] {control['task']} {control['name']} shape={tuple(control['shape'])} "
            f"{_summary_metrics(control['task'], control['metrics'])}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Manifest: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

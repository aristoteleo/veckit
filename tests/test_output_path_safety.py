"""An output typo must not overwrite the H5AD files being scored.

All fixtures are invented. Fast tests stub only the expensive metric functions;
the CLI argument parser, Python API and actual result-file writer run normally.
"""
import json
import os
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import score_h5ad as scorer  # noqa: E402
import veckit  # noqa: E402


RESULT = {"meta": {"task": "test"}, "metrics": {"example": 0.25}}


@pytest.fixture
def inputs(tmp_path):
    files = {}
    for role in ("input", "target", "reference", "wt"):
        path = tmp_path / f"{role}.h5ad"
        ad.AnnData(np.ones((4, 3), dtype=np.float32)).write_h5ad(path)
        files[role] = path
    return files


def invoke(interface, files, out, monkeypatch, task="T1"):
    if interface == "api":
        return veckit.score(task=task, **files, out=out)
    argv = ["veckit", "--task", task]
    for key, value in {**files, "out": out}.items():
        if value is not None:
            argv.extend([f"--{key}", str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    scorer.main()


def stub_metrics(monkeypatch, function):
    for task in ("t1", "t2", "t3"):
        monkeypatch.setattr(scorer, f"score_{task}", function)


@pytest.mark.parametrize("interface", ["api", "cli"])
@pytest.mark.parametrize("role", ["input", "target", "reference", "wt"])
@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
def test_rejects_input_alias_before_scoring(interface, role, alias, inputs, tmp_path, monkeypatch):
    protected = inputs[role]
    if alias == "same":
        out = protected
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        out = Path(protected.name)
    else:
        out = tmp_path / "report.json"
        try:
            if alias == "symlink":
                out.symlink_to(protected)
            else:
                os.link(protected, out)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"{alias} unavailable: {exc}")
    before = {key: path.read_bytes() for key, path in inputs.items()}
    calls = []

    def metrics(args):
        calls.append(args)
        return RESULT["meta"], RESULT["metrics"]

    stub_metrics(monkeypatch, metrics)
    with pytest.raises(ValueError, match="Output path.*--" + role):
        invoke(interface, inputs, out, monkeypatch)
    assert calls == [], "The path error must be caught before expensive scoring"
    for key, path in inputs.items():
        assert path.read_bytes() == before[key]
        assert ad.read_h5ad(path).shape == (4, 3)


@pytest.mark.parametrize("interface", ["api", "cli"])
@pytest.mark.parametrize("task", ["T1", "T2", "T3"])
@pytest.mark.parametrize("existing", [False, True])
def test_distinct_json_output_is_written(interface, task, existing, inputs, tmp_path, monkeypatch):
    out = tmp_path / "results" / "score.json"
    if existing:
        out.parent.mkdir()
        out.write_text('{"previous": true}\n', encoding="utf-8")
    before = {key: path.read_bytes() for key, path in inputs.items()}
    stub_metrics(monkeypatch, lambda args: (RESULT["meta"], RESULT["metrics"]))
    result = invoke(interface, inputs, out, monkeypatch, task=task)
    assert json.loads(out.read_text(encoding="utf-8")) == RESULT
    if interface == "api":
        assert result == RESULT
    assert all(path.read_bytes() == before[key] for key, path in inputs.items())


@pytest.mark.parametrize("interface", ["api", "cli"])
def test_no_output_path_keeps_return_or_stdout(interface, inputs, monkeypatch, capsys):
    stub_metrics(monkeypatch, lambda args: (RESULT["meta"], RESULT["metrics"]))
    result = invoke(interface, inputs, None, monkeypatch)
    if interface == "api":
        assert result == RESULT
    else:
        assert json.loads(capsys.readouterr().out) == RESULT


@pytest.mark.parametrize("interface", ["api", "cli"])
def test_default_reference_still_protects_input(interface, inputs, monkeypatch):
    files = {"input": inputs["input"], "target": inputs["target"]}
    stub_metrics(monkeypatch, lambda args: (RESULT["meta"], RESULT["metrics"]))
    before = inputs["input"].read_bytes()
    with pytest.raises(ValueError, match="Output path.*--input"):
        invoke(interface, files, inputs["input"], monkeypatch)
    assert inputs["input"].read_bytes() == before


@pytest.mark.parametrize("interface", ["api", "cli"])
def test_rechecks_alias_created_while_scoring(interface, inputs, tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    before = inputs["target"].read_bytes()

    def metrics(args):
        try:
            os.link(inputs["target"], out)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"hardlink unavailable: {exc}")
        return RESULT["meta"], RESULT["metrics"]

    stub_metrics(monkeypatch, metrics)
    with pytest.raises(ValueError, match="Output path.*--target"):
        invoke(interface, inputs, out, monkeypatch)
    assert inputs["target"].read_bytes() == before

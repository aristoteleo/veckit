"""Interruptibility regression tests (Linux/SIGINT/SIGTERM).

Guarantees that Ctrl+C (SIGINT) and SIGTERM terminate the CLI promptly (within a few seconds) with the
conventional exit codes (130 / 143), even while the process is inside a heavy native compute phase, and that
no worker child process is left behind. Uses a moderately large synthetic fixture so the frozen-probe
LogisticRegression / metric compute take a few seconds to be a realistic interrupt target.
"""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

from _fixture import make_t1_fixture                       # noqa: E402


@pytest.fixture(scope="module")
def big_fixture(tmp_path_factory):
    """Write a moderately large T1 fixture (sparse) so compute is non-trivial but fast to produce."""
    tmp = tmp_path_factory.mktemp("big")
    import anndata as ad
    from scipy import sparse
    ref, true, pred, ct_r, ct_t, ct_p, genes = make_t1_fixture(
        seed=3, cells_ref=1800, cells_true=2000, cells_pred=2000, genes=12000)
    vn = {"gene": list(genes)}
    for name, X, ct in (("ref", ref, ct_r), ("true", true, ct_t), ("pred", pred, ct_p)):
        a = ad.AnnData(X=sparse.csr_matrix(X), obs={"celltype": ct}, var=vn)
        a.write(tmp / f"{name}.h5ad")
    return tmp


def _run_cli(tmp, env):
    veckit = os.environ.get("VECKIT_BIN") or "veckit"
    return subprocess.Popen(
        [veckit, "--task", "T1",
         "--input", str(tmp / "pred.h5ad"),
         "--target", str(tmp / "true.h5ad"),
         "--reference", str(tmp / "ref.h5ad"),
         "--seed", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def _wait_stage(proc, needle, timeout=300):
    """Read the child's stderr until it logs `needle`, confirming we are inside a compute phase."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        line = proc.stderr.readline()
        if line:
            sys.stdout.write("[child] " + line)
            if needle in line:
                return True
    return False


def _orphan_veckit_children():
    import re
    out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "veckit" in l or "score_h5ad" in l]


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="no SIGINT on this platform")
def test_sigint_prompt_exit_130(big_fixture):
    env = dict(os.environ)
    proc = _run_cli(big_fixture, env)
    # wait until we're inside the heavy metric compute worker, then interrupt
    assert _wait_stage(proc, "computing metrics...", 300), "scorer never reached compute phase"
    t0 = time.time()
    proc.send_signal(signal.SIGINT)
    try:
        code = proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
        pytest.fail("CLI did not exit within 25s of SIGINT (still stuck in native call)")
    elapsed = time.time() - t0
    assert code == 130, f"expected exit code 130, got {code}"
    assert elapsed < 20, f"SIGINT took too long: {elapsed:.1f}s"
    # no leftover workers
    for l in _orphan_veckit_children():
        pytest.fail(f"orphan process left behind: {l}")


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="no SIGTERM on this platform")
def test_sigterm_prompt_exit_143(big_fixture):
    env = dict(os.environ)
    proc = _run_cli(big_fixture, env)
    assert _wait_stage(proc, "computing metrics...", 300), "scorer never reached compute phase"
    t0 = time.time()
    proc.send_signal(signal.SIGTERM)
    try:
        code = proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
        pytest.fail("CLI did not exit within 25s of SIGTERM")
    elapsed = time.time() - t0
    assert code == 143, f"expected exit code 143, got {code}"
    assert elapsed < 20, f"SIGTERM took too long: {elapsed:.1f}s"


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="no SIGINT on this platform")
def test_worker_mechanism_sigint(tmp_path):
    """Direct test of the `_run_compute` worker isolation: SIGINT terminates a long worker promptly."""
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, str(TESTS / "_worker_sleep.py")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    time.sleep(1.5)          # let it enter _run_compute's poll loop (child forked + sleeping)
    t0 = time.time()
    proc.send_signal(signal.SIGINT)
    try:
        code = proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
        pytest.fail("_run_compute worker did not exit promptly on SIGINT")
    elapsed = time.time() - t0
    assert code == 130, f"expected 130, got {code}"
    assert elapsed < 12, f"worker SIGINT took too long: {elapsed:.1f}s"

"""Standalone child process that runs the refactor's worker-isolation mechanism (`_run_compute`) on a long
native-ish call, so a parent test can send SIGINT and verify the worker is terminated promptly and the
process exits with code 130 (matching the CLI contract)."""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score_h5ad as _s


def _long_call(seconds):
    # stand-in for an uninterruptible native (sklearn/BLAS) call
    time.sleep(seconds)
    return "completed"


def main():
    try:
        result = _s._run_compute(_long_call, 3600)
        print(result)
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

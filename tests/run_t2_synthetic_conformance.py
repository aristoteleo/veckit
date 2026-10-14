from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_t2_synthetic_conformance as _t2  # noqa: E402


def main() -> int:
    _t2.test_t2_synthetic_conformance_v011()
    _t2.test_t2_public_api_repeatable()
    _t2.test_t2_setting_is_metadata_only()
    _t2.test_t2_board_anchor_partition()
    print("T2 synthetic conformance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

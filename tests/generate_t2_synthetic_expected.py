from __future__ import annotations

import argparse
import json
from pathlib import Path

from test_t2_synthetic_conformance import (
    EXPECTED_PATH,
    FIXTURE_SEED,
    UPSTREAM_COMMIT,
    make_t2_fixture,
    score_t2_public_api,
)


def build_expected() -> dict:
    fixture = make_t2_fixture(seed=FIXTURE_SEED)
    metrics = score_t2_public_api(fixture, score_seed=0)
    keys = (
        "d2_shape",
        "de_direction",
        "de_score",
        "mmd_u",
        "neighborhood_mmd",
        "occupancy_dice",
        "scale_log_ratio",
        "variogram",
    )
    # The public scorer rounds these outputs before returning them. Allow one
    # unit at each metric's published precision for platform-level drift.
    tolerances = {
        "d2_shape": 1e-5,
        "de_direction": 1e-4,
        "de_score": 1e-4,
        "mmd_u": 1e-5,
        "neighborhood_mmd": 1e-5,
        "occupancy_dice": 1e-4,
        "scale_log_ratio": 1e-4,
        "variogram": 1e-6,
    }
    return {
        "version": "veckit==0.1.1",
        "upstream_commit": UPSTREAM_COMMIT,
        "source": "synthetic-only",
        "fixture_seed": FIXTURE_SEED,
        "score_seed": 0,
        "fixture": "asymmetric_t2_small",
        "metrics": {k: metrics[k] for k in keys},
        "tolerances": tolerances,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the frozen synthetic T2 expected JSON.")
    ap.add_argument("--write", action="store_true", help="overwrite the checked-in expected JSON")
    args = ap.parse_args()

    payload = build_expected()
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.write:
        EXPECTED_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {EXPECTED_PATH}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

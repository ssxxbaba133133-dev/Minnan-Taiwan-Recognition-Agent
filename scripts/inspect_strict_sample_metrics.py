# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print CSV metrics for a contact-sheet sample JSON.")
    parser.add_argument("--sample-json", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    sample = json.loads(Path(args.sample_json).read_text(encoding="utf-8"))
    rows = {}
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row.get("output_image") or row.get("input_image")] = row

    metric_keys = [
        "body_area_ratio",
        "body_width_ratio",
        "body_height_ratio",
        "body_center_y",
        "body_bottom_ratio",
        "body_centrality",
        "roof_area_ratio",
        "roof_width_ratio",
        "roof_height_ratio",
        "roof_center_y",
        "roof_overlap_with_body",
        "strict_score",
        "sharpness",
    ]
    for idx, item in enumerate(sample, 1):
        row = rows.get(item)
        if row is None:
            print(f"{idx:02d} MISSING {Path(item).name}")
            continue
        parts = []
        for key in metric_keys:
            value = row.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={float(value):.3f}")
        print(f"{idx:02d} {Path(item).name}")
        print("    " + " ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

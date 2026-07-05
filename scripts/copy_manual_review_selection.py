# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy selected images from a contact-sheet manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude", default="", help="Comma-separated 1-based indices to exclude.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude = {int(x.strip()) for x in args.exclude.split(",") if x.strip()}

    kept = []
    rejected = []
    for item in manifest:
        idx = int(item["index"])
        src = Path(item["path"])
        if idx in exclude:
            rejected.append(item)
            continue
        dst = output_dir / src.name
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            n = 2
            while True:
                candidate = output_dir / f"{stem}_{n}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                n += 1
        shutil.copy2(src, dst)
        kept_item = dict(item)
        kept_item["output_image"] = str(dst)
        kept.append(kept_item)

    summary = {
        "manifest": str(Path(args.manifest)),
        "output_dir": str(output_dir),
        "total": len(manifest),
        "kept": len(kept),
        "rejected": len(rejected),
        "exclude": sorted(exclude),
    }
    (output_dir / "manual_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manual_review_kept.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manual_review_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

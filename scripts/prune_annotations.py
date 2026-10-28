#!/usr/bin/env python3
"""Remove annotation entries whose image file no longer exists.

Usage:
    python scripts/prune_annotations.py --session S01            # dry run: only reports
    python scripts/prune_annotations.py --session S01 --apply    # rewrite JSON (backup kept)

The previous JSON is copied to <session>.json.prune_<timestamp>.bak before rewriting.
Nothing else is changed: boxes, labels and the 'reviewed' flags of kept frames are untouched.
Exit codes: 0 ok, 2 bad input / refusing to leave an empty file.
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import data_root, load_config  # noqa: E402


def prune(ann_path, root, apply=False):
    ann_path, root = Path(ann_path), Path(root)
    doc = json.loads(ann_path.read_text(encoding="utf-8"))
    keep, removed = [], []
    for f in doc.get("frames", []):
        ok = isinstance(f, dict) and isinstance(f.get("file"), str) and (root / f["file"]).is_file()
        (keep if ok else removed).append(f)
    counts = Counter(b.get("label") for f in keep for b in f.get("boxes", []))
    removed_boxes = sum(len(f.get("boxes", [])) for f in removed if isinstance(f, dict))
    info = {"kept": len(keep), "removed": len(removed), "removed_boxes": removed_boxes,
            "kept_boxes_per_class": dict(counts), "backup": None}
    if apply and removed:
        if not keep:
            raise ValueError("no frame would remain; refusing to write an empty annotation file")
        backup = ann_path.with_name(f"{ann_path.name}.prune_{time.strftime('%Y%m%d_%H%M%S')}.bak")
        shutil.copyfile(ann_path, backup)
        doc["frames"] = keep
        tmp = ann_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, ann_path)
        info["backup"] = str(backup)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = data_root(load_config())
    path = root / "annotations" / f"{args.session}.json"
    if not path.is_file():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2
    try:
        info = prune(path, root, apply=args.apply)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"frames kept: {info['kept']} | frames without image: {info['removed']} "
          f"(holding {info['removed_boxes']} boxes)")
    print(f"boxes per class in kept frames: {info['kept_boxes_per_class']}")
    if info["removed"] == 0:
        print("nothing to prune")
    elif args.apply:
        print(f"DONE. backup: {info['backup']}")
    else:
        print("DRY RUN: nothing written. Re-run with --apply to remove those entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

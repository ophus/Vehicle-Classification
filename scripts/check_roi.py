#!/usr/bin/env python3
"""Draw the ROI polygon and your labeled boxes to PNG files: boxes whose centre is inside = green, outside = red.

Usage:
    python scripts/check_roi.py --session S01                 # 3 frames with the most outside boxes
    python scripts/check_roi.py --session S01 --frames 5 --roi-file data/roi.json
Output: artifacts/figures/roi_check_<frame_id>.png and a printed summary (counts, where the outside boxes are).
No window is opened. Open the PNGs and look: if many RED boxes are in an area you did label, the polygon is too small
(redo it: python scripts/pick_roi.py --session S01 --overwrite).
Exit codes: 0 ok, 2 error.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import artifacts_dir, data_root, load_config  # noqa: E402
from common.io import imread_unicode, imwrite_unicode  # noqa: E402


def centre_inside(poly, bbox):
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def roi_report(frames, poly):
    ins, outs = [], []
    per_class = Counter()
    per_frame = {}
    for f in frames:
        n_out = 0
        for b in f.get("boxes", []):
            if centre_inside(poly, b["bbox"]):
                ins.append(b)
            else:
                outs.append(b)
                per_class[b["label"]] += 1
                n_out += 1
        per_frame[f["frame_id"]] = n_out
    cy = [(b["bbox"][1] + b["bbox"][3]) / 2 for b in outs]
    hh = lambda bs: float(np.median([b["bbox"][3] - b["bbox"][1] for b in bs])) if bs else None  # noqa: E731
    return {"total": len(ins) + len(outs), "outside": len(outs), "outside_per_class": dict(per_class),
            "outside_centre_y": [min(cy), float(np.median(cy)), max(cy)] if cy else None,
            "median_height_inside": hh(ins), "median_height_outside": hh(outs), "outside_per_frame": per_frame}


def draw(img, poly, boxes):
    out = img.copy()
    cv2.polylines(out, [poly.astype(np.int32)], True, (0, 255, 255), 2)
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        col = (0, 200, 0) if centre_inside(poly, b["bbox"]) else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--roi-file")
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config()
    root = data_root(cfg)
    try:
        roi = json.loads(Path(args.roi_file or root / "roi.json").read_text(encoding="utf-8"))
        poly = np.array(roi["polygon"], dtype=np.float32).reshape(-1, 1, 2)
        doc = json.loads((root / "annotations" / f"{args.session}.json").read_text(encoding="utf-8"))
        if doc["image_size"] != roi["image_size"]:
            raise ValueError(f"ROI size {roi['image_size']} != session size {doc['image_size']}")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    frames = [f for f in doc["frames"] if f.get("reviewed", True)]
    rep = roi_report(frames, poly)
    pct = 100.0 * rep["outside"] / max(1, rep["total"])
    print(f"boxes: {rep['total']} | centre OUTSIDE roi: {rep['outside']} ({pct:.0f}%) | by class: {rep['outside_per_class']}")
    if rep["outside"]:
        lo, med, hi = rep["outside_centre_y"]
        print(f"outside boxes: centre y min/median/max = {lo:.0f}/{med:.0f}/{hi:.0f} px | median height "
              f"outside {rep['median_height_outside']:.0f}px vs inside "
              + (f"{rep['median_height_inside']:.0f}px" if rep["median_height_inside"] is not None else "n/a"))
    out_dir = artifacts_dir(cfg) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(frames, key=lambda f: -rep["outside_per_frame"][f["frame_id"]])[:args.frames]
    for f in worst:
        img = imread_unicode(root / f["file"])
        if img is None:
            print(f"cannot read {f['file']}", file=sys.stderr)
            continue
        p = out_dir / f"roi_check_{f['frame_id']}.png"
        imwrite_unicode(p, draw(img, poly, f.get("boxes", [])), ".png")
        print(f"saved {p}  ({rep['outside_per_frame'][f['frame_id']]} boxes outside)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

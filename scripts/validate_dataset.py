#!/usr/bin/env python3
"""Validate annotation JSON files under <data_root>/annotations/.

Usage:
    python scripts/validate_dataset.py                 # structure + files exist
    python scripts/validate_dataset.py --deep          # also decode images: size check, corrupt, duplicates
    python scripts/validate_dataset.py --report artifacts/metrics/dataset_report.json

Annotation file (one per session, <session_id>.json):
{
  "session_id": "S01",
  "image_size": [1920, 1080],                      # [width, height]
  "frames": [
    {"frame_id": "S01_000120",
     "file": "processed/frames/S01/S01_000120.jpg",   # relative to data_root
     "boxes": [
       {"bbox": [x1, y1, x2, y2],                     # pixels, x1<x2, y1<y2
        "label": "motorbike|car|truck|unknown",
        "fine_label": "pickup",                       # optional, must map to `label` (see configs)
        "occlusion": 0,                               # optional 0/1/2
        "truncated": 0}]}]                            # optional 0/1
}
Checks: JSON/schema, label set, fine_label->label consistency, bbox validity and bounds,
tiny boxes (warning), near-identical boxes in a frame (warning), duplicate frame_id,
missing files, and with --deep: corrupt images, size mismatch, byte-identical duplicate images.
Exit codes: 0 no errors (warnings allowed), 1 errors found, 2 nothing to validate.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import data_root, load_config  # noqa: E402


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_file(ann_path, cfg, root, deep=False, global_hashes=None):
    ann_path, root = Path(ann_path), Path(root)
    global_hashes = {} if global_hashes is None else global_hashes
    res = {"session": ann_path.stem, "errors": [], "warnings": [], "frames": 0,
           "boxes": 0, "counts": Counter(), "fine_counts": Counter()}
    E, W = res["errors"], res["warnings"]

    try:
        doc = json.loads(ann_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        E.append(f"cannot read JSON: {exc}")
        return res
    if not isinstance(doc, dict):
        E.append("top level must be an object")
        return res
    for key in ("session_id", "image_size", "frames"):
        if key not in doc:
            E.append(f"missing top-level key '{key}'")
    if E:
        return res

    sid = doc["session_id"]
    if sid != ann_path.stem:
        W.append(f"session_id '{sid}' differs from file name '{ann_path.stem}'")
    size = doc["image_size"]
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size)):
        E.append("image_size must be [width, height] positive ints")
        return res
    W_img, H_img = size
    if not isinstance(doc["frames"], list):
        E.append("'frames' must be a list")
        return res

    classes = set(cfg["classes"])
    f2c = cfg["fine_to_class"]
    min_px = cfg["annotation"]["min_box_px"]
    tol = cfg["annotation"]["bbox_tolerance_px"]
    seen_ids = set()

    for i, fr in enumerate(doc["frames"]):
        tag = f"{sid}[{i}]"
        if not isinstance(fr, dict):
            E.append(f"{tag}: frame must be an object")
            continue
        res["frames"] += 1
        fid, rel, boxes = fr.get("frame_id"), fr.get("file"), fr.get("boxes")
        if not isinstance(fid, str) or not fid:
            E.append(f"{tag}: missing frame_id")
        elif fid in seen_ids:
            E.append(f"{tag}: duplicate frame_id '{fid}'")
        else:
            seen_ids.add(fid)
        tag = f"{sid}:{fid or i}"
        if not isinstance(rel, str) or not rel:
            E.append(f"{tag}: missing 'file'")
        else:
            path = root / rel
            if not path.is_file():
                E.append(f"{tag}: image file not found: {rel}")
            elif deep:
                raw = path.read_bytes()
                digest = hashlib.md5(raw).hexdigest()
                if digest in global_hashes:
                    W.append(f"{tag}: byte-identical to {global_hashes[digest]}")
                else:
                    global_hashes[digest] = f"{sid}:{fid}"
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    E.append(f"{tag}: image cannot be decoded (corrupt?)")
                elif (img.shape[1], img.shape[0]) != (W_img, H_img):
                    E.append(f"{tag}: image size {img.shape[1]}x{img.shape[0]} != image_size {W_img}x{H_img}")
        if not isinstance(boxes, list):
            E.append(f"{tag}: 'boxes' must be a list (may be empty)")
            continue

        valid = []
        for j, b in enumerate(boxes):
            bt = f"{tag}#{j}"
            if not isinstance(b, dict):
                E.append(f"{bt}: box must be an object")
                continue
            res["boxes"] += 1
            label = b.get("label")
            if label not in classes:
                E.append(f"{bt}: label {label!r} not in {sorted(classes)}")
            fine = b.get("fine_label")
            if fine is not None:
                if fine not in f2c:
                    E.append(f"{bt}: fine_label {fine!r} not in config fine_to_class")
                elif label in classes and f2c[fine] != label:
                    E.append(f"{bt}: fine_label {fine!r} maps to {f2c[fine]!r}, but label is {label!r}")
                else:
                    res["fine_counts"][fine] += 1
            if "occlusion" in b and b["occlusion"] not in (0, 1, 2):
                E.append(f"{bt}: occlusion must be 0, 1 or 2")
            if "truncated" in b and b["truncated"] not in (0, 1):
                E.append(f"{bt}: truncated must be 0 or 1")
            bb = b.get("bbox")
            if not (isinstance(bb, list) and len(bb) == 4 and all(_is_num(v) for v in bb)):
                E.append(f"{bt}: bbox must be 4 finite numbers [x1,y1,x2,y2]")
                continue
            x1, y1, x2, y2 = bb
            if not (x1 < x2 and y1 < y2):
                E.append(f"{bt}: degenerate bbox {bb}")
                continue
            if x1 < -tol or y1 < -tol or x2 > W_img + tol or y2 > H_img + tol:
                E.append(f"{bt}: bbox {bb} outside image {W_img}x{H_img}")
            elif min(x2 - x1, y2 - y1) < min_px:
                W.append(f"{bt}: tiny bbox {bb} (< {min_px}px)")
            if label in classes:
                res["counts"][label] += 1
            valid.append((bb, label, j))

        for a in range(len(valid)):
            for c in range(a + 1, len(valid)):
                if valid[a][1] == valid[c][1] and iou(valid[a][0], valid[c][0]) > 0.95:
                    W.append(f"{tag}: boxes #{valid[a][2]} and #{valid[c][2]} nearly identical (IoU>0.95)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--deep", action="store_true", help="decode images, check size and duplicates")
    ap.add_argument("--report", help="write JSON report to this path")
    ap.add_argument("--config", help="path to config json (default configs/default.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = data_root(cfg)
    files = sorted((root / "annotations").glob("*.json"))
    if not files:
        print(f"ERROR: no annotation files in {root / 'annotations'}", file=sys.stderr)
        return 2

    hashes, results = {}, []
    for f in files:
        results.append(validate_file(f, cfg, root, deep=args.deep, global_hashes=hashes))

    total = Counter()
    n_err = n_warn = 0
    print(f"data_root: {root}")
    for r in results:
        total.update(r["counts"])
        n_err += len(r["errors"])
        n_warn += len(r["warnings"])
        print(f"- {r['session']}: {r['frames']} frames, {r['boxes']} boxes, "
              f"{dict(r['counts'])}, {len(r['errors'])} errors, {len(r['warnings'])} warnings")
        for m in r["errors"][:20]:
            print(f"    ERROR   {m}")
        for m in r["warnings"][:10]:
            print(f"    warning {m}")
        if len(r["errors"]) > 20 or len(r["warnings"]) > 10:
            print("    ... (truncated; see --report for the full list)")
    print(f"TOTAL boxes per class: {dict(total)} | errors: {n_err} | warnings: {n_warn}")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            [{**r, "counts": dict(r["counts"]), "fine_counts": dict(r["fine_counts"])} for r in results],
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report: {out}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cut labeled boxes and background windows into fixed-size patches, per split.

Usage:
    python scripts/build_patches.py --roi-poly "100,200;1200,200;1300,640;50,640"
    python scripts/build_patches.py --neg-per-frame 6 --min-side 16 --seed 42 [--overwrite]

Input : <data_root>/splits/splits.json (make_splits.py), annotations, frame images.
Use --roi-file data/roi.json (made by scripts/pick_roi.py) or --roi-poly.
Output: <data_root>/processed/patches/patches_{train,val,test}.npz with arrays
        X (N,S,S,3 uint8, BGR), y (N, int), class_names, frame_id, session, bbox (N,4), occlusion, source
        y indexes class_names = classes + ["background"].  source: 'gt' or 'neg'.
Patch size/margin come from configs/default.json ("patch"); the SAME crop function
(src/common/patches.py) must be used at inference time.

Background (negative) windows: sizes are drawn from the real box sizes of that split; position is random
inside --roi-poly (pixel polygon "x,y;x,y;..."); a window is rejected if it covers > --max-cover of any
labeled box (or vice versa). WITHOUT --roi-poly, windows may fall in sky/trees/far field, where unlabeled
vehicles exist -> label noise. Always give the polygon of the region you actually labeled.
Boxes with min side < --min-side are skipped (too small to classify) and counted in the report.
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
from common.config import data_root, load_config  # noqa: E402
from common.geometry import coverage  # noqa: E402
from common.io import imread_unicode  # noqa: E402
from common.patches import crop_patch  # noqa: E402


def parse_poly(text):
    pts = [tuple(float(v) for v in pair.split(",")) for pair in text.split(";") if pair.strip()]
    if len(pts) < 3 or any(len(p) != 2 for p in pts):
        raise ValueError('--roi-poly needs >= 3 points like "x,y;x,y;x,y"')
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def poly_arg(text):
    try:
        return parse_poly(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"cannot parse {text!r}: use REAL pixel numbers like \"100,200;1200,200;1300,640;50,640\". "
            "Easier: run  python scripts/pick_roi.py --session S01  and use --roi-file data/roi.json")


def load_roi_file(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    poly = parse_poly(";".join(f"{x},{y}" for x, y in doc["polygon"]))
    return poly, list(doc["image_size"])


def sample_negatives(img_wh, gt_boxes, size_pool, n, rng, roi_poly=None, max_cover=0.2, tries=60):
    W, H = img_wh
    out = []
    if not size_pool:
        return out
    for _ in range(n * tries):
        if len(out) >= n:
            break
        w, h = size_pool[rng.integers(len(size_pool))]
        if w >= W or h >= H:
            continue
        x1, y1 = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        box = [x1, y1, x1 + w, y1 + h]
        if roi_poly is not None and cv2.pointPolygonTest(roi_poly, (x1 + w / 2.0, y1 + h / 2.0), False) < 0:
            continue
        if any(coverage(box, g) > max_cover for g in gt_boxes):
            continue
        out.append(box)
    return out


def build_split(frames, root, classes, size, margin, neg_per_frame, min_side, roi_poly, seed, max_cover=0.2):
    """frames: list of (session, frame_dict). Returns (arrays dict, report dict)."""
    bg = len(classes)
    rng = np.random.default_rng(seed)
    pool = [(int(b["bbox"][2] - b["bbox"][0]), int(b["bbox"][3] - b["bbox"][1]))
            for _, f in frames for b in f.get("boxes", [])
            if min(b["bbox"][2] - b["bbox"][0], b["bbox"][3] - b["bbox"][1]) >= min_side]
    X, rows = [], []
    skipped_small = neg_short = gt_outside = 0
    for sid, f in frames:
        img = imread_unicode(root / f["file"])
        if img is None:
            raise FileNotFoundError(f"cannot read image: {root / f['file']}")
        H, W = img.shape[:2]
        gts = []
        for b in f.get("boxes", []):
            bb = b["bbox"]
            gts.append(bb)
            if roi_poly is not None and cv2.pointPolygonTest(
                    roi_poly, ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0), False) < 0:
                gt_outside += 1
            if min(bb[2] - bb[0], bb[3] - bb[1]) < min_side:
                skipped_small += 1
                continue
            X.append(crop_patch(img, bb, size, margin))
            rows.append((classes.index(b["label"]), f["frame_id"], sid, bb, b.get("occlusion", 0), "gt"))
        negs = sample_negatives((W, H), gts, pool, neg_per_frame, rng, roi_poly, max_cover)
        neg_short += neg_per_frame - len(negs)
        for bb in negs:
            X.append(crop_patch(img, bb, size, margin))
            rows.append((bg, f["frame_id"], sid, bb, 0, "neg"))
    n = len(rows)
    arrays = {
        "X": np.stack(X) if X else np.zeros((0, size, size, 3), np.uint8),
        "y": np.array([r[0] for r in rows], dtype=np.int64),
        "frame_id": np.array([r[1] for r in rows], dtype=str),
        "session": np.array([r[2] for r in rows], dtype=str),
        "bbox": np.array([r[3] for r in rows], dtype=np.int32).reshape(n, 4),
        "occlusion": np.array([r[4] for r in rows], dtype=np.int8),
        "source": np.array([r[5] for r in rows], dtype=str),
        "class_names": np.array(list(classes) + ["background"], dtype=str),
    }
    names = arrays["class_names"]
    report = {"patches": n, "per_class": {str(names[k]): int(v) for k, v in sorted(Counter(arrays["y"].tolist()).items())},
              "skipped_small_boxes": skipped_small, "negatives_not_found": neg_short,
              "gt_outside_roi": gt_outside}
    return arrays, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--neg-per-frame", type=int, default=6)
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--max-cover", type=float, default=0.2)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--roi-poly", type=poly_arg)
    g.add_argument("--roi-file", help="roi.json made by scripts/pick_roi.py")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    root = data_root(cfg)
    split_path = root / "splits" / "splits.json"
    if not split_path.is_file():
        print(f"ERROR: {split_path} not found; run make_splits.py first", file=sys.stderr)
        return 2
    sp_doc = json.loads(split_path.read_text(encoding="utf-8"))
    out_dir = root / "processed" / "patches"
    existing = [out_dir / f"patches_{s}.npz" for s in ("train", "val", "test") if (out_dir / f"patches_{s}.npz").exists()]
    if existing and not args.overwrite:
        print(f"ERROR: {existing[0]} exists; use --overwrite to rebuild", file=sys.stderr)
        return 2
    roi = args.roi_poly
    if args.roi_file:
        try:
            roi, roi_size = load_roi_file(args.roi_file)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: bad ROI file: {exc}", file=sys.stderr)
            return 2
    if roi is None:
        print("WARNING: no --roi-poly/--roi-file. Background windows may contain unlabeled vehicles (far field) -> noisy negatives.")

    ann = {}
    for p in (root / "annotations").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        ann[d["session_id"]] = {f["frame_id"]: f for f in d["frames"]}
        if args.roi_file and d["image_size"] != roi_size:
            print(f"ERROR: ROI image size {roi_size} != session {d['session_id']} size {d['image_size']}", file=sys.stderr)
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    p_cfg = cfg["patch"]
    summary = {}
    try:
        for sp in ("train", "val", "test"):
            frames = [(sid, ann[sid][fid]) for sid, ids in sp_doc["splits"][sp].items() for fid in ids]
            arrays, rep = build_split(frames, root, cfg["classes"], p_cfg["size"], p_cfg["margin"],
                                      args.neg_per_frame, args.min_side, roi, args.seed, args.max_cover)
            np.savez_compressed(out_dir / f"patches_{sp}.npz", **arrays)
            summary[sp] = rep
            print(f"{sp:5s}: {rep['patches']:5d} patches {rep['per_class']} | skipped small boxes: "
                  f"{rep['skipped_small_boxes']} | negatives not found: {rep['negatives_not_found']}"
                  + (f" | GT boxes centred OUTSIDE roi: {rep['gt_outside_roi']}" if roi is not None else ""))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    (out_dir / "patches_summary.json").write_text(json.dumps(
        {"patch": p_cfg, "args": {k: v for k, v in vars(args).items() if k != "roi_poly"}, "splits": summary},
        indent=1), encoding="utf-8")
    print(f"saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

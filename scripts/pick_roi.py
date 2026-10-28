#!/usr/bin/env python3
"""Click the ROI polygon on one frame and save it to <data_root>/roi.json.

Usage:
    python scripts/pick_roi.py --session S01                 # uses the first frame of the session
    python scripts/pick_roi.py --session S01 --frame S01_000120
    python scripts/pick_roi.py --image path/to/frame.jpg [--overwrite]

Draw the polygon around the area where you LABELED vehicles (the road lanes facing the camera);
leave out sky, trees, the opposite carriageway and the far field you did not label.
Mouse: left-click = add point, right-click = undo.  Keys: u undo, c clear, Enter finish, q/Esc cancel.
Output: roi.json {"image_size": [W,H], "polygon": [[x,y],...]} and the equivalent --roi-poly string.
Then:  python scripts/build_patches.py --roi-file data/roi.json
Exit codes: 0 saved, 1 cancelled, 2 error.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotation.roi import RoiPicker  # noqa: E402
from annotation.store import IMG_EXT  # noqa: E402
from common.config import data_root, load_config  # noqa: E402
from common.io import imread_unicode  # noqa: E402

WIN = "ROI picker (Enter = finish)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session")
    g.add_argument("--image")
    ap.add_argument("--frame", help="frame_id inside the session (default: first frame)")
    ap.add_argument("--max-width", type=int, default=1500)
    ap.add_argument("--max-height", type=int, default=820)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = data_root(load_config())
    out = root / "roi.json"
    if out.exists() and not args.overwrite:
        print(f"ERROR: {out} exists; use --overwrite to replace it", file=sys.stderr)
        return 2
    if args.image:
        path = Path(args.image)
    else:
        fdir = root / "processed" / "frames" / args.session
        files = sorted(p for p in fdir.glob("*") if p.suffix.lower() in IMG_EXT) if fdir.is_dir() else []
        if args.frame:
            files = [p for p in files if p.stem == args.frame]
        if not files:
            print(f"ERROR: no matching frames in {fdir}", file=sys.stderr)
            return 2
        path = files[0]
    img = imread_unicode(path)
    if img is None:
        print(f"ERROR: cannot read image {path}", file=sys.stderr)
        return 2

    picker = RoiPicker(img, args.max_width, args.max_height)
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, picker.on_mouse)
        while True:
            cv2.imshow(WIN, picker.render())
            if not picker.on_key(cv2.waitKey(30)):
                break
    except cv2.error as exc:
        print("ERROR: OpenCV GUI is unavailable. Make sure 'opencv-python' (not '-headless') is installed.\n"
              f"Details: {exc}", file=sys.stderr)
        return 2
    finally:
        cv2.destroyAllWindows()
    if not picker.done:
        print("cancelled, nothing saved")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"image_size": [picker.W, picker.H], "polygon": [list(p) for p in picker.points], "source_image": path.name}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(f"saved {out}\nequivalent: --roi-poly \"{picker.polygon_string()}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())

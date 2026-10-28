#!/usr/bin/env python3
"""Simple bounding-box labeling tool (OpenCV window). Writes <data_root>/annotations/<session>.json.

Usage:
    python scripts/label_tool.py --session S01
    python scripts/label_tool.py --session S01 --max-width 1300 --max-height 750

Reads frames from <data_root>/processed/frames/<session>/ (made by extract_frames.py).
Progress is auto-saved whenever you change frame; a .bak copy of the previous JSON is kept.
Re-running resumes at the first frame you have not marked as reviewed (press n/Space to review).

Mouse:  left-drag = draw box with the current class | right-click = delete the box under the cursor
Keys:   1..4 class (motorbike, car, truck, unknown)   f = cycle fine label (taxi, pickup, bus, ...)
        o = occlusion 0/1/2 for the NEXT box          t = truncated 0/1 for the NEXT box
        n or Space = next frame (marks reviewed)      b = back     j = jump to first unreviewed
        u = undo last box                             s = save     q or Esc = save and quit
Rules for what to label: docs/labeling_guideline.md.  Check the result with:
    python scripts/validate_dataset.py --deep
Exit codes: 0 ok, 2 bad input or no GUI available.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotation.store import AnnotationStore  # noqa: E402
from annotation.ui import LabelSession  # noqa: E402
from common.config import data_root, load_config  # noqa: E402

WIN = "label tool (q = save and quit)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--max-width", type=int, default=1500)
    ap.add_argument("--max-height", type=int, default=820)
    ap.add_argument("--start", type=int, help="start at this frame number (0-based)")
    args = ap.parse_args()

    cfg = load_config()
    try:
        store = AnnotationStore(args.session, data_root(cfg), cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    sess = LabelSession(store, args.max_width, args.max_height, start=args.start)
    print(f"{len(store)} frames, image {store.image_size[0]}x{store.image_size[1]}, "
          f"display scale {sess.scale:.2f}. Annotation file: {store.ann_path}")

    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, sess.on_mouse)
        first = True
        while True:
            cv2.imshow(WIN, sess.render())
            if not sess.on_key(cv2.waitKey(30)):
                break
            if not first:
                try:
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
            first = False
    except cv2.error as exc:
        store.save()
        print("ERROR: OpenCV GUI is unavailable. Make sure 'opencv-python' (not '-headless') is installed.\n"
              f"Details: {exc}", file=sys.stderr)
        return 2
    finally:
        cv2.destroyAllWindows()
    store.save(force=True)
    done = sum(1 for f in store.frames if f["reviewed"])
    print(f"saved {store.ann_path} | reviewed {done}/{len(store)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Extract frames from one self-recorded video into a session folder.

Usage:
    python scripts/extract_frames.py --video data/raw/S01.mp4 --session-id S01 --every 15
    python scripts/extract_frames.py --video clip.mp4 --session-id S02 --fps 2 --crop 0,0,1920,1000

Output (under <data_root>/processed/frames/<session_id>/):
    <session_id>_<source_frame_no:06d>.jpg
    frames_index.csv   (session_id, frame_id, source_frame, time_s, file)

Notes:
    --every N   keep every N-th source frame (takes priority over --fps)
    --fps F     keep about F frames per second of video
    --crop x,y,w,h  crop before saving (e.g. to remove a player UI bar if you screen-recorded)
    Refuses to write into a non-empty session folder unless --overwrite is given.
    Use the ORIGINAL video file, not screenshots of a player.
Exit codes: 0 ok, 2 bad input / cannot open video / output exists.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import data_root, load_config  # noqa: E402
from common.io import imwrite_unicode  # noqa: E402

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_crop(text):
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0 or parts[0] < 0 or parts[1] < 0:
        raise ValueError("--crop must be x,y,w,h with w,h > 0 and x,y >= 0")
    return tuple(parts)


def extract(video, session_id, out_dir, every=None, fps=None, crop=None,
            max_frames=None, jpeg_quality=95, overwrite=False):
    """Return (rows, info). Raises ValueError / FileNotFoundError / FileExistsError on bad input."""
    if not SESSION_RE.match(session_id):
        raise ValueError("session_id may only contain letters, digits, '_' and '-'")
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video (codec/path problem): {video}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if every is not None:
            step = every
        elif fps is not None:
            if src_fps <= 0:
                raise ValueError("video reports no FPS; use --every instead of --fps")
            step = max(1, round(src_fps / fps))
        else:
            raise ValueError("give --every or --fps")
        if step < 1:
            raise ValueError("step must be >= 1")
        if crop is not None:
            cx, cy, cw, ch = crop
            if cx + cw > width or cy + ch > height:
                raise ValueError(f"--crop exceeds frame size {width}x{height}")

        out_dir = Path(out_dir)
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace files")
        out_dir.mkdir(parents=True, exist_ok=True)

        rows, idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                if crop is not None:
                    frame = frame[cy:cy + ch, cx:cx + cw]
                frame_id = f"{session_id}_{idx:06d}"
                path = out_dir / f"{frame_id}.jpg"
                if not imwrite_unicode(path, frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise OSError(f"failed to write {path}")
                t = idx / src_fps if src_fps > 0 else float("nan")
                rows.append({"session_id": session_id, "frame_id": frame_id,
                             "source_frame": idx, "time_s": f"{t:.3f}", "file": path.name})
                if max_frames and len(rows) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    with open(out_dir / "frames_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "frame_id", "source_frame", "time_s", "file"])
        w.writeheader()
        w.writerows(rows)
    info = {"src_fps": src_fps, "reported_frames": n_frames, "read_frames": idx,
            "width": width, "height": height, "step": step, "saved": len(rows)}
    return rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--session-id", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--every", type=int)
    g.add_argument("--fps", type=float)
    ap.add_argument("--crop", type=parse_crop)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out-dir", help="default: <data_root>/processed/frames/<session_id>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        data_root(load_config()) / "processed" / "frames" / args.session_id)
    try:
        _, info = extract(args.video, args.session_id, out_dir, every=args.every, fps=args.fps,
                          crop=args.crop, max_frames=args.max_frames, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"video: {info['width']}x{info['height']} @ {info['src_fps']:.2f} fps, "
          f"read {info['read_frames']} frames (container reported {info['reported_frames']})")
    print(f"saved {info['saved']} frames (every {info['step']}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

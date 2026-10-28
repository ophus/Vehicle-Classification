"""Annotation storage for one session: load/create JSON, edit boxes, atomic save.

JSON layout is the one documented in scripts/validate_dataset.py. Extra per-frame key
"reviewed" (bool) lets the labeling tool resume where you stopped.
"""
import json
import os
import shutil
from pathlib import Path

from common.io import imread_unicode

IMG_EXT = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 5  # drag smaller than this (pixels) is treated as an accidental click


class AnnotationStore:
    def __init__(self, session_id, root, cfg, frames_dir=None, ann_path=None):
        self.session_id = session_id
        self.root = Path(root)
        self.classes = list(cfg["classes"])
        self.fine_by_class = {c: [f for f, cc in cfg["fine_to_class"].items() if cc == c]
                              for c in self.classes}
        self.frames_dir = Path(frames_dir) if frames_dir else self.root / "processed" / "frames" / session_id
        self.ann_path = Path(ann_path) if ann_path else self.root / "annotations" / f"{session_id}.json"
        self.frames, self.orphans, self.dirty = [], [], False
        self.image_size = None
        self._load()

    # ---- loading -------------------------------------------------------
    def _load(self):
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"frames folder not found: {self.frames_dir}")
        files = sorted(p for p in self.frames_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"no .jpg/.png images in {self.frames_dir}")
        first = imread_unicode(files[0])
        if first is None:
            raise ValueError(f"cannot read image: {files[0]}")
        self.image_size = [int(first.shape[1]), int(first.shape[0])]

        existing = {}
        if self.ann_path.is_file():
            doc = json.loads(self.ann_path.read_text(encoding="utf-8"))
            if doc.get("session_id") != self.session_id:
                raise ValueError(f"{self.ann_path.name} belongs to session {doc.get('session_id')!r}")
            if doc.get("image_size") != self.image_size:
                raise ValueError(f"image_size in JSON {doc.get('image_size')} != frames {self.image_size}")
            existing = {f["frame_id"]: f for f in doc.get("frames", [])}

        present = set()
        for p in files:
            fid = p.stem
            present.add(fid)
            rel = os.path.relpath(p, self.root).replace("\\", "/")
            fr = existing.get(fid) or {"frame_id": fid, "boxes": [], "reviewed": False}
            fr["file"] = rel
            fr.setdefault("boxes", [])
            fr.setdefault("reviewed", False)
            self.frames.append(fr)
        # entries whose image vanished are kept in the JSON (never silently dropped)
        self.orphans = [f for fid, f in existing.items() if fid not in present]

    def __len__(self):
        return len(self.frames)

    # ---- editing -------------------------------------------------------
    def default_fine(self, label):
        fines = self.fine_by_class[label]
        return label if label in fines else fines[-1]

    def add_box(self, i, bbox, label, fine_label=None, occlusion=0, truncated=0):
        """Add a box (image pixel coords). Returns False if it is too small to be intentional."""
        if label not in self.classes:
            raise ValueError(f"unknown class {label!r}")
        if fine_label is None:
            fine_label = self.default_fine(label)
        elif fine_label not in self.fine_by_class[label]:
            raise ValueError(f"fine_label {fine_label!r} does not belong to class {label!r}")
        W, H = self.image_size
        x1, y1, x2, y2 = (float(v) for v in bbox)
        x1, x2 = sorted((min(max(x1, 0.0), W), min(max(x2, 0.0), W)))
        y1, y2 = sorted((min(max(y1, 0.0), H), min(max(y2, 0.0), H)))
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            return False
        self.frames[i]["boxes"].append({
            "bbox": [int(round(v)) for v in (x1, y1, x2, y2)],
            "label": label, "fine_label": fine_label,
            "occlusion": int(occlusion), "truncated": int(truncated)})
        self.dirty = True
        return True

    def remove_last(self, i):
        boxes = self.frames[i]["boxes"]
        if not boxes:
            return False
        boxes.pop()
        self.dirty = True
        return True

    def remove_at(self, i, x, y):
        """Remove the smallest box containing point (x, y) so overlapped small boxes stay reachable."""
        best, best_area = None, None
        for k, b in enumerate(self.frames[i]["boxes"]):
            x1, y1, x2, y2 = b["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best_area:
                    best, best_area = k, area
        if best is None:
            return False
        del self.frames[i]["boxes"][best]
        self.dirty = True
        return True

    def set_reviewed(self, i, value=True):
        if self.frames[i]["reviewed"] != value:
            self.frames[i]["reviewed"] = value
            self.dirty = True

    def first_unreviewed(self):
        for i, f in enumerate(self.frames):
            if not f["reviewed"]:
                return i
        return 0

    # ---- saving --------------------------------------------------------
    def save(self, force=False):
        if not (self.dirty or force or not self.ann_path.exists()):
            return False
        self.ann_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"session_id": self.session_id, "image_size": self.image_size,
               "frames": self.frames + self.orphans}
        tmp = self.ann_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        if self.ann_path.exists():
            shutil.copyfile(self.ann_path, self.ann_path.with_suffix(".json.bak"))
        os.replace(tmp, self.ann_path)
        self.dirty = False
        return True

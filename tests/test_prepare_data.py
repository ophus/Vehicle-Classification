"""Tests for prune_annotations.py, make_splits.py, build_patches.py (logic only; synthetic fixtures)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import argparse  # noqa: E402

import build_patches as bp  # noqa: E402
import make_splits as ms  # noqa: E402
import prune_annotations as pa  # noqa: E402
import validate_dataset as vd  # noqa: E402
import cv2  # noqa: E402
from annotation.roi import RoiPicker  # noqa: E402
from common.config import load_config  # noqa: E402
from common.geometry import coverage, iou  # noqa: E402
from common.io import imwrite_unicode  # noqa: E402
from common.patches import crop_patch  # noqa: E402

CFG = load_config()
W, H = 320, 200
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_session(root, sid, n, step=30, reviewed=True, write_images=True, boxes_fn=None):
    """Create n frames (frame numbers 0, step, 2*step...) + annotation JSON. Returns doc."""
    fdir = root / "processed" / "frames" / sid
    fdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        fid = f"{sid}_{i * step:06d}"
        rel = f"processed/frames/{sid}/{fid}.jpg"
        if write_images:
            rng = np.random.default_rng(i)
            imwrite_unicode(root / rel, rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
        boxes = boxes_fn(i) if boxes_fn else [
            {"bbox": [20, 100, 90, 150], "label": "car", "fine_label": "car", "occlusion": 0, "truncated": 0},
            {"bbox": [150, 120, 190, 190], "label": "motorbike", "fine_label": "motorbike", "occlusion": 0, "truncated": 0}]
        r = reviewed(i) if callable(reviewed) else reviewed
        frames.append({"frame_id": fid, "file": rel, "boxes": boxes, "reviewed": r})
    doc = {"session_id": sid, "image_size": [W, H], "frames": frames}
    (root / "annotations").mkdir(exist_ok=True)
    (root / "annotations" / f"{sid}.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


class PruneTests(unittest.TestCase):
    def test_dry_run_then_apply(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_session(root, "S01", 10)
            for i in range(4, 10):                                   # delete 6 images
                (root / "processed/frames/S01" / f"S01_{i * 30:06d}.jpg").unlink()
            path = root / "annotations" / "S01.json"
            before = path.read_text(encoding="utf-8")
            info = pa.prune(path, root, apply=False)
            self.assertEqual((info["kept"], info["removed"]), (4, 6))
            self.assertEqual(path.read_text(encoding="utf-8"), before)          # dry run changed nothing
            info = pa.prune(path, root, apply=True)
            self.assertTrue(Path(info["backup"]).is_file())
            self.assertEqual(Path(info["backup"]).read_text(encoding="utf-8"), before)
            res = vd.validate_file(path, CFG, root, deep=True)
            self.assertEqual(res["errors"], [])
            self.assertEqual(res["frames"], 4)

    def test_refuses_to_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_session(root, "S01", 3, write_images=False)
            with self.assertRaises(ValueError):
                pa.prune(root / "annotations" / "S01.json", root, apply=True)


class SplitTests(unittest.TestCase):
    def load(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for sid, n in kw.items():
                make_session(root, sid, n, write_images=False)
            return ms.load_sessions(root / "annotations")[0]

    def all_ids(self, res, split):
        return {(sid, f) for sid, ids in res["splits"][split].items() for f in ids}

    def test_block_mode_no_overlap_reviewed_only_and_guard(self):
        sessions = self.load(S01=83)
        res = ms.make_splits(sessions, RATIOS, seed=1, block_frames=8, guard=2)
        self.assertEqual(res["mode"], "block")
        tr, va, te = (self.all_ids(res, s) for s in ("train", "val", "test"))
        self.assertFalse(tr & va or tr & te or va & te)
        dropped = {(sid, f) for sid, ids in res["dropped_guard"].items() for f in ids}
        self.assertFalse(dropped & (tr | va | te))
        self.assertEqual(len(tr | va | te) + len(dropped), 83)
        # guard works: no train frame directly adjacent in time to a val/test frame
        order = [f["frame_id"] for f in sessions["S01"]]
        label = {fid: s for s in ("train", "val", "test") for (_, fid) in self.all_ids(res, s)}
        for a, b in zip(order, order[1:]):
            if a in label and b in label and label[a] != label[b]:
                self.assertNotIn("train", (label[a], label[b]))
        self.assertTrue(va and te)

    def test_apply_guard_drops_the_lower_priority_side_exactly(self):
        seq = [[f"f{i}", sp] for i, sp in enumerate(["train"] * 4 + ["val"] * 4 + ["train"] * 4)]
        self.assertEqual(ms.apply_guard(seq, 2), {"f2", "f3", "f8", "f9"})          # train side only
        seq = [[f"g{i}", sp] for i, sp in enumerate(["val"] * 4 + ["test"] * 4)]
        self.assertEqual(ms.apply_guard(seq, 2), {"g2", "g3"})                      # val gives way to test
        seq = [[f"h{i}", sp] for i, sp in enumerate(["test"] * 3 + ["train"] * 3)]
        self.assertEqual(ms.apply_guard(seq, 2), {"h3", "h4"})                      # train after test

    def test_deterministic_and_seed_changes_result(self):
        sessions = self.load(S01=83)
        a = ms.make_splits(sessions, RATIOS, seed=7)
        b = ms.make_splits(sessions, RATIOS, seed=7)
        c = ms.make_splits(sessions, RATIOS, seed=8)
        self.assertEqual(a["splits"], b["splits"])
        self.assertNotEqual(a["splits"], c["splits"])

    def test_session_mode_keeps_sessions_whole(self):
        sessions = self.load(S01=30, S02=20, S03=10, S04=10, S05=10)
        res = ms.make_splits(sessions, RATIOS, seed=3)
        self.assertEqual(res["mode"], "session")
        owner = {}
        for sp, d in res["splits"].items():
            for sid in d:
                self.assertNotIn(sid, owner)             # a session lives in exactly one split
                owner[sid] = sp
        self.assertEqual(len(owner), 5)
        self.assertEqual(res["dropped_guard"], {})

    def test_too_little_data_raises(self):
        with self.assertRaises(ValueError):
            ms.make_splits(self.load(S01=10), RATIOS, block_frames=8)

    def test_unreviewed_frames_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_session(root, "S01", 20, write_images=False)
            doc = json.loads((root / "annotations/S01.json").read_text())
            for f in doc["frames"][10:]:
                f["reviewed"] = False
            (root / "annotations/S01.json").write_text(json.dumps(doc))
            sessions, excluded = ms.load_sessions(root / "annotations")
            self.assertEqual(len(sessions["S01"]), 10)
            self.assertEqual(len(excluded["S01"]), 10)

    def test_stats_count_boxes(self):
        res = ms.make_splits(self.load(S01=40), RATIOS, seed=1, block_frames=5, guard=1)
        total = sum(st["boxes_per_class"].get("car", 0) for st in res["stats"].values())
        used = sum(st["frames"] for st in res["stats"].values())
        self.assertEqual(total, used)                    # fixture has exactly 1 car per frame


class PatchTests(unittest.TestCase):
    def test_geometry(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(coverage([0, 0, 100, 100], [10, 10, 20, 20]), 1.0)   # small box inside big one

    def test_crop_patch_shape_and_clipping(self):
        img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
        self.assertEqual(crop_patch(img, [10, 10, 60, 50], 64, 0.1).shape, (64, 64, 3))
        self.assertEqual(crop_patch(img, [0, 0, 20, 20], 64, 0.5).shape, (64, 64, 3))      # margin clipped at border
        with self.assertRaises(ValueError):
            crop_patch(img, [10, 10, 11, 11], 64, 0.0)

    def test_crop_patch_content_includes_margin_on_all_sides(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[:, :, 0] = np.arange(200, dtype=np.uint8)[None, :]          # value = column index
        img[:, :, 1] = np.arange(100, dtype=np.uint8)[:, None]          # value = row index
        # bbox 40x40 at (100,20); margin 0.5 -> x 80..160, y 0..80 ; size 80 => identity resize
        np.testing.assert_array_equal(crop_patch(img, [100, 20, 140, 60], 80, 0.5), img[0:80, 80:160])

    def test_parse_poly(self):
        self.assertEqual(bp.parse_poly("0,0;10,0;10,10").shape, (3, 1, 2))
        with self.assertRaises(ValueError):
            bp.parse_poly("0,0;10,0")

    def test_negatives_avoid_gt_and_respect_roi(self):
        rng = np.random.default_rng(0)
        gt = [[20, 100, 90, 150], [150, 120, 190, 190]]
        roi = bp.parse_poly("0,80;320,80;320,200;0,200")
        negs = bp.sample_negatives((W, H), gt, [(40, 40), (60, 50)], 30, rng, roi, max_cover=0.2)
        self.assertGreater(len(negs), 10)
        for b in negs:
            self.assertTrue(all(coverage(b, g) <= 0.2 for g in gt))
            cy = (b[1] + b[3]) / 2
            self.assertGreaterEqual(cy, 80)

    def test_build_split_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def boxes(i):
                b = [{"bbox": [20, 100, 90, 150], "label": "car", "fine_label": "car"},
                     {"bbox": [150, 120, 190, 190], "label": "motorbike", "fine_label": "motorbike", "occlusion": 1}]
                if i == 0:
                    b.append({"bbox": [250, 10, 258, 18], "label": "truck", "fine_label": "truck"})   # 8px: too small
                return b
            doc = make_session(root, "S01", 4, boxes_fn=boxes)
            frames = [("S01", f) for f in doc["frames"]]
            roi = bp.parse_poly("0,80;320,80;320,200;0,200")
            arrays, rep = bp.build_split(frames, root, CFG["classes"], 64, 0.1, 5, 16, roi, seed=1)
            classes = CFG["classes"]
            self.assertEqual(arrays["X"].shape[1:], (64, 64, 3))
            self.assertEqual(arrays["X"].dtype, np.uint8)
            self.assertEqual(rep["skipped_small_boxes"], 1)
            self.assertEqual(rep["per_class"]["car"], 4)
            self.assertEqual(rep["per_class"]["motorbike"], 4)
            self.assertNotIn("truck", rep["per_class"])
            self.assertTrue(rep["per_class"]["background"] > 0)
            self.assertEqual(list(arrays["class_names"]), classes + ["background"])
            gt_mask = arrays["source"] == "gt"
            self.assertTrue(np.all(arrays["y"][~gt_mask] == len(classes)))
            self.assertTrue(np.all(arrays["y"][gt_mask] < len(classes)))
            occ = arrays["occlusion"][arrays["y"] == classes.index("motorbike")]
            self.assertTrue(np.all(occ == 1))
            # same seed -> identical negatives
            arrays2, _ = bp.build_split(frames, root, classes, 64, 0.1, 5, 16, roi, seed=1)
            np.testing.assert_array_equal(arrays["bbox"], arrays2["bbox"])

    def test_missing_image_raises(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = make_session(root, "S01", 2)
            (root / doc["frames"][0]["file"]).unlink()
            with self.assertRaises(FileNotFoundError):
                bp.build_split([("S01", f) for f in doc["frames"]], root, CFG["classes"], 64, 0.1, 2, 16, None, 1)



class RoiTests(unittest.TestCase):
    def picker(self, max_w=1500, max_h=820):
        return RoiPicker(np.zeros((100, 200, 3), np.uint8), max_w, max_h)

    def click(self, p, x, y):
        p.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + p.HUD_H)

    def test_points_scale_undo_and_finish(self):
        p = self.picker(max_w=100, max_h=100)                        # scale 0.5
        self.assertAlmostEqual(p.scale, 0.5)
        self.click(p, 10, 5)
        self.click(p, 90, 5)
        self.click(p, 50, 45)
        self.assertEqual(p.points, [(20, 10), (180, 10), (100, 90)])
        p.on_mouse(cv2.EVENT_RBUTTONDOWN, 5, 50)                     # right click = undo
        self.assertEqual(len(p.points), 2)
        self.assertTrue(p.on_key(13))                                # Enter with 2 points: refused
        self.assertFalse(p.done)
        self.click(p, 50, 45)
        self.assertFalse(p.on_key(13))                               # accepted -> picker closes
        self.assertTrue(p.done)
        self.assertEqual(p.polygon_string(), "20,10;180,10;100,90")
        self.assertEqual(p.render().shape, (p.HUD_H + 50, p.MIN_CANVAS_W, 3))

    def test_clicks_clipped_and_hud_ignored(self):
        p = self.picker()
        p.on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 10)                    # inside HUD bar: ignored
        self.assertEqual(p.points, [])
        p.on_mouse(cv2.EVENT_LBUTTONDOWN, 5000, p.HUD_H + 5000)
        self.assertEqual(p.points, [(199, 99)])

    def test_cancel_keys(self):
        p = self.picker()
        self.assertFalse(p.on_key(ord("q")))
        self.assertFalse(p.done)

    def test_poly_arg_gives_helpful_error_and_roi_file_roundtrip(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            bp.poly_arg("x1,y1;x2,y2;x3,y3")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "roi.json"
            f.write_text(json.dumps({"image_size": [W, H], "polygon": [[0, 80], [320, 80], [320, 200], [0, 200]]}))
            poly, size = bp.load_roi_file(f)
            self.assertEqual(poly.shape, (4, 1, 2))
            self.assertEqual(size, [W, H])

    def test_gt_outside_roi_is_counted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = make_session(root, "S01", 2)               # boxes at y>=100: inside a roi starting at y=80
            frames = [("S01", f) for f in doc["frames"]]
            small_roi = bp.parse_poly("0,0;320,0;320,60;0,60")     # excludes every labeled box centre
            _, rep = bp.build_split(frames, root, CFG["classes"], 64, 0.1, 0, 16, small_roi, seed=1)
            self.assertEqual(rep["gt_outside_roi"], 4)
            good = bp.parse_poly("0,80;320,80;320,200;0,200")
            _, rep = bp.build_split(frames, root, CFG["classes"], 64, 0.1, 0, 16, good, seed=1)
            self.assertEqual(rep["gt_outside_roi"], 0)

if __name__ == "__main__":
    unittest.main()

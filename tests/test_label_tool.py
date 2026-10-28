"""Headless tests for the labeling tool logic (no window is opened).

Only cv2.imshow / waitKey / setMouseCallback wiring in scripts/label_tool.py is NOT covered here:
it needs a real display and must be checked by hand.
Fixtures are throw-away synthetic images, not a dataset.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_dataset as vd  # noqa: E402
from annotation.store import AnnotationStore  # noqa: E402
from annotation.ui import LabelSession  # noqa: E402
from common.config import load_config  # noqa: E402
from common.io import imwrite_unicode  # noqa: E402

CFG = load_config()
W, H = 200, 100
HUD = LabelSession.HUD_H
DOWN, MOVE, UP, RDOWN = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN


def drag(ui, x0, y0, x1, y1):
    """Simulate a drag in DISPLAY coordinates (y without HUD offset)."""
    ui.on_mouse(DOWN, x0, y0 + HUD)
    ui.on_mouse(MOVE, (x0 + x1) // 2, (y0 + y1) // 2 + HUD)
    ui.on_mouse(UP, x1, y1 + HUD)


class LabelToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fdir = self.root / "processed" / "frames" / "S01"
        fdir.mkdir(parents=True)
        for i in range(3):
            rng = np.random.default_rng(i)
            imwrite_unicode(fdir / f"S01_{i:06d}.jpg", rng.integers(0, 255, (H, W, 3), dtype=np.uint8))

    def tearDown(self):
        self.tmp.cleanup()

    def open(self, **kw):
        store = AnnotationStore("S01", self.root, CFG)
        return store, LabelSession(store, max_w=kw.get("max_w", 200), max_h=kw.get("max_h", 100))

    def test_draw_box_with_class_and_fine_then_resume(self):
        store, ui = self.open()
        ui.on_key(ord("2"))   # car
        ui.on_key(ord("f"))   # fine: car -> taxi
        self.assertEqual((ui.cls, ui.fine), ("car", "taxi"))
        ui.on_key(ord("o"))   # occlusion 1
        drag(ui, 10, 10, 60, 50)
        b = store.frames[0]["boxes"][0]
        self.assertEqual(b, {"bbox": [10, 10, 60, 50], "label": "car", "fine_label": "taxi",
                             "occlusion": 1, "truncated": 0})
        ui.on_key(ord("n"))   # next frame: marks reviewed + autosave
        self.assertEqual(ui.idx, 1)

        store2 = AnnotationStore("S01", self.root, CFG)          # reopen from disk
        self.assertEqual(store2.frames[0]["boxes"][0]["fine_label"], "taxi")
        self.assertTrue(store2.frames[0]["reviewed"])
        self.assertEqual(store2.first_unreviewed(), 1)            # resumes at frame 1
        res = vd.validate_file(store2.ann_path, CFG, self.root, deep=True)
        self.assertEqual(res["errors"], [])

    def test_scaled_display_maps_back_to_image_coords(self):
        store, ui = self.open(max_w=100, max_h=100)               # scale 0.5
        self.assertAlmostEqual(ui.scale, 0.5)
        drag(ui, 5, 5, 30, 25)
        self.assertEqual(store.frames[0]["boxes"][0]["bbox"], [10, 10, 60, 50])
        self.assertEqual(ui.render().shape, (HUD + 50, ui.MIN_CANVAS_W, 3))

    def test_tiny_drag_ignored_and_clipping(self):
        store, ui = self.open()
        drag(ui, 20, 20, 22, 22)
        self.assertEqual(store.frames[0]["boxes"], [])
        drag(ui, 150, 60, 400, 300)                                # dragged outside -> clipped
        self.assertEqual(store.frames[0]["boxes"][0]["bbox"], [150, 60, 200, 100])

    def test_undo_and_right_click_delete_smallest(self):
        store, ui = self.open()
        drag(ui, 10, 10, 100, 90)
        drag(ui, 30, 30, 60, 60)
        ui.on_mouse(RDOWN, 40, 40 + HUD)                           # inside both -> removes the small one
        self.assertEqual([b["bbox"] for b in store.frames[0]["boxes"]], [[10, 10, 100, 90]])
        ui.on_key(ord("u"))
        self.assertEqual(store.frames[0]["boxes"], [])

    def test_unknown_class_default_fine_is_other(self):
        _, ui = self.open()
        ui.on_key(ord("4"))
        self.assertEqual((ui.cls, ui.fine), ("unknown", "other"))

    def test_quit_keys_and_navigation_bounds(self):
        _, ui = self.open()
        self.assertTrue(ui.on_key(-1))
        ui.on_key(ord("b"))
        self.assertEqual(ui.idx, 0)
        for _ in range(5):
            ui.on_key(ord("n"))
        self.assertEqual(ui.idx, 2)
        self.assertFalse(ui.on_key(ord("q")))
        self.assertFalse(ui.on_key(27))

    def test_save_keeps_backup_and_orphans(self):
        store, ui = self.open()
        drag(ui, 10, 10, 60, 50)
        ui.on_key(ord("s"))
        (self.root / "processed" / "frames" / "S01" / "S01_000002.jpg").unlink()   # image disappears
        store2 = AnnotationStore("S01", self.root, CFG)
        self.assertEqual(len(store2), 2)
        self.assertEqual(len(store2.orphans), 1)
        store2.add_box(0, [1, 1, 50, 50], "truck")
        store2.save()
        doc = json.loads(store2.ann_path.read_text(encoding="utf-8"))
        self.assertEqual(len(doc["frames"]), 3)                    # orphan not dropped
        self.assertTrue(store2.ann_path.with_suffix(".json.bak").exists())

    def test_wrong_session_or_missing_folder(self):
        store, ui = self.open()
        drag(ui, 10, 10, 60, 50)
        ui.on_key(ord("s"))
        (self.root / "annotations" / "S02.json").write_text(
            (self.root / "annotations" / "S01.json").read_text())
        with self.assertRaises(ValueError):
            AnnotationStore("S02", self.root, CFG, frames_dir=self.root / "processed" / "frames" / "S01")
        with self.assertRaises(FileNotFoundError):
            AnnotationStore("S99", self.root, CFG)

    def test_bad_fine_label_rejected(self):
        store, _ = self.open()
        with self.assertRaises(ValueError):
            store.add_box(0, [1, 1, 50, 50], "car", fine_label="bus")


if __name__ == "__main__":
    unittest.main()

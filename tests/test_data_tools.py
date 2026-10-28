"""Unit tests for scripts/validate_dataset.py and scripts/extract_frames.py.

Fixtures are tiny synthetic images/videos created in a temp dir ONLY to test the code paths.
They are not a dataset and nothing here says anything about model quality.

Run:  python -m unittest discover -s tests -v
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

import extract_frames  # noqa: E402
import validate_dataset as vd  # noqa: E402
from common.config import load_config  # noqa: E402
from common.io import imread_unicode, imwrite_unicode  # noqa: E402

CFG = load_config()
W, H = 200, 100


def make_image(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)


def box(bbox, label="car", **kw):
    return {"bbox": bbox, "label": label, **kw}


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "annotations").mkdir()
        (self.root / "processed").mkdir()
        for i in (1, 2):
            imwrite_unicode(self.root / "processed" / f"f{i}.png", make_image(i), ".png")
        # f3 is a byte-identical copy of f1
        (self.root / "processed" / "f3.png").write_bytes((self.root / "processed" / "f1.png").read_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def run_val(self, frames, deep=True):
        p = self.root / "annotations" / "S01.json"
        p.write_text(json.dumps({"session_id": "S01", "image_size": [W, H], "frames": frames}))
        return vd.validate_file(p, CFG, self.root, deep=deep)

    def frame(self, n, boxes):
        return {"frame_id": f"S01_{n}", "file": f"processed/f{n}.png", "boxes": boxes}

    def test_valid_file_has_no_errors(self):
        r = self.run_val([self.frame(1, [box([10, 10, 60, 50]), box([100, 20, 150, 90], "motorbike")])])
        self.assertEqual(r["errors"], [])
        self.assertEqual(dict(r["counts"]), {"car": 1, "motorbike": 1})

    def test_bbox_outside_image(self):
        r = self.run_val([self.frame(1, [box([10, 10, 250, 50])])])
        self.assertTrue(any("outside image" in m for m in r["errors"]))

    def test_degenerate_bbox(self):
        r = self.run_val([self.frame(1, [box([50, 10, 50, 50])])])
        self.assertTrue(any("degenerate" in m for m in r["errors"]))

    def test_unknown_label(self):
        r = self.run_val([self.frame(1, [box([10, 10, 60, 50], "bus")])])
        self.assertTrue(any("not in" in m for m in r["errors"]))

    def test_fine_label_mismatch(self):
        r = self.run_val([self.frame(1, [box([10, 10, 60, 50], "truck", fine_label="pickup")])])
        self.assertTrue(any("maps to" in m for m in r["errors"]))

    def test_fine_label_ok(self):
        r = self.run_val([self.frame(1, [box([10, 10, 60, 50], "unknown", fine_label="minibus")])])
        self.assertEqual(r["errors"], [])

    def test_missing_file(self):
        r = self.run_val([{"frame_id": "S01_9", "file": "processed/nope.png", "boxes": []}])
        self.assertTrue(any("not found" in m for m in r["errors"]))

    def test_duplicate_frame_id(self):
        f = self.frame(1, [])
        r = self.run_val([f, dict(f)])
        self.assertTrue(any("duplicate frame_id" in m for m in r["errors"]))

    def test_byte_identical_images_warned_in_deep_mode(self):
        r = self.run_val([self.frame(1, []), self.frame(3, [])])
        self.assertTrue(any("byte-identical" in m for m in r["warnings"]))

    def test_size_mismatch_in_deep_mode(self):
        imwrite_unicode(self.root / "processed" / "f2.png", np.zeros((50, 50, 3), np.uint8), ".png")
        r = self.run_val([self.frame(2, [])])
        self.assertTrue(any("image size" in m for m in r["errors"]))

    def test_corrupt_image(self):
        (self.root / "processed" / "f2.png").write_bytes(b"not an image")
        r = self.run_val([self.frame(2, [])])
        self.assertTrue(any("cannot be decoded" in m for m in r["errors"]))

    def test_tiny_and_duplicate_boxes_are_warnings(self):
        r = self.run_val([self.frame(1, [box([10, 10, 14, 14]), box([30, 30, 80, 80]), box([30, 30, 80, 80])])])
        self.assertEqual(r["errors"], [])
        self.assertTrue(any("tiny" in m for m in r["warnings"]))
        self.assertTrue(any("nearly identical" in m for m in r["warnings"]))

    def test_bad_json(self):
        p = self.root / "annotations" / "S01.json"
        p.write_text("{oops")
        r = vd.validate_file(p, CFG, self.root)
        self.assertTrue(r["errors"])


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "clip.avi"
        vw = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (W, H))
        for i in range(30):
            vw.write(make_image(i))
        vw.release()

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_n(self):
        rows, info = extract_frames.extract(self.video, "S01", self.dir / "out", every=10)
        self.assertEqual([r["source_frame"] for r in rows], [0, 10, 20])
        self.assertEqual((info["width"], info["height"]), (W, H))
        self.assertTrue((self.dir / "out" / "frames_index.csv").is_file())
        self.assertIsNotNone(imread_unicode(self.dir / "out" / "S01_000010.jpg"))

    def test_fps_mode(self):
        rows, info = extract_frames.extract(self.video, "S01", self.dir / "out", fps=2)
        self.assertEqual(info["step"], 5)
        self.assertEqual(len(rows), 6)

    def test_crop(self):
        extract_frames.extract(self.video, "S01", self.dir / "out", every=15, crop=(10, 20, 100, 50))
        img = imread_unicode(self.dir / "out" / "S01_000000.jpg")
        self.assertEqual(img.shape[:2], (50, 100))

    def test_refuses_overwrite(self):
        extract_frames.extract(self.video, "S01", self.dir / "out", every=10)
        with self.assertRaises(FileExistsError):
            extract_frames.extract(self.video, "S01", self.dir / "out", every=10)
        extract_frames.extract(self.video, "S01", self.dir / "out", every=10, overwrite=True)

    def test_bad_inputs(self):
        with self.assertRaises(FileNotFoundError):
            extract_frames.extract(self.dir / "missing.mp4", "S01", self.dir / "o", every=5)
        with self.assertRaises(ValueError):
            extract_frames.extract(self.video, "bad id!", self.dir / "o", every=5)
        with self.assertRaises(ValueError):
            extract_frames.extract(self.video, "S01", self.dir / "o", every=5, crop=(0, 0, 999, 999))
        bad = self.dir / "bad.mp4"
        bad.write_bytes(b"garbage")
        with self.assertRaises(ValueError):
            extract_frames.extract(bad, "S01", self.dir / "o2", every=5)


class UnicodePathTests(unittest.TestCase):
    def test_roundtrip_non_ascii_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "thư mục" / "ảnh xe.png"
            p.parent.mkdir()
            img = make_image(0)
            self.assertTrue(imwrite_unicode(p, img, ".png"))
            np.testing.assert_array_equal(imread_unicode(p), img)


if __name__ == "__main__":
    unittest.main()

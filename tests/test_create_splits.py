"""Tests for session-level split generation; fixtures are software-only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.config import load_config  # noqa: E402
from common.io import imwrite_unicode  # noqa: E402
from create_splits import create_manifest, split_sessions  # noqa: E402


class SessionSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_nonempty(self):
        ids = [f"S{i:02d}" for i in range(1, 8)]
        one = split_sessions(ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=17)
        two = split_sessions(ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=17)
        self.assertEqual(one, two)
        self.assertTrue(all(one[name] for name in ("train", "val", "test")))
        flattened = [sid for group in one.values() for sid in group]
        self.assertEqual(sorted(flattened), sorted(ids))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_requires_three_sessions_and_valid_ratios(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            split_sessions(["S01"], {"train": 0.7, "val": 0.15, "test": 0.15})
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            split_sessions(["S01", "S02", "S03"], {"train": 0.8, "val": 0.15, "test": 0.15})

    def test_manifest_rejects_invalid_annotation_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann_dir = root / "annotations"
            ann_dir.mkdir()
            (ann_dir / "S01.json").write_text("{bad json", encoding="utf-8")
            cfg = load_config()
            with self.assertRaisesRegex(ValueError, "cannot create splits"):
                create_manifest(ann_dir, root, cfg, {"train": 0.7, "val": 0.15, "test": 0.15})

    def test_manifest_uses_sessions_and_validates_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann_dir = root / "annotations"
            ann_dir.mkdir()
            frames_dir = root / "frames"
            frames_dir.mkdir()
            for i in range(3):
                sid = f"S{i:02d}"
                image = frames_dir / f"{sid}.png"
                imwrite_unicode(image, np.zeros((20, 30, 3), dtype=np.uint8), ".png")
                doc = {"session_id": sid, "image_size": [30, 20], "frames": [
                    {"frame_id": sid + "_000", "file": f"frames/{sid}.png", "boxes": []}]}
                (ann_dir / f"{sid}.json").write_text(json.dumps(doc), encoding="utf-8")
            manifest = create_manifest(ann_dir, root, load_config(),
                                       {"train": 0.7, "val": 0.15, "test": 0.15}, seed=3)
            self.assertEqual(manifest["group_by"], "session")
            self.assertEqual(sum(map(len, manifest["sessions"].values())), 3)


if __name__ == "__main__":
    unittest.main()

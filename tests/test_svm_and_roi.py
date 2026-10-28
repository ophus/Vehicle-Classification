"""Tests for features/hog_color.py, scripts/train_svm.py and scripts/check_roi.py.

The synthetic patches (flat colours + noise) only prove the CODE path works and that the pipeline can
separate trivially different inputs. They say NOTHING about performance on real vehicles.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_roi as cr  # noqa: E402
import train_svm as ts  # noqa: E402
from common.config import load_config  # noqa: E402
from features.hog_color import extract_features, feature_dim  # noqa: E402

CFG = load_config()
NAMES = ["motorbike", "car", "truck", "unknown", "background"]


def fake_patches(n_per_class, seed, classes=(0, 1, 4)):
    """class 0: red-ish, 1: blue-ish, 2: green-ish (BGR), 3: yellow-ish, 4: gray; plus noise."""
    rng = np.random.default_rng(seed)
    base = {0: (30, 30, 200), 1: (200, 30, 30), 2: (30, 200, 30), 3: (30, 200, 200), 4: (110, 110, 110)}
    X, y = [], []
    for c in classes:
        for _ in range(n_per_class):
            p = np.clip(np.array(base[c], np.float32) + rng.normal(0, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
            X.append(p)
            y.append(c)
    return {"X": np.stack(X), "y": np.array(y, np.int64), "class_names": np.array(NAMES, dtype=str)}


class FeatureTests(unittest.TestCase):
    def test_dimension_and_determinism(self):
        X = fake_patches(2, 0)["X"]
        F = extract_features(X, CFG["features"])
        self.assertEqual(F.shape, (6, feature_dim(64, CFG["features"])))
        self.assertEqual(F.shape[1], 1764 + 768 + 48)
        self.assertEqual(F.dtype, np.float32)
        np.testing.assert_array_equal(F, extract_features(X, CFG["features"]))
        self.assertFalse(np.allclose(F[0], F[-1]))

    def test_color_histogram_is_ycrcb(self):
        red = np.zeros((1, 64, 64, 3), np.uint8)
        red[..., 2] = 255                                   # BGR red
        hist = extract_features(red, CFG["features"])[0][-48:]
        expected = np.zeros(48, np.float32)
        expected[[4, 16 + 15, 32 + 5]] = 1.0                # Y~76 -> bin 4, Cr=255 -> bin 15, Cb~85 -> bin 5
        np.testing.assert_allclose(hist, expected, atol=1e-6)

    def test_wrong_patch_shape_rejected(self):
        with self.assertRaises(ValueError):
            from features.hog_color import extract_one, make_hog
            extract_one(np.zeros((32, 32, 3), np.uint8), CFG["features"], make_hog(64, CFG["features"]["hog"]))


class RemapTests(unittest.TestCase):
    def test_drop_and_merge(self):
        y = np.array([0, 1, 2, 3, 4, 2])
        keep, ny, names = ts.remap(y, NAMES, drop=["truck", "unknown"])
        self.assertEqual(names, ["motorbike", "car", "background"])
        self.assertEqual(ny.tolist(), [0, 1, 2])
        keep, ny, names = ts.remap(y, NAMES, merge={"truck": "other", "unknown": "other"})
        self.assertEqual(names, ["motorbike", "car", "other", "background"])
        self.assertEqual(ny.tolist(), [0, 1, 2, 2, 3, 2])
        self.assertTrue(keep.all())

    def test_merge_into_earlier_existing_class(self):
        keep, ny, names = ts.remap(np.array([3, 4, 1]), NAMES, merge={"unknown": "motorbike"})
        self.assertEqual(names, ["motorbike", "car", "truck", "background"])
        self.assertEqual(ny.tolist(), [0, 3, 1])

    def test_bad_names(self):
        with self.assertRaises(ValueError):
            ts.remap(np.array([0]), NAMES, drop=["plane"])
        with self.assertRaises(ValueError):
            ts.remap(np.array([0]), NAMES, drop=["background"])

    def test_parse_pairs(self):
        self.assertEqual(ts.parse_pairs("truck=other,unknown=other"), {"truck": "other", "unknown": "other"})
        with self.assertRaises(Exception):
            ts.parse_pairs("truck")


class TrainEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        self.art = Path(self.tmp.name) / "art"
        pdir = self.data / "processed" / "patches"
        pdir.mkdir(parents=True)
        np.savez_compressed(pdir / "patches_train.npz", **fake_patches(20, 1, classes=(0, 1, 2, 4)))
        np.savez_compressed(pdir / "patches_val.npz", **fake_patches(8, 2, classes=(0, 1, 4)))
        self.env = {"VR_DATA_ROOT": str(self.data), "VR_ARTIFACTS_DIR": str(self.art)}

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *argv):
        with mock.patch.dict(os.environ, self.env), mock.patch.object(sys, "argv", ["train_svm.py", *argv]):
            return ts.main()

    def test_train_save_reload_and_outputs(self):
        self.assertEqual(self.run_main("--name", "t1", "--drop", "unknown", "--c-grid", "0.1,1"), 0)
        card = json.loads((self.art / "models" / "t1_card.json").read_text())
        self.assertEqual(card["class_order"], ["motorbike", "car", "truck", "background"])
        self.assertTrue(card["roundtrip_predictions_identical"])
        met = json.loads((self.art / "metrics" / "t1_val.json").read_text())
        self.assertGreater(met["accuracy"], 0.9)                       # trivially separable synthetic data
        self.assertEqual(met["per_class"]["truck"]["support"], 0)      # absent from val -> reported, not crashed
        self.assertNotIn("test", json.dumps(met["split"]))
        self.assertTrue((self.art / "figures" / "t1_val_confusion.png").is_file())
        self.assertEqual(joblib.load(self.art / "models" / "t1.joblib")["svm"].class_weight, "balanced")
        bundle = joblib.load(self.art / "models" / "t1.joblib")
        X = fake_patches(3, 5, classes=(0, 1, 4))["X"]
        pred = bundle["svm"].predict(bundle["scaler"].transform(extract_features(X, bundle["feature_cfg"])))
        self.assertEqual(pred.tolist(), [0] * 3 + [1] * 3 + [3] * 3)   # motorbike, car, background(index 3)

    def test_scaler_is_fitted_on_train_only_and_C_chosen_by_val(self):
        self.assertEqual(self.run_main("--name", "t5", "--drop", "unknown", "--c-grid", "1e-9,1"), 0)
        bundle = joblib.load(self.art / "models" / "t5.joblib")
        d = np.load(self.data / "processed" / "patches" / "patches_train.npz")
        keep = d["y"] != 3
        expected_mean = extract_features(d["X"][keep], CFG["features"]).mean(axis=0)
        np.testing.assert_allclose(bundle["scaler"].mean_, expected_mean, rtol=1e-4, atol=1e-5)   # train only
        self.assertEqual(bundle["C"], 1.0)                                                         # tiny C underfits on val

    def test_refuses_overwrite(self):
        self.assertEqual(self.run_main("--name", "t2", "--drop", "unknown", "--c-grid", "1"), 0)
        self.assertEqual(self.run_main("--name", "t2", "--drop", "unknown", "--c-grid", "1"), 2)
        self.assertEqual(self.run_main("--name", "t2", "--drop", "unknown", "--c-grid", "1", "--overwrite"), 0)

    def test_class_without_train_patches_is_an_error(self):
        self.assertEqual(self.run_main("--name", "t3", "--c-grid", "1"), 2)       # 'unknown' has no train patches

    def test_test_split_is_never_read(self):
        pdir = self.data / "processed" / "patches"
        self.assertFalse((pdir / "patches_test.npz").exists())                    # no test file at all
        self.assertEqual(self.run_main("--name", "t4", "--drop", "unknown", "--c-grid", "1"), 0)


class RoiReportTests(unittest.TestCase):
    def test_report_counts_inside_outside(self):
        poly = np.array([[0, 100], [200, 100], [200, 200], [0, 200]], np.float32).reshape(-1, 1, 2)
        frames = [{"frame_id": "a", "boxes": [{"bbox": [10, 120, 50, 180], "label": "car"},      # centre y 150: inside
                                              {"bbox": [10, 10, 50, 40], "label": "motorbike"}]},  # centre y 25: outside
                  {"frame_id": "b", "boxes": [{"bbox": [100, 20, 140, 60], "label": "motorbike"}]}]
        rep = cr.roi_report(frames, poly)
        self.assertEqual((rep["total"], rep["outside"]), (3, 2))
        self.assertEqual(rep["outside_per_class"], {"motorbike": 2})
        self.assertEqual(rep["outside_per_frame"], {"a": 1, "b": 1})
        self.assertEqual(rep["outside_centre_y"][0], 25)

    def test_draw_colours_inside_green_outside_red(self):
        poly = np.array([[0, 50], [100, 50], [100, 100], [0, 100]], np.float32).reshape(-1, 1, 2)
        img = np.zeros((100, 100, 3), np.uint8)
        out = cr.draw(img, poly, [{"bbox": [10, 60, 40, 90]}, {"bbox": [60, 5, 90, 30]}])
        self.assertEqual(tuple(out[75, 10]), (0, 200, 0))     # left edge of inside box
        self.assertEqual(tuple(out[15, 60]), (0, 0, 255))     # left edge of outside box


if __name__ == "__main__":
    unittest.main()

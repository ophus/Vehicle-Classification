"""HOG + color features for the classical pipeline (baseline B). Shared by training and inference.

Library-provided parts (must be stated in the report): cv2.HOGDescriptor (HOG computation),
cv2.cvtColor / cv2.resize (color conversion, binning). Feature layout (default config, 64x64 patch):
    HOG on grayscale: 7x7 blocks x 4 cells x 9 bins = 1764
    spatial binning of YCrCb resized to 16x16:      768
    YCrCb histogram, 16 bins per channel:             48   -> 2580 dims
"""
import cv2
import numpy as np


def make_hog(size, hog_cfg):
    return cv2.HOGDescriptor((size, size), (hog_cfg["block"],) * 2, (hog_cfg["stride"],) * 2,
                             (hog_cfg["cell"],) * 2, hog_cfg["nbins"])


def feature_dim(size, cfg):
    h = cfg["hog"]
    blocks = ((size - h["block"]) // h["stride"] + 1) ** 2
    cells_per_block = (h["block"] // h["cell"]) ** 2
    return blocks * cells_per_block * h["nbins"] + 3 * cfg["spatial"] ** 2 + 3 * cfg["hist_bins"]


def extract_one(patch, cfg, hog):
    size = hog.winSize[0]
    if patch.shape[:2] != (size, size) or patch.dtype != np.uint8:
        raise ValueError(f"patch must be {size}x{size} uint8, got {patch.shape} {patch.dtype}")
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    f_hog = hog.compute(gray).ravel()
    ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    sp = cfg["spatial"]
    f_sp = cv2.resize(ycc, (sp, sp), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    f_hist = np.concatenate([np.histogram(ycc[:, :, c], bins=cfg["hist_bins"], range=(0, 256))[0]
                             for c in range(3)]).astype(np.float32) / (size * size)
    return np.concatenate([f_hog.astype(np.float32), f_sp, f_hist])


def extract_features(X, cfg):
    """X: (N, S, S, 3) uint8 BGR -> (N, D) float32."""
    if len(X) == 0:
        return np.zeros((0, 0), np.float32)
    size = X.shape[1]
    hog = make_hog(size, cfg["hog"])
    return np.stack([extract_one(p, cfg, hog) for p in X])

"""Patch cropping used by BOTH dataset building and (later) inference, so the two never drift apart."""
import cv2


def crop_patch(img, bbox, size, margin):
    """Crop bbox [x1,y1,x2,y2] enlarged by `margin` (fraction of w/h) on each side, clip to the image,
    then warp to size x size (aspect ratio is NOT preserved, same as the sliding-window path will do)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = max(0, int(round(x1 - margin * w))), max(0, int(round(y1 - margin * h)))
    cx2, cy2 = min(W, int(round(x2 + margin * w))), min(H, int(round(y2 + margin * h)))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        raise ValueError(f"degenerate crop from bbox {bbox}")
    patch = img[cy1:cy2, cx1:cx2]
    interp = cv2.INTER_AREA if patch.shape[0] * patch.shape[1] > size * size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interp)

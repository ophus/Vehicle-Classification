"""Image IO that works with non-ASCII paths on Windows (cv2.imread/imwrite do not)."""
import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img, ext=".jpg", params=None):
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True

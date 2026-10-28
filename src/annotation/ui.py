"""Interaction logic of the labeling tool, independent from any window (so it is testable).

scripts/label_tool.py only wires this class to OpenCV's imshow/waitKey/setMouseCallback.
HUD text is ASCII on purpose (cv2.putText cannot draw Vietnamese diacritics).
"""
import cv2
import numpy as np

from common.io import imread_unicode

COLORS = {"motorbike": (0, 200, 0), "car": (255, 120, 0), "truck": (0, 0, 255), "unknown": (0, 200, 255)}
HELP = "drag=box  RightClick=delete  1-4=class  f=fine  o=occl  t=trunc | n/Space=next  b=back  u=undo  j=next unreviewed  s=save  q=quit"


class LabelSession:
    HUD_H = 58
    MIN_CANVAS_W = 900

    def __init__(self, store, max_w=1500, max_h=820, start=None):
        self.store = store
        self.max_w, self.max_h = max_w, max_h
        self.classes = store.classes
        self.cls_i, self.fine_i = 0, 0
        self.occ, self.trunc = 0, 0
        self.drag_start = self.drag_now = None
        self.message = ""
        self.idx = store.first_unreviewed() if start is None else max(0, min(start, len(store) - 1))
        self._set_class(0)
        self._load_current()

    # ---- state helpers -------------------------------------------------
    @property
    def cls(self):
        return self.classes[self.cls_i]

    @property
    def fine_list(self):
        return self.store.fine_by_class[self.cls]

    @property
    def fine(self):
        return self.fine_list[self.fine_i]

    def _set_class(self, i):
        self.cls_i = i
        self.fine_i = self.fine_list.index(self.store.default_fine(self.cls))

    def _load_current(self):
        fr = self.store.frames[self.idx]
        W, H = self.store.image_size
        self.scale = min(1.0, self.max_w / W, self.max_h / H)
        self.locked = False
        img = imread_unicode(self.store.root / fr["file"])
        if img is None:
            self.img, self.locked = np.full((H, W, 3), 80, np.uint8), True
            self.message = "CANNOT READ IMAGE - frame locked"
        elif [img.shape[1], img.shape[0]] != [W, H]:
            self.img, self.locked = np.full((H, W, 3), 80, np.uint8), True
            self.message = f"size {img.shape[1]}x{img.shape[0]} != session {W}x{H} - frame locked"
        else:
            self.img = img

    def goto(self, i, mark_reviewed=False):
        if mark_reviewed:
            self.store.set_reviewed(self.idx, True)
        new = max(0, min(i, len(self.store) - 1))
        if new == self.idx and i >= len(self.store):
            self.message = "last frame reached - press q to save and quit"
        self.store.save()
        self.idx = new
        self._load_current()

    # ---- coordinates ---------------------------------------------------
    def to_img(self, x, y):
        return x / self.scale, (y - self.HUD_H) / self.scale

    # ---- events --------------------------------------------------------
    def on_mouse(self, event, x, y, flags=0, param=None):
        if self.locked:
            return
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.HUD_H:
            self.drag_start = self.drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            self.drag_start = self.drag_now = None
            ax, ay = self.to_img(x0, y0)
            bx, by = self.to_img(x, y)
            ok = self.store.add_box(self.idx, [ax, ay, bx, by], self.cls, self.fine, self.occ, self.trunc)
            self.message = f"added {self.cls}/{self.fine}" if ok else "drag too small - ignored"
        elif event == cv2.EVENT_RBUTTONDOWN and y >= self.HUD_H:
            ix, iy = self.to_img(x, y)
            self.message = "box deleted" if self.store.remove_at(self.idx, ix, iy) else "no box here"

    def on_key(self, k):
        """Return False when the user asks to quit."""
        if k is None or k == -1:
            return True
        k &= 0xFF
        if k in (27, ord("q")):
            return False
        ch = chr(k)
        if ch in "123456789"[:len(self.classes)]:
            self._set_class(int(ch) - 1)
            self.message = f"class = {self.cls}"
        elif ch == "f":
            self.fine_i = (self.fine_i + 1) % len(self.fine_list)
            self.message = f"fine = {self.fine}"
        elif ch == "o":
            self.occ = (self.occ + 1) % 3
        elif ch == "t":
            self.trunc ^= 1
        elif ch in ("n", " "):
            self.goto(self.idx + 1, mark_reviewed=True)
        elif ch == "b":
            self.goto(self.idx - 1)
        elif ch == "j":
            self.goto(self.store.first_unreviewed())
        elif ch == "u":
            self.message = "last box removed" if self.store.remove_last(self.idx) else "nothing to undo"
        elif ch == "s":
            self.store.save(force=True)
            self.message = "saved"
        return True

    # ---- drawing -------------------------------------------------------
    def render(self):
        W, H = self.store.image_size
        dw, dh = int(round(W * self.scale)), int(round(H * self.scale))
        view = cv2.resize(self.img, (dw, dh), interpolation=cv2.INTER_AREA) if self.scale != 1.0 else self.img.copy()
        for b in self.store.frames[self.idx]["boxes"]:
            x1, y1, x2, y2 = (int(round(v * self.scale)) for v in b["bbox"])
            col = COLORS.get(b["label"], (255, 255, 255))
            cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
            tag = f"{b['label']}/{b['fine_label']}" + (f" o{b['occlusion']}" if b.get("occlusion") else "") \
                + (" T" if b.get("truncated") else "")
            cv2.putText(view, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if self.drag_start and self.drag_now:
            p0 = (self.drag_start[0], self.drag_start[1] - self.HUD_H)
            p1 = (self.drag_now[0], self.drag_now[1] - self.HUD_H)
            cv2.rectangle(view, p0, p1, (255, 255, 255), 1)

        canvas = np.zeros((self.HUD_H + dh, max(dw, self.MIN_CANVAS_W), 3), np.uint8)
        canvas[self.HUD_H:self.HUD_H + dh, :dw] = view
        fr = self.store.frames[self.idx]
        l1 = f"[{self.idx + 1}/{len(self.store)}] {fr['frame_id']}  boxes:{len(fr['boxes'])}  " \
             f"reviewed:{'yes' if fr['reviewed'] else 'no'}  scale:{self.scale:.2f}"
        l2 = f"NEXT BOX -> class:{self.cls} ({self.cls_i + 1})  fine:{self.fine}  occl:{self.occ}  trunc:{self.trunc}   {self.message}"
        for n, (txt, col) in enumerate(((l1, (255, 255, 255)), (l2, COLORS[self.cls]), (HELP, (170, 170, 170)))):
            cv2.putText(canvas, txt, (6, 16 + 18 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return canvas

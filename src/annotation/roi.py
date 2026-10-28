"""Logic of the ROI polygon picker (window-independent, so it is testable).

The polygon marks the region you actually labeled (e.g. the road lanes facing the camera).
It is used to sample background patches only inside that region.
"""
import cv2
import numpy as np

HELP = "left-click=add point  right-click/u=undo  c=clear  Enter=finish(>=3 pts)  q/Esc=cancel"


class RoiPicker:
    HUD_H = 40
    MIN_CANVAS_W = 820

    def __init__(self, img, max_w=1500, max_h=820):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.scale = min(1.0, max_w / self.W, max_h / self.H)
        self.points = []
        self.done = False
        self.message = ""

    def _to_img(self, x, y):
        ix = int(round(min(max(x / self.scale, 0), self.W - 1)))
        iy = int(round(min(max((y - self.HUD_H) / self.scale, 0), self.H - 1)))
        return ix, iy

    def on_mouse(self, event, x, y, flags=0, param=None):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.HUD_H:
            self.points.append(self._to_img(x, y))
            self.message = f"{len(self.points)} point(s)"
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def on_key(self, k):
        """Return False when the picker should close (check `done` to see if the polygon was accepted)."""
        if k is None or k == -1:
            return True
        k &= 0xFF
        if k in (27, ord("q")):
            return False
        if k in (ord("u"), 8) and self.points:
            self.points.pop()
        elif k == ord("c"):
            self.points = []
        elif k in (13, 10):
            if len(self.points) >= 3:
                self.done = True
                return False
            self.message = "need at least 3 points"
        return True

    def polygon_string(self):
        return ";".join(f"{x},{y}" for x, y in self.points)

    def render(self):
        dw, dh = int(round(self.W * self.scale)), int(round(self.H * self.scale))
        view = cv2.resize(self.img, (dw, dh), interpolation=cv2.INTER_AREA) if self.scale != 1.0 else self.img.copy()
        pts = np.array([(int(round(x * self.scale)), int(round(y * self.scale))) for x, y in self.points], np.int32)
        if len(pts) >= 3:
            overlay = view.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            view = cv2.addWeighted(overlay, 0.25, view, 0.75, 0)
        if len(pts) >= 2:
            cv2.polylines(view, [pts], len(pts) >= 3, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(view, tuple(int(v) for v in p), 4, (0, 0, 255), -1)
        canvas = np.zeros((self.HUD_H + dh, max(dw, self.MIN_CANVAS_W), 3), np.uint8)
        canvas[self.HUD_H:, :dw] = view
        cv2.putText(canvas, f"ROI picker  image {self.W}x{self.H}  points:{len(self.points)}  {self.message}",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, HELP, (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        return canvas

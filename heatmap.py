import numpy as np
import cv2


class GazeHeatmap:
    def __init__(self, width, height, decay=0.98, blob_sigma=25):
        self.width = width
        self.height = height
        self.decay = decay
        self.blob_sigma = blob_sigma
        self.accum = np.zeros((height, width), dtype=np.float32)

        k = int(blob_sigma * 3) | 1
        ax = np.arange(-k // 2 + 1, k // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        self._kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * blob_sigma ** 2)).astype(np.float32)

    def add_point(self, x, y):
        x, y = int(x), int(y)
        kh, kw = self._kernel.shape
        x0, y0 = x - kw // 2, y - kh // 2
        x1, y1 = x0 + kw, y0 + kh

        tx0, ty0 = max(x0, 0), max(y0, 0)
        tx1, ty1 = min(x1, self.width), min(y1, self.height)
        if tx0 >= tx1 or ty0 >= ty1:
            return

        kx0, ky0 = tx0 - x0, ty0 - y0
        kx1, ky1 = kx0 + (tx1 - tx0), ky0 + (ty1 - ty0)

        self.accum[ty0:ty1, tx0:tx1] += self._kernel[ky0:ky1, kx0:kx1]

    def step_decay(self):
        if self.decay < 1.0:
            self.accum *= self.decay

    def render(self, colormap=cv2.COLORMAP_JET, alpha=0.55, background=None):
        norm = self.accum.copy()
        m = norm.max()
        if m > 1e-6:
            norm = (norm / m * 255).astype(np.uint8)
        else:
            norm = norm.astype(np.uint8)

        colored = cv2.applyColorMap(norm, colormap)

        if background is None:
            return colored

        bg = cv2.resize(background, (self.width, self.height))
        mask = (norm > 5).astype(np.float32)[..., None]
        blended = bg.astype(np.float32) * (1 - alpha * mask) + colored.astype(np.float32) * (alpha * mask)
        return blended.astype(np.uint8)

    def save(self, path, background=None):
        cv2.imwrite(path, self.render(background=background))
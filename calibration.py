import pickle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def build_calibration_targets(screen_w, screen_h, margin_ratio=0.08, grid=(3, 3)):
    cols, rows = grid
    mx = int(screen_w * margin_ratio)
    my = int(screen_h * margin_ratio)
    xs = np.linspace(mx, screen_w - mx, cols)
    ys = np.linspace(my, screen_h - my, rows)
    points = [(int(x), int(y)) for y in ys for x in xs]
    return points


class GazeCalibrator:
    def __init__(self, degree=2, alpha=1.0):
        self.model_x = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
        self.model_y = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
        self._fitted = False

    def fit(self, features, targets):
        features = np.asarray(features)
        targets = np.asarray(targets)
        self.model_x.fit(features, targets[:, 0])
        self.model_y.fit(features, targets[:, 1])
        self._fitted = True

    def predict(self, feature_vec):
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fit yet.")
        f = np.asarray(feature_vec).reshape(1, -1)
        x = self.model_x.predict(f)[0]
        y = self.model_y.predict(f)[0]
        return float(x), float(y)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
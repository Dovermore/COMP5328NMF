# The following class deals with all the pre-processing steps to be done on the
# image data

import numpy as np
from sklearn.base import TransformerMixin


# Rescale image to between 0 and 1
class ImageNormalizer(TransformerMixin):
    def __init__(self, min=0, max=255):
        self.min = min
        self.max = max
        self.adaptive = False
        if self.min is None or self.max is None:
            self.adaptive = True

    def fit(self, X, y=None, override=False):
        if override or self.adaptive:
            self.min = np.ndarray.min(X)
            self.max = np.ndarray.max(X)
        return self

    def transform(self, X):
        range = self.max - self.min
        X = (X - self.min) / range
        return X

    def invtransform(self, X):
        range = self.max - self.min
        X = X * range + self.min
        return X

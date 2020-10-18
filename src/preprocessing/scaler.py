# The following class deals with all the pre-processing steps to be done on the
# image data

# Author: Lupita Sahu
import numpy as np
from sklearn.base import TransformerMixin


# Rescale image to between 0 and 1
class ImageNormalizer(TransformerMixin):
    def __init__(self, min=0, max=255):
        self.min = min
        self.max = max

    def fit(self, X, y=None, override=False):
        if override or self.min is None or self.max is None:
            self.min = np.ndarray.min(X)
            self.max = np.ndarray.max(X)
        return self

    # Salt and pepper algs here.
    def transform(self, X):
        range = self.max - self.min
        X = (X - self.min) / range
        return X

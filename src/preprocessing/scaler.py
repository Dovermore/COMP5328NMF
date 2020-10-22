"""
File: noise.py
Author: Calvin Huang/Lupita Sahu
Github: https://github.com/dovermore
Description: This is part of the assignment one for Advanced machine learning
             it integrates sklearn like interface and constructs basic scaling module.
"""

import numpy as np
from sklearn.base import TransformerMixin


class ImageNormalizer(TransformerMixin):
    """
    This data transformer normalises input image to the range of 0 to 1.
    """
    def __init__(self, min=0, max=255):
        """
        Creates image normalizer class with the given value considered as min and max value of image
        set min or max to None to enable adaptive normalization (automatic finding of min and max
        from input array)

        Args:
            min: min value in the array
            max: max value in the array
        """
        self.min = min
        self.max = max
        self.adaptive = False
        if self.min is None or self.max is None:
            self.adaptive = True

    def fit(self, X, y=None, override=False):
        """
        Finds the min max value of array if the transformer is in adaptive mode or override is True

        Args:
            X: Input array
            y: not used
            override: If override the existing min max value

        Returns:
            self to chain commands
        """
        if override or self.adaptive:
            self.min = np.ndarray.min(X)
            self.max = np.ndarray.max(X)
        return self

    def transform(self, X):
        """
        Transform data to range of zero to one

        Args:
            X: Input array

        Returns:
            Normalised data
        """
        range = self.max - self.min
        X = (X - self.min) / range
        return X

    def invtransform(self, X):
        """
        Inverse transform data from zero to one back to original scale

        Args:
            X: Input array

        Returns:
            Original data before transformation
        """
        range = self.max - self.min
        X = X * range + self.min
        return X

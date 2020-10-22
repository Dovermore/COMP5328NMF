"""
File: noise.py
Author: Calvin Huang/Lupita Sahu
Github: https://github.com/dovermore
Description: This is part of the assignment one for Advanced machine learning
             it integrates sklearn like interface and constructs basic noise adding module.
"""
import numpy as np
from sklearn.base import TransformerMixin


class SaltNPepper(TransformerMixin):
    """
    This class deals with creating salt and pepper noise which is controlled by 2 parameters.
    """
    def __init__(self, p, r):
        """
        Creates the basic salt and pepper noise data transformer

        Args:
            p: p for noise level(0-1)
            r: r for salt/pepper ratio (0-1)
        """
        self.p = p
        self.r = r

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Add salt and pepper noise to a copy of input data X

        Args:
            X: Input array

        Returns:
            X with noise added
        """
        X = X.T
        out = np.copy(X)

        num_salt = np.ceil(self.p * X[0].size * self.r)
        num_pepper = np.ceil(self.p * X[0].size * (1. - self.r))
        for img in out:
            # Salt mode
            coords = tuple(np.random.randint(0, i - 1, int(num_salt))
                            for i in img.shape)
            img[coords] = 255
            # Pepper mode
            coords = tuple(np.random.randint(0, i - 1, int(num_pepper))
                            for i in img.shape)
            img[coords] = 0
        return out.T

class Gaussian(TransformerMixin):
    """
    This class deals with creating salt and pepper noise which is controlled by 2 parameters.
    """
    def __init__(self, mean=0, sigma=20):
        """
        Creates the basic gaussian noise data transformer

        Args:
            mean: mean of gaussian noise
            sigma: standard deviation of gaussian noise
        """
        self.mean = mean
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Add gaussian noise to a copy of input data X

        Args:
            X: Input array

        Returns:
            X with noise added
        """
        out = X + np.random.normal(self.mean, self.sigma, X.shape)
        return out

import numpy as np
from sklearn.base import TransformerMixin


# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
# The following class deals with creating salt and pepper noise which
# controlled by 2 parameters.
# Parameters: p for noise level(0-1), r for salt/pepper ratio (0-1)
class SaltNPepper(TransformerMixin):
    def __init__(self, p, r):
        self.p = p
        self.r = r

    def fit(self, X, y=None):
        return self

    # Salt and pepper algs here.
    def transform(self, X):
        X = X.T
        out = np.copy(X)

        num_pepper = np.ceil(self.p * X[0].size * (1. - self.r))
        num_salt = np.ceil(self.p * X[0].size * self.r)
        for img in out:
            # Salt mode
            coords = tuple([np.random.randint(0, i - 1, int(num_salt))
                            for i in img.shape])
            img[coords] = 1
            # Pepper mode
            coords = tuple([np.random.randint(0, i - 1, int(num_pepper))
                            for i in img.shape])
            img[coords] = 0
        return out.T


class Gaussian(TransformerMixin):
    def __init__(self, mean=0, sigma=20):
        self.mean = mean
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    # Salt and pepper algs here.
    def transform(self, X):
        """Add Gaussian noise to an image"""
        out = X + np.random.normal(self.mean, self.sigma, X.shape)
        return out

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
    def fit_transform(self, X):
        out = np.copy(X)
        # Salt mode
        num_salt = np.ceil(self.p * X.size * self.r)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt))
                  for i in X.shape])
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(self.p * X.size * (1. - self.r))
        coords = tuple([np.random.randint(0, i - 1, int(num_pepper))
                  for i in X.shape])
        out[coords] = 0
        return out

# This file holds NMF algorithm with l2 norm loss
# Author: Calvin Huang (zhuq9812)

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy as sp


class NmfL2Estimator(BaseEstimator, TransformerMixin):
    """
    Base class for nmf l2 estimator. Uses sklearn skeleton for better coherence
    with other parts of the codes.
    """

    def __init__(self, ):
        """
        TODO

        inputs: dimension
        """
        # store hyper parameters
        # TODO
        self.R = np.zeros(1)
        pass

    def fit(self, X, y=None):
        """
        TODO
        """
        pass

    def transform(self, X, y=None):
        """
        TODO
        """
        pass

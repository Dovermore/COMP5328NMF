# This file holds NMF algorithm with l2 norm loss
# Author: Calvin Huang (zhuq9812)

import numpy as np
from .base_nmf import BaseNmfEstimator


class NmfL2Estimator(BaseNmfEstimator):
    """
    Base class for nmf l2 estimator. Uses sklearn skeleton for better coherence
    with other parts of the codes.

    For now only the function for updating D, R, loss should be updated
    """

    def get_next_D(self, X, D, R):
        """
        Compute the next value of D based on given input, D and R

        This is the update rule for l2
        """
        next_D = D * ((X @ R.T) / (D @ R @ R.T))
        return next_D

    def get_next_R(self, X, D, R):
        """
        Compute the next value of R based on given input, D and R

        This is the update rule for l2
        """
        next_R = R * ((D.T @ X) / (D.T @ D @ R))
        return next_R

    @classmethod
    def loss(cls, X, D, R):
        """
        use the default l2 loss
        """
        return super().loss(X, D, R)

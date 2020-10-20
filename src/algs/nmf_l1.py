# This file holds NMF algorithm with l1 norm loss
# Author: Calvin Huang (zhuq9812) & Matthew Dong (mdon9995)
# Special Thanks: jordancoding13

import numpy as np
from .base_nmf import BaseNmfEstimator


class NmfL1Estimator(BaseNmfEstimator):
    """
    Base class for nmf l1 estimator. Uses sklearn skeleton for better coherence
    with other parts of the codes.

    For now only the function for updating D, R, loss should be updated
    """

    def get_next_D(self, X, D, R):
        """
        Compute the next value of D based on given input, D and R

        This is the update rule for l1
        """
        eps = X.var() / D.shape[1]
        W = 1 / (np.sqrt(np.square(X - (D @ R))) + eps ** 2)

        denom_D = (W * (D @ R)) @ (R.T)
        denom_D[denom_D == 0] = np.finfo(np.float32).eps

        next_D = D * ((W * X) @ (R.T)) / denom_D
        return next_D

    def get_next_R(self, X, D, R):
        """
        Compute the next value of R based on given input, D and R

        This is the update rule for l1
        """
        eps = X.var() / D.shape[1]
        W = 1 / (np.sqrt(np.square(X - (D @ R))) + eps ** 2)

        denom_R = D.T @ (W * (D @ R))
        denom_R[denom_R == 0] = np.finfo(np.float32).eps

        next_R = R * (D.T @ (W * X)) / denom_R
        return next_R

    @classmethod
    def loss(cls, X, D, R):
        """
        use the default l1 loss
        """
        return np.sum(np.abs(X - (D @ R)))

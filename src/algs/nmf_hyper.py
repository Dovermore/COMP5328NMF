# This file holds NMF algorithm with hyperplane norm loss
# Author: Calvin Huang (zhuq9812)

import numpy as np
from .base_nmf import BaseNmfEstimator


class NmfHyperEstimator(BaseNmfEstimator):
    """
    Base class for nmf hypersurface estimator.
    Uses sklearn skeleton for better coherence with other parts of the codes.

    For now only the function for updating D, R, loss should be updated
    """

    def __init__(self, tau=0.5, c=0.5, alpha0=1, beta0=1,
                 max_armijo=1e4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.tau = tau
        self.beta0 = beta0
        self.alpha0 = alpha0
        self.max_armijo = int(max_armijo)

    def _update_RD(self, X):
        next_R = self.get_next_R(X, self.D, self.R)
        next_D = self.get_next_D(X, self.D, self.R)
        return next_R, next_D

    def get_next_D(self, X, D, R):
        """
        Compute the next value of D based on given input, D and R

        This is the update rule for l2
        """
        dD = ((D @ R @ R.T) - (X @ R.T)) / np.sqrt(1 + np.linalg.norm(X - D @ R))
        alpha = self.armijo_search_D(X, D, R, dD)
        next_D = D - alpha * dD
        return next_D

    def get_next_R(self, X, D, R):
        """
        Compute the next value of R based on given input, D and R

        This is the update rule for l2
        """
        dR = ((D.T @ D @ R) - (D.T @ X)) / np.sqrt(1 + np.linalg.norm(X - D @ R))
        beta = self.armijo_search_R(X, D, R, dR)
        next_R = R - beta * dR
        return next_R

    @classmethod
    def loss(cls, X, D, R):
        """
        use the default l2 loss
        """
        return np.sqrt(1 + super().loss(X, D, R)) - 1

    def armijo_search_D(self, X, D, R, dD):
        m = -dD @ dD.T
        alpha = self.alpha0
        t = -self.c * m
        for i in range(self.max_armijo):
            diff = self.loss(X, D, R) - self.loss(X, D - alpha * dD, R)
            if diff > alpha * t:
                break
            alpha = alpha * self.tau
        return alpha

    def armijo_search_R(self, X, D, R, dR):
        m = -dR @ dR.T
        beta = self.beta0
        t = -self.c * m
        for i in range(self.max_armijo):
            diff = self.loss(X, D, R) - self.loss(X, D, R - beta * dR)
            print(diff.shape, m.shape, dR.shape)
            if diff > beta * t:
                break
            beta = beta * self.tau
        return beta

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

    def __init__(self, k=2, tau=0.5, c=0.5, alpha0=1, beta0=1,
                 max_armijo=1e4, max_iter=1e5, output_image=False,
                 verbose=0, log_interval=100):
        super().__init__(k=k, max_iter=max_iter, output_image=output_image,
                         verbose=verbose, log_interval=log_interval)
        self.tau = tau
        self.c = c
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_armijo = int(max_armijo)

        # stop counter for early stopping
        self.stop_counter = 0

    def _update_RD(self, X):
        next_R = self.get_next_R(X, self.D, self.R)
        next_D = self.get_next_D(X, self.D, self.R)
        return next_R, next_D

    def _terminate(self, X, R, D, next_R, next_D):
        if self.loss(X, D, R) - self.loss(X, next_D, next_R) < 0:
            self.stop_counter += 1
        if self.stop_counter >= 100:
            return True
        return False

    def fit(self, X, y=None):
        self.stop_counter = 0
        super().fit(X, y)
        return self

    def get_next_D(self, X, D, R):
        """
        Compute the next value of D based on given input, D and R

        This is the update rule for l2
        """
        dD = ((D @ R @ R.T) - (X @ R.T)) / (self.loss(X, D, R) + 1)
        alpha = self.armijo_search_D(X, D, R, dD)
        next_D = D - alpha * dD
        return next_D

    def get_next_R(self, X, D, R):
        """
        Compute the next value of R based on given input, D and R

        This is the update rule for l2
        """
        dR = ((D.T @ D @ R) - (D.T @ X)) / (self.loss(X, D, R) + 1)
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
        m = -np.linalg.norm(dD.T @ dD)
        alpha = self.alpha0
        t = -self.c * m
        diff = -np.inf
        for i in range(self.max_armijo):
            diff = self.loss(X, D, R) - self.loss(X, D - alpha * dD, R)
            # print(i, "alpha", alpha, diff, alpha * t)
            if diff >= alpha * t:
                break
            alpha = alpha * self.tau
        # print(i, "alpha", alpha, diff, alpha * t)
        return alpha

    def armijo_search_R(self, X, D, R, dR):
        m = -np.linalg.norm(dR.T @ dR)
        beta = self.beta0
        t = -self.c * m
        diff = -np.inf
        for i in range(self.max_armijo):
            diff = self.loss(X, D, R) - self.loss(X, D, R - beta * dR)
            # print(i, "beta", beta, diff, beta * t)
            if diff >= beta * t:
                break
            beta = beta * self.tau
        # print(i, "beta", beta, diff, beta * t)
        return beta

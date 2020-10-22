"""
File: nmf_hyper.py
Author: Calvin Huang
Email: zhua9812@uni.sydney.edu.au
Github: https://github.com/dovermore
Description: This implements the hyperplane loss NMF algorithm,
             Implementation based around
             Hyperplane-based nonnegative matrix factorization with label information 10.1016/j.ins.2019.04.026
             and
             Armijo, Larry. Minimization of functions having Lipschitz continuous first partial derivatives. Pacific J.
             Math. 16 (1966), no. 1, 1--3. https://projecteuclid.org/euclid.pjm/1102995080
"""

import numpy as np
from .base_nmf import BaseNmfEstimator


class NmfHyperEstimator(BaseNmfEstimator):
    """
    Base class for nmf hypersurface estimator.
    Uses sklearn skeleton for better coherence with other parts of the codes.
    """
    def __init__(self,
                 n_components=2,
                 tau=0.5,
                 c=0.5,
                 alpha0=1,
                 beta0=1,
                 init="random",
                 max_armijo=100,
                 max_iter=200,
                 output_image=False,
                 verbose=0,
                 log_interval=np.inf):
        """
        Initialises the Hyper plane NMF estimator

        Args:
            n_components: Hidden dimension of encoding
            tau: The rate at which armijo search decays
            c: The ratio of loss and parameter change at which tarmijo search reqires
            alpha0: The starting value of alpha
            beta0: The starting value of beta
            init: Initialisation strategy
            max_armijo: Maximum number of iterations to run armijo search
            max_iter: maximum number of iteration to perform
            output_image: If the fit_transform should produce the reconstructed image instead of encoding
            verbose: The verbosity of algorithm during training. 1 for loss monitoring, 2 for parameter monitoring
            log_interval: The interval of which to log the algorithm. (only used when verbosity is not 0)
        """
        super().__init__(n_components=n_components,
                         init=init,
                         max_iter=max_iter,
                         output_image=output_image,
                         verbose=verbose,
                         log_interval=log_interval)
        self.tau = tau
        self.c = c
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_armijo = int(max_armijo)

        # stop counter for early stopping
        self.stop_counter = 0

        # For improving performance (monitor max value and starting value)
        self.sum_alpha = 0
        self.amijo_iter_alpha = 0
        self.amijo_call_alpha = 0

        self.sum_beta = 0
        self.amijo_iter_beta = 0
        self.amijo_call_beta = 0

    def _update_DR(self, X):
        """
        Update D and R simultaneously

        Args:
            X: Input array
        """
        next_D = self.get_next_D(X, self.D, self.R)
        next_R = self.get_next_R(X, self.D, self.R)
        return next_D, next_R

    def _terminate(self, X, D, R, next_D, next_R):
        """
        Determine if the algorithm should terminate from the given updates

        Args:
            X: Input array
            D: Dictionary array
            R: Representation array
            next_D: The updated D
            next_R: The updated R

        Returns:
            True if should terminate and finish training, False otherwise.
        """
        if self.loss(X, D, R) - self.loss(X, next_D, next_R) < 0:
            self.stop_counter += 1
        if self.stop_counter >= 20:
            return True
        return False

    def fit(self, X, y=None):
        """
        Fit the model for given input X

        Args:
            X: Input array (shape m x n, where m is feature_size, n is sample size)
            y: not used

        Returns:
            self to chain with transformer.
        """
        self.stop_counter = 0
        super().fit(X, y)
        return self

    def get_next_D(self, X, D, R):
        """
        Compute the next value of D based on given input, D and R

        This is the update rule for l2

        Args:
            X: Input array
            D: Dictionary array
            R: Representation array

        Returns:
            Updated value of D
        """
        dD = ((D @ R @ R.T) - (X @ R.T)) / (self.loss(X, D, R) + 1)
        alpha = self.armijo_search_D(X, D, R, dD)
        next_D = D - alpha * dD
        return next_D

    def get_next_R(self, X, D, R):
        """
        Compute the next value of R based on given input, D and R

        This is the update rule for l2

        Args:
            X: Input array
            D: Dictionary array
            R: Representation array

        Returns:
            Updated value of R
        """
        dR = ((D.T @ D @ R) - (D.T @ X)) / (self.loss(X, D, R) + 1)
        beta = self.armijo_search_R(X, D, R, dR)
        next_R = R - beta * dR
        return next_R

    @classmethod
    def loss(cls, X, D, R):
        """
        Hyperplane loss

        Returns:
            computed loss according to the formula in paper


        """
        return np.sqrt(1 + super().loss(X, D, R)) - 1

    def armijo_search_D(self, X, D, R, dD):
        """
        Armijo search for finding optimal value for D

        Args:
            X: Input array
            D: Dictionary array
            R: Representation array
            dD: the differentiaion of Loss with respect to D

        Returns:
            The optimal update step size for updating D
        """
        self.amijo_call_alpha += 1

        m = -np.linalg.norm(dD.T @ dD)
        alpha = self.alpha0
        t = -self.c * m
        diff = -np.inf
        i = 0
        for i in range(self.max_armijo):
            diff = self.loss(X, D, R) - self.loss(X, D - alpha * dD, R)
            # print(i, "alpha", alpha, diff, alpha * t)
            if diff >= alpha * t:
                break
            alpha = alpha * self.tau
        # print(i, "alpha", alpha, diff, alpha * t)

        self.amijo_iter_alpha += i
        self.sum_alpha += alpha
        return alpha

    def armijo_search_R(self, X, D, R, dR):
        """
        Armijo search for finding optimal value for R

        Args:
            X: Input array
            D: Dictionary array
            R: Representation array
            dR: the differentiaion of Loss with respect to R

        Returns:
            The optimal update step size for updating R
        """
        self.amijo_call_beta += 1

        m = -np.linalg.norm(dR.T @ dR)
        beta = self.beta0
        t = -self.c * m
        diff = -np.inf
        i = 0
        for i in range(self.max_armijo):
            diff = self.loss(X, D, R) - self.loss(X, D, R - beta * dR)
            # print(i, "beta", beta, diff, beta * t)
            if diff >= beta * t:
                break
            beta = beta * self.tau
        # print(i, "beta", beta, diff, beta * t)

        self.amijo_iter_beta += i
        self.sum_beta += beta
        return beta

    @property
    def armijo_stats_alpha(self):
        "reports status of armijo search during training for debug and hand specify value"
        return {"avg_alpha": self.sum_alpha / self.amijo_call_alpha,
                "avg_iter": self.amijo_iter_alpha / self.amijo_call_alpha}

    @property
    def armijo_stats_beta(self):
        "reports status of armijo search during training for debug and hand specify value"
        return {"avg_beta": self.sum_beta / self.amijo_call_beta,
                "avg_iter": self.amijo_iter_beta / self.amijo_call_beta}

# This file holds NMF algorithm with l2 norm loss
# Author: Calvin Huang (zhuq9812)


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class BaseNmfEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, max_iter=1e5, output_image=False, verbose=0):
        """
        TODO: docstring
        """
        # store hyper parameters
        self.k = k
        self.max_iter = max_iter
        self.verbose = verbose
        self.output_image = output_image

        # Use null matrix as placeholder
        self.D = np.zeros(0)
        self.R = np.zeros(0)

    def init_DR(self, X):
        """
        TODO: docstring
        Initializes R and D matrix from give input matrix X
        """
        # d: input feature dimension
        # n: number of examples
        d, n = X.shape

        # Initialize D and R usd in the fitting process
        self.D = np.abs(np.random.randn(d, self.k))
        self.R = np.abs(np.random.randn(self.k, n))

    @classmethod
    def loss(cls, X, D, R):
        """
        Defaulting to l2 loss for good measure
        """
        # residual = X - D @ R
        residual = X - D @ R
        # square of l2 frob norm
        return np.sum(residual ** 2)

    def fit(self, X, y=None):
        """
        TODO
        """
        # Initialize R and D
        self.init_DR(X)

        # At most max iter times
        iter = 0
        while iter < self.max_iter:
            if self.verbose > 0:
                loss = self.loss(X, self.D, self.R)
                print("Iteration: %-4d | loss: %-10.3f" % (iter, loss))

                # Log size of matrices
                D_avg = np.mean(self.D)
                R_avg = np.mean(self.R)
                if self.verbose > 1:
                    print("                |  avgD: %-10.3f, avgR: %-10.3f" %
                          (D_avg, R_avg))
            next_R, next_D = self._update_DR(X)
            iter += 1

            # arrive at stable values, break the loop
            if self._terminate(X, self.R, self.D, next_R, next_D):
                break

            # else assign and continue to next iteration
            self.R = next_R
            self.D = next_D

    def _update_DR(self, X):
        """Default updating option is to update R then based on that update D"""
        # get Rn+1 based on Rn, Dn
        next_R = self.get_next_R(X, self.D, self.R)
        # get Dn+1 based on Rn+1, Dn
        next_D = self.get_next_D(X, self.D, next_R)
        return next_R, next_D

    def _terminate(self, X, D, R, next_D, next_R):
        return np.array_equal(self.R, next_R) and np.array_equal(self.D, next_D)

    def transform(self, X, y=None):
        """
        Takes input and transform according to fitted model
        """
        # Make sure the model is fitted
        assert self.R.shape != (0, )

        # Find optimal R by minimising with the already computed value of D
        R = self.R
        while iter < self.max_iter:
            # get Rn+1 based on Rn, Dn
            next_R = self.get_next_R(X, self.D, R)
            iter += 1
            # arrive at stable values, break the loop
            if np.array_equal(R, next_R):
                break
            # else assign and continue to next iteration
            R = next_R
        if self.output_image:
            return self.D @ R
        return R

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        # Fit then return the lower dimension R values
        if self.output_image:
            return self.D @ self.R
        return R

    def get_next_R(self, X, D, R):
        """
        Compute the next value of R based on given input, D and R

        Placeholder function to be overriden
        """
        return R

    def get_next_D(self, X, D, R):
        """
        Compute the next value of D based on given input, D and R

        Placeholder function to be overriden
        """
        return D

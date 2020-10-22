"""
File: base_nmf.py
Author: Calvin Huang
Email: zhua9812@uni.sydney.edu.au
Github: https://github.com/dovermore
Description: This is part of the assignment one for Advanced machine learning
             it integrates sklearn like interface and constructs basic training
             framwork.
"""



from sklearn.base import BaseEstimator, TransformerMixin
from .svd_init import svd_init
import numpy as np


class BaseNmfEstimator(BaseEstimator, TransformerMixin):
    """ The logic are pretty simple in this case, so I didn't add much comments
        This class tries to immitate how sklearn model works so we are able to
        easily design evaluation funciton with sklearn model to sort of verify
        the correctness of the model
    """
    def __init__(self,
                 n_components=2,
                 init=None,
                 max_iter=200,
                 output_image=False,
                 verbose=0,
                 log_interval=np.inf):
        """
        Constructor of Base NMF estimator class

        Args:
            n_components: number of hidden components
            init: the initialisation strategy used
            max_iter: maximum number of iteration to perform
            output_image: If the fit_transform should produce the reconstructed image instead of encoding
            verbose: The verbosity of algorithm during training. 1 for loss monitoring, 2 for parameter monitoring
            log_interval: The interval of which to log the algorithm. (only used when verbosity is not 0)
        """
        # store hyper parameters
        self.n_components = n_components
        self.init = init.lower() if isinstance(init, str) else init
        self.max_iter = max_iter
        self.verbose = verbose
        self.output_image = output_image
        self.log_interval = log_interval

        # Use null matrix as placeholder
        self.D = np.zeros(0)
        self.R = np.zeros(0)

    def init_DR(self, X):
        """
        Initializes R and D matrix from give input matrix X

        Args:
            X: Input array
        """
        # d: input feature dimension
        # n: number of examples
        d, n = X.shape

        # Initialize D and R usd in the fitting process
        if self.init is None or self.init == "random":
            factor = np.sqrt(X.mean() / self.n_components)
            self.D = np.abs(np.random.randn(d, self.n_components)) * factor
            self.R = np.abs(np.random.randn(self.n_components, n)) * factor
        elif self.init == "zero":
            self.D = np.zeros((d, self.n_components))
            self.R = np.zeros((self.n_components, n))
        elif self.init == "svd":
            self.D, self.R = svd_init(X, p=self.n_components)
        else:
            raise NotImplementedError(f"Found unknown init method {self.init}."
                                      + "Choices are: None, 'random', 'zero', "
                                      + "'svd'")

    @classmethod
    def loss(cls, X, D, R):
        """
        Defaulting to l2 loss for good measure

        Args:
            X: Input array
            D: Dictionary array (W in other context)
            R: Representation array (H in other context)
        """
        # residual = X - D @ R
        residual = X - D @ R
        # square of l2 frob norm
        return np.sum(residual ** 2)

    def fit(self, X, y=None):
        """
        Fit the model for given input X

        Args:
            X: Input array (shape m x n, where m is feature_size, n is sample size)
            y: not used
        """
        # Initialize R and D
        self.init_DR(X)

        # At most max iter times
        iter = 0
        while iter < self.max_iter:
            if self.verbose > 0 or (iter % self.log_interval == 0 and
                                    self.log_interval != np.inf):
                loss = self.loss(X, self.D, self.R)
                print("Iteration: %-4d | loss: %-10.3f" % (iter, loss))

                # Log size of matrices
                D_avg = np.mean(self.D)
                R_avg = np.mean(self.R)
                if self.verbose > 1:
                    print("                |  avgD: %-10.3f, avgR: %-10.3f" %
                          (D_avg, R_avg))
            next_D, next_R = self._update_DR(X)
            iter += 1

            # arrive at stable values, break the loop
            if self._terminate(X, self.D, self.R, next_D, next_R):
                break

            # else assign and continue to next iteration
            self.R = next_R
            self.D = next_D
        return self

    def _update_DR(self, X):
        """
        Default updating option is to update R then based on that update D

        Args:
            X: Input array
        """
        # get Rn+1 based on Rn, Dn
        next_R = self.get_next_R(X, self.D, self.R)
        # get Dn+1 based on Rn+1, Dn
        next_D = self.get_next_D(X, self.D, next_R)
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
        return np.allclose(self.D, next_D) and np.allclose(self.R, next_R)

    def transform(self, X, y=None):
        """
        Takes input and transform according to fitted model

        Args:
            X: Input array (shape m x n, where m is feature_size, n is sample size)
            y: not used

        Returns:
            The encoded array from X
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
            if self._terminate(X, self.D, R, self.D, next_R):
                break
            # else assign and continue to next iteration
            R = next_R
        if self.output_image:
            return self.D @ R
        return R

    def fit_transform(self, X, **fit_params):
        """
        Fit the model and transform the data to encoded state

        Args:
            X: Input array
            **fit_params: other parameters to fed into method fit.

        Returns:
            The representation array of X
        """
        self.fit(X, **fit_params)
        # Fit then return the lower dimension R values
        if self.output_image:
            return self.D @ self.R
        return self.R

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

    @property
    def components_(self):
        """
        This is added to make the class compatible with sklearn API

        Returns:
            The dictionary of model
        """
        return self.D

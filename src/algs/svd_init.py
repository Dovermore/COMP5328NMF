# This file holds SVD initialisation method for NMF
# Author: Calvin Huang (zhuq9812)
# Original: https://arxiv.org/pdf/1204.2311.pdf
import numpy as np


def svd_init(X, p=None, info=0.9):
    """
    Defines SVD init strategy for MNF algorithm

    Args:
        X: Input data
        p: Number of hidden_dimensions to use
        info: The mount of information ot retain

    Returns:

    """
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    if p is None:
        sum_s = np.sum(s)
        p = 0
        dsum = 0
        while dsum / sum_s < info and p < np.min(X.shape):
            dsum = dsum + s[p]
            p += 1
    W = np.abs(u[:, 1:p])
    H = np.abs((s*np.eye(s.shape[0]))[1:p, :] @ vh.T)

    return W, H

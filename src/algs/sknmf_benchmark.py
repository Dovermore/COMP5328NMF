from sklearn.decomposition import NMF


# Transpose X for correct shape
class ModifiedNMF:
    """
    This class is NOT used in the actual training, but rather converted to same format as
    the other NMF does to make sure the losses and accuracies are reasonable.
    """

    def __init__(self, *args, **kwargs):
        self.nmf = NMF(*args, **kwargs)

    def fit(self, X, *args, **kwargs):
        X = X.T
        self.nmf.fit(X, *args, **kwargs)
        return self

    def transform(self, X, *args, **kwargs):
        X = X.T
        return self.nmf.transform(X, *args, **kwargs).T

    def fit_transform(self, X, *args, **kwargs):
        X = X.T
        return self.nmf.fit_transform(X, *args, **kwargs).T

    @property
    def components_(self):
        return self.nmf.components_.T

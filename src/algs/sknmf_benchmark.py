from sklearn.decomposition import NMF


# Transpose X for correct shape
class ModifiedNMF:
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

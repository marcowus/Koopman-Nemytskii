import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

class RBFMap(BaseEstimator, TransformerMixin):
    def __init__(self, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.rbf_sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)

    def fit(self, X, y=None):
        self.rbf_sampler.fit(X)
        return self

    def transform(self, X):
        return self.rbf_sampler.transform(X)

class PolyMap(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)

    def fit(self, X, y=None):
        self.poly.fit(X)
        return self

    def transform(self, X):
        return self.poly.transform(X)

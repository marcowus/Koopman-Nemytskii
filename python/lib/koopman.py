import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from .features import RBFMap, PolyMap
from .rrr import ReducedRankRegressor

class KoopmanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel_type='rbf', rank=10, rbf_gamma=1.0, rbf_components=100, poly_degree=2, reg_alpha=1e-5):
        self.kernel_type = kernel_type
        self.rank = rank
        self.rbf_gamma = rbf_gamma
        self.rbf_components = rbf_components
        self.poly_degree = poly_degree
        self.reg_alpha = reg_alpha
        self.model = None
        self.feature_map_input = None
        self.feature_map_output = None
        self.C = None # Recovery matrix: Psi(x) -> x

    def fit(self, X, U, X_next):
        """
        Fit the Koopman operator and the recovery map.
        X: State at time k (N, n_x)
        U: Control at time k (N, n_u)
        X_next: State at time k+1 (N, n_x)
        """
        X = np.asarray(X)
        X_next = np.asarray(X_next)
        if U is not None:
            U = np.asarray(U)
            Z = np.hstack([X, U])
        else:
            Z = X

        # Initialize Feature Maps
        if self.kernel_type == 'rbf':
            self.feature_map_input = RBFMap(gamma=self.rbf_gamma, n_components=self.rbf_components, random_state=42)
            self.feature_map_output = RBFMap(gamma=self.rbf_gamma, n_components=self.rbf_components, random_state=42)
        elif self.kernel_type == 'poly':
            self.feature_map_input = PolyMap(degree=self.poly_degree)
            self.feature_map_output = PolyMap(degree=self.poly_degree)

        # Lift inputs Z -> Psi(Z)
        Psi_Z = self.feature_map_input.fit_transform(Z)

        # Lift targets X_next -> Psi(X_next)
        Psi_X_next = self.feature_map_output.fit_transform(X_next)

        # 1. Fit Koopman Operator (RRR): Psi(X_next) ~ Psi(Z) * K
        self.model = ReducedRankRegressor(rank=self.rank, alpha=self.reg_alpha)
        self.model.fit(Psi_Z, Psi_X_next)

        # 2. Fit Recovery Map: X_next ~ Psi(X_next) * C
        # We use Ridge regression for stability
        self.recovery_model = Ridge(alpha=1e-5, fit_intercept=False) # Simple linear map
        self.recovery_model.fit(Psi_X_next, X_next)

        return self

    def predict(self, X, U):
        """
        Predict the state x_{k+1} given x_k and u_k.
        """
        X = np.asarray(X)
        if U is not None:
            U = np.asarray(U)
            Z = np.hstack([X, U])
        else:
            Z = X

        # 1. Lift input
        Psi_Z = self.feature_map_input.transform(Z)

        # 2. Predict lifted state
        Psi_X_next_pred = self.model.predict(Psi_Z)

        # 3. Recover state
        X_next_pred = self.recovery_model.predict(Psi_X_next_pred)

        return X_next_pred

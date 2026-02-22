import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import svd

class ReducedRankRegressor(BaseEstimator, RegressorMixin):
    """
    Reduced Rank Ridge Regression (RRR)
    Minimize ||Y - X B||_F^2 + alpha * ||B||_F^2
    subject to rank(B) <= rank
    """
    def __init__(self, rank=None, alpha=1e-5):
        self.rank = rank
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self._fitted_rank = None

    def fit(self, X, Y):
        """
        Fit the model to data matrix X and target Y.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if self.rank is None:
            self._fitted_rank = min(X.shape[1], Y.shape[1])
        else:
            self._fitted_rank = self.rank

        # 1. Compute Ordinary Least Squares (or Ridge) solution
        # B_ols = (X^T X + alpha I)^-1 X^T Y
        # To avoid explicit inversion, use lstsq or solve
        if self.alpha > 0:
            # Regularized least squares
            # Solve (X^T X + alpha I) B = X^T Y
            # Or use augmented system
            n_features = X.shape[1]
            XTX = X.T @ X
            XTX_reg = XTX + self.alpha * np.eye(n_features)
            XTY = X.T @ Y
            try:
                B_ols = np.linalg.solve(XTX_reg, XTY)
            except np.linalg.LinAlgError:
                 # Fallback to lstsq if singular even with regularization
                 B_ols = np.linalg.lstsq(XTX_reg, XTY, rcond=None)[0]
        else:
            # Standard OLS
            B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]

        # 2. Compute fitted values Y_hat
        Y_hat = X @ B_ols

        # 3. Perform PCA on Y_hat to find the dominant subspace
        # We need the top r principal components of Y_hat
        # Y_hat = U S V^T
        # The projection matrix onto the subspace is V_r V_r^T
        # Note: numpy/scipy svd returns V^T as the third output (vh)
        # SVD of Y_hat: (m, p) -> U (m, m), S (min(m,p)), Vh (p, p)
        # We need V (p, p), so take Vh.T
        try:
            _, _, Vh = svd(Y_hat, full_matrices=False)
            V = Vh.T
        except np.linalg.LinAlgError:
            # Fallback if SVD fails
            print("Warning: SVD failed in RRR, falling back to OLS")
            self.coef_ = B_ols
            return self

        # Top r singular vectors
        Vr = V[:, :self._fitted_rank]

        # Projection matrix P = Vr Vr^T
        P = Vr @ Vr.T

        # 4. Compute Reduced Rank Coefficients
        # B_rrr = B_ols @ P
        self.coef_ = B_ols @ P

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet.")
        return X @ self.coef_

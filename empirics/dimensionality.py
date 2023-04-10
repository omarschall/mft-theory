import numpy as np

def compute_empirical_covariance(X):
    """The matrix X must be of shape (samples, features)"""

    n_samples = X.shape[0]
    cov = X.T.dot(X)/(n_samples - 1)

    return cov

def compute_participation_coefficient(cov):
    """Do this."""

    n_features = cov.shape[0]
    frob_norm = np.sum(np.square(cov))
    return np.trace(cov)**2 / frob_norm / n_features

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

def random_derangement(n):
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = np.random.randint(j+1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)

def compute_approximate_participation_coefficient(X, demean=True,
                                                  n_derangments=10):
    """From a data matrix X, compute the PR by estimating off-diagonals of
    X. X must have shape (samples, features)."""

    if demean:
        X = X - X.mean(0)

    pr = []
    for i in range(n_derangments):
        avg_X_var_squared = np.square((X * X).sum() / (X.shape[0] - 1))
        avg_X_squared_var = np.square((X * X).sum(0) / (X.shape[0] - 1)).sum()

        shuffle_idx = random_derangement(X.shape[1])
        psi_00_estimate = np.square((X * X[:, shuffle_idx]).sum(0) / (X.shape[0] - 1)).sum()

        pr.append(avg_X_var_squared / (avg_X_squared_var + X.shape[1] * psi_00_estimate) / X.shape[1])

    return np.mean(pr)
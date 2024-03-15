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

    psi_00_estimates = []
    avg_X_var_squared = np.square((X * X).sum() / (X.shape[0] - 1))
    avg_X_squared_var = np.square((X * X).sum(0) / (X.shape[0] - 1)).sum()
    for i in range(n_derangments):
        shuffle_idx = random_derangement(X.shape[1])
        psi_00_estimates.append(np.square((X * X[:, shuffle_idx]).sum(0) / (X.shape[0] - 1)).sum())
    psi_00 = np.mean(psi_00_estimates)
    pr = avg_X_var_squared / (avg_X_squared_var + X.shape[1] * psi_00) / X.shape[1]

    return pr

import numpy as np

def shared_component_variance_analysis(X, time_chunk_steps):
    """
    Help from Chat-GPT. Perform shared component variance analysis on the data matrix X.

    Parameters:
    - X: Data matrix of shape [time, units]
    - time_chunk_steps: Size of the chunks to split time points

    Returns:
    - other stuff
    """

    # Split data into two groups
    F = X[:, :X.shape[1] // 2]
    G = X[:, X.shape[1] // 2:]

    # Alternate chunks for training and testing
    n_chunks = X.shape[0] // time_chunk_steps
    train_indices = np.hstack([range(i * time_chunk_steps, (i + 1) * time_chunk_steps) for i in range(n_chunks) if i % 2 == 0])
    test_indices = np.hstack([range(i * time_chunk_steps, (i + 1) * time_chunk_steps) for i in range(n_chunks) if i % 2 != 0])
    F_train = F[train_indices]
    G_train = G[train_indices]
    F_test = F[test_indices]
    G_test = G[test_indices]

    # Compute the covariance matrix and its svd
    C = F_train.T.dot(G_train) / (F_train.shape[0] - 1)
    U, S, VT = np.linalg.svd(C)

    reliable_var = np.sum(U.dot(F_test.T) * ((VT.T.dot(G_test.T))), axis=1)/(F_test.shape[0]-1)

    return reliable_var




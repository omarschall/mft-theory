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

def shared_variance_components_analsys(X):
    """Do SVCA on an array X where we can split into test and train data points arbitrary (WLOG first and second halfs
    for each.)"""

    T, N = X.shape
    F_train, G_train = X[:T//2,:N//2], X[:T//2,N//2:]
    F_test, G_test = X[T//2:,:N//2], X[T//2:,N//2:]

    C = F_train.T.dot(G_train) / (T//2 - 1)
    U, S, VT = np.linalg.svd(C)
    print('done with svd)')
    Sk = np.einsum('ik, it, tj, kj -> k', U, F_test.T, G_test, VT)
    print('done with first Sk')
    Sk_tot_1 = np.einsum('ik, it, tj, jk -> k', U, F_test.T, F_test, U)
    print('done with Sk_tot_1')
    Sk_tot_2 = np.einsum('ki, it, tj, kj -> k', VT, G_test.T, G_test, VT)
    Sk_tot = Sk_tot_1 + Sk_tot_2
    percent_reliable_var = (Sk/(T//2)) / (Sk_tot/T)

    return percent_reliable_var

def compute_psi_tau(X, demean=True, n_derangements=10):
    """Compute the psi_tau for a data matrix X. X must have shape (samples, features)."""

    if demean:
        X = X - X.mean(0)

    psi_tau_estimates = []
    for i in range(n_derangements):
        shuffle_idx = random_derangement(X.shape[1])
        fourier_X = np.fft.rfft(X, axis=0, norm='ortho')
        fourier_cross_covs = np.conjugate(fourier_X) * fourier_X[:, shuffle_idx]
        psi_tau = (np.abs(np.fft.irfft(fourier_cross_covs, axis=0))**2).mean(1)
        psi_tau_estimates.append(psi_tau)
    psi_tau_estimates = np.array(psi_tau_estimates)

    return psi_tau_estimates.mean(0)
import numpy as np

def get_spectral_radius(M):
    """Calculates the spectral radius of a matrix."""

    eigs, _ = np.linalg.eig(M)

    return np.amax(np.absolute(eigs))

def norm(z):
    """Computes the L2 norm of a numpy array."""

    return np.sqrt(np.sum(np.square(z)))
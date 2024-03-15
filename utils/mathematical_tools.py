import numpy as np

def get_spectral_radius(M):
    """Calculates the spectral radius of a matrix."""

    eigs, _ = np.linalg.eig(M)

    return np.amax(np.absolute(eigs))

def norm(z):
    """Computes the L2 norm of a numpy array."""

    return np.sqrt(np.sum(np.square(z)))

import numpy as np

def generate_real_matrix(eigenvalues):
    # Initialize an empty block diagonal matrix
    B = np.zeros((0, 0))

    # Process each eigenvalue
    for eigenvalue in eigenvalues:
        if np.iscomplex(eigenvalue):
            # For complex eigenvalues, ensure they come in conjugate pairs and form a 2x2 block
            a = eigenvalue.real
            b = eigenvalue.imag
            block = np.array([[a, -b], [b, a]])
            # Append the block to the diagonal of B
            B = np.block([[B, np.zeros((B.shape[0], block.shape[1]))],
                          [np.zeros((block.shape[0], B.shape[1])), block]])
        else:
            # For real eigenvalues, form a 1x1 block
            block = np.array([[eigenvalue]])
            # Append the block to the diagonal of B
            B = np.block([[B, np.zeros((B.shape[0], block.shape[1]))],
                          [np.zeros((block.shape[0], B.shape[1])), block]])

    # Generate a random invertible matrix P
    P = np.random.randn(B.shape[0], B.shape[0])
    while np.linalg.det(P) == 0:
        P = np.random.randn(B.shape[0], B.shape[0])  # Ensure P is invertible

    # Compute the inverse of P
    P_inv = np.linalg.inv(P)

    # Apply the similarity transformation M = P^-1 * B * P
    M = P_inv @ B @ P

    return M

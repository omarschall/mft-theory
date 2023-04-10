import numpy as np

def psi_phi_in_fourier_space(omega_1, omega_2, alpha, nu, Delta_omega):
    """Calculate the cross-covariance in Fourier space for the activations, ie
    Eq. 6 in David's paper with a = phi.

    The discretization used in the fft of Delta_omega must match that of any
    vectorized inputs to the arguments omega_1 and omega_2."""

    gamma_phi = alpha**4
    Phi_phi = np.multiply.outer((1 + omega_1**2), (1 + omega_2**2)) / (nu**2)

    F1 = gamma_phi * (1 + nu**2 / (np.multiply.outer((1 + omega_1**2), (1 + omega_2**2)) - nu**2))
    F2 = 1 + 2 * np.real(nu / (np.multiply.outer((1 + (1j)*omega_1), (1 + (1j)*omega_2)) - nu)) * Phi_phi
    F3 = np.multiply.outer(Delta_omega, Delta_omega)

    return F1 * F2 * F3


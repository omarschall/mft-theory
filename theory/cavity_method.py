import numpy as np
from .basic_dmft import *
from .theory_utils import gaussian_integral
from functools import partial

def psi_phi_in_fourier_space(omega_1, omega_2, alpha, g, Delta_omega):
    """Calculate the cross-covariance in Fourier space for the activations, ie
    Eq. 6 in David's paper with a = phi.

    The discretization used in the fft of Delta_omega must match that of any
    vectorized inputs to the arguments omega_1 and omega_2."""

    nu = alpha**2 * g**2
    gamma_phi = alpha**4
    Phi_phi = np.multiply.outer((1 + omega_1**2), (1 + omega_2**2)) / (nu**2)

    F1 = gamma_phi * (1 + nu**2 / (np.multiply.outer((1 + omega_1**2), (1 + omega_2**2)) - nu**2))
    F2 = 1 + 2 * np.real(nu / (np.multiply.outer((1 + (1j)*omega_1), (1 + (1j)*omega_2)) - nu)) * Phi_phi
    F3 = np.multiply.outer(Delta_omega, Delta_omega)

    return F1 * F2 * F3

def compute_Delta_omega(g, T=10, dT=0.01):

    Delta_0 = solve_for_Delta_0(g=g, Delta_0_init=70)
    time, Delta_T, _ = solve_for_Delta_T(g=g, Delta_0=Delta_0, T=T, dT=dT)

    Delta_omega = np.fft.fft(Delta_T)
    omega = np.fft.fftfreq(n=int(T/dT),d=dT)
    return omega, Delta_omega

def phi_prime_x(z, Delta_0):

    return 1/np.cosh(np.sqrt(Delta_0)*z)**2 * np.sqrt(Delta_0)

#def compute_alpha(g):#
#
#    Delta_0 = solve_for_Delta_0(g=g, Delta_0_init=70)
#    f = partial(phi_prime_x, Delta_0=Delta_0)
#    return gaussian_integral(f)